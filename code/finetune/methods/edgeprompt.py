"""EdgePrompt finetuning method."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from code.finetune.encoders.prompt_encoder import resolve_edgeprompt_add_self_loops_from_cfg
from code.finetune.prompts.edgeprompt import EdgePrompt, EdgePromptPlus
from code.finetune.registry import register
from code.finetune.task_heads import TaskAwareObjective, build_task_aware_classifier
from code.finetune.task_base import FinetuneTask
from code.utils import compute_supervised_metrics


@register("edgeprompt")
class FinetuneEdgePrompt(FinetuneTask):
    """EdgePrompt / EdgePromptplus finetuning."""

    def __init__(self, cfg):
        super().__init__(cfg)
        ds_cfg = cfg.finetune.dataset
        self.task_level = ds_cfg.task_level
        self.task_level_raw = str(getattr(ds_cfg, "task_level_raw", ds_cfg.task_level) or ds_cfg.task_level).lower()
        hidden_dim = int(getattr(cfg.model, "hidden_dim", 1) or 1)
        self.repr_dim = int(getattr(cfg.model, "out_dim", hidden_dim) or hidden_dim)
        self.objective = TaskAwareObjective(cfg, task_level=self.task_level, repr_dim=self.repr_dim)
        self.num_classes = int(self.objective.num_classes or 2)
        self.task_type = self.objective.task_type
        self.label_dim = self.objective.label_dim

        method_cfg = getattr(cfg.finetune, "edgeprompt", None)
        self.method_cfg = method_cfg

        use_plus = bool(getattr(method_cfg, "plus", True)) if method_cfg is not None else True

        raw_anchors = getattr(method_cfg, "num_anchors", None) if method_cfg is not None else None
        num_anchors = None if raw_anchors is None else int(raw_anchors)
        if num_anchors is None:
            # Follow official defaults: node-style downstream uses more anchors.
            num_anchors = 10 if self.task_level_raw == "node" else 5

        raw_add_self_loops = getattr(method_cfg, "add_self_loops", None) if method_cfg is not None else None

        if raw_add_self_loops is None:
            # Official graph EdgePromptplus does not add self-loops in prompt construction.
            if self.task_level_raw == "graph":
                add_self_loops = False
            else:
                add_self_loops = resolve_edgeprompt_add_self_loops_from_cfg(cfg)
        else:
            add_self_loops = bool(raw_add_self_loops)

        model_name = str(getattr(getattr(cfg, "model", None), "name", "gcn") or "gcn").lower()
        if use_plus and model_name != "gcn" and add_self_loops:
            print(
                "[Finetune][EdgePrompt] Overriding finetune.edgeprompt.add_self_loops=True "
                "to False for non-GCN backbones."
            )
            add_self_loops = False

        self.force_mean_pooling = (
            bool(getattr(method_cfg, "force_mean_pooling", True)) if method_cfg is not None else True
        )
        self.pin_bn_eval_when_frozen = (
            bool(getattr(method_cfg, "pin_bn_eval_when_frozen", False)) if method_cfg is not None else False
        )

        dim_list = [cfg.model.in_dim] + [cfg.model.hidden_dim] * (cfg.model.num_layers - 1)
        if use_plus:
            self.prompt = EdgePromptPlus(
                dim_list=dim_list,
                num_anchors=num_anchors,
                add_self_loops=add_self_loops,
            )
        else:
            self.prompt = EdgePrompt(
                dim_list=dim_list,
                add_self_loops=add_self_loops,
            )
        self.prompt_type = "EdgePromptplus" if use_plus else "EdgePrompt"
        self.classifier = build_task_aware_classifier(
            input_dim=self.repr_dim,
            task_type=self.task_type,
            label_dim=self.label_dim,
            num_classes=self.objective.num_classes,
        )

    def parameters_to_optimize(self):
        return list(self.prompt.parameters()) + list(self.classifier.parameters())

    def build_optimizers(self, model: nn.Module):
        del model  # interface compatibility
        method_cfg = self.method_cfg
        lr = float(getattr(method_cfg, "lr", self.cfg.finetune.lr)) if method_cfg is not None else float(self.cfg.finetune.lr)
        weight_decay = (
            float(getattr(method_cfg, "weight_decay", self.cfg.finetune.weight_decay))
            if method_cfg is not None
            else float(self.cfg.finetune.weight_decay)
        )
        params = list(self.parameters_to_optimize())
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        return {"primary": optimizer}

    def _set_encoder_train_mode(self, model) -> None:
        """
        Keep dropout behavior for training.
        Optionally pin BatchNorm to eval when pretrained encoder weights are frozen.
        """
        model.train()
        if not bool(getattr(self.cfg.finetune, "freeze_pretrained", False)):
            return
        if not self.pin_bn_eval_when_frozen:
            return
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def _forward(
        self,
        model,
        data,
        device,
        mask_attr: str = "train_mask",
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        data = data.to(device)
        node_repr, graph_repr = model(data, prompt=self.prompt, prompt_type=self.prompt_type)
        pooling_mode = "mean" if self.force_mean_pooling else None
        if self.force_mean_pooling:
            graph_repr = None
        return self.objective.forward_with_model_outputs(
            classifier=self.classifier,
            node_repr=node_repr,
            graph_repr=graph_repr,
            data=data,
            device=device,
            mask_attr=mask_attr,
            graph_pooling_mode=pooling_mode,
            return_outputs=return_logits,
        )

    def train_epoch(self, model, loader, device, optimizers=None):
        self._set_encoder_train_mode(model)
        self.prompt.train()
        self.classifier.train()
        optimizer = optimizers.get("primary") if isinstance(optimizers, dict) else optimizers
        if optimizer is None:
            raise ValueError("Optimizer is required for EdgePrompt finetune.")

        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        for data in loader:
            optimizer.zero_grad()
            loss, acc = self._forward(model, data, device, mask_attr="train_mask")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc += float(acc)
            num_batches += 1

        if num_batches == 0:
            return 0.0, {}
        metric_name = "train_mae" if self.task_type == "regression" else "train_acc"
        return total_loss / num_batches, {metric_name: total_acc / num_batches}

    def evaluate_split(self, model, loader, device, prefix: str, mask_attr: str) -> Dict[str, float]:
        model.eval()
        self.prompt.eval()
        self.classifier.eval()
        total_loss = 0.0
        num_batches = 0
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for data in loader:
                loss, _acc, logits, labels = self._forward(
                    model,
                    data,
                    device,
                    mask_attr=mask_attr,
                    return_logits=True,
                )
                total_loss += loss.item()
                num_batches += 1
                all_logits.append(logits.detach().cpu())
                all_labels.append(labels.detach().cpu())

        if num_batches == 0:
            return {}

        metrics = {
            f"{prefix}_loss": total_loss / num_batches,
        }
        if all_logits and all_labels:
            logits = torch.cat(all_logits, dim=0)
            labels = torch.cat(all_labels, dim=0)
            task_type = str(getattr(self.cfg.finetune.dataset, "task_type", "classification") or "classification").lower()
            for key, value in compute_supervised_metrics(
                logits=logits,
                labels=labels,
                task_type=task_type,
            ).items():
                metrics[f"{prefix}_{key}"] = float(value)
        return metrics
