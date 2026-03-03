"""EdgePrompt finetuning method."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from code.finetune.encoders.prompt_encoder import resolve_edgeprompt_add_self_loops_from_cfg
from code.finetune.prompts.edgeprompt import EdgePrompt, EdgePromptPlus
from code.finetune.registry import register
from code.finetune.task_base import FinetuneTask
from code.pretrain.methods.utils import get_batch_vector, pool_nodes
from code.utils import compute_supervised_metrics


@register("edgeprompt")
class FinetuneEdgePrompt(FinetuneTask):
    """EdgePrompt / EdgePromptplus finetuning."""

    def __init__(self, cfg):
        super().__init__(cfg)
        ds_cfg = cfg.finetune.dataset
        self.task_level = ds_cfg.task_level
        self.task_level_raw = str(getattr(ds_cfg, "task_level_raw", ds_cfg.task_level) or ds_cfg.task_level).lower()
        self.num_classes = int(getattr(ds_cfg, "num_classes", 2) or 2)
        self.task_type = str(getattr(ds_cfg, "task_type", "classification") or "classification").lower()

        if self.task_type != "classification":
            raise ValueError("EdgePrompt currently supports classification tasks only.")

        method_cfg = getattr(cfg.finetune, "edgeprompt", None)
        legacy_cfg = getattr(cfg.finetune, "prompt", None)
        self.method_cfg = method_cfg

        use_plus = True
        if method_cfg is not None and hasattr(method_cfg, "plus"):
            use_plus = bool(getattr(method_cfg, "plus"))
        elif legacy_cfg is not None and hasattr(legacy_cfg, "edgeprompt_plus"):
            use_plus = bool(getattr(legacy_cfg, "edgeprompt_plus"))

        raw_anchors = None
        if method_cfg is not None and hasattr(method_cfg, "num_anchors"):
            raw_anchors = getattr(method_cfg, "num_anchors")
        elif legacy_cfg is not None and hasattr(legacy_cfg, "edgeprompt_num_anchors"):
            raw_anchors = getattr(legacy_cfg, "edgeprompt_num_anchors")
        num_anchors = None if raw_anchors is None else int(raw_anchors)
        if num_anchors is None:
            # Follow official defaults: node-style downstream uses more anchors.
            num_anchors = 10 if self.task_level_raw == "node" else 5

        raw_add_self_loops = None
        if method_cfg is not None and hasattr(method_cfg, "add_self_loops"):
            raw_add_self_loops = getattr(method_cfg, "add_self_loops")
        elif legacy_cfg is not None and hasattr(legacy_cfg, "edgeprompt_add_self_loops"):
            raw_add_self_loops = getattr(legacy_cfg, "edgeprompt_add_self_loops")

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
        self.classifier = nn.Linear(cfg.model.out_dim, self.num_classes)

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

        if self.task_level == "node":
            logits = self.classifier(node_repr)
            mask = getattr(data, mask_attr, None)
            if mask is None:
                mask = torch.ones(data.num_nodes, dtype=torch.bool, device=device)
            labels = data.y[mask].long()
            loss = F.cross_entropy(logits[mask], labels)
            pred = logits.argmax(dim=-1)
            acc = (pred[mask] == labels).float().mean().item()
            if return_logits:
                return loss, acc, logits[mask], labels
            return loss, acc

        batch = get_batch_vector(data)
        if self.force_mean_pooling:
            graph_repr = pool_nodes(
                node_repr,
                batch,
                mode="mean",
                data=data,
            )
        elif graph_repr is None:
            graph_repr = pool_nodes(
                node_repr,
                batch,
                mode=self.cfg.model.graph_pooling,
                data=data,
            )
        logits = self.classifier(graph_repr)
        labels = data.y.view(-1).long()
        loss = F.cross_entropy(logits, labels)
        pred = logits.argmax(dim=-1)
        acc = (pred == labels).float().mean().item()
        if return_logits:
            return loss, acc, logits, labels
        return loss, acc

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
        return total_loss / num_batches, {"train_acc": total_acc / num_batches}

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
