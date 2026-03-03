"""GPF prompt finetuning method."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from code.finetune.prompts.gpf import GPFPlusPrompt, GPFPrompt
from code.finetune.registry import register
from code.finetune.supervised import FinetuneSupervised
from code.finetune.task_base import FinetuneTask
from code.utils import compute_supervised_metrics


@register("gpf")
class FinetuneGPF(FinetuneTask):
    """GPF/GPF-plus finetuning with a prompted input and supervised task head."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.supervised_head = FinetuneSupervised(cfg)
        self.task_type = str(getattr(self.supervised_head, "task_type", "classification") or "classification").lower()
        self.task_level = str(getattr(self.supervised_head, "task_level", "graph") or "graph").lower()
        if self.task_level == "edge":
            raise ValueError("GPF requires node/graph batches. Set finetune.dataset.induced=True for edge tasks.")

        method_cfg = getattr(cfg.finetune, "gpf", None)
        plus_flag = self._to_bool(getattr(method_cfg, "plus", False)) if method_cfg else False
        legacy_variant = (
            str(
                getattr(
                    method_cfg,
                    "variant",
                    getattr(method_cfg, "tuning_type", getattr(method_cfg, "prompt_type", "gpf")),
                )
            )
            .lower()
            .replace("-", "_")
        )
        variant = "gpf_plus" if plus_flag else legacy_variant
        p_num = int(getattr(method_cfg, "p_num", getattr(method_cfg, "pnum", 20))) if method_cfg else 20

        self.prompt_in_dim = int(getattr(cfg.model, "in_dim", 0) or 0)
        if self.prompt_in_dim <= 0:
            raise ValueError("GPF requires model.in_dim > 0.")

        if variant in {"gpf_plus", "gpfplus", "gpf_plus_att", "gpfplusatt"}:
            self.prompt = GPFPlusPrompt(in_channels=self.prompt_in_dim, p_num=p_num)
            self.prompt_variant = "gpf_plus"
        elif variant == "gpf":
            self.prompt = GPFPrompt(in_channels=self.prompt_in_dim)
            self.prompt_variant = "gpf"
        else:
            raise ValueError(f"Unknown gpf variant: {variant}")

        # Official GPF scripts tune the prediction head depth (num_layers).
        self.head_layers = (
            int(getattr(method_cfg, "head_layers", getattr(method_cfg, "num_layers", 1)))
            if method_cfg is not None
            else 1
        )
        self.head_layers = max(1, self.head_layers)
        self.head_hidden_dim = (
            int(getattr(method_cfg, "head_hidden_dim", 0))
            if method_cfg is not None
            else 0
        )
        self.head_dropout = (
            float(getattr(method_cfg, "head_dropout", 0.0))
            if method_cfg is not None
            else 0.0
        )
        self.update_pretrained = self._resolve_update_pretrained(method_cfg)
        self.freeze_encoder_bn_when_frozen = (
            self._to_bool(getattr(method_cfg, "freeze_encoder_bn_when_frozen", True))
            if method_cfg is not None
            else True
        )
        self._encoder_frozen_notice_printed = False
        self._maybe_upgrade_prediction_head()

    @staticmethod
    def _to_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"1", "true", "yes", "y", "on"}:
                return True
            if token in {"0", "false", "no", "n", "off"}:
                return False
        return bool(value)

    def _resolve_update_pretrained(self, method_cfg) -> bool:
        raw = None
        if method_cfg is not None and hasattr(method_cfg, "update_pretrained"):
            raw = getattr(method_cfg, "update_pretrained")
        if raw is None:
            return not bool(getattr(self.cfg.finetune, "freeze_pretrained", False))
        return self._to_bool(raw)

    def _maybe_upgrade_prediction_head(self) -> None:
        if self.head_layers <= 1:
            return
        classifier = self.supervised_head.classifier
        if not isinstance(classifier, nn.Linear):
            return

        in_dim = int(classifier.in_features)
        out_dim = int(classifier.out_features)
        hidden_dim = int(self.head_hidden_dim) if int(self.head_hidden_dim) > 0 else in_dim

        layers = []
        for idx in range(self.head_layers - 1):
            input_dim = in_dim if idx == 0 else hidden_dim
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if self.head_dropout > 0:
                layers.append(nn.Dropout(self.head_dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.supervised_head.classifier = nn.Sequential(*layers)

    def parameters_to_optimize(self):
        return list(self.prompt.parameters()) + list(self.supervised_head.classifier.parameters())

    def build_optimizers(self, model):
        method_cfg = getattr(self.cfg.finetune, "gpf", None)
        base_lr = float(getattr(method_cfg, "lr", self.cfg.finetune.lr)) if method_cfg else float(self.cfg.finetune.lr)
        base_wd = (
            float(getattr(method_cfg, "weight_decay", self.cfg.finetune.weight_decay))
            if method_cfg
            else float(self.cfg.finetune.weight_decay)
        )
        prompt_lr = float(getattr(method_cfg, "prompt_lr", base_lr)) if method_cfg else base_lr
        prompt_wd = float(getattr(method_cfg, "prompt_weight_decay", base_wd)) if method_cfg else base_wd
        head_lr_scale = float(getattr(method_cfg, "head_lr_scale", 1.0)) if method_cfg else 1.0
        head_lr = float(getattr(method_cfg, "head_lr", base_lr * head_lr_scale)) if method_cfg else base_lr * head_lr_scale
        head_wd = float(getattr(method_cfg, "head_weight_decay", base_wd)) if method_cfg else base_wd
        encoder_lr = float(getattr(method_cfg, "encoder_lr", base_lr)) if method_cfg else base_lr
        encoder_wd = float(getattr(method_cfg, "encoder_weight_decay", base_wd)) if method_cfg else base_wd
        optimizer_name = str(getattr(method_cfg, "optimizer", "adam") if method_cfg else "adam").lower()

        if not self.update_pretrained:
            # Keep behavior consistent with official prompt-only tuning even
            # when global freeze_pretrained is disabled.
            frozen = 0
            for param in model.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen += param.numel()
            if frozen > 0 and not self._encoder_frozen_notice_printed:
                print(f"[Finetune][gpf] Frozen encoder parameters for prompt-only tuning (count={frozen}).")
                self._encoder_frozen_notice_printed = True

        param_groups = [
            {
                "params": list(self.prompt.parameters()),
                "lr": prompt_lr,
                "weight_decay": prompt_wd,
            },
            {
                "params": list(self.supervised_head.classifier.parameters()),
                "lr": head_lr,
                "weight_decay": head_wd,
            },
        ]
        if self.update_pretrained:
            encoder_params = [param for param in model.parameters() if param.requires_grad]
            if encoder_params:
                param_groups.append(
                    {
                        "params": encoder_params,
                        "lr": encoder_lr,
                        "weight_decay": encoder_wd,
                    }
                )

        if optimizer_name == "adam":
            optimizer_cls = torch.optim.Adam
        elif optimizer_name == "adamw":
            optimizer_cls = torch.optim.AdamW
        else:
            raise ValueError("gpf.optimizer must be one of: adam, adamw.")

        optimizer = optimizer_cls(param_groups)
        return {"primary": optimizer}

    def _set_encoder_train_mode(self, model) -> None:
        if self.update_pretrained:
            model.train()
            return

        model.train()
        if not self.freeze_encoder_bn_when_frozen:
            return
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def _apply_prompt(self, data, device):
        data = data.to(device)
        x = getattr(data, "x", None)
        if x is None:
            raise ValueError("GPF requires node features in `data.x`.")
        if int(x.size(-1)) != self.prompt_in_dim:
            raise ValueError(
                f"GPF prompt dim mismatch: expected {self.prompt_in_dim}, got {int(x.size(-1))}. "
                "Ensure finetune dataset features match pretrained model input dimension."
            )
        prompted = data.clone()
        prompted.x = self.prompt.add(prompted.x)
        return prompted

    def train_epoch(self, model, loader, device, optimizers=None):
        optimizer = optimizers.get("primary") if isinstance(optimizers, dict) else optimizers
        if optimizer is None:
            raise ValueError("GPF requires an optimizer.")

        self._set_encoder_train_mode(model)
        self.train()
        total_loss = 0.0
        total_primary = 0.0
        num_batches = 0

        for data in loader:
            optimizer.zero_grad()
            prompted = self._apply_prompt(data, device)
            loss, primary = self.supervised_head._forward(
                model=model,
                data=prompted,
                device=device,
                mask_attr="train_mask",
            )
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_primary += float(primary)
            num_batches += 1

        if num_batches == 0:
            return 0.0, {}

        avg_loss = total_loss / num_batches
        avg_primary = total_primary / num_batches
        if self.task_type == "regression":
            return avg_loss, {"train_mae": avg_primary}
        return avg_loss, {"train_acc": avg_primary}

    def evaluate_split(self, model, loader, device, prefix: str, mask_attr: str) -> Dict[str, float]:
        model.eval()
        self.eval()
        total_loss = 0.0
        num_batches = 0
        logits_buffer = []
        labels_buffer = []

        with torch.no_grad():
            for data in loader:
                prompted = self._apply_prompt(data, device)
                loss, _primary, logits, labels = self.supervised_head._forward(
                    model=model,
                    data=prompted,
                    device=device,
                    mask_attr=mask_attr,
                    return_outputs=True,
                )
                total_loss += float(loss.item())
                num_batches += 1
                if logits is not None and labels is not None:
                    logits_buffer.append(torch.as_tensor(logits).detach().cpu())
                    labels_buffer.append(torch.as_tensor(labels).detach().cpu())

        if num_batches == 0:
            return {}

        metrics = {
            f"{prefix}_loss": total_loss / num_batches,
        }
        if logits_buffer and labels_buffer:
            logits = torch.cat(
                [x.view(-1, 1) if x.dim() == 1 else x.view(x.size(0), -1) for x in logits_buffer],
                dim=0,
            )
            if any(y.dim() > 1 for y in labels_buffer):
                labels = torch.cat(
                    [y.view(-1, 1) if y.dim() == 1 else y.view(y.size(0), -1) for y in labels_buffer],
                    dim=0,
                )
                if labels.dim() == 2 and labels.size(1) == 1:
                    labels = labels.view(-1)
            else:
                labels = torch.cat([y.view(-1) for y in labels_buffer], dim=0)

            for key, value in compute_supervised_metrics(
                logits=logits,
                labels=labels,
                task_type=self.task_type,
            ).items():
                metrics[f"{prefix}_{key}"] = float(value)
        return metrics
