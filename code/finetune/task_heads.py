"""Shared task-aware prediction helpers for finetuning methods."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from code.pretrain.methods.utils import get_batch_vector, pool_nodes


def resolve_task_output_dim(
    *,
    task_type: str,
    label_dim: int,
    num_classes: int | None,
) -> int:
    """Resolve the prediction dimension for the finetune task."""
    task = str(task_type or "classification").lower()
    label_dim = max(1, int(label_dim or 1))
    if task == "regression":
        return label_dim
    if label_dim > 1:
        return label_dim
    return max(2, int(num_classes or 1))


def build_task_aware_classifier(
    *,
    input_dim: int,
    task_type: str,
    label_dim: int,
    num_classes: int | None,
) -> nn.Linear:
    """Construct a linear head sized for the finetune task."""
    return nn.Linear(
        in_features=int(input_dim),
        out_features=resolve_task_output_dim(
            task_type=task_type,
            label_dim=label_dim,
            num_classes=num_classes,
        ),
    )


class TaskAwareObjective:
    """Task-aware label shaping, loss computation, and model-output selection."""

    def __init__(
        self,
        cfg,
        *,
        task_level: str | None = None,
        task_type: str | None = None,
        label_dim: int | None = None,
        num_classes: int | None = None,
        repr_dim: int | None = None,
    ):
        self.cfg = cfg
        ds_cfg = getattr(getattr(cfg, "finetune", None), "dataset", None) or getattr(cfg, "dataset", None)
        self.task_level = str(task_level or getattr(ds_cfg, "task_level", "graph") or "graph").lower()
        self.task_type = str(task_type or getattr(ds_cfg, "task_type", "classification") or "classification").lower()
        self.label_dim = max(1, int(label_dim if label_dim is not None else (getattr(ds_cfg, "label_dim", 1) or 1)))
        raw_num_classes = num_classes if num_classes is not None else getattr(ds_cfg, "num_classes", None)
        self.num_classes = None if raw_num_classes in (None, "") else int(raw_num_classes)
        self.repr_dim = int(repr_dim) if repr_dim is not None else None
        self.output_dim = resolve_task_output_dim(
            task_type=self.task_type,
            label_dim=self.label_dim,
            num_classes=self.num_classes,
        )

    @property
    def is_single_label_classification(self) -> bool:
        return self.task_type == "classification" and self.label_dim <= 1

    @property
    def is_multilabel_classification(self) -> bool:
        return self.task_type == "classification" and self.label_dim > 1

    @property
    def primary_metric_name(self) -> str:
        return "mae" if self.task_type == "regression" else "acc"

    @staticmethod
    def _align_last_dim(x: torch.Tensor, target_dim: int | None) -> torch.Tensor:
        if target_dim is None:
            return x
        current_dim = int(x.size(-1))
        if current_dim == target_dim:
            return x
        if current_dim > target_dim:
            return x[:, :target_dim]
        pad = x.new_zeros((x.size(0), target_dim - current_dim))
        return torch.cat([x, pad], dim=-1)

    @staticmethod
    def _reshape_targets_with_batch_size(labels: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(labels)
        if tensor.dim() == 0:
            tensor = tensor.view(1, 1)
        elif tensor.dim() == 1:
            if batch_size > 0 and tensor.numel() % batch_size == 0:
                tensor = tensor.view(batch_size, -1)
            else:
                tensor = tensor.view(-1, 1)
        else:
            if tensor.size(0) != batch_size and batch_size > 0 and tensor.numel() % batch_size == 0:
                tensor = tensor.view(batch_size, -1)
            elif tensor.size(0) == batch_size:
                tensor = tensor.view(batch_size, -1)
        if tensor.size(0) != batch_size:
            raise ValueError(
                f"Unable to align {name} with batch size {batch_size}: got shape {tuple(tensor.shape)}."
            )
        return tensor

    @staticmethod
    def _binary_targets_and_valid(labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target = torch.as_tensor(labels).float()
        valid = torch.isfinite(target)
        uses_signed = bool((target < 0).any().item()) if target.numel() else False
        if uses_signed:
            valid = valid & (target != 0)
            target = (target + 1.0) / 2.0
        target = target.clamp(min=0.0, max=1.0)
        return target, valid

    def _prepare_single_label_targets(self, labels: torch.Tensor, batch_size: int) -> torch.Tensor:
        tensor = torch.as_tensor(labels)
        if tensor.dim() > 1:
            tensor = self._reshape_targets_with_batch_size(tensor, batch_size, "labels")[:, 0]
        else:
            tensor = tensor.view(-1)
        if tensor.numel() != batch_size:
            raise ValueError(
                f"Expected {batch_size} single-label targets, got {tensor.numel()} elements."
            )
        if tensor.dtype.is_floating_point:
            rounded = tensor.round()
            if torch.allclose(tensor, rounded, atol=1e-6):
                tensor = rounded
            else:
                tensor = (tensor > 0.5).to(tensor.dtype)
        return tensor.long()

    def select_representations_and_labels(
        self,
        *,
        node_repr: torch.Tensor,
        graph_repr: torch.Tensor | None,
        data,
        device: torch.device,
        mask_attr: str = "train_mask",
        graph_pooling_mode: str | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_repr = self._align_last_dim(node_repr, self.repr_dim)
        if graph_repr is not None:
            graph_repr = self._align_last_dim(graph_repr, self.repr_dim)

        if self.task_level == "node":
            mask = getattr(data, mask_attr, None)
            if mask is None:
                mask = torch.ones(data.num_nodes, dtype=torch.bool, device=device)
            else:
                mask = torch.as_tensor(mask, dtype=torch.bool, device=device).view(-1)
            return node_repr[mask], torch.as_tensor(data.y)[mask]

        batch = get_batch_vector(data)
        if graph_repr is None:
            graph_repr = pool_nodes(
                x=node_repr,
                batch=batch,
                mode=graph_pooling_mode or self.cfg.model.graph_pooling,
                data=data,
            )
        graph_repr = self._align_last_dim(graph_repr, self.repr_dim)
        return graph_repr, torch.as_tensor(data.y)

    def loss_from_logits(
        self,
        *,
        logits: torch.Tensor,
        labels: torch.Tensor,
        return_outputs: bool = False,
    ):
        if self.task_type == "regression":
            preds = self._reshape_targets_with_batch_size(logits, int(logits.size(0)), "logits").float()
            target = self._reshape_targets_with_batch_size(labels, int(logits.size(0)), "labels").float().to(preds.device)
            valid = torch.isfinite(target)
            safe_target = torch.where(valid, target, torch.zeros_like(target))
            diff = preds - safe_target
            valid_f = valid.float()
            denom = valid_f.sum().clamp(min=1.0)
            loss = ((diff * diff) * valid_f).sum() / denom
            mae = float((diff.abs() * valid_f).sum().item() / float(denom.item()))
            pred_out = preds.view(-1) if preds.size(1) == 1 else preds
            target_out = target.view(-1) if target.size(1) == 1 else target
            if return_outputs:
                return loss, mae, pred_out, target_out
            return loss, mae

        logits_mat = self._reshape_targets_with_batch_size(logits, int(logits.size(0)), "logits")
        labels_tensor = torch.as_tensor(labels, device=logits_mat.device)

        if self.is_multilabel_classification:
            targets = self._reshape_targets_with_batch_size(labels_tensor, int(logits_mat.size(0)), "labels").float()
            targets, valid = self._binary_targets_and_valid(targets)
            targets = targets.to(logits_mat.device)
            valid = valid.to(logits_mat.device)
            safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
            loss_mat = F.binary_cross_entropy_with_logits(
                logits_mat.float(),
                safe_targets,
                reduction="none",
            )
            valid_f = valid.float()
            denom = valid_f.sum().clamp(min=1.0)
            loss = (loss_mat * valid_f).sum() / denom
            probs = torch.sigmoid(logits_mat.float())
            pred = (probs >= 0.5).float()
            acc = float((((pred == targets).float() * valid_f).sum() / denom).item())
            if return_outputs:
                return loss, acc, logits_mat, labels_tensor
            return loss, acc

        if logits_mat.size(1) == 1:
            targets = self._reshape_targets_with_batch_size(labels_tensor, int(logits_mat.size(0)), "labels").view(-1).float()
            targets, valid = self._binary_targets_and_valid(targets)
            targets = targets.to(logits_mat.device)
            valid = valid.to(logits_mat.device)
            logits_vec = logits_mat.view(-1).float()
            if bool(valid.any().item()):
                loss = F.binary_cross_entropy_with_logits(logits_vec[valid], targets[valid])
                pred = (torch.sigmoid(logits_vec[valid]) >= 0.5).long()
                truth = targets[valid].long()
                acc = float((pred == truth).float().mean().item())
            else:
                loss = logits_vec.sum() * 0.0
                acc = 0.0
            if return_outputs:
                return loss, acc, logits_vec.unsqueeze(-1), labels_tensor.view(-1)
            return loss, acc

        class_labels = self._prepare_single_label_targets(labels_tensor, int(logits_mat.size(0))).to(logits_mat.device)
        loss = F.cross_entropy(logits_mat, class_labels)
        pred = logits_mat.argmax(dim=-1)
        acc = float((pred == class_labels).float().mean().item())
        if return_outputs:
            return loss, acc, logits_mat, class_labels
        return loss, acc

    def forward_with_classifier(
        self,
        *,
        classifier: nn.Module,
        representations: torch.Tensor,
        labels: torch.Tensor,
        input_dim: int | None = None,
        return_outputs: bool = False,
    ):
        reps = self._align_last_dim(representations, input_dim or self.repr_dim)
        logits = classifier(reps)
        return self.loss_from_logits(logits=logits, labels=labels, return_outputs=return_outputs)

    def forward_with_model_outputs(
        self,
        *,
        classifier: nn.Module,
        node_repr: torch.Tensor,
        graph_repr: torch.Tensor | None,
        data,
        device: torch.device,
        mask_attr: str = "train_mask",
        graph_pooling_mode: str | None = None,
        return_outputs: bool = False,
    ):
        representations, labels = self.select_representations_and_labels(
            node_repr=node_repr,
            graph_repr=graph_repr,
            data=data,
            device=device,
            mask_attr=mask_attr,
            graph_pooling_mode=graph_pooling_mode,
        )
        return self.forward_with_classifier(
            classifier=classifier,
            representations=representations,
            labels=labels,
            input_dim=getattr(classifier, "in_features", self.repr_dim),
            return_outputs=return_outputs,
        )


__all__ = [
    "TaskAwareObjective",
    "build_task_aware_classifier",
    "resolve_task_output_dim",
]
