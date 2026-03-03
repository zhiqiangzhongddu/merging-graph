import torch
import torch.nn.functional as F
from torch import nn

from ..base import PretrainTask
from ..registry import register
from .utils import get_batch_vector, pool_nodes


@register("supervised")
class Supervised(PretrainTask):
    """Supervised pretraining task.
    
    Reference: Hu et al. "Strategies for Pre-training Graph Neural Networks" ICLR 2020.
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        ds_cfg = getattr(getattr(cfg, "pretrain", None), "dataset", None) or getattr(cfg, "dataset", None)
        self.task_level = ds_cfg.task_level
        self.task_type = str(getattr(ds_cfg, "task_type", "classification") or "classification").lower()
        self.label_dim = int(getattr(ds_cfg, "label_dim", 1) or 1)
        num_classes = int(getattr(ds_cfg, "num_classes", 1) or 1)
        if self.task_type == "regression":
            out_dim = 1
        elif self.label_dim > 1:
            # Multi-task binary classification (e.g., MoleculeNet).
            out_dim = self.label_dim
        elif str(self.task_level).lower() in {"graph", "edge"} and num_classes <= 2:
            # Binary graph/edge tasks are trained with a single-logit BCE head.
            out_dim = 1
        else:
            out_dim = num_classes
        self.classifier = nn.Linear(
            in_features=cfg.model.out_dim, 
            out_features=out_dim
        )

    @staticmethod
    def _prepare_labels(labels: torch.Tensor) -> torch.Tensor:
        return labels

    def _forward(self, model, data, device, mask_attr: str = "train_mask", return_outputs: bool = False):
        data = data.to(device)
        node_repr, graph_repr = model(data)

        if self.task_level == "node":
            logits = self.classifier(node_repr)
            mask = getattr(data, mask_attr, None)
            if mask is None:
                mask = torch.ones(data.num_nodes, dtype=torch.bool, device=device)
            logits_used = logits[mask]
            labels = self._prepare_labels(data.y[mask])
        elif self.task_level == "edge":
            edge_label_index = getattr(data, "edge_label_index", None)
            if edge_label_index is None:
                raise ValueError("Edge-level supervised pretraining requires edge_label_index.")
            src, dst = edge_label_index
            # Edge representation from endpoint interaction (pretrain-gnns style dot-product family).
            edge_repr = node_repr[src] * node_repr[dst]
            logits_used = self.classifier(edge_repr)
            edge_label = getattr(data, "edge_label", None)
            if edge_label is None:
                y = getattr(data, "y", None)
                if y is None:
                    raise ValueError("Edge-level supervised pretraining requires edge_label or y.")
                y = torch.as_tensor(y, device=device)
                if y.dim() > 1:
                    y = y.view(y.size(0), -1)[:, 0]
                y = y.view(-1)
                batch = getattr(data, "batch", None)
                if batch is not None and y.numel() > 1:
                    edge_batch = batch[src]
                    if int(edge_batch.max().item()) < y.numel():
                        edge_label = y[edge_batch]
                    else:
                        edge_label = y[:1].expand(edge_label_index.size(1))
                else:
                    edge_label = y[:1].expand(edge_label_index.size(1))
            labels = self._prepare_labels(edge_label)
        else:
            batch = get_batch_vector(data)
            if graph_repr is None:
                graph_repr = pool_nodes(
                    x=node_repr,
                    batch=batch,
                    mode=self.cfg.model.graph_pooling
                )
            logits_used = self.classifier(graph_repr)
            labels = self._prepare_labels(data.y)

        if logits_used.numel() == 0:
            zero = node_repr.sum() * 0.0
            if return_outputs:
                empty = logits_used.view(0, 1) if logits_used.dim() < 2 else logits_used
                return zero, 0.0, empty, torch.as_tensor(labels).view(-1)[:0]
            return zero, 0.0

        if self.task_type == "regression":
            pred = logits_used.view(-1).float()
            target = torch.as_tensor(labels).view(-1).float()
            loss = F.mse_loss(pred, target)
            mae = float((pred - target).abs().mean().item())
            if return_outputs:
                return loss, mae, pred, target
            return loss, mae

        labels_tensor = torch.as_tensor(labels)
        # Multi-task binary classification with optional missing labels:
        # - labels in {-1, 1} (or {-1, 0, 1}) are mapped to {0, 1}
        # - 0 is treated as missing only when {-1, 1} encoding is detected.
        if labels_tensor.dim() > 1 and labels_tensor.size(-1) > 1:
            targets = labels_tensor.float()
            valid = torch.isfinite(targets)
            uses_signed = bool((targets < 0).any().item())
            if uses_signed:
                valid = valid & (targets != 0)
                targets = (targets + 1.0) / 2.0
            targets = targets.clamp(min=0.0, max=1.0)

            loss_mat = F.binary_cross_entropy_with_logits(
                logits_used.float(),
                targets,
                reduction="none",
            )
            valid_f = valid.float()
            denom = valid_f.sum().clamp(min=1.0)
            loss = (loss_mat * valid_f).sum() / denom

            probs = torch.sigmoid(logits_used.float())
            pred = (probs >= 0.5).float()
            acc = float(((pred == targets).float() * valid_f).sum().item() / float(denom.item()))
            if return_outputs:
                return loss, acc, logits_used, labels_tensor
            return loss, acc

        # Binary single-target classification (logit shape [N, 1]).
        if logits_used.dim() == 1 or (logits_used.dim() == 2 and logits_used.size(-1) == 1):
            logits_vec = logits_used.view(-1).float()
            targets = labels_tensor.view(-1).float()
            valid = torch.isfinite(targets)
            uses_signed = bool((targets < 0).any().item())
            if uses_signed:
                valid = valid & (targets != 0)
                targets = (targets + 1.0) / 2.0
            targets = targets.clamp(min=0.0, max=1.0)
            if valid.any():
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

        class_labels = labels_tensor
        if class_labels.dim() > 1:
            class_labels = class_labels.view(class_labels.size(0), -1)[:, 0]
        else:
            class_labels = class_labels.view(-1)
        if class_labels.dtype.is_floating_point:
            rounded = class_labels.round()
            if torch.allclose(class_labels, rounded, atol=1e-6):
                class_labels = rounded
            else:
                class_labels = (class_labels > 0.5).to(class_labels.dtype)
        class_labels = class_labels.long()
        loss = F.cross_entropy(logits_used, class_labels)
        pred = logits_used.argmax(dim=-1)
        acc = float((pred == class_labels).float().mean().item())
        if return_outputs:
            return loss, acc, logits_used, class_labels
        return loss, acc

    def step(self, model, data, device):
        loss, primary = self._forward(model=model, data=data, device=device, mask_attr="train_mask")
        if self.task_type == "regression":
            return loss, {"train_mae": primary}
        return loss, {"train_acc": primary}

    def evaluate(self, model, data, device):
        loss, primary = self._forward(model=model, data=data, device=device, mask_attr="val_mask")
        if self.task_type == "regression":
            return loss, {"val_mae": primary}
        return loss, {"val_acc": primary}
