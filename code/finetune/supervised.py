import torch
import torch.nn.functional as F
from torch import nn

from code.pretrain.base import PretrainTask
from code.pretrain.methods.utils import get_batch_vector, pool_nodes


class FinetuneSupervised(PretrainTask):
    """Supervised finetuning task that uses the finetune dataset config for heads."""

    def __init__(self, cfg):
        super().__init__(cfg)
        ds_cfg = getattr(getattr(cfg, "finetune", None), "dataset", None) or getattr(cfg, "dataset", None)
        self.task_level = ds_cfg.task_level
        self.task_type = str(getattr(ds_cfg, "task_type", "classification") or "classification").lower()
        self.label_dim = int(getattr(ds_cfg, "label_dim", 1) or 1)
        hidden_dim = int(getattr(cfg.model, "hidden_dim", 1) or 1)
        out_repr_dim = int(getattr(cfg.model, "out_dim", hidden_dim) or hidden_dim)
        self.repr_dim = out_repr_dim

        num_classes = int(getattr(ds_cfg, "num_classes", 1) or 1)
        if self.task_type == "regression":
            out_dim = 1
        elif self.label_dim > 1:
            out_dim = self.label_dim
        else:
            # Single-label classification uses a multi-class CE head,
            # including binary tasks.
            out_dim = max(2, num_classes)
        self.classifier = nn.Linear(
            in_features=self.repr_dim,
            out_features=out_dim,
        )

    @staticmethod
    def _prepare_labels(labels: torch.Tensor) -> torch.Tensor:
        return labels

    @staticmethod
    def _align_last_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
        current_dim = int(x.size(-1))
        if current_dim == target_dim:
            return x
        if current_dim > target_dim:
            return x[:, :target_dim]
        pad = x.new_zeros((x.size(0), target_dim - current_dim))
        return torch.cat([x, pad], dim=-1)

    def _forward(self, model, data, device, mask_attr: str = "train_mask", return_outputs: bool = False):
        data = data.to(device)
        node_repr, graph_repr = model(data)
        node_repr = self._align_last_dim(node_repr, self.repr_dim)
        if graph_repr is not None:
            graph_repr = self._align_last_dim(graph_repr, self.repr_dim)

        if self.task_level == "node":
            logits = self.classifier(node_repr)
            mask = getattr(data, mask_attr, None)
            if mask is None:
                mask = torch.ones(data.num_nodes, dtype=torch.bool, device=device)
            logits_used = logits[mask]
            labels = self._prepare_labels(data.y[mask])
        else:
            batch = get_batch_vector(data)
            if graph_repr is None:
                graph_repr = pool_nodes(
                    x=node_repr,
                    batch=batch,
                    mode=self.cfg.model.graph_pooling,
                    data=data,
                )
            graph_repr = self._align_last_dim(graph_repr, self.repr_dim)
            logits_used = self.classifier(graph_repr)
            labels = self._prepare_labels(data.y)

        if self.task_type == "regression":
            pred = logits_used.view(-1).float()
            target = torch.as_tensor(labels).view(-1).float()
            loss = F.mse_loss(pred, target)
            mae = float((pred - target).abs().mean().item())
            if return_outputs:
                return loss, mae, pred, target
            return loss, mae

        labels_tensor = torch.as_tensor(labels)
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
