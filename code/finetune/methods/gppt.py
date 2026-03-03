"""GPPT prompt finetuning method."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from code.finetune.prompts.gppt import GPPTPrompt
from code.finetune.registry import register
from code.finetune.task_base import FinetuneTask
from code.pretrain.methods.utils import get_batch_vector
from code.utils import compute_supervised_metrics


@register("gppt")
class FinetuneGPPT(FinetuneTask):
    """Official GPPT-style prompt tuning with fixed structure/task tokens."""

    def __init__(self, cfg):
        super().__init__(cfg)
        ds_cfg = cfg.finetune.dataset
        self.task_level = str(getattr(ds_cfg, "task_level", "graph") or "graph").lower()
        self.task_type = str(getattr(ds_cfg, "task_type", "classification") or "classification").lower()
        self.label_dim = int(getattr(ds_cfg, "label_dim", 1) or 1)
        self.num_classes = max(2, int(getattr(ds_cfg, "num_classes", 2) or 2))

        if self.task_type != "classification":
            raise ValueError("GPPT currently supports classification tasks only.")
        if self.label_dim > 1:
            raise ValueError("GPPT currently supports single-label classification only.")
        if self.task_level == "edge":
            raise ValueError("GPPT requires node/graph batches. Set finetune.dataset.induced=True for edge tasks.")

        method_cfg = getattr(cfg.finetune, "gppt", None)
        raw_center_num = int(getattr(method_cfg, "center_num", 0)) if method_cfg is not None else 0
        if raw_center_num > 0:
            center_num = raw_center_num
        else:
            center_num = self.num_classes

        hidden_dim = int(getattr(cfg.model, "hidden_dim", 1) or 1)
        self.repr_dim = int(getattr(cfg.model, "out_dim", hidden_dim) or hidden_dim)
        self.constraint_weight = (
            float(getattr(method_cfg, "constraint_weight", 1e-2)) if method_cfg is not None else 1e-2
        )
        self.update_structure_every_step = (
            self._to_bool(getattr(method_cfg, "update_structure_every_step", True))
            if method_cfg is not None
            else True
        )
        self.update_structure_from_mask = (
            self._to_bool(getattr(method_cfg, "update_structure_from_mask", True))
            if method_cfg is not None
            else True
        )
        method_force_freeze = (
            self._to_bool(getattr(method_cfg, "force_freeze_encoder", False))
            if method_cfg is not None
            else False
        )
        self.freeze_encoder_effective = bool(method_force_freeze)
        self._encoder_frozen_notice_printed = False
        concat_neighbor = self._to_bool(getattr(method_cfg, "concat_neighbor", True)) if method_cfg is not None else True
        structure_mode = str(getattr(method_cfg, "structure_mode", "concat")) if method_cfg is not None else "concat"
        task_mode_default = "concat" if concat_neighbor else "neighbor"
        task_mode = str(getattr(method_cfg, "task_mode", task_mode_default)) if method_cfg is not None else task_mode_default

        self.prompt = GPPTPrompt(
            in_channels=self.repr_dim,
            center_num=center_num,
            num_classes=self.num_classes,
            structure_mode=structure_mode,
            task_mode=task_mode,
            add_self_loops_in_conv=(
                self._to_bool(getattr(method_cfg, "add_self_loops", False)) if method_cfg is not None else False
            ),
            kmeans_max_iter=int(getattr(method_cfg, "kmeans_max_iter", 100)) if method_cfg is not None else 100,
            kmeans_tol=float(getattr(method_cfg, "kmeans_tol", 1e-4)) if method_cfg is not None else 1e-4,
            kmeans_restarts=int(getattr(method_cfg, "kmeans_restarts", 1)) if method_cfg is not None else 1,
            use_sklearn_kmeans=(
                self._to_bool(getattr(method_cfg, "use_sklearn_kmeans", True)) if method_cfg is not None else True
            ),
            kmeans_random_state=int(getattr(method_cfg, "kmeans_random_state", 0)) if method_cfg is not None else 0,
            kmeans_n_init=int(getattr(method_cfg, "kmeans_n_init", 10)) if method_cfg is not None else 10,
        )
        self._prompt_initialized = False

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

    @staticmethod
    def _align_last_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
        current_dim = int(x.size(-1))
        if current_dim == target_dim:
            return x
        if current_dim > target_dim:
            return x[:, :target_dim]
        pad = x.new_zeros((x.size(0), target_dim - current_dim))
        return torch.cat([x, pad], dim=-1)

    @staticmethod
    def _prepare_labels(labels: torch.Tensor) -> torch.Tensor:
        labels = torch.as_tensor(labels)
        if labels.dim() > 1:
            labels = labels.view(labels.size(0), -1)[:, 0]
        else:
            labels = labels.view(-1)
        if labels.dtype.is_floating_point:
            rounded = labels.round()
            if torch.allclose(labels, rounded, atol=1e-6):
                labels = rounded
            else:
                labels = (labels > 0.5).to(labels.dtype)
        return labels.long()

    @staticmethod
    def _match_label_size(labels: torch.Tensor, target_size: int) -> torch.Tensor:
        labels = labels.view(-1)
        if target_size <= 0:
            return labels[:0]
        if labels.numel() == target_size:
            return labels
        if labels.numel() == 0:
            return torch.zeros(target_size, dtype=torch.long, device=labels.device)
        if labels.numel() == 1:
            return labels.repeat(target_size)
        if labels.numel() > target_size:
            return labels[:target_size]
        repeat = (target_size + labels.numel() - 1) // labels.numel()
        return labels.repeat(repeat)[:target_size]

    def parameters_to_optimize(self):
        return self.prompt.parameters()

    def build_optimizers(self, model: nn.Module):
        if self.freeze_encoder_effective:
            frozen = 0
            for param in model.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen += param.numel()
            if frozen > 0 and not self._encoder_frozen_notice_printed:
                print(f"[Finetune][gppt] Forced frozen encoder parameters (count={frozen}).")
                self._encoder_frozen_notice_printed = True
        else:
            # Allow GPPT to jointly tune encoder + prompt when not frozen.
            unfrozen = 0
            for param in model.parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    unfrozen += param.numel()
            if unfrozen > 0 and not self._encoder_frozen_notice_printed:
                print(f"[Finetune][gppt] Unfroze encoder parameters (count={unfrozen}).")
                self._encoder_frozen_notice_printed = True

        method_cfg = getattr(self.cfg.finetune, "gppt", None)
        lr = float(getattr(method_cfg, "lr", 2e-3)) if method_cfg is not None else 2e-3
        weight_decay = float(getattr(method_cfg, "weight_decay", 5e-4)) if method_cfg is not None else 5e-4
        params = list(self.prompt.parameters())
        encoder_params = [param for param in model.parameters() if param.requires_grad]
        if encoder_params:
            params.extend(encoder_params)
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        return {"primary": optimizer}

    def _orthogonality_constraint(self) -> torch.Tensor:
        weights = self.prompt.get_TaskToken()
        if not weights:
            return next(self.prompt.parameters()).new_tensor(0.0)
        total = weights[0].new_tensor(0.0)
        for weight in weights:
            gram = weight @ weight.t()
            ident = torch.eye(weight.size(0), device=weight.device, dtype=weight.dtype)
            total = total + torch.norm(gram - ident, p="fro")
        return total / len(weights)

    def _set_encoder_mode(self, model: nn.Module) -> None:
        if self.freeze_encoder_effective:
            model.eval()
        else:
            model.train()

    def _graph_vote_logits(
        self,
        node_logits: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: Optional[int] = None,
    ) -> torch.Tensor:
        if num_graphs is None:
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        if num_graphs <= 0:
            return node_logits.new_zeros((0, self.num_classes))

        pred = node_logits.argmax(dim=-1)
        one_hot = F.one_hot(pred, num_classes=self.num_classes).to(node_logits.dtype)
        votes = node_logits.new_zeros((num_graphs, self.num_classes))
        votes.scatter_add_(0, batch.view(-1, 1).expand_as(one_hot), one_hot)
        return votes

    def _node_forward(
        self,
        model: nn.Module,
        data,
        device: torch.device,
        mask_attr: str,
    ) -> Optional[Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        data = data.to(device)
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            raise ValueError("GPPT requires edge_index for node-level tuning.")

        node_repr, _ = model(data)
        node_repr = self._align_last_dim(node_repr, self.repr_dim)
        logits = self.prompt(node_repr, edge_index)

        mask = getattr(data, mask_attr, None)
        if mask is None:
            mask = torch.ones(data.num_nodes, dtype=torch.bool, device=device)
        else:
            mask = torch.as_tensor(mask, dtype=torch.bool, device=device).view(-1)
            if mask.numel() > data.num_nodes:
                mask = mask[: data.num_nodes]
            elif mask.numel() < data.num_nodes:
                pad = torch.zeros(data.num_nodes - mask.numel(), dtype=torch.bool, device=device)
                mask = torch.cat([mask, pad], dim=0)
        if not bool(mask.any().item()):
            return None

        labels = self._prepare_labels(data.y).to(device)
        labels = self._match_label_size(labels, data.num_nodes)
        logits_used = logits[mask]
        labels_used = labels[mask]
        loss = F.cross_entropy(logits_used, labels_used)
        pred = logits_used.argmax(dim=-1)
        acc = float((pred == labels_used).float().mean().item())
        structure_for_update = self.prompt.get_mid_h()
        if structure_for_update is not None:
            if self.update_structure_from_mask:
                structure_for_update = structure_for_update[mask]
            structure_for_update = structure_for_update.detach()
        return loss, acc, logits_used, labels_used, structure_for_update

    def _graph_forward(
        self,
        model: nn.Module,
        data,
        device: torch.device,
    ) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        data = data.to(device)
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            raise ValueError("GPPT requires edge_index for graph-level tuning.")

        node_repr, _ = model(data)
        node_repr = self._align_last_dim(node_repr, self.repr_dim)
        node_logits = self.prompt(node_repr, edge_index)
        batch = get_batch_vector(data).long()
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0

        graph_labels = self._prepare_labels(data.y).to(device)
        if num_graphs > 0:
            graph_labels = self._match_label_size(graph_labels, num_graphs)

        if num_graphs > 0:
            graph_losses = []
            for graph_id in range(num_graphs):
                node_mask = batch == graph_id
                if not bool(node_mask.any().item()):
                    continue
                node_label = graph_labels[graph_id].view(1).repeat(int(node_mask.sum().item()))
                graph_losses.append(F.cross_entropy(node_logits[node_mask], node_label))
            if graph_losses:
                loss = torch.stack(graph_losses, dim=0).mean()
            else:
                loss = node_logits.sum() * 0.0
        else:
            loss = node_logits.sum() * 0.0

        graph_votes = self._graph_vote_logits(node_logits=node_logits, batch=batch, num_graphs=num_graphs)
        graph_logits = (
            torch.softmax(graph_votes.float(), dim=-1).to(node_logits.dtype)
            if graph_votes.numel() > 0
            else graph_votes
        )
        pred = graph_logits.argmax(dim=-1) if graph_logits.numel() > 0 else graph_labels.new_zeros((0,), dtype=torch.long)
        acc = float((pred == graph_labels).float().mean().item()) if graph_labels.numel() > 0 else 0.0
        structure_for_update = self.prompt.get_mid_h()
        if structure_for_update is not None:
            structure_for_update = structure_for_update.detach()
        return loss, acc, graph_logits, graph_labels, structure_for_update

    def _maybe_initialize_prompt(self, model: nn.Module, loader, device: torch.device) -> None:
        if self._prompt_initialized:
            return

        node_reprs = []
        edge_indices = []
        node_labels = []
        train_indices = []
        node_offset = 0

        model_was_training = model.training
        model.eval()
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                edge_index = getattr(data, "edge_index", None)
                if edge_index is None:
                    continue

                node_repr, _ = model(data)
                node_repr = self._align_last_dim(node_repr, self.repr_dim)
                num_nodes = int(node_repr.size(0))
                if num_nodes <= 0:
                    continue

                node_reprs.append(node_repr.detach())
                edge_indices.append(edge_index + node_offset)
                if self.task_level == "node":
                    labels = self._prepare_labels(data.y).to(device)
                    labels = self._match_label_size(labels, num_nodes)
                    node_labels.append(labels)
                    train_mask = getattr(data, "train_mask", None)
                    if train_mask is None:
                        local_idx = torch.arange(num_nodes, device=device)
                    else:
                        local_mask = torch.as_tensor(train_mask, dtype=torch.bool, device=device).view(-1)
                        if local_mask.numel() > num_nodes:
                            local_mask = local_mask[:num_nodes]
                        elif local_mask.numel() < num_nodes:
                            pad = torch.zeros(num_nodes - local_mask.numel(), dtype=torch.bool, device=device)
                            local_mask = torch.cat([local_mask, pad], dim=0)
                        local_idx = torch.nonzero(local_mask, as_tuple=False).view(-1)
                    train_indices.append(local_idx + node_offset)
                else:
                    batch = get_batch_vector(data).long()
                    graph_labels = self._prepare_labels(data.y).to(device)
                    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
                    if num_graphs > 0:
                        graph_labels = self._match_label_size(graph_labels, num_graphs)
                        labels = graph_labels[batch]
                    else:
                        labels = graph_labels.new_zeros((0,), dtype=torch.long)
                    node_labels.append(labels)
                    train_indices.append(torch.arange(num_nodes, device=device, dtype=torch.long) + node_offset)
                node_offset += num_nodes

        model.train(model_was_training)

        if not node_reprs or not edge_indices or not node_labels:
            return

        all_repr = torch.cat(node_reprs, dim=0)
        all_edge_index = torch.cat(edge_indices, dim=1)
        all_labels = torch.cat(node_labels, dim=0)
        all_indices = (
            torch.cat(train_indices, dim=0) if train_indices else torch.arange(all_repr.size(0), device=device)
        )
        self.prompt.weigth_init(
            h=all_repr,
            edge_index=all_edge_index,
            label=all_labels,
            index=all_indices,
        )
        self._prompt_initialized = True

    def train_epoch(self, model, loader, device, optimizers=None):
        optimizer = optimizers.get("primary") if isinstance(optimizers, dict) else optimizers
        if optimizer is None:
            raise ValueError("GPPT requires an optimizer.")

        self._maybe_initialize_prompt(model=model, loader=loader, device=device)
        self._set_encoder_mode(model)
        self.prompt.train()

        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        for data in loader:
            optimizer.zero_grad()
            structure_for_update = None
            if self.task_level == "node":
                outputs = self._node_forward(model=model, data=data, device=device, mask_attr="train_mask")
                if outputs is None:
                    continue
                loss, acc, _logits, _labels, structure_for_update = outputs
            else:
                loss, acc, _logits, _labels, structure_for_update = self._graph_forward(
                    model=model,
                    data=data,
                    device=device,
                )

            if self.constraint_weight > 0:
                loss = loss + self.constraint_weight * self._orthogonality_constraint()

            loss.backward()
            optimizer.step()

            if self.update_structure_every_step:
                self.prompt.update_StructureToken_weight(structure_for_update)

            total_loss += float(loss.item())
            total_acc += float(acc)
            num_batches += 1

        if num_batches == 0:
            return 0.0, {}
        return total_loss / num_batches, {"train_acc": total_acc / num_batches}

    def evaluate_split(self, model, loader, device, prefix: str, mask_attr: str) -> Dict[str, float]:
        model.eval()
        self.prompt.eval()

        total_loss = 0.0
        num_batches = 0
        logits_buffer = []
        labels_buffer = []

        with torch.no_grad():
            for data in loader:
                if self.task_level == "node":
                    outputs = self._node_forward(model=model, data=data, device=device, mask_attr=mask_attr)
                    if outputs is None:
                        continue
                    loss, _acc, logits, labels, _structure_for_update = outputs
                else:
                    loss, _acc, logits, labels, _structure_for_update = self._graph_forward(
                        model=model,
                        data=data,
                        device=device,
                    )

                total_loss += float(loss.item())
                num_batches += 1
                logits_buffer.append(torch.as_tensor(logits).detach().cpu())
                labels_buffer.append(torch.as_tensor(labels).detach().cpu())

        if num_batches == 0:
            return {}

        metrics = {f"{prefix}_loss": total_loss / num_batches}
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
