"""GraphPrompt finetuning method."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from code.finetune.prompts.graphprompt import (
    GraphPrompt,
    GraphPromptPlusStageWise,
    compute_class_centers,
)
from code.finetune.registry import register
from code.finetune.task_base import FinetuneTask
from code.pretrain.methods.utils import get_batch_vector, pool_nodes
from code.utils import compute_supervised_metrics


@register("graphprompt")
class FinetuneGraphPrompt(FinetuneTask):
    """Prototype-based GraphPrompt finetuning with a feature-weight prompt."""

    def __init__(self, cfg):
        super().__init__(cfg)
        ds_cfg = cfg.finetune.dataset
        self.task_level = str(getattr(ds_cfg, "task_level", "graph") or "graph").lower()
        self.task_level_raw = str(getattr(ds_cfg, "task_level_raw", self.task_level) or self.task_level).lower()
        self.is_induced = bool(getattr(ds_cfg, "induced", False))
        self.task_type = str(getattr(ds_cfg, "task_type", "classification") or "classification").lower()
        self.label_dim = int(getattr(ds_cfg, "label_dim", 1) or 1)
        self.num_classes = max(2, int(getattr(ds_cfg, "num_classes", 2) or 2))

        if self.task_type != "classification":
            raise ValueError("GraphPrompt currently supports classification tasks only.")
        if self.label_dim > 1:
            raise ValueError("GraphPrompt currently supports single-label classification only.")
        if self.task_level == "edge":
            raise ValueError("GraphPrompt requires node/graph batches. Set finetune.dataset.induced=True for edge tasks.")
        if self.task_level_raw == "node" and self.is_induced:
            print(
                "[Finetune][GraphPrompt] Running node GraphPrompt with induced subgraphs. "
                "Official GraphPrompt uses non-induced node training; consider finetune.dataset.induced=False."
            )

        hidden_dim = int(getattr(cfg.model, "hidden_dim", 1) or 1)
        out_dim = int(getattr(cfg.model, "out_dim", hidden_dim) or hidden_dim)
        self.repr_dim = out_dim

        method_cfg = getattr(cfg.finetune, "graphprompt", None)
        self.use_plus = bool(getattr(method_cfg, "plus", False)) if method_cfg else False
        self.p_num = int(getattr(method_cfg, "p_num", 4)) if method_cfg else 4
        self.prompt_init = str(getattr(method_cfg, "init", "xavier")) if method_cfg else "xavier"
        self.prompt_init_std = float(getattr(method_cfg, "init_std", 0.02)) if method_cfg else 0.02
        self.update_pretrained = bool(getattr(method_cfg, "update_pretrained", False)) if method_cfg else False
        self.tau = float(getattr(method_cfg, "tau", 0.1)) if method_cfg else 0.1
        raw_score_mode = str(getattr(method_cfg, "score_mode", "neg_distance") if method_cfg else "neg_distance").lower()
        if raw_score_mode == "auto":
            # Backward-compatible alias.
            raw_score_mode = "neg_distance"
        self.score_mode = raw_score_mode

        raw_reduction = str(getattr(method_cfg, "loss_reduction", "mean") if method_cfg else "mean").lower()
        if raw_reduction == "auto":
            # Backward-compatible alias.
            raw_reduction = "mean"
        self.loss_reduction = raw_reduction

        raw_train_center_mode = str(getattr(method_cfg, "train_center_mode", "batch") if method_cfg else "batch").lower()
        if raw_train_center_mode == "auto":
            # Backward-compatible alias.
            raw_train_center_mode = "batch"
        self.train_center_mode = raw_train_center_mode

        raw_eval_center_mode = str(getattr(method_cfg, "eval_center_mode", "train") if method_cfg else "train").lower()
        if raw_eval_center_mode == "auto":
            # Backward-compatible alias.
            raw_eval_center_mode = "train"
        self.eval_center_mode = raw_eval_center_mode

        raw_graph_pooling_mode = str(getattr(method_cfg, "graph_pooling", "sum") if method_cfg else "sum").lower()
        if raw_graph_pooling_mode == "auto":
            # Backward-compatible alias.
            raw_graph_pooling_mode = "sum"
        self.graph_pooling_mode = raw_graph_pooling_mode
        self.prompt_dropout = float(getattr(method_cfg, "prompt_dropout", 0.0)) if method_cfg else 0.0
        self.embedding_scalar = float(getattr(method_cfg, "embedding_scalar", 1e3)) if method_cfg else 1e3
        self.center_momentum = float(getattr(method_cfg, "center_momentum", 0.9)) if method_cfg else 0.9
        if self.score_mode not in {"neg_distance", "distance", "cosine"}:
            raise ValueError("graphprompt.score_mode must be one of: neg_distance, distance, cosine (auto maps to neg_distance).")
        if self.loss_reduction not in {"mean", "sum"}:
            raise ValueError("graphprompt.loss_reduction must be one of: mean, sum (auto maps to mean).")
        if self.train_center_mode not in {"batch", "train", "ema"}:
            raise ValueError("graphprompt.train_center_mode must be one of: batch, train, ema (auto maps to batch).")
        if self.eval_center_mode not in {"train", "batch"}:
            raise ValueError("graphprompt.eval_center_mode must be one of: train, batch (auto maps to train).")
        if self.graph_pooling_mode not in {"encoder", "sum", "add", "mean", "max", "target"}:
            raise ValueError(
                "graphprompt.graph_pooling must be one of: encoder, sum, add, mean, max, target (auto maps to sum)."
            )
        if self.prompt_dropout < 0.0 or self.prompt_dropout >= 1.0:
            raise ValueError("graphprompt.prompt_dropout must be in [0.0, 1.0).")

        if self.use_plus:
            self.prompt = GraphPromptPlusStageWise(
                in_channels=int(getattr(cfg.model, "in_dim", self.repr_dim) or self.repr_dim),
                hidden_channels=int(getattr(cfg.model, "hidden_dim", self.repr_dim) or self.repr_dim),
                out_channels=self.repr_dim,
                num_layers=int(getattr(cfg.model, "num_layers", 1) or 1),
                p_num=self.p_num,
                init=self.prompt_init,
                init_std=self.prompt_init_std,
            )
        else:
            self.prompt = GraphPrompt(
                in_channels=self.repr_dim,
                init=self.prompt_init,
                init_std=self.prompt_init_std,
            )
        self.latest_centers: Optional[torch.Tensor] = None
        self.prototype_bank: Optional[torch.Tensor] = None

    def parameters_to_optimize(self):
        return self.prompt.parameters()

    def build_optimizers(self, model: nn.Module):
        method_cfg = getattr(self.cfg.finetune, "graphprompt", None)
        prompt_lr = float(getattr(method_cfg, "prompt_lr", self.cfg.finetune.lr)) if method_cfg else float(self.cfg.finetune.lr)
        prompt_wd = (
            float(getattr(method_cfg, "prompt_weight_decay", 1e-5))
            if method_cfg
            else 1e-5
        )
        encoder_lr = float(getattr(method_cfg, "encoder_lr", self.cfg.finetune.lr)) if method_cfg else float(self.cfg.finetune.lr)
        encoder_wd = (
            float(getattr(method_cfg, "encoder_weight_decay", self.cfg.finetune.weight_decay))
            if method_cfg
            else float(self.cfg.finetune.weight_decay)
        )
        amsgrad = bool(getattr(method_cfg, "amsgrad", True)) if method_cfg else True

        param_groups = [
            {
                "params": list(self.prompt.parameters()),
                "lr": prompt_lr,
                "weight_decay": prompt_wd,
            }
        ]
        encoder_params = [param for param in model.parameters() if param.requires_grad] if self.update_pretrained else []
        if encoder_params:
            param_groups.append(
                {
                    "params": encoder_params,
                    "lr": encoder_lr,
                    "weight_decay": encoder_wd,
                }
            )

        optimizer = torch.optim.AdamW(param_groups, amsgrad=amsgrad)
        return {"primary": optimizer}

    def _set_encoder_train_mode(self, model: nn.Module) -> None:
        """
        Match official GraphPrompt prompt-tuning behavior:
        keep pretrained encoder in eval mode unless explicitly updating it.
        """
        if not self.update_pretrained:
            model.eval()
            return
        has_trainable = any(param.requires_grad for param in model.parameters())
        if has_trainable:
            model.train()
        else:
            model.eval()

    def _supports_stagewise_plus_backbone(self, model: nn.Module) -> bool:
        if not (self.use_plus and isinstance(self.prompt, GraphPromptPlusStageWise)):
            return False
        required_attrs = ("convs", "model_type", "dropout", "act")
        if any(not hasattr(model, attr) for attr in required_attrs):
            return False
        model_type = str(getattr(model, "model_type", "") or "").lower()
        if model_type not in {"gcn", "gin", "gat", "mlp"}:
            return False
        return True

    def _forward_with_stage_prompt(self, model: nn.Module, data, stage_id: int):
        """
        Manual GNNEncoder forward with a single stage prompt injected.

        This mirrors the extension idea where each prompt stage produces one
        representation branch, then branches are mixed by learnable weights.
        """
        x = data.x
        edge_index = getattr(data, "edge_index", None)
        batch = getattr(data, "batch", None)
        model_type = str(getattr(model, "model_type", "") or "").lower()
        convs = getattr(model, "convs")
        bns = getattr(model, "bns", [])
        use_batchnorm = bool(getattr(model, "use_batchnorm", False))
        act = getattr(model, "act")
        dropout = getattr(model, "dropout")

        if stage_id == 0 and self.prompt.has_stage(0):
            x = self.prompt.apply_stage(0, x)

        for idx, conv in enumerate(convs):
            if model_type == "mlp":
                x = conv(x)
            else:
                if edge_index is None:
                    raise ValueError("edge_index is required for stage-wise GraphPrompt+ on GNN backbones.")
                x = conv(x, edge_index)

            is_last = idx == len(convs) - 1
            if not is_last:
                x = act(x)
                if use_batchnorm and idx < len(bns):
                    x = bns[idx](x)
                x = dropout(x)
                if stage_id == 1 and idx == 0 and self.prompt.has_stage(1):
                    x = self.prompt.apply_stage(1, x)
                if stage_id == 2 and idx == 1 and self.prompt.has_stage(2):
                    x = self.prompt.apply_stage(2, x)
                continue

            if use_batchnorm and idx < len(bns):
                x = act(x)
                x = bns[idx](x)

        node_repr = x
        if stage_id == 3 and self.prompt.has_stage(3):
            node_repr = self.prompt.apply_stage(3, node_repr)

        graph_repr = None
        if batch is not None and hasattr(model, "pool"):
            graph_repr = model.pool(node_repr, batch)
        return node_repr, graph_repr

    def _extract_stagewise_plus_embeddings(
        self,
        model: nn.Module,
        data,
        device: torch.device,
        mask_attr: str,
        apply_prompt_dropout: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        data = data.to(device)
        batch = get_batch_vector(data)

        if self.task_level == "node":
            mask = getattr(data, mask_attr, None)
            if mask is None:
                mask = torch.ones(data.num_nodes, dtype=torch.bool, device=device)
            labels = self._prepare_labels(data.y[mask]).to(device)
        else:
            labels = self._prepare_labels(data.y).to(device)

        mixed_embeddings = None
        for stage_id, coeff in self.prompt.iter_stage_coefficients():
            node_repr, stage_graph_repr = self._forward_with_stage_prompt(model=model, data=data, stage_id=stage_id)
            node_repr = self._align_last_dim(node_repr, self.repr_dim)

            if self.task_level == "node":
                stage_embeddings = node_repr[mask]
            else:
                if self.graph_pooling_mode == "encoder":
                    graph_repr = stage_graph_repr
                    if graph_repr is None:
                        graph_repr = pool_nodes(
                            x=node_repr,
                            batch=batch,
                            mode=self.cfg.model.graph_pooling,
                            data=data,
                        )
                else:
                    graph_repr = pool_nodes(
                        x=node_repr,
                        batch=batch,
                        mode=self.graph_pooling_mode,
                        data=data,
                    )
                stage_embeddings = self._align_last_dim(graph_repr, self.repr_dim)

            weighted = coeff * stage_embeddings
            mixed_embeddings = weighted if mixed_embeddings is None else (mixed_embeddings + weighted)

        if mixed_embeddings is None:
            raise RuntimeError("Stage-wise GraphPrompt+ produced no embeddings.")

        embeddings = mixed_embeddings * self.embedding_scalar
        if apply_prompt_dropout and self.prompt_dropout > 0.0:
            embeddings = F.dropout(embeddings, p=self.prompt_dropout, training=True)
        return embeddings, labels

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

    def _extract_embeddings_and_labels(
        self,
        model: nn.Module,
        data,
        device: torch.device,
        mask_attr: str,
        apply_prompt_dropout: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_plus and isinstance(self.prompt, GraphPromptPlusStageWise):
            if not self._supports_stagewise_plus_backbone(model):
                model_name = model.__class__.__name__
                raise ValueError(
                    "[Finetune][GraphPrompt+] Stage-wise prompting requires a standard GNNEncoder "
                    f"backbone (gcn/gin/gat/mlp); got {model_name}."
                )
            return self._extract_stagewise_plus_embeddings(
                model=model,
                data=data,
                device=device,
                mask_attr=mask_attr,
                apply_prompt_dropout=apply_prompt_dropout,
            )

        data = data.to(device)
        node_repr, _graph_repr = model(data)
        node_repr = self._align_last_dim(node_repr, self.repr_dim)
        prompted_node_repr = self.prompt(node_repr) * self.embedding_scalar
        if apply_prompt_dropout and self.prompt_dropout > 0.0:
            prompted_node_repr = F.dropout(prompted_node_repr, p=self.prompt_dropout, training=True)

        if self.task_level == "node":
            mask = getattr(data, mask_attr, None)
            if mask is None:
                mask = torch.ones(data.num_nodes, dtype=torch.bool, device=device)
            labels = self._prepare_labels(data.y[mask]).to(device)
            embeddings = prompted_node_repr[mask]
        else:
            if self.graph_pooling_mode == "encoder":
                graph_repr = _graph_repr
                if graph_repr is None:
                    batch = get_batch_vector(data)
                    graph_repr = pool_nodes(
                        x=node_repr,
                        batch=batch,
                        mode=self.cfg.model.graph_pooling,
                        data=data,
                    )
                graph_repr = self._align_last_dim(graph_repr, self.repr_dim)
                graph_repr = self.prompt(graph_repr)
                graph_repr = graph_repr * self.embedding_scalar
            else:
                batch = get_batch_vector(data)
                graph_repr = pool_nodes(
                    x=prompted_node_repr,
                    batch=batch,
                    mode=self.graph_pooling_mode,
                    data=data,
                )
            if apply_prompt_dropout and self.prompt_dropout > 0.0:
                graph_repr = F.dropout(graph_repr, p=self.prompt_dropout, training=True)
            labels = self._prepare_labels(data.y).to(device)
            embeddings = graph_repr

        return embeddings, labels

    def _similarity_logits(self, embeddings: torch.Tensor, centers: torch.Tensor, is_train: bool) -> torch.Tensor:
        if self.score_mode == "cosine":
            logits = F.cosine_similarity(embeddings.unsqueeze(1), centers.unsqueeze(0), dim=-1)
            return logits / max(self.tau, 1e-12)
        # Official GraphPrompt scripts use reciprocal-normalized distance for training
        # and negative normalized distance for evaluation.
        n = embeddings.size(0)
        k = centers.size(0)
        emb_power = torch.sum(embeddings * embeddings, dim=1, keepdim=True).expand(n, k)
        center_power = torch.sum(centers * centers, dim=1).expand(n, k)
        distance = emb_power + center_power - 2 * torch.mm(embeddings, centers.transpose(0, 1))
        normed_distance = F.normalize(distance, dim=1)
        if self.score_mode == "distance":
            return normed_distance
        if is_train:
            return torch.reciprocal(normed_distance.clamp_min(1e-12))
        return -1.0 * normed_distance

    def _fill_missing_centers(
        self,
        centers: torch.Tensor,
        counts: torch.Tensor,
        fallback: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if fallback is None:
            return centers
        present = counts.view(-1) > 0
        if bool(present.all().item()):
            return centers
        mixed = centers.clone()
        mixed[~present] = fallback[~present]
        return mixed

    def _compute_reference_centers(
        self,
        model: nn.Module,
        loader,
        device: torch.device,
        mask_attr: str,
    ) -> Optional[torch.Tensor]:
        accum_centers = torch.zeros(self.num_classes, self.repr_dim, device=device)
        accum_counts = torch.zeros(self.num_classes, 1, device=device)
        observed = 0

        # Prototype extraction must run in eval mode to avoid dropout/BN noise.
        model_was_training = model.training
        prompt_was_training = self.prompt.training
        model.eval()
        self.prompt.eval()
        try:
            with torch.no_grad():
                for data in loader:
                    embeddings, labels = self._extract_embeddings_and_labels(
                        model=model,
                        data=data,
                        device=device,
                        mask_attr=mask_attr,
                        apply_prompt_dropout=False,
                    )
                    if embeddings.numel() == 0:
                        continue
                    centers, counts = compute_class_centers(embeddings, labels, self.num_classes)
                    accum_centers += centers * counts
                    accum_counts += counts
                    observed += int(labels.numel())
        finally:
            if model_was_training:
                model.train()
            if prompt_was_training:
                self.prompt.train()

        if observed == 0:
            return None
        return accum_centers / accum_counts.clamp_min(1.0)

    def train_epoch(self, model, loader, device, optimizers=None):
        optimizer = optimizers.get("primary") if isinstance(optimizers, dict) else optimizers
        if optimizer is None:
            raise ValueError("GraphPrompt requires an optimizer.")

        self._set_encoder_train_mode(model)
        self.prompt.train()

        total_loss = 0.0
        total_loss_denom = 0
        total_acc = 0.0
        num_batches = 0

        accum_centers = torch.zeros(self.num_classes, self.repr_dim, device=device)
        accum_counts = torch.zeros(self.num_classes, 1, device=device)
        epoch_train_centers = None
        if self.train_center_mode == "train":
            epoch_train_centers = self._compute_reference_centers(
                model=model,
                loader=loader,
                device=device,
                mask_attr="train_mask",
            )
        warm_start_centers = None
        if self.train_center_mode in {"batch", "ema"}:
            warm_start_centers = self.prototype_bank if self.prototype_bank is not None else self.latest_centers
            if warm_start_centers is None:
                warm_start_centers = self._compute_reference_centers(
                    model=model,
                    loader=loader,
                    device=device,
                    mask_attr="train_mask",
                )
        runtime_bank = warm_start_centers.detach().clone() if warm_start_centers is not None else None

        for data in loader:
            optimizer.zero_grad()
            embeddings, labels = self._extract_embeddings_and_labels(
                model=model,
                data=data,
                device=device,
                mask_attr="train_mask",
                apply_prompt_dropout=True,
            )
            if embeddings.numel() == 0:
                continue

            batch_centers, batch_counts = compute_class_centers(embeddings, labels, self.num_classes)
            centers_for_loss = batch_centers
            if self.train_center_mode == "train" and epoch_train_centers is not None:
                centers_for_loss = epoch_train_centers
            elif runtime_bank is not None:
                centers_for_loss = self._fill_missing_centers(
                    centers=batch_centers,
                    counts=batch_counts,
                    fallback=runtime_bank,
                )

            logits = self._similarity_logits(embeddings, centers_for_loss, is_train=True)
            loss = F.cross_entropy(logits, labels, reduction=self.loss_reduction)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                total_acc += float((pred == labels).float().mean().item())
                accum_centers += batch_centers.detach() * batch_counts.detach()
                accum_counts += batch_counts.detach()
                if runtime_bank is None:
                    runtime_bank = batch_centers.detach().clone()
                else:
                    present = batch_counts.view(-1) > 0
                    runtime_bank[present] = batch_centers.detach()[present]
                if self.train_center_mode == "ema":
                    if self.prototype_bank is None:
                        self.prototype_bank = batch_centers.detach().clone()
                    else:
                        present = batch_counts.view(-1) > 0
                        self.prototype_bank[present] = (
                            self.center_momentum * self.prototype_bank[present]
                            + (1.0 - self.center_momentum) * batch_centers.detach()[present]
                        )

            total_loss += float(loss.item())
            total_loss_denom += int(labels.numel()) if self.loss_reduction == "sum" else 1
            num_batches += 1

        if num_batches == 0:
            self.latest_centers = None
            return 0.0, {}

        mean_loss = total_loss / max(1, total_loss_denom)
        self.latest_centers = (accum_centers / accum_counts.clamp_min(1.0)).detach()
        if self.prototype_bank is None:
            self.prototype_bank = self.latest_centers.clone()
        return mean_loss, {"train_acc": total_acc / num_batches}

    def on_epoch_end(self, model: nn.Module, loader, device):
        self.latest_centers = self._compute_reference_centers(
            model=model,
            loader=loader,
            device=device,
            mask_attr="train_mask",
        )
        if self.latest_centers is not None:
            self.prototype_bank = self.latest_centers.detach().clone()
        return None

    def evaluate_split(self, model, loader, device, prefix: str, mask_attr: str) -> Dict[str, float]:
        model.eval()
        self.prompt.eval()

        total_loss = 0.0
        total_loss_denom = 0
        num_batches = 0
        all_logits = []
        all_labels = []

        reference_centers = None
        if self.eval_center_mode == "train":
            reference_centers = self.latest_centers
            if reference_centers is None:
                reference_centers = self._compute_reference_centers(
                    model=model,
                    loader=loader,
                    device=device,
                    mask_attr=mask_attr,
                )
            if reference_centers is None:
                return {}

        with torch.no_grad():
            for data in loader:
                embeddings, labels = self._extract_embeddings_and_labels(
                    model=model,
                    data=data,
                    device=device,
                    mask_attr=mask_attr,
                    apply_prompt_dropout=False,
                )
                if embeddings.numel() == 0:
                    continue
                centers = reference_centers
                if self.eval_center_mode == "batch":
                    centers, counts = compute_class_centers(embeddings, labels, self.num_classes)
                    centers = self._fill_missing_centers(
                        centers=centers,
                        counts=counts,
                        fallback=self.latest_centers,
                    )
                logits = self._similarity_logits(embeddings, centers, is_train=False)
                loss = F.cross_entropy(logits, labels, reduction=self.loss_reduction)
                total_loss += float(loss.item())
                total_loss_denom += int(labels.numel()) if self.loss_reduction == "sum" else 1
                num_batches += 1
                all_logits.append(logits.detach().cpu())
                all_labels.append(labels.detach().cpu())

        if num_batches == 0:
            return {}

        metrics = {
            f"{prefix}_loss": total_loss / max(1, total_loss_denom),
        }
        if all_logits and all_labels:
            logits = torch.cat(all_logits, dim=0)
            labels = torch.cat(all_labels, dim=0)
            for key, value in compute_supervised_metrics(
                logits=logits,
                labels=labels,
                task_type=self.task_type,
            ).items():
                metrics[f"{prefix}_{key}"] = float(value)
        return metrics
