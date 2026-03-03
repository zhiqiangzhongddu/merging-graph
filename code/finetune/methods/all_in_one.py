"""All-in-One prompt finetuning method."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from code.finetune.prompts.all_in_one import HeavyPrompt
from code.finetune.registry import register
from code.finetune.task_base import FinetuneTask
from code.pretrain.methods.utils import get_batch_vector, pool_nodes
from code.utils import compute_supervised_metrics


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad = flag


def _is_few_shot_split(split) -> bool:
    """Return True when split uses few-shot form (shots_per_class, val_weight, test_weight)."""
    if not isinstance(split, (list, tuple)) or len(split) != 3:
        return False
    first, val_ratio, test_ratio = split
    try:
        first_val = float(first)
        shots_like = first_val.is_integer() and first_val >= 1.0
    except Exception:
        shots_like = False
    if not shots_like:
        return False
    try:
        val = float(val_ratio)
        test = float(test_ratio)
    except Exception:
        return False
    return val >= 0.0 and test >= 0.0 and (val + test) > 0.0


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


def _none_if_str_none(value):
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"none", "null", ""}:
            return None
    return value


@register("all_in_one")
class FinetuneAllInOne(FinetuneTask):
    """Alternating prompt/answer optimization on graph-level batches."""

    def __init__(self, cfg):
        super().__init__(cfg)
        ds_cfg = cfg.finetune.dataset
        self.task_level = str(getattr(ds_cfg, "task_level", "graph") or "graph").lower()
        self.task_level_raw = str(getattr(ds_cfg, "task_level_raw", self.task_level) or self.task_level).lower()
        self.task_type = str(getattr(ds_cfg, "task_type", "classification") or "classification").lower()
        self.label_dim = int(getattr(ds_cfg, "label_dim", 1) or 1)

        if self.task_level == "node":
            raise ValueError("All-in-One requires graph-level batches (use induced=True for node datasets).")
        if self.task_type != "classification":
            raise ValueError("All-in-One currently supports classification only.")
        if self.label_dim > 1:
            raise ValueError("All-in-One currently supports single-label classification only.")

        method_cfg = getattr(cfg.finetune, "all_in_one", None)
        token_num = int(getattr(method_cfg, "token_num", 10)) if method_cfg else 10
        cross_prune = float(getattr(method_cfg, "cross_prune", 0.1)) if method_cfg else 0.1
        inner_prune = float(getattr(method_cfg, "inner_prune", 0.3)) if method_cfg else 0.3
        # Reference downstream scripts use epochs=1000 by default for
        # All-in-One. Keep this method-specific so other finetune methods can
        # use different global epoch budgets.
        self.total_epochs = int(getattr(method_cfg, "total_epochs", 1000)) if method_cfg else 1000
        # Speed-up: answer phase has frozen encoder + frozen prompt, so cache
        # graph embeddings once per outer epoch and optimize the head on cache.
        self.cache_answer_embeddings = (
            _to_bool(getattr(method_cfg, "cache_answer_embeddings", True)) if method_cfg else True
        )
        # Strict parity mode with the reference All-in-One head uses
        # Linear + Softmax as the answering head while still optimizing with CE.
        self.answer_with_softmax = (
            _to_bool(getattr(method_cfg, "answer_with_softmax", False)) if method_cfg else False
        )
        # Historical best runs in this repo consistently used bidirectional
        # prompt-graph links and excluded prompt nodes from readout on induced
        # node tasks, while reference defaults are one-way links with
        # prompt-inclusive pooling.
        default_bidirectional = self.task_level_raw == "node"
        default_exclude_prompt = self.task_level_raw == "node"
        bidirectional_cross_edges_cfg = _none_if_str_none(
            getattr(method_cfg, "bidirectional_cross_edges", None) if method_cfg else None
        )
        exclude_prompt_from_pooling_cfg = _none_if_str_none(
            getattr(method_cfg, "exclude_prompt_from_pooling", None) if method_cfg else None
        )
        self.bidirectional_cross_edges = (
            _to_bool(bidirectional_cross_edges_cfg)
            if bidirectional_cross_edges_cfg is not None
            else default_bidirectional
        )
        self.exclude_prompt_from_pooling = (
            _to_bool(exclude_prompt_from_pooling_cfg)
            if exclude_prompt_from_pooling_cfg is not None
            else default_exclude_prompt
        )
        self.is_few_shot = _is_few_shot_split(getattr(ds_cfg, "fixed_split", None))
        # Default alternating schedule aligned with reference task scripts:
        # - induced node-task path: answer/prompt = 50/50
        # - graph-task few-shot: answer/prompt = 50/50
        # - graph-task standard split: answer/prompt = 5/1
        if self.task_level_raw == "node":
            default_answer_epoch = 50
            default_prompt_epoch = 50
        else:
            default_answer_epoch = 50 if self.is_few_shot else 5
            default_prompt_epoch = 50 if self.is_few_shot else 1
        answer_epoch_cfg = int(getattr(method_cfg, "answer_epoch", -1)) if method_cfg else -1
        prompt_epoch_cfg = int(getattr(method_cfg, "prompt_epoch", -1)) if method_cfg else -1
        self.answer_epoch = max(1, answer_epoch_cfg if answer_epoch_cfg > 0 else default_answer_epoch)
        self.prompt_epoch = max(1, prompt_epoch_cfg if prompt_epoch_cfg > 0 else default_prompt_epoch)
        # Reference defaults differ by tasker:
        # - node_task returns answer-loss
        # - graph_task returns prompt-loss
        self._monitor_prompt_loss = self.task_level_raw != "node"

        num_classes = int(getattr(ds_cfg, "num_classes", 2) or 2)
        hidden_dim = int(getattr(cfg.model, "hidden_dim", 1) or 1)
        out_dim = int(getattr(cfg.model, "out_dim", hidden_dim) or hidden_dim)
        self.repr_dim = out_dim

        self.prompt = HeavyPrompt(
            token_dim=int(cfg.model.in_dim),
            token_num=token_num,
            cross_prune=cross_prune,
            inner_prune=inner_prune,
            bidirectional_cross_edges=self.bidirectional_cross_edges,
        )
        if self.answer_with_softmax:
            self.answering = nn.Sequential(
                nn.Linear(self.repr_dim, max(2, num_classes)),
                nn.Softmax(dim=1),
            )
        else:
            self.answering = nn.Linear(self.repr_dim, max(2, num_classes))
        self.criterion = nn.CrossEntropyLoss()
        self._encoder_frozen_notice_printed = False
        self._answer_cache_notice_printed = False

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

    def get_effective_epochs(self, original_epochs: int) -> int:
        total_epochs = int(self.total_epochs) if int(self.total_epochs) > 0 else int(original_epochs)
        if self.answer_epoch > 0:
            effective = total_epochs // self.answer_epoch
            return max(1, effective)
        return total_epochs

    def parameters_to_optimize(self):
        return list(self.prompt.parameters()) + list(self.answering.parameters())

    def build_optimizers(self, model: nn.Module):
        # All-in-One freezes encoder parameters by default.
        frozen_count = 0
        for param in model.parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_count += param.numel()
        if frozen_count > 0 and not self._encoder_frozen_notice_printed:
            print(f"[Finetune][all_in_one] Forced frozen encoder parameters (count={frozen_count}).")
            self._encoder_frozen_notice_printed = True

        method_cfg = getattr(self.cfg.finetune, "all_in_one", None)
        prompt_lr = float(getattr(method_cfg, "prompt_lr", 1e-6)) if method_cfg else 1e-6
        prompt_wd_cfg = _none_if_str_none(getattr(method_cfg, "prompt_weight_decay", None) if method_cfg else None)
        if prompt_wd_cfg is None:
            prompt_wd = 5e-4 if self.task_level_raw == "node" else float(self.cfg.finetune.weight_decay)
        else:
            prompt_wd = float(prompt_wd_cfg)
        answer_lr = float(getattr(method_cfg, "answer_lr", self.cfg.finetune.lr)) if method_cfg else float(self.cfg.finetune.lr)
        answer_wd_cfg = _none_if_str_none(getattr(method_cfg, "answer_weight_decay", None) if method_cfg else None)
        if answer_wd_cfg is None:
            answer_wd = 5e-4 if self.task_level_raw == "node" else float(self.cfg.finetune.weight_decay)
        else:
            answer_wd = float(answer_wd_cfg)

        prompt_opt = torch.optim.Adam(self.prompt.parameters(), lr=prompt_lr, weight_decay=prompt_wd)
        answer_opt = torch.optim.Adam(self.answering.parameters(), lr=answer_lr, weight_decay=answer_wd)
        return {"prompt": prompt_opt, "answer": answer_opt, "primary": answer_opt}

    def _graph_embeddings(self, model: nn.Module, batch, device):
        batch = batch.to(device)
        prompted = self.prompt(batch)
        node_repr, graph_repr = model(prompted)
        pool_mode = str(getattr(self.cfg.model, "graph_pooling", "mean") or "mean").lower()
        if pool_mode == "sum":
            pool_mode = "add"
        batch_vec = get_batch_vector(prompted)

        if pool_mode == "target":
            # For induced node tasks, target pooling depends on ptr/target_node on
            # the full prompted graph; do not drop nodes before selecting targets.
            graph_repr = pool_nodes(
                x=node_repr,
                batch=batch_vec,
                mode=pool_mode,
                data=prompted,
            )
        elif self.exclude_prompt_from_pooling and hasattr(prompted, "prompt_node_mask"):
            prompt_mask = torch.as_tensor(prompted.prompt_node_mask, dtype=torch.bool, device=node_repr.device)
            node_mask = ~prompt_mask
            if bool(node_mask.any().item()):
                graph_repr = pool_nodes(
                    x=node_repr[node_mask],
                    batch=batch_vec[node_mask],
                    mode=pool_mode,
                    data=prompted,
                )
            elif graph_repr is None:
                graph_repr = pool_nodes(
                    x=node_repr,
                    batch=batch_vec,
                    mode=pool_mode,
                    data=prompted,
                )
        elif graph_repr is None:
            graph_repr = pool_nodes(
                x=node_repr,
                batch=batch_vec,
                mode=pool_mode,
                data=prompted,
            )
        graph_repr = self._align_last_dim(graph_repr, self.repr_dim)
        labels = self._prepare_labels(prompted.y).to(device)
        return graph_repr, labels

    def _run_epoch(self, model, loader, device, optimizer, train_prompt: bool):
        model.eval()
        if train_prompt:
            self.prompt.train()
            self.answering.eval()
            _set_requires_grad(self.prompt, True)
            _set_requires_grad(self.answering, False)
        else:
            self.prompt.eval()
            self.answering.train()
            _set_requires_grad(self.prompt, False)
            _set_requires_grad(self.answering, True)

        total_loss = 0.0
        num_batches = 0
        for data in loader:
            optimizer.zero_grad(set_to_none=True)
            if train_prompt:
                graph_repr, labels = self._graph_embeddings(model, data, device)
            else:
                with torch.no_grad():
                    graph_repr, labels = self._graph_embeddings(model, data, device)
            logits = self.answering(graph_repr)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            num_batches += 1

        if num_batches == 0:
            return 0.0
        return total_loss / num_batches

    def _build_answer_cache(self, model, loader, device):
        model.eval()
        self.prompt.eval()
        _set_requires_grad(self.prompt, False)
        _set_requires_grad(self.answering, True)

        cache = []
        with torch.no_grad():
            for data in loader:
                graph_repr, labels = self._graph_embeddings(model, data, device)
                cache.append((graph_repr.detach(), labels.detach()))
        return cache

    def _run_answer_epoch_from_cache(self, cache, optimizer):
        if not cache:
            return 0.0

        self.answering.train()
        self.prompt.eval()
        _set_requires_grad(self.prompt, False)
        _set_requires_grad(self.answering, True)

        if len(cache) > 1:
            order = torch.randperm(len(cache), device=cache[0][0].device).tolist()
        else:
            order = [0]

        total_loss = 0.0
        for idx in order:
            graph_repr, labels = cache[idx]
            optimizer.zero_grad(set_to_none=True)
            logits = self.answering(graph_repr)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        return total_loss / len(cache)

    def train_epoch(self, model, loader, device, optimizers=None):
        if not isinstance(optimizers, dict):
            raise ValueError("All-in-One requires named optimizers.")

        answer_opt = optimizers.get("answer")
        prompt_opt = optimizers.get("prompt")
        if answer_opt is None or prompt_opt is None:
            raise ValueError("All-in-One optimizers are missing.")

        answer_loss = 0.0
        prompt_loss = 0.0
        if self.cache_answer_embeddings and self.answer_epoch > 0:
            if not self._answer_cache_notice_printed:
                print("[Finetune][all_in_one] Enabled cached answering-phase embeddings.")
                self._answer_cache_notice_printed = True
            answer_cache = self._build_answer_cache(model, loader, device)
            for _ in range(self.answer_epoch):
                answer_loss = self._run_answer_epoch_from_cache(answer_cache, answer_opt)
        else:
            for _ in range(self.answer_epoch):
                answer_loss = self._run_epoch(model, loader, device, answer_opt, train_prompt=False)
        for _ in range(self.prompt_epoch):
            prompt_loss = self._run_epoch(model, loader, device, prompt_opt, train_prompt=True)

        monitored_loss = prompt_loss if self._monitor_prompt_loss else answer_loss
        return monitored_loss, {
            "train_answer_loss": float(answer_loss),
            "train_prompt_loss": float(prompt_loss),
        }

    def evaluate_split(self, model, loader, device, prefix: str, mask_attr: str) -> Dict[str, float]:
        del mask_attr  # graph-level evaluation does not use masks

        model.eval()
        self.prompt.eval()
        self.answering.eval()
        total_loss = 0.0
        num_batches = 0
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for data in loader:
                graph_repr, labels = self._graph_embeddings(model, data, device)
                logits = self.answering(graph_repr)
                loss = self.criterion(logits, labels)
                total_loss += float(loss.item())
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
            metric_logits = logits
            if self.answer_with_softmax:
                # compute_supervised_metrics applies softmax for multiclass inputs.
                # Convert probabilities back to logit-space so metrics use exactly
                # the model probabilities instead of a second softmax.
                metric_logits = torch.log(logits.clamp_min(1e-12))
            for key, value in compute_supervised_metrics(
                logits=metric_logits,
                labels=labels,
                task_type=self.task_type,
            ).items():
                metrics[f"{prefix}_{key}"] = float(value)
        return metrics
