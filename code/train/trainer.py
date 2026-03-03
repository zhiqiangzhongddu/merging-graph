import json
import os
import time
from numbers import Integral
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim

from code.data_loader import create_dataset, dataset_info, log_split_instance_counts, make_loaders
from code.model import build_encoder_from_cfg
from code.pretrain.base import PretrainTask
from code.pretrain.checkpoint import cfg_to_dict, save_checkpoint
from code.pretrain.methods.utils import get_batch_vector, pool_nodes
from code.utils import compute_supervised_metrics, ensure_dir, format_split_for_name, set_seed


def _shared_split_root(cfg) -> str:
    ds_cfg = getattr(getattr(cfg, "data_preparation", None), "dataset", None)
    return getattr(ds_cfg, "split_root", "data/splits")


def _shared_induced_root(cfg, fallback: str = "") -> str:
    ds_cfg = getattr(getattr(cfg, "data_preparation", None), "dataset", None)
    root = getattr(ds_cfg, "induced_root", "")
    return root or fallback


def _split_dataset_name(base_name: str, task_level: str, seed: int) -> str:
    name = str(base_name)
    if any(tag in name for tag in ("_node_seed", "_graph_seed", "_edge_seed")):
        return name
    return f"{name}_{task_level}_seed{int(seed)}"


class TrainSupervised(PretrainTask):
    """Supervised training head that uses the train dataset config."""

    def __init__(self, cfg):
        super().__init__(cfg)
        ds_cfg = getattr(getattr(cfg, "train", None), "dataset", None) or getattr(cfg, "dataset", None)
        self.task_level = ds_cfg.task_level
        self.task_type = str(getattr(ds_cfg, "task_type", "classification") or "classification").lower()
        out_dim = 1 if self.task_type == "regression" else int(getattr(ds_cfg, "num_classes", 1) or 1)
        self.classifier = nn.Linear(
            in_features=cfg.model.out_dim,
            out_features=out_dim,
        )

    @staticmethod
    def _prepare_labels(labels: torch.Tensor) -> torch.Tensor:
        if labels.dim() > 1:
            if labels.size(-1) == 1:
                labels = labels.view(-1)
            else:
                labels = labels[:, 0].view(-1)
        else:
            labels = labels.view(-1)
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
        else:
            batch = get_batch_vector(data)
            if graph_repr is None:
                graph_repr = pool_nodes(
                    x=node_repr,
                    batch=batch,
                    mode=self.cfg.model.graph_pooling,
                )
            logits_used = self.classifier(graph_repr)
            labels = self._prepare_labels(data.y)

        if self.task_type == "regression":
            pred = logits_used.view(-1).float()
            target = labels.view(-1).float()
            loss = F.mse_loss(pred, target)
            mae = float((pred - target).abs().mean().item())
            if return_outputs:
                return loss, mae, pred, target
            return loss, mae

        class_labels = labels
        if class_labels.dtype.is_floating_point:
            rounded = class_labels.round()
            if torch.allclose(class_labels, rounded, atol=1e-6):
                class_labels = rounded
            else:
                class_labels = (class_labels > 0.5).to(class_labels.dtype)
        class_labels = class_labels.long()
        loss = F.cross_entropy(logits_used, class_labels)
        pred = logits_used.argmax(dim=-1)
        acc = (pred == class_labels).float().mean().item()
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


class TrainRunner:
    """Train a model from scratch using cfg.train settings."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
        set_seed(seed=self.cfg.seed)
        # Define summary fields up front so skip paths are safe for run-level aggregation.
        self.best_metrics: Dict[str, float] = {}
        self.best_epoch = None
        self.best_metric = float("nan")
        self.monitor_name = "val_acc"
        self.monitor_mode = "max"
        self.train_history = []

        ds_cfg = cfg.train.dataset
        raw_task_level = ds_cfg.task_level
        self.task_level_raw = raw_task_level
        self.split = self._resolve_split()
        self.run_name = self._build_run_name()
        self.run_group = self._build_run_group()
        self.run_dir = os.path.join(self.cfg.train.checkpoint_dir, self.run_group)
        # Backward-compatible lookup for checkpoints saved by the previous layout.
        self.legacy_run_dir = os.path.join(self.cfg.train.checkpoint_dir, self.run_name)
        self._skip_due_to_existing_checkpoint = False
        ckpt_path = self._existing_checkpoint_path()
        if getattr(self.cfg.train, "skip_if_exists", False) and ckpt_path is not None:
            self._skip_due_to_existing_checkpoint = True
            print(f"[Train] Checkpoint already exists, skipping: {ckpt_path}")
            return

        induced = getattr(ds_cfg, "induced", False)
        self.dataset = create_dataset(
            name=ds_cfg.name,
            root=ds_cfg.root,
            task_level=raw_task_level,
            feat_reduction=ds_cfg.feat_reduction,
            feat_reduction_dim=getattr(ds_cfg, "feat_reduction_dim", 100),
            induced=induced,
            induced_min_size=getattr(ds_cfg, "induced_min_size", 10),
            induced_max_size=getattr(ds_cfg, "induced_max_size", 30),
            induced_max_hops=getattr(ds_cfg, "induced_max_hops", 5),
            split_root=_shared_split_root(self.cfg),
            induced_root=_shared_induced_root(self.cfg, getattr(ds_cfg, "induced_root", "")),
            split=self.split,
            seed=self.cfg.seed,
        )
        effective_task_level = "graph" if induced else raw_task_level
        ds_cfg.task_level = effective_task_level

        self.dataset_meta = dataset_info(
            dataset=self.dataset,
            task_level=raw_task_level,
            name=ds_cfg.name,
            induced=induced,
        )
        cfg.model.in_dim = cfg.model.in_dim or self.dataset_meta["num_node_features"]
        if getattr(ds_cfg, "num_classes", None) is None:
            ds_cfg.num_classes = self.dataset_meta.get("num_classes")

        self.model = build_encoder_from_cfg(
            cfg=cfg,
            in_dim=cfg.model.in_dim,
        ).to(self.device)
        model_name = getattr(cfg.model, "name", "model")
        print(f"[Train] Encoder architecture ({model_name}):\n{self.model}")
        self.task = TrainSupervised(cfg).to(self.device)
        print(f"[Train] Task head:\n{self.task}")
        params = list(self.model.parameters()) + list(self.task.parameters())
        self.optimizer = optim.Adam(
            params=params,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )

        self._init_monitoring()
        ensure_dir(self.run_dir)
        self.train_loader, self.val_loader, self.test_loader = self._make_loaders(induced=induced)
        log_split_instance_counts(
            self.train_loader,
            self.val_loader,
            self.test_loader,
            task_level=self.task_level_raw,
            split=self.split,
            induced=induced,
            prefix="[Train][Split]",
        )

    def _init_monitoring(self) -> None:
        task_type = str(getattr(self.cfg.train.dataset, "task_type", "classification") or "classification").lower()
        task_level = str(getattr(self, "task_level_raw", self.cfg.train.dataset.task_level) or "").lower()
        split = getattr(self, "split", None)

        # Monitoring policy:
        # - Few-shot: training loss (validation split is intentionally empty)
        # - Classification: accuracy
        # - Edge prediction: AUC (link-task selection)
        # - Regression: training loss (loss-based selection)
        if self._is_few_shot_split(split):
            self.monitor_name = "train_loss"
            self.monitor_mode = "min"
            self.best_metric = float("inf")
        elif task_level == "edge":
            self.monitor_name = "val_auc"
            self.monitor_mode = "max"
            self.best_metric = float("-inf")
        elif task_type == "regression":
            self.monitor_name = "train_loss"
            self.monitor_mode = "min"
            self.best_metric = float("inf")
        else:
            self.monitor_name = "val_acc"
            self.monitor_mode = "max"
            self.best_metric = float("-inf")
        self.best_epoch = None
        self.best_metrics: Dict[str, float] = {}

    @staticmethod
    def _is_few_shot_split(split) -> bool:
        if not isinstance(split, (tuple, list)) or not split:
            return False
        first = split[0]
        if isinstance(first, bool):
            return False
        if isinstance(first, Integral):
            return True
        if isinstance(first, float) and first.is_integer():
            return True
        return False

    def _make_loaders(self, induced: bool):
        ds_cfg = self.cfg.train.dataset
        split = self.split
        split_task_level = self.task_level_raw if induced else ds_cfg.task_level
        loader_task_level = self._loader_task_level(induced=induced)
        split_dataset_name = _split_dataset_name(ds_cfg.name, split_task_level, self.cfg.seed)
        return make_loaders(
            dataset=self.dataset,
            dataset_name=split_dataset_name,
            task_level=loader_task_level,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            split=split,
            seed=self.cfg.seed,
            induced=induced,
            split_root=_shared_split_root(self.cfg),
        )

    def _loader_task_level(self, induced: bool) -> str:
        # Induced edge runs keep edge split semantics (ratios sum to <= 1.0) and
        # should not be treated as graph-level fixed splits.
        if induced and str(self.task_level_raw).lower() == "edge":
            return "edge"
        return self.cfg.train.dataset.task_level

    def _resolve_split(self) -> Tuple[float, float, float]:
        ds_cfg = self.cfg.train.dataset
        split = getattr(ds_cfg, "fixed_split", None)
        if split is None:
            split = getattr(self.cfg.train, "fixed_split", None)
        if split is None:
            return (0.8, 0.1, 0.1)
        parts = list(split)
        first = parts[0] if parts else None
        is_few_shot = False
        try:
            first_val = float(first)
            is_few_shot = float(first_val).is_integer()
        except Exception:
            is_few_shot = isinstance(first, Integral) and not isinstance(first, bool)
        if is_few_shot:
            val_ratio = float(parts[1])
            test_ratio = float(parts[2])
            if abs(val_ratio) > 1e-6 or abs(test_ratio - 1.0) > 1e-6:
                raise ValueError("Few-shot split must be (shots, 0.0, 1.0).")
        return tuple(parts)

    def _build_run_name(self) -> str:
        dataset_cfg = self.cfg.train.dataset
        train_cfg = self.cfg.train
        model_cfg = self.cfg.model
        split_tag = format_split_for_name(self.split)
        model_name = getattr(model_cfg, "name", "")
        raw_task_level = getattr(self, "task_level_raw", getattr(dataset_cfg, "task_level", ""))
        parts = [
            "train",
            dataset_cfg.name,
            f"induced{int(getattr(dataset_cfg, 'induced', False))}",
            split_tag,
            f"task{raw_task_level}",
            model_name,
            f"h{model_cfg.hidden_dim}",
            f"o{model_cfg.out_dim}",
            f"l{model_cfg.num_layers}",
            f"e{train_cfg.epochs}",
            f"lr{train_cfg.lr:g}" if isinstance(train_cfg.lr, (int, float)) else f"lr{train_cfg.lr}",
            f"bs{train_cfg.batch_size}",
            f"seed{self.cfg.seed}",
        ]
        return "_".join(str(p) for p in parts if p not in ("", None))

    def _build_run_group(self) -> str:
        dataset_cfg = self.cfg.train.dataset
        raw_task_level = getattr(self, "task_level_raw", getattr(dataset_cfg, "task_level", ""))
        return f"{dataset_cfg.name}-{raw_task_level}"

    def _checkpoint_path(self) -> str:
        return os.path.join(self.run_dir, f"{self.run_name}.pt")

    def _legacy_checkpoint_path(self) -> str:
        return os.path.join(self.legacy_run_dir, f"{self.run_name}.pt")

    def _existing_checkpoint_path(self) -> str | None:
        ckpt_path = self._checkpoint_path()
        if os.path.isfile(ckpt_path):
            return ckpt_path
        legacy_ckpt = self._legacy_checkpoint_path()
        if os.path.isfile(legacy_ckpt):
            return legacy_ckpt
        return None

    def get_checkpoint_path_for_metrics(self) -> str:
        return self._existing_checkpoint_path() or self._checkpoint_path()

    def _log_path(self) -> str:
        return os.path.join(self.run_dir, f"{self.run_name}_log.json")

    def _save_training_log(self) -> None:
        payload = {
            "config": cfg_to_dict(self.cfg),
            "dataset_meta": self.dataset_meta,
            "history": self.train_history,
            "best": {
                "epoch": self.best_epoch,
                "metric": self.best_metric,
                "monitor": self.monitor_name,
            },
        }
        with open(self._log_path(), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _is_improved(self, metric: float) -> bool:
        if metric is None:
            return False
        if self.monitor_mode == "max":
            return metric > self.best_metric
        return metric < self.best_metric

    def _save_best_checkpoint(
        self,
        epoch: int,
        train_loss: float,
        train_logs: Dict[str, float],
        val_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        monitor_value: float,
    ) -> None:
        if not self._is_improved(monitor_value):
            return

        self.best_metric = monitor_value
        self.best_epoch = epoch
        metrics = {
            "train_loss": train_loss,
            "best_epoch": epoch,
            self.monitor_name: monitor_value,
            **train_logs,
            **val_metrics,
            **test_metrics,
        }
        save_checkpoint(
            path=self._checkpoint_path(),
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            cfg=self.cfg,
            dataset_meta=self.dataset_meta,
            metrics=metrics,
        )
        self.best_metrics = metrics
        self._save_training_log()
        print(f"[Train] Best epoch updated: epoch={epoch} {self.monitor_name}={monitor_value:.4f}")

    def train_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        self.task.train()
        total_loss = 0.0
        logs: Dict[str, float] = {}

        for data in self.train_loader:
            self.optimizer.zero_grad()
            loss, log = self.task.step(
                model=self.model,
                data=data,
                device=self.device,
            )
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            for k, v in log.items():
                logs[k] = logs.get(k, 0.0) + float(v)

        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        logs = {k: v / num_batches for k, v in logs.items()}
        return avg_loss, logs

    def _evaluate_split(self, loader, prefix: str, mask_attr: str) -> Dict[str, float]:
        self.model.eval()
        self.task.eval()
        total_loss = 0.0
        num_batches = 0
        logits_buffer = []
        labels_buffer = []

        with torch.no_grad():
            for data in loader:
                loss, _primary, logits, labels = self.task._forward(  # pylint: disable=protected-access
                    model=self.model,
                    data=data,
                    device=self.device,
                    mask_attr=mask_attr,
                    return_outputs=True,
                )
                total_loss += loss.item()
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
            logits = torch.cat([x.view(-1) if x.dim() == 1 else x for x in logits_buffer], dim=0)
            labels = torch.cat([y.view(-1) for y in labels_buffer], dim=0)
            task_type = str(getattr(self.cfg.train.dataset, "task_type", "classification") or "classification").lower()
            for key, value in compute_supervised_metrics(logits=logits, labels=labels, task_type=task_type).items():
                metrics[f"{prefix}_{key}"] = float(value)
        return metrics

    def fit(self) -> None:
        if getattr(self, "_skip_due_to_existing_checkpoint", False):
            return
        ensure_dir(path=self.cfg.train.checkpoint_dir)
        ensure_dir(self.run_dir)
        ckpt_path = self._existing_checkpoint_path()
        if getattr(self.cfg.train, "skip_if_exists", False) and ckpt_path is not None:
            print(f"[Train] Checkpoint already exists, skipping: {ckpt_path}")
            return
        if os.path.isfile(self._checkpoint_path()):
            print(f"[Train] Overwriting existing checkpoint: {self._checkpoint_path()}")

        patience = int(getattr(self.cfg.train, "early_stopping", 0) or 0)
        epochs_since_improvement = 0

        for epoch in range(1, self.cfg.train.epochs + 1):
            start = time.time()
            train_loss, train_logs = self.train_epoch(epoch)
            val_metrics = self._evaluate_split(self.val_loader, prefix="val", mask_attr="val_mask")
            test_metrics = self._evaluate_split(self.test_loader, prefix="test", mask_attr="test_mask")

            duration = time.time() - start
            log_parts = [
                f"[Train][Epoch {epoch}/{self.cfg.train.epochs}]",
                f"train_loss={train_loss:.4f}",
            ]
            for metrics in (train_logs, val_metrics, test_metrics):
                for k, v in metrics.items():
                    if k == "batch_size":
                        continue
                    log_parts.append(f"{k}={v:.4f}")
            log_parts.append(f"time={duration:.1f}s")
            print(" ".join(log_parts))

            merged_metrics = {k: float(v) for k, v in train_logs.items()}
            merged_metrics.update({k: float(v) for k, v in val_metrics.items()})
            merged_metrics.update({k: float(v) for k, v in test_metrics.items()})
            self.train_history.append(
                {
                    "epoch": epoch,
                    "loss": float(train_loss),
                    "duration_sec": float(duration),
                    "metrics": merged_metrics,
                }
            )

            monitor_value = val_metrics.get(self.monitor_name)
            if monitor_value is None:
                monitor_value = train_loss

            improved = self._is_improved(float(monitor_value))
            self._save_best_checkpoint(
                epoch=epoch,
                train_loss=train_loss,
                train_logs=train_logs,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                monitor_value=float(monitor_value),
            )

            if patience > 0:
                if improved:
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    print(
                        f"[Train] Early stopping at epoch {epoch} "
                        f"(no improvement in {patience} epochs)."
                    )
                    break

        print(
            f"[Train] Complete. Best {self.monitor_name}: "
            f"{self.best_metric:.4f} at epoch {self.best_epoch}."
        )
        for metric_name in ("test_acc", "test_micro_f1", "test_macro_f1", "test_auc", "test_mae", "test_mse"):
            metric_value = self.best_metrics.get(metric_name)
            if metric_value is not None:
                print(f"[Train] Best-epoch {metric_name}={metric_value:.4f}")
