import json
import math
import os
import time
from typing import Any, Dict, Tuple

import torch
from torch import optim
from torch_geometric.loader import DataLoader

from code.data_loader import (
    SingleGraphDataLoader,
    create_dataset,
    dataset_info,
    log_split_instance_counts,
    make_loaders,
)
from code.model import build_encoder_from_cfg
from code.pretrain.checkpoint import cfg_to_dict, save_checkpoint
from code.pretrain.registry import build_pretrain_task
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


def _checkpoint_dataset_dir_name(dataset_name: str) -> str:
    name = str(dataset_name or "").strip()
    if not name:
        return "unknown_dataset"
    return name.replace("\\", "_").replace("/", "_")


class PretrainRunner:
    """Class to handle pretraining of graph models."""
    def __init__(self, cfg):
        # Initialize settings
        self.cfg = cfg
        self.device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
        set_seed(seed=self.cfg.seed)
        # Define summary fields up front so skip paths are safe for run-level aggregation.
        self.best_metrics: Dict[str, float] = {}
        self.best_epoch = None
        self.best_metric = float("nan")
        self.monitor_name = "train_loss"
        self.monitor_mode = "min"
        self.train_history = []

        # Load dataset
        ds_cfg = cfg.pretrain.dataset
        raw_task_level = ds_cfg.task_level
        # Keep the original task level for naming to disambiguate edge runs.
        self.task_level_raw = raw_task_level
        self.split = self._resolve_split()
        self.use_dataset_splits = self._uses_dataset_splits()
        self.run_name = self._build_run_name()
        dataset_dir = _checkpoint_dataset_dir_name(ds_cfg.name)
        self.run_dir = os.path.join(self.cfg.pretrain.checkpoint_dir, dataset_dir)
        self._skip_due_to_existing_checkpoint = False
        ckpt_path = self._checkpoint_path()
        if os.path.isfile(ckpt_path) and getattr(self.cfg.pretrain, "skip_if_exists", True):
            self._skip_due_to_existing_checkpoint = True
            print(f"[Pretrain] Checkpoint already exists, skipping: {ckpt_path}")
            return
        induced = getattr(ds_cfg, "induced", False)
        split_root_for_dataset = _shared_split_root(self.cfg)
        split_for_dataset = self.split
        if not self.use_dataset_splits:
            # Unsupervised pretraining methods operate on full data and do not need
            # persisted train/val/test split files.
            if not (str(raw_task_level).lower() == "edge" and induced):
                split_root_for_dataset = ""
                split_for_dataset = None
        if cfg.pretrain.method == "graphcl" and str(raw_task_level).lower() == "node" and not induced:
            raise ValueError(
                f"{cfg.pretrain.method} requires graph instances in each batch. "
                "For node datasets, set pretrain.dataset.induced=True."
            )
        self.dataset = create_dataset(
            name=ds_cfg.name,
            root=ds_cfg.root,
            task_level=raw_task_level,
            feat_reduction=ds_cfg.feat_reduction,
            feat_reduction_dim=getattr(ds_cfg, "feat_reduction_svd_dim", getattr(ds_cfg, "feat_reduction_dim", 100)),
            persist_feature_svd=ds_cfg.feat_reduction,  # Persist only when feature reduction is enabled.
            feature_svd_dir=getattr(ds_cfg, "feature_svd_dir", "data/feature_svd"),
            induced=induced,
            induced_min_size=getattr(ds_cfg, "induced_min_size", 10),
            induced_max_size=getattr(ds_cfg, "induced_max_size", 30),
            induced_max_hops=getattr(ds_cfg, "induced_max_hops", 5),
            cache_induced=getattr(ds_cfg, "cache_induced", True),
            split_root=split_root_for_dataset,
            induced_root=_shared_induced_root(self.cfg, getattr(ds_cfg, "induced_root", "")),
            split=split_for_dataset,
            seed=self.cfg.seed,
            force_reload_raw=getattr(ds_cfg, "force_reload_raw", False),
        )
        # Induced subgraphs are handled as graph-level datasets downstream
        effective_task_level = "graph" if induced else raw_task_level
        ds_cfg.task_level = effective_task_level
        # Get dataset meta info
        self.dataset_meta = dataset_info(
            dataset=self.dataset, 
            task_level=raw_task_level,
            name=ds_cfg.name,
            induced=induced,
        )
        # Ensure model input dim matches actual feature dim when not reducing features.
        if not getattr(ds_cfg, "feat_reduction", False):
            cfg.model.in_dim = self.dataset_meta.get("num_node_features")
        cfg.model.in_dim = cfg.model.in_dim or self.dataset_meta["num_node_features"]
        if getattr(ds_cfg, "num_classes", None) is None:
            ds_cfg.num_classes = self.dataset_meta.get("num_classes")
        if getattr(ds_cfg, "label_dim", None) is None:
            ds_cfg.label_dim = self.dataset_meta.get("label_dim")
        # Validate config
        task_type = str(getattr(ds_cfg, "task_type", "classification") or "classification").lower()
        if cfg.pretrain.method == "supervised" and task_type != "regression" and not ds_cfg.num_classes:
            raise ValueError("num_classes is required for supervised pretraining")

        # Build encoder model
        self.model = build_encoder_from_cfg(
            cfg=cfg, 
            in_dim=cfg.model.in_dim
        ).to(self.device)
        model_name = getattr(cfg.model, "name", "model")
        print(f"[Pretrain] Encoder architecture ({model_name}):\n{self.model}")
        # Build pretraining task
        self.task = build_pretrain_task(
            name=cfg.pretrain.method, 
            cfg=cfg
        ).to(self.device)
        print(f"[Pretrain] Task head ({cfg.pretrain.method}):\n{self.task}")
        # Create optimizer
        task_params = list(self.task.parameters_to_optimize())
        params = list(self.model.parameters()) + task_params
        self.optimizer = optim.Adam(
            params=params,
            lr=cfg.pretrain.lr,
            weight_decay=cfg.pretrain.weight_decay,
        )

        # Prepare training and validation loaders and settings
        self._init_monitoring()
        self.train_history = []
        ensure_dir(self.run_dir)
        self.train_loader, self.val_loader, self.test_loader = self._make_loaders(induced=induced)
        if self.use_dataset_splits:
            log_split_instance_counts(
                self.train_loader,
                self.val_loader,
                self.test_loader,
                task_level=self._loader_task_level(induced=induced),
                split=self.split,
                induced=induced,
                prefix="[Pretrain][Split]",
            )
        else:
            print(
                f"[Pretrain] Using full dataset for {self.cfg.pretrain.method} "
                "(no explicit split loading)."
            )
        if self.cfg.pretrain.method == "graphcl" and len(self.train_loader) == 0:
            raise ValueError(
                f"{self.cfg.pretrain.method} train loader is empty after drop_last. "
                "Lower pretrain.batch_size or provide more training graphs."
            )

    def _init_monitoring(self) -> None:
        """Set monitoring mode and defaults based on the pretrain method."""
        is_supervised = self.cfg.pretrain.method == "supervised"
        method = str(getattr(self.cfg.pretrain, "method", "")).lower()
        task_type = str(getattr(self.cfg.pretrain.dataset, "task_type", "classification") or "classification").lower()
        task_level = str(getattr(self, "task_level_raw", self.cfg.pretrain.dataset.task_level) or "").lower()
        label_dim = int(getattr(self.cfg.pretrain.dataset, "label_dim", 1) or 1)

        if method == "context_pred":
            # Baseline contextpred reports the balanced objective (loss_pos + loss_neg).
            self.monitor_mode = "min"
            self.monitor_name = "balanced_loss"
        elif not is_supervised:
            self.monitor_mode = "min"
            self.monitor_name = "train_loss"
        elif task_level == "edge":
            # Link-prediction pretraining selection uses validation AUC.
            self.monitor_mode = "max"
            self.monitor_name = "val_auc"
        elif task_type == "classification" and label_dim > 1:
            # Multi-task binary classification (e.g., MoleculeNet) follows AUC selection.
            self.monitor_mode = "max"
            self.monitor_name = "val_auc"
        elif task_type == "regression":
            # Regression-style runs rely on loss for model selection.
            self.monitor_mode = "min"
            self.monitor_name = "train_loss"
        else:
            self.monitor_mode = "max"
            self.monitor_name = "val_acc"
        self.best_metric = float("-inf") if self.monitor_mode == "max" else float("inf")
        self.best_epoch = None
        self.best_metrics: Dict[str, float] = {}

    def _uses_dataset_splits(self) -> bool:
        """Only supervised pretraining relies on explicit train/val/test splits."""
        return str(getattr(self.cfg.pretrain, "method", "")).lower() == "supervised"

    def _make_loaders(self, induced: bool):
        """Create train/val/test loaders with the appropriate split."""
        if not self.use_dataset_splits:
            return self._make_full_dataset_loaders(induced=induced)

        ds_cfg = self.cfg.pretrain.dataset
        split = self.split
        split_task_level = self.task_level_raw if induced else ds_cfg.task_level
        loader_task_level = self._loader_task_level(induced=induced)
        split_dataset_name = _split_dataset_name(ds_cfg.name, split_task_level, self.cfg.seed)
        return make_loaders(
            dataset=self.dataset,
            dataset_name=split_dataset_name,
            task_level=loader_task_level,
            batch_size=self.cfg.pretrain.batch_size,
            num_workers=self.cfg.pretrain.num_workers,
            split=split,
            seed=self.cfg.seed,
            induced=induced,
            split_root=_shared_split_root(self.cfg),
            edge_pred_cfg=getattr(self.cfg.pretrain, "edge_pred", None),
            drop_last_train=bool(self.cfg.pretrain.method == "graphcl"),
        )

    def _make_full_dataset_loaders(self, induced: bool):
        """
        Create train loader without split files for unsupervised pretraining.
        Validation/test loaders are intentionally omitted.
        """
        task_level = self._loader_task_level(induced=induced)
        if task_level in {"node", "edge"} and not induced:
            data = self.dataset[0]
            train_loader = SingleGraphDataLoader(data)
            return train_loader, None, None

        train_loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.cfg.pretrain.batch_size,
            num_workers=self.cfg.pretrain.num_workers,
            shuffle=True,
            drop_last=bool(self.cfg.pretrain.method == "graphcl"),
        )
        return train_loader, None, None

    def _loader_task_level(self, induced: bool) -> str:
        # Keep edge split semantics for induced edge tasks.
        if induced and str(self.task_level_raw).lower() == "edge":
            return "edge"
        return self.cfg.pretrain.dataset.task_level

    def _resolve_split(self) -> Tuple[float, float, float]:
        ds_cfg = self.cfg.pretrain.dataset
        split = getattr(ds_cfg, "fixed_split", None)
        return tuple(split) if split is not None else (0.8, 0.1, 0.1)

    def _build_run_name(self) -> str:
        """Generate run name from requested cfg (validated against loaded dataset)."""
        dataset_cfg = self.cfg.pretrain.dataset
        model_cfg = self.cfg.model
        pretrain_cfg = self.cfg.pretrain
        split_tag = format_split_for_name(self.split) if self.use_dataset_splits else ""
        model_name = getattr(model_cfg, "name", "")
        ds_name = self.ds_name_used if hasattr(self, "ds_name_used") else dataset_cfg.name
        ds_task = self.ds_task_used if hasattr(self, "ds_task_used") else getattr(self, "task_level_raw", dataset_cfg.task_level)
        ds_induced = self.ds_induced_used if hasattr(self, "ds_induced_used") else getattr(dataset_cfg, "induced", False)
        parts = [
            pretrain_cfg.method,
            ds_name,
            f"task{ds_task}",
            f"induced{int(ds_induced)}",
            split_tag,
            model_name,
            f"h{model_cfg.hidden_dim}",
            f"o{model_cfg.out_dim}",
            f"l{model_cfg.num_layers}",
            f"e{pretrain_cfg.epochs}",
            f"lr{pretrain_cfg.lr:g}",
            f"bs{pretrain_cfg.batch_size}",
            f"seed{self.cfg.seed}",
        ]
        return "_".join(str(p) for p in parts if p)

    def _checkpoint_path(self) -> str:
        return os.path.join(self.run_dir, f"{self.run_name}.pt")

    def _log_path(self) -> str:
        return os.path.join(self.run_dir, f"{self.run_name}_log.json")

    def _architecture_path(self) -> str:
        return os.path.join(self.run_dir, f"{self.run_name}_architecture.json")

    def _artifact_config_dict(self) -> Dict[str, Any]:
        """
        Return config dict used for artifact serialization.
        For methods that do not use splits, omit fixed_split entirely.
        """
        cfg_dict = cfg_to_dict(self.cfg)
        if self.use_dataset_splits:
            return cfg_dict
        pretrain_cfg = cfg_dict.get("pretrain")
        if isinstance(pretrain_cfg, dict):
            dataset_cfg = pretrain_cfg.get("dataset")
            if isinstance(dataset_cfg, dict):
                dataset_cfg.pop("fixed_split", None)
        return cfg_dict

    def _save_training_log(self) -> None:
        """Save the full pretraining log and config alongside the checkpoint."""
        log_path = self._log_path()
        artifact_cfg = self._artifact_config_dict()
        payload = {
            "config": artifact_cfg,
            "dataset_meta": self.dataset_meta,
            "history": self.train_history,
            "best": {
                "epoch": self.best_epoch, 
                "metric": self.best_metric, 
                "monitor": self.monitor_name
            },
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _save_model_architecture(self) -> None:
        """Persist the model and task definitions in a readable JSON file."""
        arch_path = self._architecture_path()
        artifact_cfg = self._artifact_config_dict()
        payload = {
            "model": str(self.model),
            "task": str(self.task),
            "config": artifact_cfg,
            "dataset_meta": self.dataset_meta,
        }
        with open(arch_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _is_improved(self, metric: float) -> bool:
        if metric is None or not math.isfinite(float(metric)):
            return False
        if self.monitor_mode == "max":
            return metric > self.best_metric
        return metric < self.best_metric

    def _save_best_checkpoint(
        self,
        epoch: int,
        train_loss: float,
        logs: Dict[str, float],
        monitor_value: float,
    ) -> None:
        """Persist the best-performing checkpoint only."""
        if not self._is_improved(monitor_value):
            return

        self.best_metric = monitor_value
        self.best_epoch = epoch
        metrics = {
            "train_loss": train_loss,
            "best_epoch": epoch,
            self.monitor_name: monitor_value,
            **logs,
        }
        # Ensure run directory exists and does not belong to another dataset/method.
        ensure_dir(self.run_dir)
        # Guard against accidental overwrite of a different run.
        existing_log = self._log_path()
        if os.path.isfile(existing_log):
            try:
                with open(existing_log, "r", encoding="utf-8") as f:
                    prev = json.load(f)
                prev_cfg = prev.get("config", {})
                prev_pretrain = (prev_cfg.get("pretrain") or {})
                prev_ds = prev_pretrain.get("dataset") or {}
                prev_name = str(prev_ds.get("name", "")).lower()
                prev_task = str(prev_ds.get("task_level", "")).lower()
                prev_method = str(prev_pretrain.get("method", "")).lower()
                prev_induced = bool(prev_ds.get("induced", False))
                prev_has_induced = "induced" in prev_ds
                curr_name = str(self.cfg.pretrain.dataset.name).lower()
                curr_method = str(getattr(self.cfg.pretrain, "method", "")).lower()
                curr_task_cfg = str(self.cfg.pretrain.dataset.task_level).lower()
                curr_task_raw = str(getattr(self, "task_level_raw", curr_task_cfg)).lower()
                curr_induced = bool(getattr(self.cfg.pretrain.dataset, "induced", False))
                task_match = (not prev_task) or (prev_task in {curr_task_cfg, curr_task_raw})
                name_match = (not prev_name) or (prev_name == curr_name)
                method_match = (not prev_method) or (prev_method == curr_method)
                induced_match = (not prev_has_induced) or (prev_induced == curr_induced)
                if not (name_match and method_match and task_match and induced_match):
                    raise RuntimeError(
                        f"Refusing to overwrite existing checkpoint at {self.run_dir} "
                        f"(found dataset={prev_name}, method={prev_method}, task={prev_task}, induced={prev_induced}; "
                        f"current={curr_name}/{curr_method}/{curr_task_raw}|{curr_task_cfg}/{curr_induced}). "
                        f"Please set pretrain.run_name explicitly."
                    )
            except RuntimeError:
                raise
            except Exception:
                pass
        save_checkpoint(
            path=self._checkpoint_path(),
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            cfg=self._artifact_config_dict(),
            dataset_meta=self.dataset_meta,
            metrics=metrics,
        )
        self.best_metrics = metrics
        self._save_training_log()
        self._save_model_architecture()
        print(f"[Best epoch updated] epoch={epoch} {self.monitor_name}={monitor_value:.4f}")

    def train_epoch(self, epoch: int) -> Tuple[float, Dict]:
        """Train the model for one epoch."""
        self.model.train()
        self.task.train()
        total_loss = 0.0
        logs: Dict[str, float] = {}

        for data in self.train_loader:
            self.optimizer.zero_grad()
            loss, log = self.task.step(
                model=self.model, 
                data=data, 
                device=self.device
            )
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            for k, v in log.items():
                logs[k] = logs.get(k, 0.0) + float(v)

        num_batches = len(self.train_loader)
        if num_batches == 0:
            raise RuntimeError("Train loader is empty; unable to run a pretraining epoch.")
        avg_loss = total_loss / num_batches
        logs = {k: v / num_batches for k, v in logs.items()}
        return avg_loss, logs

    def _evaluate_supervised(self, loader, prefix: str, mask_attr: str) -> Dict[str, float]:
        """Run evaluation for supervised pretraining."""
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
            task_type = str(getattr(self.cfg.pretrain.dataset, "task_type", "classification") or "classification").lower()
            for key, value in compute_supervised_metrics(logits=logits, labels=labels, task_type=task_type).items():
                metrics[f"{prefix}_{key}"] = float(value)
        return metrics

    def fit(self):
        """Run the pretraining process."""
        if getattr(self, "_skip_due_to_existing_checkpoint", False):
            return
        ensure_dir(
            path=self.cfg.pretrain.checkpoint_dir
        )
        ensure_dir(self.run_dir)
        ckpt_path = self._checkpoint_path()
        if os.path.isfile(ckpt_path) and getattr(self.cfg.pretrain, "skip_if_exists", True):
            print(f"[Pretrain] Checkpoint already exists, skipping: {ckpt_path}")
            return

        patience = int(getattr(self.cfg.pretrain, "early_stopping", 0) or 0)
        epochs_since_improvement = 0

        for epoch in range(1, self.cfg.pretrain.epochs + 1):
            start = time.time()
            loss, logs = self.train_epoch(epoch)
            duration = time.time() - start
            log_parts = [
                f"[Epoch {epoch}/{self.cfg.pretrain.epochs}]",
                f"train_loss={loss:.4f}",
            ]
            if logs:
                for k, v in logs.items():
                    if k == "batch_size":
                        continue
                    log_parts.append(f"{k}={v:.4f}")

            val_metrics: Dict[str, float] = {}
            test_metrics: Dict[str, float] = {}
            if self.cfg.pretrain.method == "supervised":
                val_metrics = self._evaluate_supervised(self.val_loader, prefix="val", mask_attr="val_mask")
                test_metrics = self._evaluate_supervised(self.test_loader, prefix="test", mask_attr="test_mask")
                if val_metrics:
                    for k, v in val_metrics.items():
                        if k == "batch_size":
                            continue
                        log_parts.append(f"{k}={v:.4f}")
                if test_metrics:
                    for k, v in test_metrics.items():
                        if k == "batch_size":
                            continue
                        log_parts.append(f"{k}={v:.4f}")

            log_parts.append(f"time={duration:.1f}s")
            log_str = " ".join(log_parts)
            print(log_str)

            merged_metrics = {k: float(v) for k, v in logs.items()}
            merged_metrics.update({k: float(v) for k, v in val_metrics.items()})
            merged_metrics.update({k: float(v) for k, v in test_metrics.items()})
            self.train_history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss),
                    "duration_sec": float(duration),
                    "metrics": merged_metrics,
                }
            )

            if self.monitor_name == "train_loss":
                monitor_value = loss
            else:
                monitor_value = merged_metrics.get(self.monitor_name)
                if monitor_value is None:
                    monitor_value = loss
            monitor_value = float(monitor_value)
            if not math.isfinite(monitor_value):
                monitor_value = float(loss)

            improved = self._is_improved(monitor_value)
            self._save_best_checkpoint(
                epoch=epoch,
                train_loss=loss,
                logs=merged_metrics,
                monitor_value=monitor_value,
            )

            if patience > 0:
                if improved:
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    print(
                        f"[Pretrain] Early stopping at epoch {epoch} "
                        f"(no improvement in {patience} epochs)."
                    )
                    break
        # If no checkpoint was saved (e.g., monitor_value never improved), save the last state.
        if not os.path.isfile(self._checkpoint_path()):
            last_metrics = self.train_history[-1]["metrics"] if self.train_history else {}
            save_checkpoint(
                path=self._checkpoint_path(),
                model=self.model,
                optimizer=self.optimizer,
                epoch=self.train_history[-1]["epoch"] if self.train_history else 0,
                cfg=self._artifact_config_dict(),
                dataset_meta=self.dataset_meta,
                metrics=last_metrics,
            )
            self.best_metric = self.best_metric if self.best_metric is not None else float("inf")
            print(f"[Pretrain] No best checkpoint was saved; wrote last state to {self._checkpoint_path()}")

        print(f"Pretraining complete. Best {self.monitor_name}: {self.best_metric:.4f} at epoch {self.best_epoch}.")
        if hasattr(self, "best_metrics"):
            for metric_name in ("test_acc", "test_micro_f1", "test_macro_f1", "test_auc", "test_mae", "test_mse"):
                metric_value = self.best_metrics.get(metric_name)
                if metric_value is not None:
                    print(f"[Pretrain] Best-epoch {metric_name}={metric_value:.4f}")
