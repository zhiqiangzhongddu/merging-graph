import json
import os
import time
from numbers import Integral
from typing import Any, Dict, Tuple

import torch
from torch import optim

from code.data_loader import create_dataset, dataset_info, log_split_instance_counts, make_loaders
from code.model import build_encoder_from_cfg
from code.pretrain.checkpoint import cfg_to_dict, save_checkpoint
from code.utils import compute_supervised_metrics, ensure_dir, format_split_for_name, set_seed
from .encoders.prompt_encoder import build_prompt_encoder_from_cfg
from .supervised import FinetuneSupervised
from .registry import build_finetune_task
from .task_base import FinetuneTask

_LOG_EXCLUDED_METRICS = {"batch_size", "test_loss"}
_BEST_TEST_METRICS_TO_PRINT = ("test_acc", "test_micro_f1", "test_macro_f1", "test_auc", "test_mae", "test_mse")


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


class FinetuneRunner:
    """Fine-tune a pretrained encoder on a target dataset."""

    def __init__(self, cfg, pretrained_checkpoint: str, pretrained_run_name: str = None):
        if not os.path.isfile(pretrained_checkpoint):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_checkpoint}")

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

        self.pretrained_checkpoint = pretrained_checkpoint
        self.pretrained_run_name = pretrained_run_name or os.path.splitext(os.path.basename(pretrained_checkpoint))[0]
        self.checkpoint = torch.load(pretrained_checkpoint, map_location="cpu")
        self.pretrain_cfg = self.checkpoint.get("cfg", {}) or {}
        self.pretrain_dataset_meta = self.checkpoint.get("dataset", {}) or {}
        pretrain_block = self.pretrain_cfg.get("pretrain", {}) if isinstance(self.pretrain_cfg, dict) else {}
        pretrain_ds_block = pretrain_block.get("dataset", {}) if isinstance(pretrain_block, dict) else {}

        self.pretrain_dataset_name = (
            pretrain_ds_block.get("name")
            or ((self.pretrain_cfg.get("dataset") or {}).get("name") if isinstance(self.pretrain_cfg, dict) else None)
        )
        self.pretrain_task_level = (
            pretrain_ds_block.get("task_level")
            or ((self.pretrain_cfg.get("dataset") or {}).get("task_level") if isinstance(self.pretrain_cfg, dict) else None)
        )
        self.pretrain_method = (
            pretrain_block.get("method") if isinstance(pretrain_block, dict) else None
        )

        self._apply_pretrained_model_cfg()

        target_ds = self._get_target_dataset_cfg()
        raw_task_level = target_ds["task_level"]
        self.task_level_raw = raw_task_level
        self.split = self._resolve_split()
        self.finetune_method = (
            getattr(getattr(cfg, "finetune", None), "method", "supervised") or "supervised"
        ).lower().replace("-", "_")
        self.freeze_pretrained_effective = self._effective_freeze_pretrained()
        self.run_name = self._build_run_name()
        self.run_group = _checkpoint_dataset_dir_name(target_ds["name"])
        self.run_dir = os.path.join(self.cfg.finetune.checkpoint_dir, self.run_group)
        # Backward-compatible lookup for checkpoints saved by the previous layout.
        self.legacy_run_dir = os.path.join(self.cfg.finetune.checkpoint_dir, self.run_name)
        self._skip_due_to_existing_checkpoint = False
        ckpt_path = self._existing_checkpoint_path()
        if getattr(self.cfg.finetune, "skip_if_exists", True) and ckpt_path is not None:
            self._skip_due_to_existing_checkpoint = True
            print(f"[Finetune] Checkpoint already exists, skipping: {ckpt_path}")
            return

        induced = target_ds["induced"]
        self.dataset = create_dataset(
            name=target_ds["name"],
            root=target_ds["root"],
            task_level=raw_task_level,
            feat_reduction=target_ds["feat_reduction"],
            feat_reduction_dim=target_ds["feat_reduction_dim"],
            feature_svd_dir=target_ds["feature_svd_dir"],
            induced=induced,
            induced_min_size=target_ds["induced_min_size"],
            induced_max_size=target_ds["induced_max_size"],
            induced_max_hops=target_ds["induced_max_hops"],
            induced_root=_shared_induced_root(self.cfg, target_ds.get("induced_root", "")),
            split=self.split,
            seed=self.cfg.seed,
            split_root=_shared_split_root(self.cfg),
            force_reload_raw=target_ds.get("force_reload_raw", False),
        )
        effective_task_level = "graph" if induced else raw_task_level
        target_cfg = self.cfg.finetune.dataset
        target_cfg.task_level = effective_task_level
        target_cfg.task_level_raw = raw_task_level
        target_cfg.name = target_ds["name"]
        target_cfg.induced = induced

        self.dataset_meta = dataset_info(
            dataset=self.dataset,
            task_level=raw_task_level,
            name=target_ds["name"],
            induced=induced,
        )
        cfg.model.in_dim = cfg.model.in_dim or self.dataset_meta["num_node_features"]
        if getattr(target_cfg, "num_classes", None) is None:
            target_cfg.num_classes = self.dataset_meta.get("num_classes")
        if getattr(target_cfg, "label_dim", None) is None:
            target_cfg.label_dim = self.dataset_meta.get("label_dim")

        if self.finetune_method == "edgeprompt":
            self.model = build_prompt_encoder_from_cfg(
                cfg=cfg,
                in_dim=cfg.model.in_dim,
            ).to(self.device)
        else:
            self.model = build_encoder_from_cfg(
                cfg=cfg,
                in_dim=cfg.model.in_dim,
            ).to(self.device)
        missing, unexpected = self.model.load_state_dict(
            self.checkpoint.get("model_state", {}),
            strict=False,
        )
        if missing:
            print(f"[Finetune] Missing keys when loading pretrained weights: {missing}")
        if unexpected:
            print(f"[Finetune] Unexpected keys when loading pretrained weights: {unexpected}")
        model_name = getattr(cfg.model, "name", "model")
        print(f"[Finetune] Encoder architecture ({model_name}):\n{self.model}")

        if self.finetune_method == "supervised":
            self.task = FinetuneSupervised(cfg).to(self.device)
        else:
            self.task = build_finetune_task(self.finetune_method, cfg).to(self.device)
        self._maybe_freeze_pretrained_encoder()
        print(f"[Finetune] Task head:\n{self.task}")
        self.task_supports_epoch = isinstance(self.task, FinetuneTask)
        self.optimizers = None
        if self.task_supports_epoch:
            optimizers = self.task.build_optimizers(self.model)
            if optimizers is None:
                params = [p for p in self.model.parameters() if p.requires_grad] + list(
                    self.task.parameters_to_optimize()
                )
                self.optimizer = optim.Adam(
                    params=params,
                    lr=cfg.finetune.lr,
                    weight_decay=cfg.finetune.weight_decay,
                )
                self.optimizers = {"primary": self.optimizer}
            elif isinstance(optimizers, dict):
                self.optimizers = optimizers
                primary = optimizers.get("primary")
                if primary is None:
                    primary = next(iter(optimizers.values()))
                self.optimizer = primary
            else:
                self.optimizer = optimizers
                self.optimizers = {"primary": optimizers}
        else:
            params = [p for p in self.model.parameters() if p.requires_grad] + list(self.task.parameters())
            self.optimizer = optim.Adam(
                params=params,
                lr=cfg.finetune.lr,
                weight_decay=cfg.finetune.weight_decay,
            )
            self.optimizers = {"primary": self.optimizer}

        self._init_monitoring()
        self.train_history = []
        ensure_dir(self.run_dir)
        self.train_loader, self.val_loader, self.test_loader = self._make_loaders(induced=induced)
        log_split_instance_counts(
            self.train_loader,
            self.val_loader,
            self.test_loader,
            task_level=self._loader_task_level(induced=induced),
            split=self.split,
            induced=induced,
            prefix="[Finetune][Split]",
        )

    def _apply_pretrained_model_cfg(self) -> None:
        """Align model/dataset hyperparameters with the pretrained run when available."""
        if not isinstance(self.pretrain_cfg, dict):
            return
        model_cfg = self.pretrain_cfg.get("model", {}) or {}
        model_name = model_cfg.get("name")
        if model_name:
            self.cfg.model.name = model_name
        # Also pull pretrain-time input dimension so loading state_dict doesn't mismatch.
        for attr in ("in_dim", "hidden_dim", "out_dim", "num_layers", "activation", "dropout", "graph_pooling", "use_batchnorm"):
            if attr in model_cfg:
                setattr(self.cfg.model, attr, model_cfg[attr])
        gat_cfg = model_cfg.get("gat", {}) or {}
        if isinstance(gat_cfg, dict) and "heads" in gat_cfg:
            self.cfg.model.gat.heads = gat_cfg["heads"]

    def _get_target_dataset_cfg(self) -> Dict[str, Any]:
        """Resolve target dataset settings, allowing finetune overrides."""
        finetune_cfg = getattr(self.cfg, "finetune", None)
        target_cfg = getattr(getattr(self.cfg, "finetune", None), "dataset", None)
        pretrain_ds = getattr(getattr(self.cfg, "pretrain", None), "dataset", None)
        name = getattr(target_cfg, "name", None) or getattr(pretrain_ds, "name", None)
        task_level = getattr(target_cfg, "task_level", None) or getattr(pretrain_ds, "task_level", None)
        induced = getattr(target_cfg, "induced", None)
        if induced is None and pretrain_ds is not None:
            induced = getattr(pretrain_ds, "induced", False)
        root = getattr(target_cfg, "root", None) or getattr(pretrain_ds, "root", None)
        split_root = _shared_split_root(self.cfg)
        feat_reduction = getattr(target_cfg, "feat_reduction", True)
        feat_reduction_dim = getattr(
            target_cfg,
            "feat_reduction_svd_dim",
            getattr(target_cfg, "feat_reduction_dim", None),
        )
        if feat_reduction_dim is None:
            feat_reduction_dim = getattr(
                pretrain_ds,
                "feat_reduction_svd_dim",
                getattr(pretrain_ds, "feat_reduction_dim", 100),
            )
        feature_svd_dir = getattr(target_cfg, "feature_svd_dir", None) or getattr(
            pretrain_ds, "feature_svd_dir", "data/feature_svd"
        )
        if feature_svd_dir in (None, ""):
            feature_svd_dir = "data/feature_svd"
        subgraph_svd = getattr(target_cfg, "subgraph_svd", False)
        induced_root = getattr(target_cfg, "induced_root", None) or getattr(pretrain_ds, "induced_root", None) or ""
        induced_min_size = getattr(target_cfg, "induced_min_size", 10)
        induced_max_size = getattr(target_cfg, "induced_max_size", 30)
        induced_max_hops = getattr(target_cfg, "induced_max_hops", 5)
        force_reload_raw = getattr(target_cfg, "force_reload_raw", False)

        method_name = (
            str(getattr(finetune_cfg, "method", "supervised") or "supervised")
            .lower()
            .replace("-", "_")
        )
        if method_name == "gpf":
            gpf_cfg = getattr(finetune_cfg, "gpf", None)
            prefer_non_induced_node = bool(getattr(gpf_cfg, "prefer_non_induced_node", True)) if gpf_cfg is not None else True
            if prefer_non_induced_node and bool(induced) and str(task_level or "").lower() == "node":
                induced = False
                print(
                    "[Finetune][gpf] Overriding finetune.dataset.induced=True to False for node-level tuning. "
                    "Set finetune.gpf.prefer_non_induced_node=False to keep induced subgraphs."
                )
        if method_name == "edgeprompt":
            edge_cfg = getattr(finetune_cfg, "edgeprompt", None)
            use_official_node_subgraphs = (
                bool(getattr(edge_cfg, "use_official_node_subgraphs", True)) if edge_cfg is not None else True
            )
            if use_official_node_subgraphs and bool(induced) and str(task_level or "").lower() == "node":
                induced_min_size = int(getattr(edge_cfg, "node_subgraph_min_size", 1)) if edge_cfg is not None else 1
                induced_max_size = int(getattr(edge_cfg, "node_subgraph_max_size", 100000)) if edge_cfg is not None else 100000
                induced_max_hops = int(getattr(edge_cfg, "node_subgraph_hops", 2)) if edge_cfg is not None else 2
                print(
                    "[Finetune][EdgePrompt] Using official-style node subgraph settings: "
                    f"hops={induced_max_hops}, min_size={induced_min_size}, max_size={induced_max_size}."
                )

        if force_reload_raw:
            feat_reduction = False
            feat_reduction_dim = 0
        return {
            "name": name,
            "task_level": task_level,
            "induced": induced,
            "root": root,
            "split_root": split_root,
            "feat_reduction": feat_reduction,
            "feat_reduction_dim": feat_reduction_dim,
            "feature_svd_dir": feature_svd_dir,
            "subgraph_svd": subgraph_svd,
            "induced_root": induced_root,
            "induced_min_size": induced_min_size,
            "induced_max_size": induced_max_size,
            "induced_max_hops": induced_max_hops,
            "force_reload_raw": force_reload_raw,
        }

    def _init_monitoring(self) -> None:
        """
        Initialize early stopping monitoring.

        Policy:
        - Explicit `finetune.monitor_metric`: respected when set to a known metric name
        - Classification: monitor `val_acc`
        - Few-shot split without validation weight: monitor `train_loss`
        - Prompt-only methods (all_in_one/gppt): monitor `train_loss`
        - Edge prediction: monitor `val_auc`
        - Regression: monitor `train_loss` (loss-based early stopping)
        - `none`/`disabled`: disable early stopping checks
        """
        monitor_metric = getattr(self.cfg.finetune, "monitor_metric", "auto")
        monitor_metric = monitor_metric.lower() if monitor_metric else "auto"
        task_type = str(getattr(self.cfg.finetune.dataset, "task_type", "classification") or "classification").lower()
        task_level = str(getattr(self, "task_level_raw", self.cfg.finetune.dataset.task_level) or "").lower()
        label_dim = int(getattr(self.cfg.finetune.dataset, "label_dim", 1) or 1)
        few_shot_no_val = self._few_shot_without_validation()
        method_name = str(getattr(self, "finetune_method", getattr(self.cfg.finetune, "method", "")) or "").lower()
        method_name = method_name.replace("-", "_")
        gpf_cfg = getattr(getattr(self.cfg, "finetune", None), "gpf", None)
        gpf_monitor_train_loss = bool(getattr(gpf_cfg, "monitor_train_loss", False)) if gpf_cfg is not None else False

        explicit_monitor_map = {
            "train_loss": ("train_loss", "min"),
            "val_loss": ("val_loss", "min"),
            "test_loss": ("test_loss", "min"),
            "train_acc": ("train_acc", "max"),
            "val_acc": ("val_acc", "max"),
            "test_acc": ("test_acc", "max"),
            "train_auc": ("train_auc", "max"),
            "val_auc": ("val_auc", "max"),
            "test_auc": ("test_auc", "max"),
            "train_micro_f1": ("train_micro_f1", "max"),
            "val_micro_f1": ("val_micro_f1", "max"),
            "test_micro_f1": ("test_micro_f1", "max"),
            "train_macro_f1": ("train_macro_f1", "max"),
            "val_macro_f1": ("val_macro_f1", "max"),
            "test_macro_f1": ("test_macro_f1", "max"),
            "train_mae": ("train_mae", "min"),
            "val_mae": ("val_mae", "min"),
            "test_mae": ("test_mae", "min"),
            "train_mse": ("train_mse", "min"),
            "val_mse": ("val_mse", "min"),
            "test_mse": ("test_mse", "min"),
        }

        if monitor_metric in ("none", "disabled"):
            self.monitor_name = None
            self.monitor_mode = None
        elif monitor_metric in explicit_monitor_map:
            self.monitor_name, self.monitor_mode = explicit_monitor_map[monitor_metric]
        elif monitor_metric != "auto":
            supported = ", ".join(sorted(["auto", "none", "disabled", *explicit_monitor_map.keys()]))
            raise ValueError(
                f"Unsupported finetune.monitor_metric='{monitor_metric}'. "
                f"Supported values: {supported}"
            )
        elif few_shot_no_val:
            # Few-shot runs without validation weights have no val samples.
            self.monitor_name = "train_loss"
            self.monitor_mode = "min"
        elif method_name in {"all_in_one", "gppt"} or (method_name == "gpf" and gpf_monitor_train_loss):
            # Prompt-only methods default to early stopping on train loss.
            self.monitor_name = "train_loss"
            self.monitor_mode = "min"
        elif task_level == "edge":
            self.monitor_name = "val_auc"
            self.monitor_mode = "max"
        elif task_type == "classification" and label_dim > 1:
            self.monitor_name = "val_auc"
            self.monitor_mode = "max"
        elif task_type == "regression":
            self.monitor_name = "train_loss"
            self.monitor_mode = "min"
        else:
            self.monitor_name = "val_acc"
            self.monitor_mode = "max"

        if self.monitor_name is None:
            self.best_metric = float("inf")
        else:
            self.best_metric = float("-inf") if self.monitor_mode == "max" else float("inf")
        self.best_epoch = None
        self.best_metrics: Dict[str, float] = {}

    def _is_few_shot_split(self) -> bool:
        """Return True when split uses few-shot form (shots_per_class, val_weight, test_weight)."""
        split = getattr(self, "split", None)
        if split is None or len(split) != 3:
            return False
        first, val_ratio, test_ratio = split
        shots_like = self._is_valid_shot_count(first)
        if not shots_like:
            return False
        try:
            val = float(val_ratio)
            test = float(test_ratio)
        except Exception:
            return False
        return val >= 0.0 and test >= 0.0 and (val + test) > 0.0

    def _few_shot_without_validation(self) -> bool:
        split = getattr(self, "split", None)
        if not self._is_few_shot_split() or split is None or len(split) != 3:
            return False
        try:
            val = float(split[1])
            test = float(split[2])
        except Exception:
            return False
        return val <= 1e-12 and test > 0.0

    @staticmethod
    def _is_valid_shot_count(value: object) -> bool:
        """Validate the 'shots' field used by few-shot split shorthand."""
        try:
            numeric = float(value)
            return numeric.is_integer() and numeric >= 1.0
        except Exception:
            return isinstance(value, Integral) and not isinstance(value, bool) and int(value) >= 1

    def _make_loaders(self, induced: bool):
        ds_cfg = self.cfg.finetune.dataset
        split = self.split
        split_task_level = self.task_level_raw if induced else ds_cfg.task_level
        loader_task_level = self._loader_task_level(induced=induced)
        split_dataset_name = _split_dataset_name(ds_cfg.name, split_task_level, self.cfg.seed)
        return make_loaders(
            dataset=self.dataset,
            dataset_name=split_dataset_name,
            task_level=loader_task_level,
            batch_size=self.cfg.finetune.batch_size,
            num_workers=self.cfg.finetune.num_workers,
            split=split,
            seed=self.cfg.seed,
            induced=induced,
            split_root=_shared_split_root(self.cfg),
        )

    def _loader_task_level(self, induced: bool) -> str:
        # Keep edge split semantics for induced edge tasks.
        if induced and str(self.task_level_raw).lower() == "edge":
            return "edge"
        return self.cfg.finetune.dataset.task_level

    def _effective_freeze_pretrained(self) -> bool:
        finetune_cfg = getattr(self.cfg, "finetune", None)
        freeze = bool(getattr(finetune_cfg, "freeze_pretrained", False))
        method = str(getattr(finetune_cfg, "method", "supervised") or "supervised").lower().replace("-", "_")

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

        if method == "gpf":
            gpf_cfg = getattr(finetune_cfg, "gpf", None)
            raw_update = getattr(gpf_cfg, "update_pretrained", None) if gpf_cfg is not None else None
            if raw_update is not None:
                update_pretrained = _to_bool(raw_update)
                if update_pretrained:
                    return False
            return freeze
        if method == "gppt":
            gppt_cfg = getattr(finetune_cfg, "gppt", None)
            if gppt_cfg is not None and hasattr(gppt_cfg, "force_freeze_encoder"):
                # GPPT method setting takes precedence so it can match official behavior.
                return _to_bool(getattr(gppt_cfg, "force_freeze_encoder"))
            return freeze
        if method == "graphprompt":
            gp_cfg = getattr(finetune_cfg, "graphprompt", None)
            # GraphPrompt follows official prompt-tuning behavior by default:
            # keep the pretrained encoder frozen unless explicitly enabled.
            update_pretrained = bool(getattr(gp_cfg, "update_pretrained", False)) if gp_cfg is not None else False
            return not update_pretrained
        return freeze

    def _maybe_freeze_pretrained_encoder(self) -> None:
        freeze = bool(getattr(self, "freeze_pretrained_effective", self._effective_freeze_pretrained()))
        if not freeze:
            return
        frozen_params = 0
        for param in self.model.parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_params += param.numel()
        print(f"[Finetune] Frozen encoder parameters (count={frozen_params})")

    def _resolve_split(self) -> Tuple[float, float, float]:
        ds_cfg = self.cfg.finetune.dataset
        split = getattr(ds_cfg, "fixed_split", None)
        if split is None:
            return (0.1, 0.1, 0.8)
        parts = list(split)
        if len(parts) != 3:
            raise ValueError(f"finetune.dataset.fixed_split must be a length-3 tuple/list, got: {split}")

        if self._is_valid_shot_count(parts[0]):
            try:
                val_ratio = float(parts[1])
                test_ratio = float(parts[2])
            except Exception as exc:
                raise ValueError("Few-shot split must be numeric: (shots_per_class, val_weight, test_weight).") from exc
            if val_ratio < 0.0 or test_ratio < 0.0:
                raise ValueError("Few-shot split requires non-negative val/test weights.")
            if (val_ratio + test_ratio) <= 0.0:
                raise ValueError("Few-shot split requires val_weight + test_weight > 0.")
        return tuple(parts)

    @staticmethod
    def _should_log_metric(metric_name: str) -> bool:
        return metric_name not in _LOG_EXCLUDED_METRICS

    def _build_run_name(self) -> str:
        dataset_cfg = self.cfg.finetune.dataset
        finetune_cfg = self.cfg.finetune
        split_tag = format_split_for_name(self.split)
        freeze_flag = int(
            bool(getattr(self, "freeze_pretrained_effective", getattr(finetune_cfg, "freeze_pretrained", False)))
        )
        raw_task_level = getattr(self, "task_level_raw", getattr(dataset_cfg, "task_level", ""))
        parts = [
            "ft",
            self.finetune_method,
            self._method_run_tag(),
            self.pretrained_run_name,
            f"to_{dataset_cfg.name}",
            f"induced{int(getattr(dataset_cfg, 'induced', False))}",
            split_tag,
            f"task{raw_task_level}",
            f"frz{freeze_flag}",
            *self._method_variant_tags(),
            f"e{finetune_cfg.epochs}",
            f"lr{finetune_cfg.lr:g}" if isinstance(finetune_cfg.lr, (int, float)) else f"lr{finetune_cfg.lr}",
            f"bs{finetune_cfg.batch_size}",
            f"seed{self.cfg.seed}",
        ]
        return "_".join(str(p) for p in parts if p not in ("", None))

    def _method_run_tag(self) -> str:
        method = str(getattr(self, "finetune_method", "") or "").lower()
        finetune_cfg = getattr(self.cfg, "finetune", None)
        if method == "edgeprompt":
            ep_cfg = getattr(finetune_cfg, "edgeprompt", None)
            plus = int(bool(getattr(ep_cfg, "plus", False))) if ep_cfg is not None else 0
            return f"plus{plus}"
        if method == "gpf":
            gpf_cfg = getattr(finetune_cfg, "gpf", None)
            plus = int(bool(getattr(gpf_cfg, "plus", False))) if gpf_cfg is not None else 0
            if plus:
                p_num = int(getattr(gpf_cfg, "p_num", 0) or 0)
                return f"plus{plus}_p{p_num}"
            return f"plus{plus}"
        if method == "graphprompt":
            gp_cfg = getattr(finetune_cfg, "graphprompt", None)
            plus = int(bool(getattr(gp_cfg, "plus", False))) if gp_cfg is not None else 0
            score_mode = str(getattr(gp_cfg, "score_mode", "auto") or "auto").lower()
            return f"plus{plus}_{score_mode}"
        return ""

    def _method_variant_tags(self):
        """
        Add method-specific tags to avoid checkpoint name collisions across variants.
        """
        method = str(getattr(self, "finetune_method", "") or "").lower()
        finetune_cfg = getattr(self.cfg, "finetune", None)
        if finetune_cfg is None:
            return []

        if method == "edgeprompt":
            edge_cfg = getattr(finetune_cfg, "edgeprompt", None)
            if edge_cfg is None:
                return []

            num_anchors = getattr(edge_cfg, "num_anchors", None)
            add_self_loops = getattr(edge_cfg, "add_self_loops", None)

            tags = []
            if num_anchors is None:
                tags.append("anchorsauto")
            else:
                tags.append(f"anchors{int(num_anchors)}")
            if add_self_loops is None:
                tags.append("loopsauto")
            else:
                tags.append(f"loops{int(bool(add_self_loops))}")
            return tags

        if method == "gpf":
            gpf_cfg = getattr(finetune_cfg, "gpf", None)
            if gpf_cfg is None:
                return []

            tags = []
            head_layers = int(getattr(gpf_cfg, "head_layers", 1) or 1)
            if head_layers > 1:
                tags.append(f"head{head_layers}")
                head_hidden_dim = int(getattr(gpf_cfg, "head_hidden_dim", 0) or 0)
                if head_hidden_dim > 0:
                    tags.append(f"hhd{head_hidden_dim}")
            head_lr_scale = float(getattr(gpf_cfg, "head_lr_scale", 1.0) or 1.0)
            if abs(head_lr_scale - 1.0) > 1e-12:
                tags.append(f"hlrs{head_lr_scale:g}")

            raw_update_pretrained = getattr(gpf_cfg, "update_pretrained", None)
            if raw_update_pretrained is not None:
                if isinstance(raw_update_pretrained, str):
                    update_token = raw_update_pretrained.strip().lower()
                    update_flag = update_token in {"1", "true", "yes", "y", "on"}
                else:
                    update_flag = bool(raw_update_pretrained)
                tags.append(f"upd{int(update_flag)}")

            optimizer_name = str(getattr(gpf_cfg, "optimizer", "adam") or "adam").lower()
            if optimizer_name != "adam":
                tags.append(optimizer_name)

            monitor_train_loss = bool(getattr(gpf_cfg, "monitor_train_loss", False))
            if monitor_train_loss:
                tags.append("mtrain")
            return tags

        return []

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
            "pretrained_from": {
                "run_name": self.pretrained_run_name,
                "checkpoint": self.pretrained_checkpoint,
                "dataset": self.pretrain_dataset_name,
                "task_level": self.pretrain_task_level,
                "method": self.pretrain_method,
                "freeze_pretrained_effective": bool(getattr(self, "freeze_pretrained_effective", False)),
                "dataset_meta": self.pretrain_dataset_meta,
            },
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

    def _resolve_monitor_value(
        self,
        train_loss: float,
        train_logs: Dict[str, float],
        val_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
    ) -> float:
        """
        Resolve the scalar used for checkpoint selection and early stopping.

        Falls back to `train_loss` when the requested metric is absent so runs
        can continue even when a split has no evaluable samples.
        """
        if self.monitor_name is None or self.monitor_name == "train_loss":
            return float(train_loss)
        if self.monitor_name.startswith("train_"):
            monitor_value = train_logs.get(self.monitor_name)
        elif self.monitor_name.startswith("val_"):
            monitor_value = val_metrics.get(self.monitor_name)
        elif self.monitor_name.startswith("test_"):
            monitor_value = test_metrics.get(self.monitor_name)
        else:
            monitor_value = val_metrics.get(self.monitor_name)
        if monitor_value is None:
            return float(train_loss)
        return float(monitor_value)

    def _monitor_uses_train_split(self) -> bool:
        """Train-only monitor metrics do not require validation passes."""
        return bool(self.monitor_name) and str(self.monitor_name).startswith("train_")

    def _save_best_checkpoint(
        self,
        epoch: int,
        train_loss: float,
        train_logs: Dict[str, float],
        val_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        monitor_value: float,
    ) -> None:
        if self.monitor_name is None:
            # Keep checkpoint/log artifacts aligned with the latest epoch when
            # explicit monitoring is disabled.
            self.best_metric = float(monitor_value)
            self.best_epoch = epoch
        else:
            if not self._is_improved(monitor_value):
                return
            self.best_metric = float(monitor_value)
            self.best_epoch = epoch
        metrics = {
            "train_loss": train_loss,
            "best_epoch": epoch,
            **train_logs,
            **val_metrics,
            **test_metrics,
        }
        if self.monitor_name is not None:
            metrics[self.monitor_name] = float(monitor_value)
        task_state = None
        try:
            task_state = self.task.state_dict()
        except Exception:
            task_state = None

        optimizer_states = {}
        if isinstance(getattr(self, "optimizers", None), dict):
            for name, opt in self.optimizers.items():
                try:
                    optimizer_states[name] = opt.state_dict()
                except Exception:
                    continue
        save_checkpoint(
            path=self._checkpoint_path(),
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            cfg=self.cfg,
            dataset_meta=self.dataset_meta,
            metrics=metrics,
            extra={
                "pretrained_from": {
                    "run_name": self.pretrained_run_name,
                    "checkpoint": self.pretrained_checkpoint,
                    "dataset": self.pretrain_dataset_name,
                    "task_level": self.pretrain_task_level,
                    "method": self.pretrain_method,
                    "dataset_meta": self.pretrain_dataset_meta,
                },
                "finetune_task_state": task_state,
                "finetune_optimizers_state": optimizer_states,
            },
        )
        self.best_metrics = metrics
        self._save_training_log()
        if self.monitor_name is None:
            print(f"[Finetune] Checkpoint updated at epoch={epoch} (monitor disabled).")
        else:
            print(f"[Finetune] Best epoch updated: epoch={epoch} {self.monitor_name}={monitor_value:.4f}")

    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
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
            task_type = str(getattr(self.cfg.finetune.dataset, "task_type", "classification") or "classification").lower()
            for key, value in compute_supervised_metrics(logits=logits, labels=labels, task_type=task_type).items():
                metrics[f"{prefix}_{key}"] = float(value)
        return metrics

    def fit(self) -> None:
        if getattr(self, "_skip_due_to_existing_checkpoint", False):
            return
        ensure_dir(path=self.cfg.finetune.checkpoint_dir)
        ensure_dir(self.run_dir)
        ckpt_path = self._existing_checkpoint_path()
        if getattr(self.cfg.finetune, "skip_if_exists", True) and ckpt_path is not None:
            print(f"[Finetune] Checkpoint already exists, skipping: {ckpt_path}")
            return
        if os.path.isfile(self._checkpoint_path()):
            print(f"[Finetune] Overwriting existing checkpoint: {self._checkpoint_path()}")

        patience = int(getattr(self.cfg.finetune, "early_stopping", 0) or 0)
        method_name = (
            str(getattr(self.cfg.finetune, "method", "supervised") or "supervised")
            .lower()
            .replace("-", "_")
        )
        if method_name == "edgeprompt":
            edge_cfg = getattr(self.cfg.finetune, "edgeprompt", None)
            disable_es = bool(getattr(edge_cfg, "disable_early_stopping", True)) if edge_cfg is not None else True
            if disable_es and patience != 0:
                print("[Finetune][EdgePrompt] Disabling early stopping (fixed-epoch training).")
                patience = 0
        if method_name == "gpf":
            gpf_cfg = getattr(self.cfg.finetune, "gpf", None)
            disable_es = bool(getattr(gpf_cfg, "disable_early_stopping", True)) if gpf_cfg is not None else True
            if disable_es and patience != 0:
                print("[Finetune][gpf] Disabling early stopping (fixed-epoch training).")
                patience = 0
        epochs_since_improvement = 0

        # Allow task to override effective epochs when needed.
        total_epochs = self.cfg.finetune.epochs
        if hasattr(self.task, "get_effective_epochs"):
            total_epochs = self.task.get_effective_epochs(total_epochs)
            if total_epochs != self.cfg.finetune.epochs:
                print(f"[Finetune] Using effective epochs: {total_epochs} (original: {self.cfg.finetune.epochs})")

        for epoch in range(1, total_epochs + 1):
            start = time.time()
            monitor_on_train = self._monitor_uses_train_split()
            if self.task_supports_epoch:
                train_loss, train_logs = self.task.train_epoch(
                    model=self.model,
                    loader=self.train_loader,
                    device=self.device,
                    optimizers=self.optimizers,
                )
                if hasattr(self.task, "on_epoch_end"):
                    self.task.on_epoch_end(self.model, self.train_loader, self.device)
                # Skip val evaluation when monitor metric is train-split only.
                if monitor_on_train:
                    val_metrics = {}
                else:
                    val_metrics = self.task.evaluate_split(
                        model=self.model,
                        loader=self.val_loader,
                        device=self.device,
                        prefix="val",
                        mask_attr="val_mask",
                    )
                test_metrics = self.task.evaluate_split(
                    model=self.model,
                    loader=self.test_loader,
                    device=self.device,
                    prefix="test",
                    mask_attr="test_mask",
                )
            else:
                train_loss, train_logs = self.train_epoch()
                # Skip val evaluation when monitor metric is train-split only.
                if monitor_on_train:
                    val_metrics = {}
                else:
                    val_metrics = self._evaluate_split(self.val_loader, prefix="val", mask_attr="val_mask")
                test_metrics = self._evaluate_split(self.test_loader, prefix="test", mask_attr="test_mask")

            duration = time.time() - start
            log_parts = [
                f"[Finetune][Epoch {epoch}/{total_epochs}]",
                f"train_loss={train_loss:.4f}",
            ]
            for metrics in (train_logs, val_metrics, test_metrics):
                for k, v in metrics.items():
                    if not self._should_log_metric(k):
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

            monitor_value = self._resolve_monitor_value(
                train_loss=train_loss,
                train_logs=train_logs,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
            )
            improved = True if self.monitor_name is None else self._is_improved(monitor_value)

            self._save_best_checkpoint(
                epoch=epoch,
                train_loss=train_loss,
                train_logs=train_logs,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                monitor_value=monitor_value,
            )

            if patience > 0:
                if improved:
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    print(
                        f"[Finetune] Early stopping at epoch {epoch} "
                        f"(no improvement in {patience} epochs)."
                    )
                    break

        if self.monitor_name is not None:
            print(
                f"[Finetune] Complete. Best {self.monitor_name}: "
                f"{self.best_metric:.4f} at epoch {self.best_epoch}."
            )
        else:
            print(f"[Finetune] Complete. Final epoch: {epoch} (early stopping disabled).")
        for metric_name in _BEST_TEST_METRICS_TO_PRINT:
            metric_value = self.best_metrics.get(metric_name)
            if metric_value is not None:
                print(f"[Finetune] Best-epoch {metric_name}={metric_value:.4f}")
