"""Shared monitoring primitives used by train, pretrain, and finetune."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional


EXPLICIT_MONITOR_MAP: Dict[str, tuple[str, str]] = {
    "balanced_loss": ("balanced_loss", "min"),
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


@dataclass(frozen=True)
class MonitorSpec:
    """Resolved monitoring configuration used by training loops."""

    name: Optional[str]
    mode: Optional[str]
    best_metric: float


def normalize_monitor_metric(raw_monitor_metric: object) -> str:
    """Normalize monitor config tokens to the canonical lowercase representation."""
    token = str(raw_monitor_metric or "auto").strip().lower()
    return token or "auto"


def _best_metric_for_mode(mode: Optional[str]) -> float:
    if mode == "max":
        return float("-inf")
    return float("inf")


def make_monitor_spec(name: Optional[str], mode: Optional[str]) -> MonitorSpec:
    """Create a monitor spec with the correct initial best metric."""
    if name is None or mode is None:
        return MonitorSpec(name=None, mode=None, best_metric=float("inf"))
    return MonitorSpec(name=name, mode=mode, best_metric=_best_metric_for_mode(mode))


def supported_monitor_metric_values() -> list[str]:
    """Return all accepted monitor_metric tokens in stable order."""
    return sorted(["auto", "disabled", "none", *EXPLICIT_MONITOR_MAP.keys()])


def resolve_explicit_monitor_spec(
    *,
    raw_monitor_metric: object,
    setting_name: str,
) -> Optional[MonitorSpec]:
    """Resolve disabled and explicit monitor settings; return None for `auto`."""
    monitor_metric = normalize_monitor_metric(raw_monitor_metric)

    if monitor_metric in {"none", "disabled"}:
        return make_monitor_spec(None, None)

    if monitor_metric in EXPLICIT_MONITOR_MAP:
        name, mode = EXPLICIT_MONITOR_MAP[monitor_metric]
        return make_monitor_spec(name, mode)

    if monitor_metric != "auto":
        supported = ", ".join(supported_monitor_metric_values())
        raise ValueError(
            f"Unsupported {setting_name}='{monitor_metric}'. Supported values: {supported}"
        )

    return None


def monitor_uses_train_split(monitor_name: Optional[str]) -> bool:
    """Return True when the selected monitor depends only on training logs."""
    if not monitor_name:
        return False
    token = str(monitor_name)
    return token == "balanced_loss" or token == "train_loss" or token.startswith("train_")


def resolve_monitor_value(
    monitor_name: Optional[str],
    *,
    train_loss: float,
    train_logs: Mapping[str, float],
    val_metrics: Mapping[str, float],
    test_metrics: Mapping[str, float],
) -> float:
    """Select the scalar used for checkpointing and early stopping."""
    if monitor_name is None or monitor_name == "train_loss":
        return float(train_loss)

    if monitor_name.startswith("train_"):
        monitor_value = train_logs.get(monitor_name)
    elif monitor_name.startswith("val_"):
        monitor_value = val_metrics.get(monitor_name)
    elif monitor_name.startswith("test_"):
        monitor_value = test_metrics.get(monitor_name)
    else:
        monitor_value = None
        for metrics in (train_logs, val_metrics, test_metrics):
            if monitor_name in metrics:
                monitor_value = metrics[monitor_name]
                break

    if monitor_value is None:
        return float(train_loss)

    monitor_value = float(monitor_value)
    if not math.isfinite(monitor_value):
        return float(train_loss)
    return monitor_value


__all__ = [
    "EXPLICIT_MONITOR_MAP",
    "MonitorSpec",
    "make_monitor_spec",
    "monitor_uses_train_split",
    "normalize_monitor_metric",
    "resolve_explicit_monitor_spec",
    "resolve_monitor_value",
    "supported_monitor_metric_values",
]
