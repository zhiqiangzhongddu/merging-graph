"""Pretrain-specific monitoring policy helpers."""

from __future__ import annotations

from code.utils.monitoring import MonitorSpec, make_monitor_spec, resolve_explicit_monitor_spec


def resolve_pretrain_monitor_spec(
    cfg,
    *,
    task_level: str,
    label_dim: int,
) -> MonitorSpec:
    """Resolve pretraining monitor selection, including method-specific auto rules."""
    spec = resolve_explicit_monitor_spec(
        raw_monitor_metric=getattr(cfg.pretrain, "monitor_metric", "auto"),
        setting_name="pretrain.monitor_metric",
    )
    if spec is not None:
        return spec

    task_type = str(getattr(cfg.pretrain.dataset, "task_type", "classification") or "classification").lower()
    task_level = str(task_level or getattr(cfg.pretrain.dataset, "task_level", "")).lower()
    label_dim = int(label_dim or 1)
    method = str(getattr(cfg.pretrain, "method", "") or "").lower()
    is_supervised = method == "supervised"

    if method == "context_pred":
        return make_monitor_spec("balanced_loss", "min")
    if not is_supervised:
        return make_monitor_spec("train_loss", "min")
    if task_level == "edge":
        return make_monitor_spec("val_auc", "max")
    if task_type == "classification" and label_dim > 1:
        return make_monitor_spec("val_micro_f1", "max")
    if task_type == "regression":
        return make_monitor_spec("val_mae", "min")
    if task_type == "classification":
        return make_monitor_spec("val_acc", "max")

    raise ValueError(
        "Unsupported monitoring context: "
        f"setting=pretrain.monitor_metric, task_type={task_type}, task_level={task_level}"
    )


__all__ = ["resolve_pretrain_monitor_spec"]
