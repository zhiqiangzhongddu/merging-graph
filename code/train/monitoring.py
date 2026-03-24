"""Train-specific monitoring policy helpers."""

from __future__ import annotations

from code.utils.monitoring import MonitorSpec, make_monitor_spec, resolve_explicit_monitor_spec


def resolve_train_monitor_spec(
    cfg,
    *,
    task_level: str,
    label_dim: int,
    few_shot_without_validation: bool = False,
) -> MonitorSpec:
    """Resolve training monitor selection, including train-specific auto policy."""
    spec = resolve_explicit_monitor_spec(
        raw_monitor_metric=getattr(cfg.train, "monitor_metric", "auto"),
        setting_name="train.monitor_metric",
    )
    if spec is not None:
        return spec

    task_type = str(getattr(cfg.train.dataset, "task_type", "classification") or "classification").lower()
    task_level = str(task_level or getattr(cfg.train.dataset, "task_level", "")).lower()
    label_dim = int(label_dim or 1)

    if few_shot_without_validation:
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
        f"setting=train.monitor_metric, task_type={task_type}, task_level={task_level}"
    )


__all__ = ["resolve_train_monitor_spec"]
