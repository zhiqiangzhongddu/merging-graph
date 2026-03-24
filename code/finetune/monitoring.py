"""Finetune-specific monitoring policy helpers."""

from __future__ import annotations

from code.utils.monitoring import MonitorSpec, make_monitor_spec, resolve_explicit_monitor_spec


def resolve_finetune_monitor_spec(
    cfg,
    *,
    task_level: str,
    label_dim: int,
    few_shot_without_validation: bool = False,
    method_name: str = "",
) -> MonitorSpec:
    """Resolve finetuning monitor selection, including prompt-specific auto rules."""
    spec = resolve_explicit_monitor_spec(
        raw_monitor_metric=getattr(cfg.finetune, "monitor_metric", "auto"),
        setting_name="finetune.monitor_metric",
    )
    if spec is not None:
        return spec

    task_type = str(getattr(cfg.finetune.dataset, "task_type", "classification") or "classification").lower()
    task_level = str(task_level or getattr(cfg.finetune.dataset, "task_level", "")).lower()
    label_dim = int(label_dim or 1)
    method_name = str(method_name or getattr(cfg.finetune, "method", "") or "").lower().replace("-", "_")
    gpf_cfg = getattr(getattr(cfg, "finetune", None), "gpf", None)
    gpf_monitor_train_loss = bool(getattr(gpf_cfg, "monitor_train_loss", False)) if gpf_cfg is not None else False

    if few_shot_without_validation:
        return make_monitor_spec("train_loss", "min")
    if method_name in {"all_in_one", "gppt"} or (method_name == "gpf" and gpf_monitor_train_loss):
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
        f"setting=finetune.monitor_metric, task_type={task_type}, task_level={task_level}"
    )


__all__ = ["resolve_finetune_monitor_spec"]
