"""Naming helpers for split and run identifiers."""

from __future__ import annotations

from numbers import Integral
from typing import Optional


def format_split_for_name(split) -> str:
    """
    Format a split tuple for inclusion in filenames, e.g., (0.8,0.1,0.1) -> "split80-10-10".
    Returns an empty string when split is None or not iterable.
    """
    try:
        parts = list(split)
    except Exception:
        return ""

    if not parts:
        return ""

    first = parts[0]
    if isinstance(first, Integral) and not isinstance(first, bool):
        val_ratio = parts[1] if len(parts) > 1 else 0.0
        test_ratio = parts[2] if len(parts) > 2 else 0.0
        try:
            val_pct = int(round(float(val_ratio) * 100))
            test_pct = int(round(float(test_ratio) * 100))
        except Exception:
            val_pct, test_pct = 0, 0
        return f"fewshot{first}-{val_pct}-{test_pct}"

    try:
        numeric = [float(part) for part in parts]
    except Exception:
        return ""
    suffix = "-".join(str(int(round(part * 100))) for part in numeric)
    return f"split{suffix}"


def build_run_name_from_cfg(cfg, include_split: Optional[bool] = None) -> str:
    """
    Mirror the pretrainer run-name convention using the current cfg.

    Args:
        include_split:
            - None: include split tag only for supervised pretraining.
            - True: always include split tag when available.
            - False: never include split tag.
    """
    dataset_cfg = getattr(getattr(cfg, "pretrain", None), "dataset", None) or getattr(cfg, "dataset", None)
    model_cfg = getattr(cfg, "model", None)
    pretrain_cfg = getattr(cfg, "pretrain", None)
    split = getattr(dataset_cfg, "fixed_split", None) if dataset_cfg else None

    dataset_name = getattr(dataset_cfg, "name", "dataset") if dataset_cfg else "dataset"
    induced_flag = int(getattr(dataset_cfg, "induced", False)) if dataset_cfg else 0
    task_level = getattr(dataset_cfg, "task_level", "") if dataset_cfg else ""
    model_name = getattr(model_cfg, "name", "model") if model_cfg else "model"
    hidden_dim = getattr(model_cfg, "hidden_dim", "")
    out_dim = getattr(model_cfg, "out_dim", "")
    num_layers = getattr(model_cfg, "num_layers", "")
    epochs = getattr(pretrain_cfg, "epochs", "")
    lr = getattr(pretrain_cfg, "lr", "")
    batch_size = getattr(pretrain_cfg, "batch_size", "")
    seed = getattr(cfg, "seed", "")
    method = getattr(pretrain_cfg, "method", "method") if pretrain_cfg else "method"
    if include_split is None:
        include_split_effective = str(method).lower() == "supervised"
    else:
        include_split_effective = bool(include_split)
    split_tag = format_split_for_name(split) if include_split_effective else ""

    parts = [
        method,
        dataset_name,
        f"task{task_level}",
        f"induced{induced_flag}",
        split_tag,
        model_name,
        f"h{hidden_dim}",
        f"o{out_dim}",
        f"l{num_layers}",
        f"e{epochs}",
        f"lr{lr:g}" if isinstance(lr, (int, float)) else f"lr{lr}" if lr != "" else "",
        f"bs{batch_size}",
        f"seed{seed}",
    ]
    return "_".join(str(part) for part in parts if part not in ("", None))


__all__ = ["build_run_name_from_cfg", "format_split_for_name"]
