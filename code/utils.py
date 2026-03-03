import os
import random
from numbers import Integral
from typing import Dict, Optional
import numpy as np
import torch


def ensure_dir(path: str):
    """Ensure that a directory exists."""
    os.makedirs(path, exist_ok=True)


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
        numeric = [float(p) for p in parts]
    except Exception:
        return ""
    suffix = "-".join(str(int(round(p * 100))) for p in numeric)
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
    return "_".join(str(p) for p in parts if p not in ("", None))


def set_seed(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _classification_labels(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.detach().view(-1).cpu()
    if labels.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    if labels.dtype.is_floating_point:
        rounded = labels.round()
        if torch.allclose(labels, rounded, atol=1e-6):
            return rounded.long()
        return (labels > 0.5).long()
    return labels.long()


def _binary_targets_and_valid(labels: torch.Tensor):
    """
    Convert binary labels to {0,1} with a validity mask.

    Supports:
    - {0,1}
    - {-1,1}
    - {-1,0,1} where 0 denotes missing label (chem setting)
    """
    target = torch.as_tensor(labels).detach().float().cpu()
    valid = torch.isfinite(target)
    uses_signed = bool((target < 0).any().item()) if target.numel() else False
    if uses_signed:
        valid = valid & (target != 0)
        target = (target + 1.0) / 2.0
    target = target.clamp(min=0.0, max=1.0)
    return target, valid


def _safe_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    try:
        from sklearn.metrics import f1_score  # pylint: disable=import-outside-toplevel
    except Exception:
        return {"micro_f1": float("nan"), "macro_f1": float("nan")}
    try:
        micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        return {"micro_f1": micro, "macro_f1": macro}
    except Exception:
        return {"micro_f1": float("nan"), "macro_f1": float("nan")}


def _safe_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score  # pylint: disable=import-outside-toplevel
    except Exception:
        return float("nan")

    try:
        unique = np.unique(y_true)
        if unique.size < 2:
            return float("nan")
        if probs.ndim == 1:
            return float(roc_auc_score(y_true, probs))
        if probs.ndim == 2 and probs.shape[1] == 2:
            return float(roc_auc_score(y_true, probs[:, 1]))
        if probs.ndim == 2 and probs.shape[1] > 2:
            return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
        return float("nan")
    except Exception:
        return float("nan")


def compute_supervised_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_type: str = "classification",
) -> Dict[str, float]:
    task = str(task_type or "classification").lower()
    logits = torch.as_tensor(logits).detach().cpu()
    labels = torch.as_tensor(labels).detach().cpu()

    if logits.numel() == 0 or labels.numel() == 0:
        return {}

    if task == "regression":
        preds = logits.view(-1).float()
        target = labels.view(-1).float()
        n = min(int(preds.numel()), int(target.numel()))
        if n <= 0:
            return {}
        preds = preds[:n]
        target = target[:n]
        diff = preds - target
        mse = float((diff * diff).mean().item())
        mae = float(diff.abs().mean().item())
        return {"mse": mse, "mae": mae}

    # Multi-task binary classification.
    if (
        logits.dim() == 2
        and labels.dim() == 2
        and logits.size(0) == labels.size(0)
        and logits.size(1) == labels.size(1)
        and logits.size(1) > 1
    ):
        target, valid = _binary_targets_and_valid(labels)
        probs = torch.sigmoid(logits.float())
        preds = (probs >= 0.5).float()

        valid_f = valid.float()
        denom = valid_f.sum().clamp(min=1.0)
        acc = float((((preds == target).float() * valid_f).sum() / denom).item())

        # Flattened micro-F1 over all valid task labels.
        valid_flat = valid.view(-1)
        y_true_flat = target.view(-1)[valid_flat].long().numpy() if valid_flat.any() else np.array([], dtype=np.int64)
        y_pred_flat = preds.view(-1)[valid_flat].long().numpy() if valid_flat.any() else np.array([], dtype=np.int64)
        if y_true_flat.size == 0:
            micro_f1 = float("nan")
        else:
            micro_f1 = _safe_f1(y_true_flat, y_pred_flat)["micro_f1"]

        # Macro-F1 averaged across tasks with at least one valid label.
        macro_f1_vals = []
        auc_vals = []
        probs_np = probs.numpy()
        target_np = target.numpy()
        valid_np = valid.numpy()
        for tid in range(target_np.shape[1]):
            mask = valid_np[:, tid]
            if not np.any(mask):
                continue
            y_t = target_np[mask, tid]
            y_p = (probs_np[mask, tid] >= 0.5).astype(np.int64)
            f1_task = _safe_f1(y_t.astype(np.int64), y_p)["macro_f1"]
            if not np.isnan(f1_task):
                macro_f1_vals.append(float(f1_task))
            if np.unique(y_t).size >= 2:
                try:
                    from sklearn.metrics import roc_auc_score  # pylint: disable=import-outside-toplevel

                    auc_vals.append(float(roc_auc_score(y_t, probs_np[mask, tid])))
                except Exception:
                    pass

        macro_f1 = float(np.mean(macro_f1_vals)) if macro_f1_vals else float("nan")
        auc = float(np.mean(auc_vals)) if auc_vals else float("nan")
        return {
            "acc": acc,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "auc": auc,
        }

    if logits.dim() == 1:
        logits = logits.unsqueeze(-1)
    if logits.dim() != 2:
        logits = logits.view(logits.size(0), -1)

    y_true = _classification_labels(labels)
    if logits.size(1) == 2 and y_true.numel() > 0 and int(y_true.min().item()) < 0:
        uniq = torch.unique(y_true)
        if torch.all((uniq == -1) | (uniq == 1)):
            y_true = (y_true > 0).long()
    n = min(int(logits.size(0)), int(y_true.numel()))
    if n <= 0:
        return {}
    logits = logits[:n]
    y_true = y_true[:n]

    if logits.size(1) == 1:
        target, valid = _binary_targets_and_valid(labels.view(-1))
        probs_pos = torch.sigmoid(logits.view(-1).float())
        if valid.any():
            probs_pos_v = probs_pos[valid]
            y_true_v = target[valid].long()
            y_pred = (probs_pos_v >= 0.5).long()
            probs = probs_pos_v.numpy()
            y_true_np = y_true_v.numpy()
            y_pred_np = y_pred.numpy()
            acc = float((y_pred == y_true_v).float().mean().item())
            f1 = _safe_f1(y_true_np, y_pred_np)
            auc = _safe_auc(y_true_np, probs)
            return {
                "acc": acc,
                "micro_f1": float(f1["micro_f1"]),
                "macro_f1": float(f1["macro_f1"]),
                "auc": float(auc),
            }
        return {
            "acc": float("nan"),
            "micro_f1": float("nan"),
            "macro_f1": float("nan"),
            "auc": float("nan"),
        }
    else:
        probs_tensor = torch.softmax(logits, dim=-1)
        y_pred = probs_tensor.argmax(dim=-1)
        probs = probs_tensor.numpy()

    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()
    acc = float((y_pred == y_true).float().mean().item())
    f1 = _safe_f1(y_true_np, y_pred_np)
    auc = _safe_auc(y_true_np, probs)
    return {
        "acc": acc,
        "micro_f1": float(f1["micro_f1"]),
        "macro_f1": float(f1["macro_f1"]),
        "auc": float(auc),
    }
