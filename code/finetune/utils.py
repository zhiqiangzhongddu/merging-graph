import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from code.utils import build_run_name_from_cfg
from code.finetune.finetuner import FinetuneRunner


def parse_finetune_tasks(tsv_path: str) -> List[Tuple[str, str, bool]]:
    """Read dataset, task_level, induced triples from a TSV file."""
    tasks: List[Tuple[str, str, bool]] = []
    if not os.path.isfile(tsv_path):
        print(f"[Finetune] Tasks TSV not found: {tsv_path}")
        return tasks

    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 3:
                continue
            dataset_name, task_level, induced_str = parts[:3]
            induced_flag = induced_str.lower() in ("true", "1", "yes", "y")
            tasks.append((dataset_name, task_level, induced_flag))

    # Keep declaration order while removing duplicates.
    seen = set()
    unique: List[Tuple[str, str, bool]] = []
    for item in tasks:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _checkpoint_dataset_dir_name(dataset_name: str) -> str:
    name = str(dataset_name or "").strip()
    if not name:
        return "unknown_dataset"
    return name.replace("\\", "_").replace("/", "_")


def _iter_checkpoint_files(root: str) -> List[str]:
    root_path = Path(root)
    if not root_path.is_dir():
        return []
    checkpoint_files: List[str] = []
    for dataset_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
        checkpoint_files.extend(
            str(path)
            for path in sorted(dataset_dir.glob("*.pt"))
            if path.is_file()
        )
    return checkpoint_files


def _extract_pretrain_meta_from_log(log_path: Path) -> Dict[str, Any]:
    with open(log_path, "r", encoding="utf-8") as f:
        log = json.load(f)
    cfg_dict = log.get("config") or log.get("cfg") or {}
    pretrain_cfg = cfg_dict.get("pretrain") or {}
    dataset_cfg = pretrain_cfg.get("dataset") or cfg_dict.get("dataset") or {}
    return {
        "dataset": dataset_cfg.get("name"),
        "task_level": dataset_cfg.get("task_level"),
        "induced": dataset_cfg.get("induced"),
        "method": pretrain_cfg.get("method"),
    }


def collect_pretrained_checkpoints(root: str) -> List[Dict[str, Any]]:
    """List pretrained checkpoint paths and basic metadata under a root directory."""
    checkpoints: List[Dict[str, Any]] = []
    if not os.path.isdir(root):
        print(f"[Finetune] Pretrained checkpoint dir not found: {root}")
        return checkpoints

    root_path = Path(root).resolve()
    for ckpt_path_str in _iter_checkpoint_files(root):
        ckpt_path = Path(ckpt_path_str).resolve()
        run_dir = ckpt_path.parent
        run_name = ckpt_path.stem
        dataset_hint = None
        try:
            rel_parts = ckpt_path.relative_to(root_path).parts
            if len(rel_parts) >= 2:
                candidate_hint = rel_parts[0]
                if candidate_hint != run_name:
                    dataset_hint = candidate_hint
        except Exception:
            dataset_hint = None
        meta = {
            "path": str(ckpt_path),
            "run_name": run_name,
            "dataset": dataset_hint,
            "task_level": None,
            "induced": None,
            "method": None,
        }

        log_candidates: List[Path] = []
        exact_log = run_dir / f"{run_name}_log.json"
        if exact_log.is_file():
            log_candidates.append(exact_log)
        else:
            log_candidates.extend(sorted(run_dir.glob("*_log.json")))

        for log_path in log_candidates:
            try:
                meta.update(_extract_pretrain_meta_from_log(log_path))
                break
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"[Finetune] Failed to read log for {run_name}: {exc}")
        checkpoints.append(meta)
    return checkpoints


def resolve_pretrained_checkpoint(cfg) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve checkpoint path for a single finetune run using pretrain/model config.
    """
    ckpt_root = getattr(getattr(cfg, "pretrain", None), "checkpoint_dir", "pretrained_models")

    run_name = build_run_name_from_cfg(cfg)
    dataset_name = str(getattr(getattr(cfg.pretrain, "dataset", None), "name", "") or "")
    dataset_dir_name = _checkpoint_dataset_dir_name(dataset_name)

    # Preferred layout: pretrained_models/<dataset>/<run_name>.pt
    candidate = os.path.join(ckpt_root, dataset_dir_name, f"{run_name}.pt")
    if os.path.isfile(candidate):
        return candidate, run_name

    # Fallback: search by dataset/method/task_level/induced when run_name differs.
    dataset_name = dataset_name.lower()
    task_level = str(getattr(getattr(cfg.pretrain, "dataset", None), "task_level", "")).lower()
    induced = bool(getattr(getattr(cfg.pretrain, "dataset", None), "induced", False))
    method = str(getattr(cfg.pretrain, "method", "")).lower()
    candidates = collect_pretrained_checkpoints(ckpt_root)
    for ckpt in candidates:
        if str(ckpt.get("run_name") or "") == run_name:
            return ckpt["path"], run_name
    for ckpt in candidates:
        if (
            str(ckpt.get("dataset") or "").lower() == dataset_name
            and str(ckpt.get("task_level") or "").lower() == task_level
            and bool(ckpt.get("induced")) == induced
            and str(ckpt.get("method") or "").lower() == method
        ):
            return ckpt["path"], ckpt.get("run_name")

    print(f"[Finetune] Could not locate checkpoint for run '{run_name}' under {ckpt_root}")
    return None, None


def _build_task_cfg(base_cfg, dataset_name: str, task_level: str, induced: bool):
    run_cfg = base_cfg.clone()
    run_cfg.finetune.dataset.name = dataset_name
    run_cfg.finetune.dataset.task_level = task_level
    run_cfg.finetune.dataset.induced = induced
    run_cfg.finetune.dataset.num_classes = None
    run_cfg.finetune.dataset.label_dim = None
    run_cfg.model.in_dim = 0
    run_cfg.finetune.run_all = False
    return run_cfg


def run_finetune_tasks(cfg) -> int:
    """Run fine-tuning for all pretrained checkpoints against target datasets."""
    tasks = parse_finetune_tasks(getattr(cfg.finetune, "tasks_tsv", ""))
    if not tasks:
        print("[Finetune] No dataset definitions found for run_all.")
        return 1

    checkpoint_root = getattr(getattr(cfg, "pretrain", None), "checkpoint_dir", "pretrained_models")
    checkpoints = collect_pretrained_checkpoints(checkpoint_root)
    if not checkpoints:
        print(f"[Finetune] No pretrained checkpoints found under {checkpoint_root}")
        return 1

    results: List[bool] = []
    for ckpt in checkpoints:
        skip_dataset = (ckpt.get("dataset") or "").lower()
        for dataset_name, task_level, induced in tasks:
            if skip_dataset and dataset_name.lower() == skip_dataset:
                continue

            print(
                f"[Finetune] Running {ckpt.get('run_name')} -> "
                f"{dataset_name} (task={task_level}, induced={induced})"
            )
            run_cfg = _build_task_cfg(cfg, dataset_name, task_level, induced)
            try:
                runner = FinetuneRunner(
                    cfg=run_cfg,
                    pretrained_checkpoint=ckpt["path"],
                    pretrained_run_name=ckpt.get("run_name"),
                )
                runner.fit()
                results.append(True)
            except Exception as exc:
                print(f"[Finetune] Failed {ckpt.get('run_name')} -> {dataset_name}: {exc}")
                results.append(False)
    return 0 if all(results) else 1
