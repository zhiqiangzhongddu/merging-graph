import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from code.utils import build_run_name_from_cfg


def _checkpoint_dataset_dir_name(dataset_name: str) -> str:
    name = str(dataset_name or "").strip()
    if not name:
        return "unknown_dataset"
    return name.replace("\\", "_").replace("/", "_")


def _iter_checkpoint_files(root: str) -> List[str]:
    root_path = Path(root)
    if not root_path.is_dir():
        return []
    return sorted(str(path) for path in root_path.rglob("*.pt") if path.is_file())


def _select_checkpoint_file(run_dir: str, run_name: str) -> Optional[str]:
    """
    Pick the best checkpoint path inside a run directory, preferring an exact
    `<run_name>.pt` match, then a file whose stem matches the directory name,
    then the first available .pt file.
    """
    pt_files = sorted(glob.glob(os.path.join(run_dir, "*.pt")))
    if not pt_files:
        return None

    exact = os.path.join(run_dir, f"{run_name}.pt")
    if os.path.isfile(exact):
        return exact

    preferred = next(
        (p for p in pt_files if os.path.splitext(os.path.basename(p))[0] == os.path.basename(run_dir)),
        None,
    )
    return preferred or pt_files[0]


def list_pretrained_checkpoints(root: str, target_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Enumerate pretrained checkpoints under a root directory, optionally filtering by run name.
    Also attempts to read basic metadata from *_log.json files when present.
    """
    checkpoints: List[Dict[str, Any]] = []
    if not os.path.isdir(root):
        return checkpoints

    root_path = Path(root).resolve()
    for ckpt_path_str in _iter_checkpoint_files(root):
        ckpt_path = Path(ckpt_path_str).resolve()
        run_dir = str(ckpt_path.parent)
        run_name = ckpt_path.stem
        if target_name and run_name != target_name:
            continue

        dataset_hint = None
        try:
            rel_parts = ckpt_path.relative_to(root_path).parts
            if len(rel_parts) >= 2 and rel_parts[0] != run_name:
                dataset_hint = rel_parts[0]
        except Exception:
            dataset_hint = None

        meta: Dict[str, Any] = {
            "path": str(ckpt_path),
            "run_name": run_name,
            "dataset": dataset_hint,
            "task_level": None,
            "induced": None,
            "method": None,
        }
        log_candidates: List[Path] = []
        exact_log = Path(run_dir) / f"{run_name}_log.json"
        if exact_log.is_file():
            log_candidates.append(exact_log)
        else:
            log_candidates.extend(sorted(Path(run_dir).glob("*_log.json")))
        for log_path in log_candidates:
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    log = json.load(f)
                cfg_dict = log.get("config") or log.get("cfg") or {}
                pretrain_cfg = cfg_dict.get("pretrain") or {}
                dataset_cfg = pretrain_cfg.get("dataset") or cfg_dict.get("dataset") or {}
                meta.update(
                    {
                        "dataset": dataset_cfg.get("name"),
                        "task_level": dataset_cfg.get("task_level"),
                        "induced": dataset_cfg.get("induced"),
                        "method": pretrain_cfg.get("method"),
                    }
                )
                break
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"[RouterDatasetGeneration] Failed to read log for {run_name}: {exc}")
        checkpoints.append(meta)
    return checkpoints


def resolve_router_checkpoint(cfg) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve a single pretrained checkpoint path using the router/pretrain cfg.
    """
    ckpt_root = getattr(getattr(cfg, "pretrain", None), "checkpoint_dir", "pretrained_models")

    run_name = build_run_name_from_cfg(cfg)
    dataset_name = str(getattr(getattr(cfg.pretrain, "dataset", None), "name", "") or "")
    dataset_dir_name = _checkpoint_dataset_dir_name(dataset_name)

    candidate = os.path.join(ckpt_root, dataset_dir_name, f"{run_name}.pt")
    if os.path.isfile(candidate):
        return candidate, run_name

    exact_matches = sorted(glob.glob(os.path.join(ckpt_root, "**", f"{run_name}.pt"), recursive=True))
    if exact_matches:
        return exact_matches[0], run_name

    print(f"[RouterDatasetGeneration] Could not locate checkpoint for run '{run_name}' under {ckpt_root}")
    return None, None


def load_available_datasets(path: Optional[str] = None) -> List[str]:
    """Read dataset names from the canonical TSV list."""
    tsv_path = path or "data/available_datasets.tsv"
    datasets: List[str] = []
    try:
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                datasets.append(stripped)
    except FileNotFoundError:
        print(f"[RouterDatasetGeneration] Available datasets list not found: {tsv_path}")
    return datasets


def list_target_datasets(
    data_root: str,
    split_root: str,
    available_list_path: Optional[str] = None,
    target_name: Optional[str] = None,
) -> List[str]:
    """
    Discover target datasets using an available-datasets list, intersected with split files.
    Falls back to filesystem discovery if the list is missing.
    """
    avail_list = load_available_datasets(available_list_path)
    avail_map = {name.lower(): name for name in avail_list}

    split_names = {}
    if os.path.isdir(split_root):
        for fname in os.listdir(split_root):
            if "_splits" not in fname or not fname.endswith(".pt"):
                continue
            base = fname.split("_splits", 1)[0]
            if base:
                split_names[base.lower()] = base

    if target_name:
        key = target_name.lower()
        if avail_map:
            if key in avail_map:
                if split_names and key not in split_names:
                    print(
                        f"[RouterDatasetGeneration] Warning: target '{target_name}' missing split file under {split_root}; "
                        "continuing without split intersection."
                    )
                return [avail_map[key]]
            return []
        if key in split_names:
            return [split_names[key]]
        return []

    if avail_map:
        candidates = []
        for key, name in avail_map.items():
            if split_names and key not in split_names:
                continue
            candidates.append(name)
        return sorted(set(candidates))

    # Fallback to filesystem discovery when available list is missing.
    data_dirs = {}
    if os.path.isdir(data_root):
        for d in os.listdir(data_root):
            if os.path.isdir(os.path.join(data_root, d)):
                data_dirs[d.lower()] = d

    candidates = []
    blacklist = {"processed", "raw", "__pycache__"}
    for lower_name, actual in data_dirs.items():
        if split_names and lower_name not in split_names:
            continue
        if actual not in blacklist:
            candidates.append(actual)
    return sorted(set(candidates))


def serialize_graph(data) -> Dict:
    """Convert a torch_geometric Data object into a JSON-serializable dictionary."""
    payload: Dict = {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
    }
    if getattr(data, "x", None) is not None:
        payload["x"] = data.x.cpu().tolist()
    if getattr(data, "edge_index", None) is not None:
        payload["edge_index"] = data.edge_index.cpu().tolist()
    if getattr(data, "y", None) is not None:
        y = data.y
        if torch.is_tensor(y):
            y = y.view(-1).cpu().tolist()
        payload["y"] = y
    return payload
