import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from code.utils import ensure_dir


def safe_filename(name: str) -> str:
    """Make a reasonably-safe filename segment from an arbitrary dataset/run name."""
    if not isinstance(name, str):
        name = str(name)
    cleaned: List[str] = []
    for ch in name.strip():
        if ch.isalnum() or ch in ("-", "_", "."):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    result = "".join(cleaned).strip("._")
    return result or "unnamed"


def _normalize_split(split: Any) -> Optional[Tuple[float, float, float]]:
    if split is None:
        return None
    try:
        vals = [float(x) for x in split]
    except Exception:
        return None
    if len(vals) != 3:
        return None
    return tuple(vals)


def key_from_run(run: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[Tuple[float, float, float]]]:
    pm = run.get("pretrained_model") or {}
    td = run.get("target_dataset") or {}
    split = td.get("split")
    if split is None:
        task = run.get("task") or {}
        split = task.get("split")
    return pm.get("run_name"), td.get("name"), _normalize_split(split)


def load_existing_records_from_dir(output_dir: str) -> Tuple[List[Dict[str, Any]], int]:
    """
    Load existing per-run records from an output directory.

    Layout:
      - `output_dir/<pretrained_run_name>.json` containing a list of dataset runs.
    Returns (records, removed_count) where removed_count counts invalid/duplicate entries dropped.
    """
    root = Path(output_dir)
    if not root.is_dir():
        return [], 0
    
    loaded: List[Dict[str, Any]] = []
    seen = set()
    removed_invalid = 0

    def _is_skipped(name: str) -> bool:
        if name == "router_records_failed_runs.json":
            return True
        return False

    # New layout: one JSON per expert at the root.
    for json_path in sorted(root.glob("*.json")):
        if _is_skipped(json_path.name):
            continue
        try:
            obj = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            removed_invalid += 1
            continue
        candidates = obj if isinstance(obj, list) else [obj]
        for run in candidates:
            if not isinstance(run, dict):
                removed_invalid += 1
                continue
            key = key_from_run(run)
            if key[0] is None or key[1] is None:
                removed_invalid += 1
                continue
            if key in seen:
                raise ValueError(
                    "Duplicate router run detected while loading existing records: "
                    f"pretrained_run={key[0]!r}, dataset={key[1]!r}, split={key[2]!r}."
                )
            seen.add(key)
            loaded.append(run)

    return loaded, removed_invalid


def _load_runs_from_path(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(obj, list):
        return [o for o in obj if isinstance(o, dict)]
    if isinstance(obj, dict):
        return [obj]
    return []


def persist_runs_to_output_dir(output_dir: str, runs: List[Dict[str, Any]]) -> List[str]:
    root = Path(output_dir)
    ensure_dir(str(root))
    written: List[str] = []
    by_expert: Dict[str, List[Dict[str, Any]]] = {}
    for run in runs:
        run_name, dataset_name, _split = key_from_run(run)
        if not run_name or not dataset_name:
            continue
        by_expert.setdefault(run_name, []).append(run)

    for run_name, expert_runs in by_expert.items():
        out_path = root / f"{safe_filename(run_name)}.json"

        # Merge existing expert file + new runs.
        merged: Dict[str, Dict[str, Any]] = {}

        for existing in _load_runs_from_path(out_path):
            _, ds, split = key_from_run(existing)
            if ds:
                key = f"{ds}:{split}" if split is not None else str(ds)
                merged[key] = existing

        for new_run in expert_runs:
            _, ds, split = key_from_run(new_run)
            if ds:
                key = f"{ds}:{split}" if split is not None else str(ds)
                merged[key] = new_run

        ordered = [merged[k] for k in sorted(merged)]
        out_path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")
        written.append(str(out_path.resolve()))
    return written
