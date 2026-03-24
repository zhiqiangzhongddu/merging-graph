import json
import os
from typing import Dict, List, Optional, Tuple


def _normalize_split(split) -> Optional[Tuple[float, float, float]]:
    if split is None:
        return None
    try:
        vals = [float(x) for x in split]
    except Exception:
        return None
    if len(vals) != 3:
        return None
    return tuple(vals)

from .recorder import RouterRecorder, MismatchedCheckpointError
from .utils import (
    list_target_datasets,
    resolve_router_checkpoint,
)


def _shared_split_root(cfg) -> str:
    ds_cfg = getattr(getattr(cfg, "data_preparation", None), "dataset", None)
    return getattr(ds_cfg, "split_root", "data/splits")


def _summarize_record(record: Dict) -> Dict:
    """Create a compact record entry for readability."""
    target = record.get("target") or {}
    node_id = None
    if isinstance(target, dict):
        node_id = target.get("node_id") or target.get("node_index")
    return {
        "target": {"node_id": node_id} if node_id is not None else target,
        "prediction": record.get("prediction"),
    }


def _format_run_result(result) -> Dict:
    """Produce hierarchical structure for a single run."""
    return {
        "pretrained_model": {
            "run_name": result.pretrained_run,
            "checkpoint": None,
        },
        "target_dataset": {
            "name": result.dataset,
            "task_level": result.task_level,
            "induced": result.induced,
            "split": result.split,
        },
        "records": [_summarize_record(r) for r in result.records],
    }


def collect_records(
    cfg,
    existing_records: Optional[List[dict]] = None,
) -> List[dict]:
    """
    Run the full (pretrained model, dataset) grid and gather hierarchical records.
    """
    ckpt_path, run_name = resolve_router_checkpoint(cfg)
    if not ckpt_path:
        print("[RouterRecorder] Unable to resolve pretrained checkpoint from cfg.pretrain.")
        return list(existing_records) if existing_records else []
    checkpoints = [{"path": ckpt_path, "run_name": run_name}]
    for ckpt in checkpoints:
        print(f"[RouterRecorder] Loaded checkpoint for run '{ckpt['run_name']}': {ckpt['path']}")

    ds_cfg = cfg.router_dataset.target_dataset
    split_root = _shared_split_root(cfg)
    target_dataset = getattr(ds_cfg, "name", None) or None
    target_datasets = list_target_datasets(
        ds_cfg.root,
        split_root,
        available_list_path=getattr(ds_cfg, "available_list", None) or "data/available_datasets.tsv",
        target_name=target_dataset,
    )
    if not target_datasets:
        if target_dataset:
            print(
                f"[RouterRecorder] Target dataset '{target_dataset}' not found with data_root={ds_cfg.root} "
                f"and split_root={split_root}"
            )
        else:
            print(f"[RouterRecorder] No target datasets found under {ds_cfg.root} with splits in {split_root}")
        return list(existing_records) if existing_records else []

    formatted: List[dict] = list(existing_records) if existing_records else []
    seen = set()
    for rec in formatted:
        pm = rec.get("pretrained_model") or {}
        td = rec.get("target_dataset") or {}
        split = td.get("split")
        if split is None:
            task = rec.get("task") or {}
            split = task.get("split")
        run_name = (pm.get("run_name"), td.get("name"), _normalize_split(split))
        seen.add(run_name)

    for ckpt in checkpoints:
        for dataset_name in target_datasets:
            key = (
                ckpt["run_name"],
                dataset_name,
                _normalize_split(getattr(cfg.router_dataset.target_dataset, "fixed_split", None)),
            )
            if key in seen:
                print(f"[RouterRecorder] Skipping existing record for {ckpt['run_name']} on {dataset_name}")
                continue

            print(f"[RouterRecorder] Processing {ckpt['run_name']} on {dataset_name}")
            run_cfg = cfg.clone()
            # When user pins dataset/task settings, keep those overrides; otherwise just set name.
            run_cfg.router_dataset.target_dataset.name = dataset_name
            if getattr(cfg.router_dataset.target_dataset, "task_level", None):
                run_cfg.router_dataset.target_dataset.task_level = cfg.router_dataset.target_dataset.task_level
            if getattr(cfg.router_dataset.target_dataset, "induced", None) is not None:
                run_cfg.router_dataset.target_dataset.induced = cfg.router_dataset.target_dataset.induced
            if getattr(cfg.router_dataset.target_dataset, "fixed_split", None):
                run_cfg.router_dataset.target_dataset.fixed_split = cfg.router_dataset.target_dataset.fixed_split
            try:
                recorder = RouterRecorder(
                    cfg=run_cfg,
                    pretrained_checkpoint=ckpt["path"],
                    pretrained_run_name=ckpt["run_name"],
                    dataset_name=dataset_name,
                )
                result = recorder.run()
                formatted_run = _format_run_result(result)
                formatted.append(formatted_run)
                seen.add(key)
            except MismatchedCheckpointError as exc:
                print(f"[RouterRecorder] Skipping run due to checkpoint mismatch: {exc}")
    return formatted
