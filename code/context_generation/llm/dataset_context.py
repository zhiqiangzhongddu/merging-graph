"""Helpers for LLM dataset context generation workflows."""

import json
import os
from typing import List, Tuple

from code.data_loader.datasets import (
    create_dataset,
    dataset_info,
    get_basic_dataset_info,
    infer_task_level,
)

from .dataset_describer import LLMDatasetDescriber


def _read_available_datasets(tsv_path: str) -> List[str]:
    """Load available dataset names from a TSV (one name per line)."""
    datasets: List[str] = []
    if not os.path.isfile(tsv_path):
        print(f"[Context generation] Available list not found: {tsv_path}")
        return datasets
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            datasets.append(stripped)
    return datasets


def _parse_induced_flag(raw: str) -> bool | None:
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    return None


def parse_available_dataset_tasks(
    tsv_path: str,
    *,
    default_task_level: str,
    default_induced: bool,
) -> List[Tuple[str, str, bool]]:
    """
    Read dataset names from an available-datasets TSV and attach defaults.
    """
    datasets = _read_available_datasets(tsv_path)
    tasks: List[Tuple[str, str, bool]] = []
    seen = set()
    for name in datasets:
        task_level = None
        induced = None
        if "\t" in name:
            parts = [part.strip() for part in name.split("\t")]
            name = parts[0]
            if len(parts) > 1 and parts[1]:
                task_level = parts[1].lower()
            if len(parts) > 2 and parts[2]:
                induced = _parse_induced_flag(parts[2])
        if not task_level:
            task_level = infer_task_level(name) or default_task_level
        if induced is None:
            induced = False if task_level == "graph" else default_induced
        item = (name, task_level, induced)
        if item in seen:
            continue
        seen.add(item)
        tasks.append(item)
    return tasks


def generate_context_for_dataset(
    *,
    cfg,
    dataset_name: str,
    task_level: str,
    induced: bool,
    describer: LLMDatasetDescriber,
    out_root: str,
    skip_induced_generation: bool = False,
) -> bool:
    load_task_level = task_level
    load_induced = induced
    if skip_induced_generation and induced:
        load_induced = False
        if task_level == "edge":
            load_task_level = "node"
        print(f"[Context generation] Skipping induced subgraph build for {dataset_name}.")

    dataset = create_dataset(
        name=dataset_name,
        root=cfg.dataset.root,
        task_level=load_task_level,
        feat_reduction=cfg.dataset.feat_reduction,
        feat_reduction_dim=cfg.dataset.feat_reduction_svd_dim,
        feature_svd_dir=getattr(cfg.dataset, "feature_svd_dir", "data/feature_svd"),
        induced=load_induced,
        induced_min_size=cfg.dataset.induced_min_size,
        induced_max_size=cfg.dataset.induced_max_size,
        induced_max_hops=cfg.dataset.induced_max_hops,
        induced_root=getattr(cfg.dataset, "induced_root", ""),
    )

    dataset_meta = get_basic_dataset_info(dataset)
    dataset_info(
        dataset=dataset,
        task_level=load_task_level,
        name=dataset_name,
        induced=load_induced,
    )

    folder_name = f"{dataset_meta.get('name', dataset_name)}_{task_level}".replace(" ", "_")
    dataset_path = os.path.join(out_root, "dataset", folder_name, "dataset_llm_context.json")
    task_path = os.path.join(out_root, "dataset", folder_name, "task_llm_context.json")
    if os.path.isfile(dataset_path) and os.path.isfile(task_path):
        print(
            "[Context generation] Found existing context. Skipping.\n"
            f"  dataset: {dataset_path}\n"
            f"  task:    {task_path}"
        )
        return True

    try:
        context = describer.describe_dataset(dataset=dataset)
    except Exception as exc:
        print(f"[Context generation] Failed to query LLM for {dataset_name}-{task_level}: {exc}")
        return False

    print(json.dumps(context, indent=2))

    base_name = context.get("dataset", {}).get("name", dataset_name)
    folder_name = f"{base_name}_{task_level}".replace(" ", "_")
    paths = describer.save_context(
        context=context,
        output_root=out_root,
        folder_name=folder_name,
    )

    print(f"[Context generation] Saved dataset context to {paths['dataset']}")
    print(f"[Context generation] Saved task context to {paths['task']}")
    return True
