"""Helpers for LLM model context generation workflows."""

import json
import os
from typing import Any, Dict, Optional

from .model_describer import LLMModelDescriber


def generate_context_for_arch(
    *,
    arch_path: str,
    describer: LLMModelDescriber,
    output_dir: str,
    dataset_override: Optional[Dict[str, Any]] = None,
    skip_if_exists: bool = True,
) -> bool:
    """
    Generate and save model context for a single architecture file.
    """
    with open(arch_path, "r", encoding="utf-8") as f:
        architecture = json.load(f)

    run_name = os.path.basename(arch_path).replace("_architecture.json", "")
    arch_cfg = architecture.get("config") or {}
    pretrain_cfg = arch_cfg.get("pretrain") or {}
    dataset_cfg = arch_cfg.get("dataset") or {}
    pretrain_dataset_cfg = pretrain_cfg.get("dataset") or {}
    override = dataset_override or {}

    existing_path = os.path.join(output_dir, "model", run_name, "model_llm_context.json")
    if skip_if_exists and os.path.isfile(existing_path):
        print(f"[Context generation] Found existing model context. Skipping: {existing_path}")
        return True

    dataset_name_candidates = []
    for candidate in (
        override.get("name"),
        dataset_cfg.get("name"),
        pretrain_dataset_cfg.get("name"),
        (architecture.get("dataset_meta") or {}).get("name"),
    ):
        if candidate:
            dataset_name_candidates.append(str(candidate))
            dataset_name_candidates.append(str(candidate).capitalize())

    # de-duplicate while preserving order
    seen = set()
    dataset_names = []
    for name in dataset_name_candidates:
        if name not in seen:
            seen.add(name)
            dataset_names.append(name)

    task_level = (
        override.get("task_level")
        or dataset_cfg.get("task_level")
        or pretrain_dataset_cfg.get("task_level")
        or "task"
    )
    folder_candidates = []
    for name in dataset_names or ["dataset"]:
        folder_candidates.append(f"{name}_{task_level}".replace(" ", "_"))

    dataset_paths = None
    dataset_context = None
    attempted = []
    for folder in folder_candidates:
        dataset_path = os.path.join(output_dir, "dataset", folder, "dataset_llm_context.json")
        task_path = os.path.join(output_dir, "dataset", folder, "task_llm_context.json")
        attempted.append((dataset_path, task_path))
        if os.path.isfile(dataset_path) and os.path.isfile(task_path):
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset_desc = json.load(f)
            with open(task_path, "r", encoding="utf-8") as f:
                task_desc = json.load(f)
            dataset_paths = {"dataset": dataset_path, "task": task_path}
            dataset_context = {
                "dataset_description": dataset_desc,
                "task_description": task_desc,
                "folder": folder,
            }
            break

    if dataset_context is None:
        print("[Context generation] Missing dataset/task context files. Checked:")
        for dp, tp in attempted:
            print(f"  - dataset: {dp}")
            print(f"  - task:    {tp}")
        return False

    try:
        context = describer.describe_model(
            architecture=architecture,
            dataset_context=dataset_context,
            pretrain_cfg=pretrain_cfg,
        )
    except Exception as exc:
        print(f"[Context generation] Failed to query LLM for {arch_path}: {exc}")
        return False

    source_paths = {"architecture": arch_path, "dataset_context": dataset_paths}
    out_path = describer.save_context(
        context=context,
        architecture=architecture,
        dataset_context=dataset_context,
        pretrain_cfg=pretrain_cfg,
        output_root=output_dir,
        run_name=run_name,
        source_paths=source_paths,
    )
    print(f"[Context generation] Saved model context to {out_path}")
    return True
