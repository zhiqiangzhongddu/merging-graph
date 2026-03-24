"""Runtime helpers for LLM-based context generation."""

from __future__ import annotations

import glob
import os
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

from code.utils import build_run_name_from_cfg, set_seed

from .dataset_context import generate_context_for_dataset, parse_available_dataset_tasks
from .dataset_describer import LLMDatasetDescriber
from .model_context import generate_context_for_arch
from .model_describer import LLMModelDescriber
from .utils import list_architecture_paths


def _sanitize_component(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _context_cfg(cfg):
    return getattr(cfg, "context", None)


def _llm_cfg(cfg):
    context_cfg = _context_cfg(cfg)
    return getattr(context_cfg, "llm", None) if context_cfg is not None else None


def _dataset_cfg(cfg):
    def _has_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        return True

    def _copy_attrs(dst: SimpleNamespace, src: Any) -> None:
        if src is None:
            return
        if hasattr(src, "keys"):
            keys = list(src.keys())
        else:
            keys = [
                key for key in dir(src)
                if not key.startswith("_") and not callable(getattr(src, key))
            ]
        for key in keys:
            try:
                value = getattr(src, key)
            except Exception:
                continue
            if _has_value(value):
                setattr(dst, key, value)

    merged = SimpleNamespace(
        name=None,
        task_level="node",
        induced=True,
        root="data/datasets",
        feat_reduction=True,
        feat_reduction_svd_dim=100,
        feature_svd_dir="data/feature_svd",
        induced_min_size=10,
        induced_max_size=30,
        induced_max_hops=5,
        induced_root="data/induced_subgraphs",
        available_datasets="",
        available_node_datasets="data/available_node_datasets.tsv",
        available_graph_datasets="data/available_graph_datasets.tsv",
    )

    train_cfg = getattr(cfg, "train", None)
    if train_cfg is not None and getattr(train_cfg, "dataset", None) is not None:
        _copy_attrs(merged, train_cfg.dataset)

    pretrain_cfg = getattr(cfg, "pretrain", None)
    if pretrain_cfg is not None and getattr(pretrain_cfg, "dataset", None) is not None:
        _copy_attrs(merged, pretrain_cfg.dataset)

    context_cfg = _context_cfg(cfg)
    if context_cfg is not None and getattr(context_cfg, "dataset", None) is not None:
        _copy_attrs(merged, context_cfg.dataset)

    return merged


def _resolve_llm_model_name(cfg) -> str:
    llm_cfg = _llm_cfg(cfg)
    if llm_cfg is not None:
        model_name = str(getattr(llm_cfg, "model_name", "") or "").strip()
        if model_name:
            return model_name
    return "gpt-5.2"


def _resolve_llm_temperature(cfg) -> float:
    llm_cfg = _llm_cfg(cfg)
    if llm_cfg is not None and hasattr(llm_cfg, "temperature"):
        return float(getattr(llm_cfg, "temperature"))
    return 0.2


def _resolve_output_dir(cfg) -> str:
    context_cfg = _context_cfg(cfg)
    base_output = getattr(context_cfg, "output_dir", "generated_context") if context_cfg is not None else "generated_context"
    return os.path.join(
        str(base_output),
        "llm",
        _sanitize_component(_resolve_llm_model_name(cfg)),
    )


def _checkpoint_dataset_dir_name(dataset_name: str) -> str:
    name = str(dataset_name or "").strip()
    if not name:
        return "unknown_dataset"
    return name.replace("\\", "_").replace("/", "_")


def _resolve_architecture_path(
    cfg,
    arch_root: str,
    run_name: str,
) -> str:
    ds_cfg = _dataset_cfg(cfg)
    dataset_name = str(getattr(ds_cfg, "name", "") or "")
    dataset_dir_name = _checkpoint_dataset_dir_name(dataset_name)

    candidate = os.path.join(arch_root, dataset_dir_name, f"{run_name}_architecture.json")
    if os.path.isfile(candidate):
        return candidate

    matches = sorted(glob.glob(os.path.join(arch_root, "**", f"{run_name}_architecture.json"), recursive=True))
    if matches:
        return matches[0]

    return candidate


def _run_all_enabled(cfg) -> bool:
    llm_cfg = _llm_cfg(cfg)
    return _to_bool(getattr(llm_cfg, "run_all", False))


def _iter_tasks_from_tsv(
    tsv_path: str,
    *,
    default_task_level: str,
    default_induced: bool,
) -> Iterable[Tuple[str, str, bool]]:
    if not tsv_path:
        return []
    return parse_available_dataset_tasks(
        tsv_path,
        default_task_level=default_task_level,
        default_induced=default_induced,
    )


def _collect_dataset_tasks(cfg) -> List[Tuple[str, str, bool]]:
    ds_cfg = _dataset_cfg(cfg)
    llm_cfg = _llm_cfg(cfg)

    default_task_level = str(getattr(ds_cfg, "task_level", "node") or "node")
    default_induced = _to_bool(getattr(ds_cfg, "induced", True))

    tasks: List[Tuple[str, str, bool]] = []
    seen = set()

    def _append(items: Iterable[Tuple[str, str, bool]]) -> None:
        for dataset_name, task_level, induced in items:
            key = (str(dataset_name), str(task_level), bool(induced))
            if key in seen:
                continue
            tasks.append(key)
            seen.add(key)

    preferred_tsv = str(getattr(llm_cfg, "available_datasets_tsv", "") or "") if llm_cfg is not None else ""
    preferred_tsv = preferred_tsv.strip()
    if not preferred_tsv:
        preferred_tsv = str(getattr(ds_cfg, "available_datasets", "") or "").strip()
    if preferred_tsv:
        _append(
            _iter_tasks_from_tsv(
                preferred_tsv,
                default_task_level=default_task_level,
                default_induced=default_induced,
            )
        )

    node_tsv = str(getattr(ds_cfg, "available_node_datasets", "") or "").strip()
    if node_tsv:
        _append(
            _iter_tasks_from_tsv(
                node_tsv,
                default_task_level="node",
                default_induced=default_induced,
            )
        )

    graph_tsv = str(getattr(ds_cfg, "available_graph_datasets", "") or "").strip()
    if graph_tsv:
        _append(
            _iter_tasks_from_tsv(
                graph_tsv,
                default_task_level="graph",
                default_induced=False,
            )
        )

    return tasks


def run_llm_dataset_from_cfg(cfg) -> int:
    set_seed(int(getattr(cfg, "seed", 0)))

    llm_cfg = _llm_cfg(cfg)
    ds_cfg = _dataset_cfg(cfg)
    out_root = _resolve_output_dir(cfg)
    model_name = _resolve_llm_model_name(cfg)
    temperature = _resolve_llm_temperature(cfg)
    skip_induced_generation = _to_bool(getattr(llm_cfg, "skip_induced_generation", False))

    describer = LLMDatasetDescriber(model=model_name, temperature=temperature)

    shim_cfg = SimpleNamespace(dataset=ds_cfg)

    if _run_all_enabled(cfg):
        tasks = _collect_dataset_tasks(cfg)
        if not tasks:
            print("[Context generation][LLM] No dataset/task entries found for run_all.")
            return 1

        results: List[bool] = []
        for dataset_name, task_level, induced in tasks:
            print(
                f"[Context generation][LLM][dataset] Processing {dataset_name} "
                f"(task level={task_level}, induced={induced})"
            )
            success = generate_context_for_dataset(
                cfg=shim_cfg,
                dataset_name=dataset_name,
                task_level=task_level,
                induced=induced,
                describer=describer,
                out_root=out_root,
                skip_induced_generation=skip_induced_generation,
            )
            results.append(bool(success))
        return 0 if all(results) else 1

    dataset_name = str(getattr(ds_cfg, "name", "") or "").strip()
    if not dataset_name:
        print("[Context generation][LLM][dataset] Missing context.dataset.name for single-dataset run.")
        return 1

    task_level = str(getattr(ds_cfg, "task_level", "node") or "node")
    induced = _to_bool(getattr(ds_cfg, "induced", True))
    success = generate_context_for_dataset(
        cfg=shim_cfg,
        dataset_name=dataset_name,
        task_level=task_level,
        induced=induced,
        describer=describer,
        out_root=out_root,
        skip_induced_generation=skip_induced_generation,
    )
    return 0 if success else 1


def run_llm_model_from_cfg(cfg) -> int:
    set_seed(int(getattr(cfg, "seed", 0)))

    llm_cfg = _llm_cfg(cfg)
    ds_cfg = _dataset_cfg(cfg)

    arch_root = str(getattr(getattr(cfg, "pretrain", None), "checkpoint_dir", "pretrained_models"))
    if llm_cfg is not None:
        override_arch_root = str(getattr(llm_cfg, "arch_root", "") or "").strip()
        if override_arch_root:
            arch_root = override_arch_root

    if not os.path.isdir(arch_root):
        print(f"[Context generation][LLM][model] Checkpoint dir not found: {arch_root}")
        return 1

    output_dir = _resolve_output_dir(cfg)
    describer = LLMModelDescriber(
        model=_resolve_llm_model_name(cfg),
        temperature=_resolve_llm_temperature(cfg),
    )

    dataset_override: Dict[str, Optional[str]] = {
        "name": getattr(ds_cfg, "name", None),
        "task_level": getattr(ds_cfg, "task_level", None),
    }
    skip_if_exists = True
    if llm_cfg is not None and hasattr(llm_cfg, "skip_if_exists"):
        skip_if_exists = _to_bool(getattr(llm_cfg, "skip_if_exists"))

    if _run_all_enabled(cfg):
        arch_paths = list_architecture_paths(arch_root)
        if not arch_paths:
            print(f"[Context generation][LLM][model] No architecture files found under {arch_root}")
            return 1

        results: List[bool] = []
        for arch_path in arch_paths:
            success = generate_context_for_arch(
                arch_path=arch_path,
                describer=describer,
                output_dir=output_dir,
                dataset_override=dataset_override,
                skip_if_exists=skip_if_exists,
            )
            results.append(bool(success))
        return 0 if all(results) else 1

    run_name = ""
    if llm_cfg is not None:
        run_name = str(getattr(llm_cfg, "run_name", "") or "").strip()
    if not run_name:
        run_name = build_run_name_from_cfg(cfg)

    arch_path = _resolve_architecture_path(
        cfg,
        arch_root,
        run_name,
    )

    if not os.path.isfile(arch_path):
        print(
            "[Context generation][LLM][model] Architecture file not found for "
            f"run '{run_name}' under {arch_root}"
        )
        return 1

    success = generate_context_for_arch(
        arch_path=arch_path,
        describer=describer,
        output_dir=output_dir,
        dataset_override=dataset_override,
        skip_if_exists=skip_if_exists,
    )
    return 0 if success else 1
