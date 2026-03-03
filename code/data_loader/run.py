"""Runtime orchestration for the `run_data_preparation.py` entrypoint."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

from code.config import cfg as base_cfg, update_cfg
from code.data_loader.dataset_prepare import prepare_datasets
from code.utils import set_seed

TargetSelection = Optional[Union[str, List[str]]]
SplitTuple = Tuple[float, float, float]


@dataclass(frozen=True)
class _PrepStage:
    """One dataset-preparation stage with its execution parameters."""

    title: str
    datasets: Union[str, List[str]]
    task_levels: Tuple[str, ...]
    induced: bool
    use_infer: bool


def _print_stage_header(title: str, char: str = "=") -> None:
    line = char * 72
    print(line)
    print(title)
    print(line)


def _normalize_targets(value) -> TargetSelection:
    """Normalize target input into either a path/string or an explicit list of names."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return [str(value)]


def _parse_target_value(raw: str) -> Union[str, List[str]]:
    """Parse CLI target value like `[cora,actor]`, `cora,actor`, or `path/to/list.tsv`."""
    text = str(raw).strip()
    if not text:
        return []

    is_bracket_list = text.startswith("[") and text.endswith("]")
    if is_bracket_list:
        text = text[1:-1]

    if "," not in text:
        single = text.strip().strip("'\"")
        if is_bracket_list:
            return [single] if single else []
        return single

    items: List[str] = []
    for part in text.split(","):
        cleaned = part.strip().strip("'\"")
        if cleaned:
            items.append(cleaned)
    return items


def _extract_target_override(argv: List[str]) -> Tuple[List[str], TargetSelection]:
    """Extract `data_preparation.target_datasets` from argv for robust list parsing."""
    cleaned: List[str] = []
    override: TargetSelection = None
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token == "data_preparation.target_datasets":
            if idx + 1 >= len(argv):
                raise ValueError("data_preparation.target_datasets requires a value")
            override = _parse_target_value(argv[idx + 1])
            idx += 2
            continue
        cleaned.append(token)
        idx += 1
    return cleaned, override


def _same_path(path_a: str, path_b: str) -> bool:
    try:
        return Path(path_a).resolve() == Path(path_b).resolve()
    except Exception:
        return str(path_a) == str(path_b)


def _coerce_split_tuple(split_def) -> SplitTuple:
    if not isinstance(split_def, (list, tuple)) or len(split_def) < 3:
        raise ValueError(f"Invalid split definition: {split_def}")
    return (float(split_def[0]), float(split_def[1]), float(split_def[2]))


def _normalize_edge_split(split_def) -> SplitTuple:
    train_pos, val_pos, test_pos = _coerce_split_tuple(split_def)
    if min(train_pos, val_pos, test_pos) < 0.0:
        raise ValueError(f"Invalid edge split definition {split_def}: all values must be >= 0.")

    total_pos = train_pos + val_pos + test_pos
    if total_pos > 1.0 + 1e-6:
        raise ValueError(
            f"Invalid edge split definition {split_def}: train/val/test positive ratios must sum to <= 1.0."
        )
    return (train_pos, val_pos, test_pos)


def _split_defs_for_task(cfg, task_level: str) -> List[SplitTuple]:
    dp_cfg = cfg.data_preparation
    if task_level == "node":
        return [_coerce_split_tuple(split_def) for split_def in getattr(dp_cfg, "node_task_splits", [])]
    if task_level == "graph":
        return [_coerce_split_tuple(split_def) for split_def in getattr(dp_cfg, "graph_task_splits", [])]
    if task_level == "edge":
        return [_normalize_edge_split(split_def) for split_def in getattr(dp_cfg, "edge_task_splits", [])]
    return []


def _resolve_split_seeds(cfg, num_splits: int) -> List[int]:
    """Resolve split seeds and append deterministic random seeds when needed."""
    raw_seeds = getattr(cfg, "seeds", None) or []
    if isinstance(raw_seeds, (int, float)):
        seeds = [int(raw_seeds)]
    else:
        seeds = [int(seed) for seed in raw_seeds]

    if not seeds:
        seeds = [int(getattr(cfg, "seed", 0))]

    rng = random.Random(int(getattr(cfg, "seed", 0)))
    seen = set(seeds)
    while len(seeds) < num_splits:
        candidate = rng.randint(0, 2**31 - 1)
        if candidate in seen:
            continue
        seeds.append(candidate)
        seen.add(candidate)
    return seeds[:num_splits]


def _build_split_generation_plan(cfg) -> Tuple[Dict[str, List[SplitTuple]], List[int]]:
    ds_cfg = cfg.data_preparation.dataset
    num_splits = int(getattr(ds_cfg, "num_splits", 5) or 0)
    if num_splits <= 0:
        num_splits = 5

    split_defs_by_task = {
        "node": _split_defs_for_task(cfg, "node"),
        "graph": _split_defs_for_task(cfg, "graph"),
        "edge": _split_defs_for_task(cfg, "edge"),
    }
    split_seeds = _resolve_split_seeds(cfg, num_splits)
    return split_defs_by_task, split_seeds


def _resolve_stage_plan(cfg, targets: TargetSelection) -> List[_PrepStage]:
    """Create stage plan based on selected targets and configured dataset lists."""
    dp_cfg = cfg.data_preparation
    ds_cfg = dp_cfg.dataset

    node_list = getattr(ds_cfg, "available_node_datasets", "")
    graph_list = getattr(ds_cfg, "available_graph_datasets", "")
    include_edge_level = bool(getattr(dp_cfg, "generate_edge_level", True))
    node_task_levels = ("node", "edge") if include_edge_level else ("node",)

    if not targets:
        stages: List[_PrepStage] = []
        if node_list:
            stages.append(
                _PrepStage(
                    title="[DataPrep] Dataset preparation (node/edge)",
                    datasets=node_list,
                    task_levels=node_task_levels,
                    induced=bool(getattr(ds_cfg, "induced", True)),
                    use_infer=False,
                )
            )
        if graph_list:
            stages.append(
                _PrepStage(
                    title="[DataPrep] Dataset preparation (graph)",
                    datasets=graph_list,
                    task_levels=("graph",),
                    induced=False,
                    use_infer=False,
                )
            )
        return stages

    if isinstance(targets, list):
        return [
            _PrepStage(
                title="[DataPrep] Dataset preparation",
                datasets=targets,
                task_levels=("node", "edge", "graph"),
                induced=bool(getattr(ds_cfg, "induced", True)),
                use_infer=True,
            )
        ]

    if node_list and _same_path(targets, node_list):
        return [
            _PrepStage(
                title="[DataPrep] Dataset preparation (node/edge)",
                datasets=targets,
                task_levels=node_task_levels,
                induced=bool(getattr(ds_cfg, "induced", True)),
                use_infer=False,
            )
        ]

    if graph_list and _same_path(targets, graph_list):
        return [
            _PrepStage(
                title="[DataPrep] Dataset preparation (graph)",
                datasets=targets,
                task_levels=("graph",),
                induced=False,
                use_infer=False,
            )
        ]

    return [
        _PrepStage(
            title="[DataPrep] Dataset preparation",
            datasets=targets,
            task_levels=("node", "edge", "graph"),
            induced=bool(getattr(ds_cfg, "induced", True)),
            use_infer=True,
        )
    ]


def _run_prepare_stage(
    cfg,
    stage: _PrepStage,
    split_defs_by_task: Dict[str, List[SplitTuple]],
    split_seeds: List[int],
) -> int:
    """Run one preparation stage using precomputed split plan."""
    _print_stage_header(stage.title, char="=")

    ds_cfg = cfg.data_preparation.dataset
    dp_cfg = cfg.data_preparation

    return prepare_datasets(
        tsv_path=stage.datasets,
        root=Path(ds_cfg.root),
        task_levels=stage.task_levels,
        feat_reduction=ds_cfg.feat_reduction,
        feat_reduction_dim=ds_cfg.feat_reduction_svd_dim,
        feature_svd_dir=getattr(ds_cfg, "feature_svd_dir", "data/feature_svd"),
        induced=stage.induced,
        induced_min_size=getattr(ds_cfg, "induced_min_size", 10),
        induced_max_size=getattr(ds_cfg, "induced_max_size", 30),
        induced_max_hops=getattr(ds_cfg, "induced_max_hops", 5),
        induced_root=getattr(ds_cfg, "induced_root", ""),
        force_reload_raw=getattr(ds_cfg, "force_reload_raw", False),
        subgraph_svd=getattr(ds_cfg, "subgraph_svd", True),
        subgraph_svd_feat_dim=getattr(ds_cfg, "subgraph_svd_feat_dim", 100),
        subgraph_svd_struct_dim=getattr(ds_cfg, "subgraph_svd_struct_dim", 100),
        subgraph_svd_matrix=getattr(ds_cfg, "subgraph_svd_matrix", "adjacency"),
        subgraph_svd_dir=getattr(ds_cfg, "subgraph_svd_dir", "data/subgraph_svd"),
        use_infer=stage.use_infer,
        split_root=getattr(ds_cfg, "split_root", ""),
        generate_edge_level=bool(getattr(dp_cfg, "generate_edge_level", True)),
        split_defs_by_task=split_defs_by_task,
        split_seeds=split_seeds,
        split_batch_size=int(getattr(cfg.train, "batch_size", 64)),
        split_num_workers=int(getattr(cfg.train, "num_workers", 0)),
    )


def run_data_preparation(cfg) -> int:
    """Main orchestrator used by `run_data_preparation.py`."""
    dp_cfg = cfg.data_preparation
    ds_cfg = dp_cfg.dataset

    # Backward compatibility: older configs may store target datasets directly
    # under `cfg.data_preparation.dataset`.
    fallback_targets = None
    if not hasattr(ds_cfg, "root"):
        fallback_targets = ds_cfg
        cfg.data_preparation.dataset = base_cfg.data_preparation.dataset.clone()

    targets = _normalize_targets(getattr(dp_cfg, "target_datasets", None))
    if (targets is None or targets == []) and fallback_targets is not None:
        targets = _normalize_targets(fallback_targets)

    stages = _resolve_stage_plan(cfg, targets)
    if not stages:
        print("[DataPrep] No datasets to prepare. Check dataset list configuration.")
        return 0

    split_defs_by_task, split_seeds = _build_split_generation_plan(cfg)
    status_codes = [
        _run_prepare_stage(cfg, stage, split_defs_by_task=split_defs_by_task, split_seeds=split_seeds)
        for stage in stages
    ]
    return 0 if all(code == 0 for code in status_codes) else 1


def run_data_preparation_from_cli(argv: Iterable[str]) -> int:
    argv_list, target_override = _extract_target_override(list(argv))
    cfg = update_cfg(base_cfg, " ".join(argv_list))
    set_seed(cfg.seed)
    if target_override is not None:
        cfg.data_preparation.target_datasets = target_override
    return run_data_preparation(cfg)
