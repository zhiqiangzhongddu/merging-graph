"""AnyGraph baseline orchestration."""

from __future__ import annotations

import ast
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from code.utils import parse_csv_list, project_path, resolve_project_path

from .conversion import main as run_anygraph_conversion_cli
from .link import main as run_anygraph_link_cli
from .node import main as run_anygraph_node_cli
from .report import build_agae_eval_report
from .runtime import (
    ANYGRAPH_HISTORY_DIR,
    ANYGRAPH_LINK_MAIN,
    ANYGRAPH_MODELS_DIR,
    ANYGRAPH_NODE_MAIN,
    ensure_anygraph_runtime_files_exist,
)


@dataclass(frozen=True)
class _AnyGraphPaths:
    dataset_root: Path
    split_root: Path
    out_root: Path
    outputs_dir: Path
    link_csv: Path
    node_csv: Path
    report_csv: Path
    index_path: Path


@dataclass(frozen=True)
class _AnyGraphIndexState:
    payload: dict
    link_datasets: List[str]
    node_datasets: List[str]
    link_records: List[dict]
    node_records: List[dict]


@dataclass(frozen=True)
class _AnyGraphStageSelection:
    link_setting: str
    node_setting: str
    link_datasets: List[str]
    node_datasets: List[str]


def _print_header(title: str, char: str = "=") -> None:
    line = char * 72
    print(line, flush=True)
    print(title, flush=True)
    print(line, flush=True)


def _format_module_cmd(module_name: str, argv: Sequence[str]) -> str:
    parts = [sys.executable, "-m", module_name, *[str(item) for item in argv]]
    return " ".join(shlex.quote(str(part)) for part in parts)


def _run_module(module_name: str, argv: Sequence[str], runner) -> None:
    printable = _format_module_cmd(module_name, argv)
    print(f"[Baseline][AnyGraph][Run] {printable}", flush=True)
    rc = int(runner(argv))
    if rc != 0:
        raise RuntimeError(f"Command failed with rc={rc}: {printable}")


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_csv(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return ",".join(items)
    return str(value).strip()


def _as_token_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return shlex.split(text) if text else []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            token = str(item).strip()
            if token:
                out.append(token)
        return out
    token = str(value).strip()
    return [token] if token else []


def _ordered_unique(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        key = str(value).strip()
        if not key or key in seen:
            continue
        out.append(key)
        seen.add(key)
    return out


def _normalize_int_list(value, default: Sequence[int]) -> List[int]:
    raw = value if value not in (None, "") else default
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raw = list(default)
        else:
            try:
                raw = ast.literal_eval(text)
            except Exception:
                raw = [int(item) for item in parse_csv_list(text)]
    if isinstance(raw, (list, tuple, set)):
        return [int(item) for item in raw]
    return [int(raw)]


def _normalize_split_def(value) -> Tuple[float, float, float]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Split definition cannot be empty.")
        try:
            value = ast.literal_eval(text)
        except Exception:
            value = [float(item) for item in parse_csv_list(text)]
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Invalid split definition '{value}'. Expected 3 values.")
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _normalize_split_list(value, default: Sequence[Sequence[float]]) -> List[Tuple[float, float, float]]:
    raw = value if value not in (None, "") else default
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raw = list(default)
        else:
            try:
                raw = ast.literal_eval(text)
            except Exception:
                raw = [chunk.strip() for chunk in text.split(";") if chunk.strip()] if ";" in text else text
    if isinstance(raw, (list, tuple)):
        if len(raw) == 0:
            return []
        if len(raw) == 3 and not isinstance(raw[0], (list, tuple)):
            return [_normalize_split_def(raw)]
        return [_normalize_split_def(item) for item in raw]
    return [_normalize_split_def(raw)]


def _split_list_arg(split_defs: Sequence[Sequence[float]]) -> str:
    return json.dumps([[float(item) for item in split_def] for split_def in split_defs])


def _resolve_token(token: str, known_datasets: Sequence[str]) -> List[str]:
    token = str(token).strip()
    if token == "":
        return []
    if "," in token:
        items = parse_csv_list(token)
    else:
        items = [token]
    known = set(known_datasets)
    unknown = [item for item in items if item not in known]
    if unknown:
        raise ValueError(
            f"Unknown datasets in setting token '{token}': {unknown}. "
            "Use dataset names that exist in the conversion index."
        )
    return _ordered_unique(items)


def _parse_dataset_setting(setting: str, known_datasets: Sequence[str]) -> Tuple[List[str], List[str], str]:
    """Return (train_datasets, test_datasets, mode) where mode in {'same','plus','in'}."""
    raw = str(setting).strip()
    if raw == "":
        return [], [], "same"
    if "+" in raw:
        idx = raw.index("+")
        train = _resolve_token(raw[:idx], known_datasets)
        test = _resolve_token(raw[idx + 1 :], known_datasets)
        return train, test, "plus"
    if "_in_" in raw:
        idx = raw.index("_in_")
        left = _resolve_token(raw[:idx], known_datasets)
        right = set(_resolve_token(raw[idx + len("_in_") :], known_datasets))
        both = [item for item in left if item in right]
        return both, both, "in"
    same = _resolve_token(raw, known_datasets)
    return same, same, "same"


def _render_dataset_setting(train_list: Sequence[str], test_list: Sequence[str], mode: str) -> str:
    train_csv = ",".join(train_list)
    test_csv = ",".join(test_list)
    if mode == "plus":
        return f"{train_csv}+{test_csv}"
    if mode == "in":
        return train_csv
    return train_csv


def _filter_setting_by_task(
    setting: str,
    *,
    link_datasets: Sequence[str],
    node_datasets: Sequence[str],
) -> Tuple[str, str]:
    known_union = _ordered_unique(list(link_datasets) + list(node_datasets))
    if not known_union:
        raise ValueError("Conversion index has no datasets; cannot derive task-specific settings.")

    train_all, test_all, mode = _parse_dataset_setting(setting, known_union)
    if len(train_all) == 0 or len(test_all) == 0:
        raise ValueError(
            f"Dataset setting resolves to empty split: '{setting}'. "
            "Provide non-empty train/test dataset names."
        )

    link_set = set(link_datasets)
    node_set = set(node_datasets)
    link_train = [item for item in train_all if item in link_set]
    link_test = [item for item in test_all if item in link_set]
    node_train = [item for item in train_all if item in node_set]
    node_test = [item for item in test_all if item in node_set]

    link_setting = ""
    node_setting = ""
    if mode == "plus":
        if link_train and link_test:
            link_setting = _render_dataset_setting(link_train, link_test, mode)
        elif link_train:
            link_setting = ",".join(link_train)
        elif link_test:
            link_setting = ",".join(link_test)

        if node_train and node_test:
            node_setting = _render_dataset_setting(node_train, node_test, mode)
        elif node_train:
            node_setting = ",".join(node_train)
        elif node_test:
            node_setting = ",".join(node_test)
    else:
        if link_train:
            link_setting = _render_dataset_setting(link_train, link_test, mode)
        if node_train:
            node_setting = _render_dataset_setting(node_train, node_test, mode)

    if link_setting == "" and node_setting == "":
        raise ValueError(
            f"Dataset setting '{setting}' contains no datasets available in converted link/node outputs."
        )
    return link_setting, node_setting


def _load_conversion_index(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse conversion index JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid conversion index payload (expected object): {path}")
    return payload


def _extract_str_list(value) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _parse_stage_dataset_names(value) -> List[str]:
    if value in (None, "", []):
        return []
    if isinstance(value, str):
        return parse_csv_list(value)
    if isinstance(value, (list, tuple, set)):
        return _ordered_unique([str(item).strip() for item in value if str(item).strip()])
    return [str(value).strip()]


def _normalize_optional_split(value) -> Optional[Tuple[float, float, float]]:
    if value in (None, "", []):
        return None
    return _normalize_split_def(value)


def _format_fixed_split(split) -> str:
    if split in (None, "", []):
        return "<all>"
    values = _normalize_split_def(split)
    parts: List[str] = []
    for value in values:
        if abs(value - round(value)) < 1e-8:
            if abs(value) <= 1.0:
                parts.append(f"{value:.1f}")
            else:
                parts.append(str(int(round(value))))
        else:
            parts.append(f"{value:g}")
    return f"({', '.join(parts)})"


def _split_matches(record: dict, fixed_split: Optional[Tuple[float, float, float]]) -> bool:
    if fixed_split is None:
        return True
    raw_split = record.get("split")
    if not isinstance(raw_split, (list, tuple)) or len(raw_split) != 3:
        return False
    try:
        record_split = tuple(float(item) for item in raw_split)
    except Exception:
        return False
    return all(abs(left - right) < 1e-8 for left, right in zip(record_split, fixed_split))


def _record_split_tag(record: dict) -> str:
    explicit = str(record.get("split_tag", "") or "").strip()
    if explicit:
        return explicit
    raw_split = record.get("split")
    if isinstance(raw_split, (list, tuple)) and len(raw_split) == 3:
        return _format_fixed_split(raw_split)
    return ""


def _pick_stage_dataset_value(stage_cfg, field: str, fallback_cfg=None):
    dataset_cfg = getattr(stage_cfg, "dataset", None)
    value = getattr(dataset_cfg, field, None) if dataset_cfg is not None else None
    if value not in (None, "", []):
        return value
    if fallback_cfg is None:
        return value
    fallback_dataset_cfg = getattr(fallback_cfg, "dataset", None)
    if fallback_dataset_cfg is None:
        return value
    return getattr(fallback_dataset_cfg, field, None)


def _checkpoint_exists(models_dir: Path, history_dir: Path, run_name: str) -> bool:
    if not run_name:
        return False
    model_path = models_dir / f"{run_name}.mod"
    history_path = history_dir / f"{run_name}.his"
    return model_path.is_file() and history_path.is_file()


def _checkpoint_paths(models_dir: Path, history_dir: Path, run_name: str) -> Tuple[Path, Path]:
    return models_dir / f"{run_name}.mod", history_dir / f"{run_name}.his"


def _preview_values(values: Sequence[str], *, limit: int = 4) -> str:
    items = [str(value).strip() for value in values if str(value).strip()]
    if not items:
        return "<none>"
    if len(items) <= limit:
        return ", ".join(items)
    return f"{', '.join(items[:limit])}, ... ({len(items)} total)"


def _normalize_mode(value, *, cfg_key: str) -> str:
    mode = str(value or "both").strip().lower()
    if mode not in {"link", "node", "both"}:
        raise ValueError(f"Unsupported {cfg_key}='{mode}'. Use link/node/both.")
    return mode


def _print_stage_info(stage_name: str, rows: Sequence[Tuple[str, str]]) -> None:
    prefix = f"[Baseline][AnyGraph][{stage_name}]"
    for key, value in rows:
        print(f"{prefix} {key}: {value}", flush=True)


def _resolve_stage_dataset_settings(
    stage_cfg,
    *,
    link_datasets: Sequence[str],
    node_datasets: Sequence[str],
    stage_cfg_key: str,
    fallback_cfg=None,
) -> Tuple[str, str]:
    def _pick(name: str) -> str:
        value = str(getattr(stage_cfg, name, "") or "").strip()
        if value or fallback_cfg is None:
            return value
        return str(getattr(fallback_cfg, name, "") or "").strip()

    link_dataset_setting = _pick("link_dataset_setting")
    node_dataset_setting = _pick("node_dataset_setting")
    mixed_dataset_setting = _pick("mixed_dataset_setting")

    if mixed_dataset_setting:
        if link_dataset_setting or node_dataset_setting:
            raise ValueError(
                f"Do not combine {stage_cfg_key}.mixed_dataset_setting with "
                f"{stage_cfg_key}.link_dataset_setting/{stage_cfg_key}.node_dataset_setting."
            )
        link_dataset_setting, node_dataset_setting = _filter_setting_by_task(
            mixed_dataset_setting,
            link_datasets=link_datasets,
            node_datasets=node_datasets,
        )
        print(
            "[Baseline][AnyGraph][SplitSetting]",
            f"source={stage_cfg_key}.mixed_dataset_setting",
            f"mixed='{mixed_dataset_setting}'",
            f"-> link='{link_dataset_setting or '<none>'}'",
            f"node='{node_dataset_setting or '<none>'}'",
            flush=True,
        )
    return link_dataset_setting, node_dataset_setting


def _resolve_stage_selected_datasets(
    stage_cfg,
    *,
    stage_cfg_key: str,
    task_name: str,
    records: Sequence[dict],
    cfg,
    fallback_cfg=None,
) -> List[str]:
    source_datasets = _ordered_unique(_parse_stage_dataset_names(_pick_stage_dataset_value(stage_cfg, "name", fallback_cfg)))
    fixed_split = _normalize_optional_split(_pick_stage_dataset_value(stage_cfg, "fixed_split", fallback_cfg))
    required_seeds = [int(seed) for seed in (getattr(cfg, "seeds", []) or [getattr(cfg, "seed", 0)])]
    has_structured_filters = bool(source_datasets or fixed_split is not None)
    has_explicit_setting = any(
        str(getattr(stage_cfg, field, "") or "").strip()
        for field in ("link_dataset_setting", "node_dataset_setting", "mixed_dataset_setting")
    )
    if has_structured_filters and has_explicit_setting:
        raise ValueError(
            f"Do not combine {stage_cfg_key}.dataset.name/{stage_cfg_key}.dataset.fixed_split with "
            f"{stage_cfg_key}.link_dataset_setting/{stage_cfg_key}.node_dataset_setting/"
            f"{stage_cfg_key}.mixed_dataset_setting."
        )
    if not has_structured_filters:
        return []

    available_sources = _ordered_unique(
        [str(record.get("source_dataset", "") or "").strip() for record in records if str(record.get("source_dataset", "") or "").strip()]
    )
    requested_sources = source_datasets or available_sources
    unknown_sources = [name for name in requested_sources if name not in set(available_sources)]
    if unknown_sources:
        raise ValueError(
            f"Unknown {task_name} source datasets in {stage_cfg_key}.dataset.name: {unknown_sources}. "
            f"Available datasets: {_preview_values(available_sources)}"
        )

    selected_records: List[dict] = []
    for source_dataset in requested_sources:
        matched_records = [
            record
            for record in records
            if str(record.get("source_dataset", "") or "").strip() == source_dataset and _split_matches(record, fixed_split)
        ]
        if not matched_records:
            raise ValueError(
                f"No {task_name} datasets found for source_dataset='{source_dataset}' "
                f"and fixed_split={_format_fixed_split(fixed_split)}."
            )

        if fixed_split is None:
            selected_records.extend(matched_records)
            continue

        per_seed = {}
        for record in matched_records:
            try:
                record_seed = int(record.get("seed", 0))
            except Exception:
                record_seed = 0
            if record_seed in per_seed:
                raise ValueError(
                    f"Duplicate {task_name} converted dataset for source_dataset='{source_dataset}', "
                    f"fixed_split={_format_fixed_split(fixed_split)}, seed={record_seed}."
                )
            per_seed[record_seed] = record

        missing_seeds = [seed for seed in required_seeds if seed not in per_seed]
        if missing_seeds:
            raise ValueError(
                f"Missing {task_name} converted datasets for source_dataset='{source_dataset}', "
                f"fixed_split={_format_fixed_split(fixed_split)}, seeds={missing_seeds}. "
                "Run step=conversion first."
            )
        selected_records.extend(per_seed[seed] for seed in required_seeds)

    selected = _ordered_unique([str(record.get("dataset", "") or "").strip() for record in selected_records])
    if not selected:
        raise ValueError(
            f"No {task_name} datasets matched {stage_cfg_key}.dataset.name/{stage_cfg_key}.dataset.fixed_split. "
            f"Available {task_name} datasets include: {_preview_values([str(r.get('dataset', '')) for r in records])}"
        )
    selected_split_tags = _ordered_unique([_record_split_tag(record) for record in selected_records if _record_split_tag(record)])
    selected_seeds = _ordered_unique([str(record.get("seed", "")).strip() for record in selected_records])
    print(
        f"[Baseline][AnyGraph][Select][{task_name}]",
        f"dataset.name={_preview_values(requested_sources)}",
        f"dataset.fixed_split={_format_fixed_split(fixed_split)}",
        f"split_tags={_preview_values(selected_split_tags)}",
        f"seeds={_preview_values(selected_seeds)}",
        f"selected={_preview_values(selected)}",
        flush=True,
    )
    return selected


def _extend_dataset_args(
    args: List[str],
    *,
    dataset_setting: str,
    datasets: Sequence[str],
    stage_cfg_key: str,
    task_name: str,
) -> None:
    if dataset_setting:
        args.extend(["--dataset_setting", dataset_setting])
        return
    if datasets:
        args.extend(["--datasets", ",".join(datasets)])
        return
    raise ValueError(
        f"No {task_name} datasets resolved. Provide {stage_cfg_key}.{task_name}_dataset_setting "
        "or run conversion first."
    )


def _require_checkpoint_tag(
    run_name: str,
    *,
    models_dir: Path,
    history_dir: Path,
    route_name: str,
    cfg_key: str,
) -> None:
    if not _checkpoint_exists(models_dir, history_dir, run_name):
        model_path, history_path = _checkpoint_paths(models_dir, history_dir, run_name)
        raise ValueError(
            f"Missing {route_name} checkpoint '{run_name}'. Expected "
            f"{model_path} and {history_path}. Run step=train first or set {cfg_key}."
        )


def _resolve_cfg_seeds(cfg) -> List[int]:
    seeds = getattr(cfg, "seeds", None) or []
    if isinstance(seeds, (int, float)):
        return [int(seeds)]
    resolved = [int(seed) for seed in seeds]
    if resolved:
        return resolved
    return [int(getattr(cfg, "seed", 0))]


def _resolve_runtime_paths(paths_cfg, output_cfg) -> _AnyGraphPaths:
    dataset_root = resolve_project_path(
        getattr(paths_cfg, "dataset_root", ""),
        default="data/datasets",
    )
    split_root = resolve_project_path(
        getattr(paths_cfg, "split_root", ""),
        default="data/splits",
    )
    out_root = resolve_project_path(
        getattr(paths_cfg, "out_root", ""),
        default="data/anygraph_data",
    )
    outputs_dir = project_path("outputs", "anygraph")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    link_csv = resolve_project_path(
        getattr(output_cfg, "link_csv", ""),
        default=outputs_dir / "anygraph_link_eval.csv",
    )
    node_csv = resolve_project_path(
        getattr(output_cfg, "node_csv", ""),
        default=outputs_dir / "anygraph_node_eval.csv",
    )
    report_csv = resolve_project_path(
        getattr(output_cfg, "report_csv", ""),
        default=outputs_dir / "anygraph_baseline_report.csv",
    )
    link_csv.parent.mkdir(parents=True, exist_ok=True)
    node_csv.parent.mkdir(parents=True, exist_ok=True)
    report_csv.parent.mkdir(parents=True, exist_ok=True)

    return _AnyGraphPaths(
        dataset_root=dataset_root,
        split_root=split_root,
        out_root=out_root,
        outputs_dir=outputs_dir,
        link_csv=link_csv,
        node_csv=node_csv,
        report_csv=report_csv,
        index_path=out_root / "conversion_index.json",
    )


def _load_index_state(index_path: Path) -> _AnyGraphIndexState:
    payload = _load_conversion_index(index_path)
    link_datasets = _extract_str_list(payload.get("link_datasets"))
    node_datasets = _extract_str_list(payload.get("node_datasets"))
    index_results = payload.get("results", []) if isinstance(payload.get("results"), list) else []
    link_records = [
        record
        for record in index_results
        if isinstance(record, dict)
        and str(record.get("task", "")).strip() == "link"
        and str(record.get("status", "")).strip() == "ok"
    ]
    node_records = [
        record
        for record in index_results
        if isinstance(record, dict)
        and str(record.get("task", "")).strip() == "node"
        and str(record.get("status", "")).strip() == "ok"
    ]
    return _AnyGraphIndexState(
        payload=payload,
        link_datasets=link_datasets,
        node_datasets=node_datasets,
        link_records=link_records,
        node_records=node_records,
    )


def _resolve_stage_selection(
    cfg,
    stage_cfg,
    *,
    stage_cfg_key: str,
    mode: str,
    index_state: _AnyGraphIndexState,
    fallback_cfg=None,
) -> _AnyGraphStageSelection:
    link_setting, node_setting = _resolve_stage_dataset_settings(
        stage_cfg,
        link_datasets=index_state.link_datasets,
        node_datasets=index_state.node_datasets,
        stage_cfg_key=stage_cfg_key,
        fallback_cfg=fallback_cfg,
    )
    link_datasets = (
        _resolve_stage_selected_datasets(
            stage_cfg,
            stage_cfg_key=stage_cfg_key,
            task_name="link",
            records=index_state.link_records,
            cfg=cfg,
            fallback_cfg=fallback_cfg,
        )
        if mode in {"link", "both"}
        else []
    )
    node_datasets = (
        _resolve_stage_selected_datasets(
            stage_cfg,
            stage_cfg_key=stage_cfg_key,
            task_name="node",
            records=index_state.node_records,
            cfg=cfg,
            fallback_cfg=fallback_cfg,
        )
        if mode in {"node", "both"}
        else []
    )
    return _AnyGraphStageSelection(
        link_setting=link_setting,
        node_setting=node_setting,
        link_datasets=link_datasets,
        node_datasets=node_datasets,
    )


def _run_conversion_step(cfg, paths: _AnyGraphPaths, conversion_cfg, conversion_task: str) -> None:
    _print_header("[Baseline][AnyGraph][Step 1] conversion", char="=")
    if bool(getattr(conversion_cfg, "skip", False)):
        print("[Baseline][AnyGraph][Conversion] status: skipped by baseline.anygraph.conversion.skip=True")
        return

    data_prep_cfg = getattr(cfg, "data_preparation", None)
    configured_edge_splits = list(getattr(data_prep_cfg, "edge_task_splits", [(0.1, 0.05, 0.1)]))
    configured_node_splits = list(getattr(data_prep_cfg, "node_task_splits", [(0.8, 0.1, 0.1)]))
    default_edge_splits = configured_edge_splits if configured_edge_splits else [(0.1, 0.05, 0.1)]
    default_node_splits = configured_node_splits if configured_node_splits else [(0.8, 0.1, 0.1)]
    conversion_seed = int(getattr(cfg, "seed", 0))
    default_split_seeds = _resolve_cfg_seeds(cfg)
    split_seeds = _normalize_int_list(
        getattr(conversion_cfg, "seeds", default_split_seeds),
        default=default_split_seeds,
    )
    edge_splits = _normalize_split_list(
        getattr(conversion_cfg, "edge_splits", default_edge_splits),
        default=default_edge_splits,
    )
    node_splits = _normalize_split_list(
        getattr(conversion_cfg, "node_splits", default_node_splits),
        default=default_node_splits,
    )
    dataset_csv = _as_csv(getattr(conversion_cfg, "dataset", ""))
    dataset_file_raw = str(getattr(conversion_cfg, "dataset_file", "") or "").strip()

    _print_stage_info(
        "Conversion",
        [
            ("task", conversion_task),
            ("dataset_root", str(paths.dataset_root)),
            ("split_root", str(paths.split_root)),
            ("out_root", str(paths.out_root)),
            ("conversion_index", str(paths.index_path)),
            ("datasets", dataset_csv or dataset_file_raw or "<all available>"),
            ("seed_count", str(len(split_seeds))),
            ("seeds", _preview_values([str(item) for item in split_seeds])),
            ("edge_split_count", str(len(edge_splits))),
            ("node_split_count", str(len(node_splits))),
        ],
    )

    convert_args = [
        "--task",
        conversion_task,
        "--dataset_root",
        str(paths.dataset_root),
        "--split_root",
        str(paths.split_root),
        "--out_root",
        str(paths.out_root),
        "--seed",
        str(conversion_seed),
        "--mask_col",
        str(_safe_int(getattr(conversion_cfg, "mask_col", 0), 0)),
        "--feat_dim",
        str(_safe_int(getattr(conversion_cfg, "feat_dim", 100), 100)),
        "--seeds",
        str(_as_csv(split_seeds) or str(conversion_seed)),
        "--edge_splits",
        _split_list_arg(edge_splits),
        "--node_splits",
        _split_list_arg(node_splits),
        "--edge_eval_payload_name",
        str(getattr(conversion_cfg, "edge_eval_payload_name", "agae_edge_eval_payload.pt")),
        "--node_output_feat_dim",
        str(_safe_int(getattr(conversion_cfg, "node_output_feat_dim", 128), 128)),
        "--index_out",
        str(paths.index_path),
    ]
    convert_args.append(
        "--feat_reduction" if bool(getattr(conversion_cfg, "feat_reduction", False)) else "--no-feat_reduction"
    )
    convert_args.append(
        "--l1_normalize_features"
        if bool(getattr(conversion_cfg, "l1_normalize_features", True))
        else "--no-l1_normalize_features"
    )
    convert_args.append(
        "--emit_node_val" if bool(getattr(conversion_cfg, "emit_node_val", True)) else "--no-emit_node_val"
    )
    if dataset_csv:
        convert_args.extend(["--dataset", dataset_csv])
    elif dataset_file_raw:
        convert_args.extend(
            ["--dataset_file", str(resolve_project_path(dataset_file_raw, default="data/available_node_datasets.tsv"))]
        )

    _run_module("code.baseline.anygraph.conversion", convert_args, run_anygraph_conversion_cli)
    print("[Baseline][AnyGraph][Conversion] status: completed")
    print(f"[Baseline][AnyGraph][Conversion] conversion_index: {paths.index_path}")


def _run_train_step(
    cfg,
    *,
    paths: _AnyGraphPaths,
    index_state: _AnyGraphIndexState,
    any_cfg,
    conversion_cfg,
    train_cfg,
    prediction_cfg,
) -> None:
    _print_header("[Baseline][AnyGraph][Step 2] train", char="=")
    train_mode = _normalize_mode(getattr(train_cfg, "mode", "both"), cfg_key="baseline.anygraph.train.mode")
    train_device = str(_safe_int(getattr(cfg, "device", 0), 0))
    train_epoch = _safe_int(getattr(train_cfg, "epoch", 100), 100)
    if train_epoch <= 0:
        raise ValueError("baseline.anygraph.train.epoch must be > 0 for step=train.")

    stage_selection = _resolve_stage_selection(
        cfg,
        train_cfg,
        stage_cfg_key="baseline.anygraph.train",
        mode=train_mode,
        index_state=index_state,
    )
    seed_scope = _resolve_cfg_seeds(cfg)
    _print_stage_info(
        "Train",
        [
            ("mode", train_mode),
            ("device", train_device),
            ("epoch", str(train_epoch)),
            ("seed_scope", _preview_values([str(seed) for seed in seed_scope])),
            (
                "link_datasets",
                stage_selection.link_setting or _preview_values(stage_selection.link_datasets or index_state.link_datasets)
                if train_mode in {"link", "both"} else "<not selected>",
            ),
            (
                "node_datasets",
                stage_selection.node_setting or _preview_values(stage_selection.node_datasets or index_state.node_datasets)
                if train_mode in {"node", "both"} else "<not selected>",
            ),
            ("models_dir", str(ANYGRAPH_MODELS_DIR)),
            ("history_dir", str(ANYGRAPH_HISTORY_DIR)),
        ],
    )

    shared_extra = _as_token_list(getattr(any_cfg, "extra_args", []))
    ran_train_route = False
    if train_mode in {"link", "both"}:
        link_cfg = prediction_cfg.link
        save_link_path = str(getattr(train_cfg, "save_link_path", "anygraph_link_run"))
        load_link_model = str(getattr(train_cfg, "load_link_model", "") or "").strip()
        if load_link_model:
            _require_checkpoint_tag(
                load_link_model,
                models_dir=ANYGRAPH_MODELS_DIR,
                history_dir=ANYGRAPH_HISTORY_DIR,
                route_name="link",
                cfg_key="baseline.anygraph.train.load_link_model",
            )
        elif bool(getattr(train_cfg, "skip_if_exists", True)) and _checkpoint_exists(
            ANYGRAPH_MODELS_DIR,
            ANYGRAPH_HISTORY_DIR,
            save_link_path,
        ):
            model_path, history_path = _checkpoint_paths(ANYGRAPH_MODELS_DIR, ANYGRAPH_HISTORY_DIR, save_link_path)
            print(
                "[Baseline][AnyGraph][Train][Link] skip:",
                f"checkpoint exists ({save_link_path})",
                f"model={model_path}",
                f"history={history_path}",
            )
        else:
            train_link_csv = paths.outputs_dir / "_train_link_eval.csv"
            link_args = [
                "--data_root",
                str((paths.out_root / "link").resolve()),
                "--result_csv",
                str(train_link_csv),
                "--save_path",
                save_link_path,
                "--gpu",
                train_device,
                "--epoch",
                str(train_epoch),
                "--tst_epoch",
                str(_safe_int(getattr(link_cfg, "tst_epoch", 1), 1)),
                "--topk",
                str(_safe_int(getattr(link_cfg, "topk", 20), 20)),
                "--eval_protocol",
                str(getattr(link_cfg, "eval_protocol", "agae")),
                "--edge_eval_threshold_mode",
                str(getattr(link_cfg, "edge_eval_threshold_mode", "val_best_acc")),
                "--edge_eval_payload_name",
                str(getattr(conversion_cfg, "edge_eval_payload_name", "agae_edge_eval_payload.pt")),
                "--edge_eval_repeat_times",
                str(_safe_int(getattr(link_cfg, "edge_eval_repeat_times", 5), 5)),
            ]
            if load_link_model:
                link_args.extend(["--load_model", load_link_model])
            _extend_dataset_args(
                link_args,
                dataset_setting=stage_selection.link_setting,
                datasets=stage_selection.link_datasets or index_state.link_datasets,
                stage_cfg_key="baseline.anygraph.train",
                task_name="link",
            )
            _print_stage_info(
                "Train",
                [
                    ("link_action", "train"),
                    ("link_save_path", save_link_path),
                    ("link_resume_from", load_link_model or "<none>"),
                    ("link_result_csv", str(train_link_csv)),
                ],
            )
            link_args.extend(shared_extra)
            _run_module("code.baseline.anygraph.link", link_args, run_anygraph_link_cli)
            ran_train_route = True

    if train_mode in {"node", "both"}:
        node_cfg = prediction_cfg.node
        save_node_path = str(getattr(train_cfg, "save_node_path", "anygraph_node_run"))
        load_node_model = str(getattr(train_cfg, "load_node_model", "") or "").strip()
        if load_node_model:
            _require_checkpoint_tag(
                load_node_model,
                models_dir=ANYGRAPH_MODELS_DIR,
                history_dir=ANYGRAPH_HISTORY_DIR,
                route_name="node",
                cfg_key="baseline.anygraph.train.load_node_model",
            )
        elif bool(getattr(train_cfg, "skip_if_exists", True)) and _checkpoint_exists(
            ANYGRAPH_MODELS_DIR,
            ANYGRAPH_HISTORY_DIR,
            save_node_path,
        ):
            model_path, history_path = _checkpoint_paths(ANYGRAPH_MODELS_DIR, ANYGRAPH_HISTORY_DIR, save_node_path)
            print(
                "[Baseline][AnyGraph][Train][Node] skip:",
                f"checkpoint exists ({save_node_path})",
                f"model={model_path}",
                f"history={history_path}",
            )
        else:
            train_node_csv = paths.outputs_dir / "_train_node_eval.csv"
            node_args = [
                "--data_root",
                str((paths.out_root / "node").resolve()),
                "--result_csv",
                str(train_node_csv),
                "--save_path",
                save_node_path,
                "--gpu",
                train_device,
                "--epoch",
                str(train_epoch),
                "--tst_epoch",
                str(_safe_int(getattr(node_cfg, "tst_epoch", 1), 1)),
                "--assignment",
                str(getattr(node_cfg, "assignment", "top1")),
            ]
            if load_node_model:
                node_args.extend(["--load_model", load_node_model])
            _extend_dataset_args(
                node_args,
                dataset_setting=stage_selection.node_setting,
                datasets=stage_selection.node_datasets or index_state.node_datasets,
                stage_cfg_key="baseline.anygraph.train",
                task_name="node",
            )
            _print_stage_info(
                "Train",
                [
                    ("node_action", "train"),
                    ("node_save_path", save_node_path),
                    ("node_resume_from", load_node_model or "<none>"),
                    ("node_result_csv", str(train_node_csv)),
                ],
            )
            node_args.extend(shared_extra)
            _run_module("code.baseline.anygraph.node", node_args, run_anygraph_node_cli)
            ran_train_route = True

    if not ran_train_route:
        print("[Baseline][AnyGraph][Train] status: no routes executed (all requested checkpoints already existed)")
    else:
        print("[Baseline][AnyGraph][Train] status: completed")


def _run_eval_step(
    cfg,
    *,
    paths: _AnyGraphPaths,
    index_state: _AnyGraphIndexState,
    any_cfg,
    conversion_cfg,
    train_cfg,
    eval_cfg,
    prediction_cfg,
) -> None:
    _print_header("[Baseline][AnyGraph][Step 3] eval", char="=")
    eval_mode = _normalize_mode(getattr(eval_cfg, "mode", "both"), cfg_key="baseline.anygraph.eval.mode")
    eval_device = str(_safe_int(getattr(cfg, "device", 0), 0))
    stage_selection = _resolve_stage_selection(
        cfg,
        eval_cfg,
        stage_cfg_key="baseline.anygraph.eval",
        mode=eval_mode,
        index_state=index_state,
        fallback_cfg=train_cfg,
    )
    seed_scope = _resolve_cfg_seeds(cfg)
    _print_stage_info(
        "Eval",
        [
            ("mode", eval_mode),
            ("device", eval_device),
            ("seed_scope", _preview_values([str(seed) for seed in seed_scope])),
            (
                "link_datasets",
                stage_selection.link_setting or _preview_values(stage_selection.link_datasets or index_state.link_datasets)
                if eval_mode in {"link", "both"} else "<not selected>",
            ),
            (
                "node_datasets",
                stage_selection.node_setting or _preview_values(stage_selection.node_datasets or index_state.node_datasets)
                if eval_mode in {"node", "both"} else "<not selected>",
            ),
            ("link_csv", str(paths.link_csv)),
            ("node_csv", str(paths.node_csv)),
            ("report_csv", str(paths.report_csv)),
        ],
    )

    shared_extra = _as_token_list(getattr(any_cfg, "extra_args", []))
    ran_eval_link = False
    ran_eval_node = False
    if eval_mode in {"link", "both"}:
        link_cfg = prediction_cfg.link
        load_link_model = str(getattr(eval_cfg, "load_link_model", "") or "").strip()
        if not load_link_model:
            load_link_model = str(getattr(train_cfg, "save_link_path", "anygraph_link_run"))
        _require_checkpoint_tag(
            load_link_model,
            models_dir=ANYGRAPH_MODELS_DIR,
            history_dir=ANYGRAPH_HISTORY_DIR,
            route_name="link",
            cfg_key="baseline.anygraph.eval.load_link_model",
        )
        link_args = [
            "--data_root",
            str((paths.out_root / "link").resolve()),
            "--result_csv",
            str(paths.link_csv),
            "--save_path",
            load_link_model,
            "--load_model",
            load_link_model,
            "--gpu",
            eval_device,
            "--epoch",
            "0",
            "--tst_epoch",
            str(_safe_int(getattr(link_cfg, "tst_epoch", 1), 1)),
            "--topk",
            str(_safe_int(getattr(link_cfg, "topk", 20), 20)),
            "--eval_protocol",
            str(getattr(link_cfg, "eval_protocol", "agae")),
            "--edge_eval_threshold_mode",
            str(getattr(link_cfg, "edge_eval_threshold_mode", "val_best_acc")),
            "--edge_eval_payload_name",
            str(getattr(conversion_cfg, "edge_eval_payload_name", "agae_edge_eval_payload.pt")),
            "--edge_eval_repeat_times",
            str(_safe_int(getattr(link_cfg, "edge_eval_repeat_times", 5), 5)),
        ]
        _extend_dataset_args(
            link_args,
            dataset_setting=stage_selection.link_setting,
            datasets=stage_selection.link_datasets or index_state.link_datasets,
            stage_cfg_key="baseline.anygraph.eval",
            task_name="link",
        )
        _print_stage_info(
            "Eval",
            [
                ("link_action", "evaluate"),
                ("link_load_model", load_link_model),
            ],
        )
        link_args.extend(shared_extra)
        _run_module("code.baseline.anygraph.link", link_args, run_anygraph_link_cli)
        ran_eval_link = True

    if eval_mode in {"node", "both"}:
        node_cfg = prediction_cfg.node
        load_node_model = str(getattr(eval_cfg, "load_node_model", "") or "").strip()
        if not load_node_model:
            load_node_model = str(getattr(train_cfg, "save_node_path", "anygraph_node_run"))
        _require_checkpoint_tag(
            load_node_model,
            models_dir=ANYGRAPH_MODELS_DIR,
            history_dir=ANYGRAPH_HISTORY_DIR,
            route_name="node",
            cfg_key="baseline.anygraph.eval.load_node_model",
        )
        node_args = [
            "--data_root",
            str((paths.out_root / "node").resolve()),
            "--result_csv",
            str(paths.node_csv),
            "--save_path",
            load_node_model,
            "--load_model",
            load_node_model,
            "--gpu",
            eval_device,
            "--epoch",
            "0",
            "--tst_epoch",
            str(_safe_int(getattr(node_cfg, "tst_epoch", 1), 1)),
            "--assignment",
            str(getattr(node_cfg, "assignment", "top1")),
        ]
        _extend_dataset_args(
            node_args,
            dataset_setting=stage_selection.node_setting,
            datasets=stage_selection.node_datasets or index_state.node_datasets,
            stage_cfg_key="baseline.anygraph.eval",
            task_name="node",
        )
        _print_stage_info(
            "Eval",
            [
                ("node_action", "evaluate"),
                ("node_load_model", load_node_model),
            ],
        )
        node_args.extend(shared_extra)
        _run_module("code.baseline.anygraph.node", node_args, run_anygraph_node_cli)
        ran_eval_node = True

    if not ran_eval_link and not ran_eval_node:
        raise ValueError("Neither link nor node evaluation route was executed.")

    report_link_csv = paths.link_csv if ran_eval_link else paths.outputs_dir / "_missing_link_eval.csv"
    report_node_csv = paths.node_csv if ran_eval_node else paths.outputs_dir / "_missing_node_eval.csv"
    for path in (report_link_csv, report_node_csv):
        if path.exists() and path.name.startswith("_missing_"):
            path.unlink()

    report_args = [
        "--link_csv",
        str(report_link_csv),
        "--node_csv",
        str(report_node_csv),
        "--out_csv",
        str(paths.report_csv),
        "--index_json",
        str(paths.index_path),
    ]
    print(f"[Baseline][AnyGraph][Run] {_format_module_cmd('code.baseline.anygraph.report', report_args)}")
    build_agae_eval_report(report_link_csv, report_node_csv, paths.report_csv, paths.index_path)
    print("[Baseline][AnyGraph][Eval] status: completed")


def run_anygraph_baseline(cfg) -> int:
    any_cfg = cfg.baseline.anygraph
    execution_cfg = any_cfg.execution
    paths_cfg = any_cfg.paths
    conversion_cfg = any_cfg.conversion
    train_cfg = any_cfg.train
    eval_cfg = any_cfg.eval
    prediction_cfg = any_cfg.prediction
    output_cfg = any_cfg.output

    step = str(getattr(execution_cfg, "step", "all") or "all").strip().lower()
    if step not in {"conversion", "train", "eval", "all"}:
        raise ValueError(
            f"Unsupported baseline.anygraph.execution.step='{step}'. Use conversion/train/eval/all."
        )
    planned_steps = ["conversion", "train", "eval"] if step == "all" else [step]

    conversion_task = str(getattr(conversion_cfg, "task", "auto") or "auto").strip().lower()
    if "conversion" in planned_steps and conversion_task not in {"auto", "all", "node", "link"}:
        raise ValueError(
            f"Unsupported baseline.anygraph.conversion.task='{conversion_task}'. Use auto/all/node/link."
        )

    train_mode = _normalize_mode(getattr(train_cfg, "mode", "both"), cfg_key="baseline.anygraph.train.mode")
    eval_mode = _normalize_mode(getattr(eval_cfg, "mode", "both"), cfg_key="baseline.anygraph.eval.mode")
    paths = _resolve_runtime_paths(paths_cfg, output_cfg)

    _print_header(f"[Baseline][AnyGraph] pipeline_steps={','.join(planned_steps)}", char="=")

    if "conversion" in planned_steps:
        _run_conversion_step(cfg, paths, conversion_cfg, conversion_task)

    if step == "conversion":
        return 0

    index_state = _load_index_state(paths.index_path)

    if "train" in planned_steps and train_mode in {"link", "both"}:
        ensure_anygraph_runtime_files_exist(ANYGRAPH_LINK_MAIN)
    if "train" in planned_steps and train_mode in {"node", "both"}:
        ensure_anygraph_runtime_files_exist(ANYGRAPH_NODE_MAIN)
    if "eval" in planned_steps and eval_mode in {"link", "both"}:
        ensure_anygraph_runtime_files_exist(ANYGRAPH_LINK_MAIN)
    if "eval" in planned_steps and eval_mode in {"node", "both"}:
        ensure_anygraph_runtime_files_exist(ANYGRAPH_NODE_MAIN)
    ANYGRAPH_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ANYGRAPH_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    if "train" in planned_steps:
        _run_train_step(
            cfg,
            paths=paths,
            index_state=index_state,
            any_cfg=any_cfg,
            conversion_cfg=conversion_cfg,
            train_cfg=train_cfg,
            prediction_cfg=prediction_cfg,
        )

    if "eval" in planned_steps:
        _run_eval_step(
            cfg,
            paths=paths,
            index_state=index_state,
            any_cfg=any_cfg,
            conversion_cfg=conversion_cfg,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            prediction_cfg=prediction_cfg,
        )

    _print_header("[Baseline][AnyGraph] completed", char="=")
    print(f"[Baseline][AnyGraph] conversion_index: {paths.index_path}")
    print(f"[Baseline][AnyGraph] link_csv: {paths.link_csv}")
    print(f"[Baseline][AnyGraph] node_csv: {paths.node_csv}")
    print(f"[Baseline][AnyGraph] report_csv: {paths.report_csv}")
    return 0
