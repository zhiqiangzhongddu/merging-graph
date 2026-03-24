"""Runtime bridges for condensation-based context generation workflows."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Sequence

from code.utils import read_name_list_file


def _context_output_dir(cfg) -> Path:
    context_cfg = getattr(cfg, "context", None)
    raw = str(getattr(context_cfg, "output_dir", "generated_context") or "generated_context").strip()
    return Path(raw)


def _cfg_block(value):
    if value is None or isinstance(value, str):
        return None
    if hasattr(value, "keys"):
        return value
    return None


def _dataset_name(dataset_cfg) -> str:
    if dataset_cfg is None:
        return ""
    value = getattr(dataset_cfg, "name", None)
    return "" if value is None else str(value).strip()


def _dataset_root(dataset_cfg, fallback: str = "data/datasets") -> str:
    if dataset_cfg is None:
        return fallback
    value = str(getattr(dataset_cfg, "root", "") or "").strip()
    return value or fallback


def _dataset_split_root(dataset_cfg, fallback: str = "data/splits") -> str:
    if dataset_cfg is None:
        return fallback
    value = str(getattr(dataset_cfg, "split_root", "") or "").strip()
    return value or fallback


def _dataset_fixed_split(dataset_cfg, fallback=(0.8, 0.1, 0.1)):
    if dataset_cfg is None:
        return fallback
    value = getattr(dataset_cfg, "fixed_split", None)
    return fallback if value is None else value


def _dataset_feat_reduction(dataset_cfg, fallback: bool = True) -> bool:
    if dataset_cfg is None:
        return fallback
    return _to_bool(getattr(dataset_cfg, "feat_reduction", fallback))


def _dataset_feat_dim(dataset_cfg, fallback: int = 100) -> int:
    if dataset_cfg is None:
        return fallback
    return int(getattr(dataset_cfg, "feat_reduction_svd_dim", fallback))


def _override_dataset_root(override_cfg, fallback_cfg, default: str = "data/datasets") -> str:
    override = _cfg_block(override_cfg)
    if override is not None:
        value = str(getattr(override, "root", "") or "").strip()
        if value:
            return value
    return _dataset_root(_cfg_block(fallback_cfg), default)


def _override_dataset_split_root(override_cfg, fallback_cfg, default: str = "data/splits") -> str:
    override = _cfg_block(override_cfg)
    if override is not None:
        value = str(getattr(override, "split_root", "") or "").strip()
        if value:
            return value
    return _dataset_split_root(_cfg_block(fallback_cfg), default)


def _override_dataset_feat_reduction(override_cfg, fallback_cfg, default: bool = True) -> bool:
    override = _cfg_block(override_cfg)
    if override is not None and hasattr(override, "feat_reduction"):
        return _to_bool(getattr(override, "feat_reduction", default))
    return _dataset_feat_reduction(_cfg_block(fallback_cfg), default)


def _override_dataset_feat_dim(override_cfg, fallback_cfg, default: int = 100) -> int:
    override = _cfg_block(override_cfg)
    if override is not None and hasattr(override, "feat_reduction_svd_dim"):
        return int(getattr(override, "feat_reduction_svd_dim", default))
    return _dataset_feat_dim(_cfg_block(fallback_cfg), default)


def _dataset_list_path(dataset_cfg, attr_name: str, fallback: str) -> str:
    if dataset_cfg is None:
        return fallback
    value = str(getattr(dataset_cfg, attr_name, "") or "").strip()
    return value or fallback


def _override_dataset_name(override_cfg, fallback_cfg) -> str:
    return _dataset_name(_cfg_block(override_cfg)) or _dataset_name(_cfg_block(fallback_cfg))


def _override_fixed_split(override_cfg, fallback_cfg, default=(0.8, 0.1, 0.1)):
    override = _cfg_block(override_cfg)
    if override is not None and getattr(override, "fixed_split", None) is not None:
        return getattr(override, "fixed_split")
    return _dataset_fixed_split(_cfg_block(fallback_cfg), default)


def _graph_condensation_root(cfg) -> Path:
    return _context_output_dir(cfg) / "graph_condensation"


def _bonsai_out_root(cfg, bonsai_cfg) -> str:
    explicit = str(getattr(bonsai_cfg, "out_root", "") or "").strip()
    if explicit:
        return explicit
    return str(_graph_condensation_root(cfg) / "bonsai")


def _bonsai_condensed_root(cfg, eval_cfg) -> str:
    explicit = str(getattr(eval_cfg, "condensed_root", "") or "").strip()
    if explicit:
        return explicit
    return str(_graph_condensation_root(cfg) / "bonsai")


def _sgdc_pretrain_root(cfg, pretrain_cfg) -> str:
    explicit = str(getattr(pretrain_cfg, "default_ckpt_dir", "") or "").strip()
    if explicit:
        return explicit
    return str(Path("outputs") / "sgdc" / "pretrain")


def _sgdc_teacher_model_name(cfg_block, default: str = "gin") -> str:
    if hasattr(cfg_block, "model"):
        return str(getattr(cfg_block, "model", default) or default).strip()
    return str(getattr(cfg_block, "train_model", default) or default).strip()


def _sgdc_pretrain_filename(dataset_name: str, model_name: str) -> str:
    return f"{dataset_name}_{model_name}_pretrain.cpt"


def _sgdc_condense_root(cfg, condense_cfg) -> str:
    explicit = str(getattr(condense_cfg, "out_root", "") or "").strip()
    if explicit:
        return explicit
    return str(_graph_condensation_root(cfg) / "sgdc")


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        if "," in text:
            return [part.strip().strip("'\"") for part in text.split(",") if part.strip()]
        return [text]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _read_dataset_list(tsv_path: str) -> List[str]:
    path = Path(tsv_path)
    if not path.is_file():
        print(f"[Context generation][Condensation] Dataset list not found: {path}")
        return []
    return read_name_list_file(path)


def _split_arg(split_value: Any, fallback: str = "0.8,0.1,0.1") -> str:
    if split_value is None:
        return fallback
    if isinstance(split_value, str):
        text = split_value.strip()
        return text or fallback
    if isinstance(split_value, (list, tuple)) and len(split_value) >= 3:
        return ",".join(str(float(split_value[idx])) for idx in range(3))
    return fallback


def _coerce_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _cfg_alias_value(block, *names: str):
    if block is None:
        return None
    for name in names:
        if hasattr(block, name):
            value = getattr(block, name)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return value
    return None


def _resolve_run_seeds(cfg) -> List[int]:
    raw_seeds = getattr(cfg, "seeds", None) or []
    if isinstance(raw_seeds, (int, float)):
        seeds = [int(raw_seeds)]
    else:
        seeds = [int(seed) for seed in raw_seeds]
    if seeds:
        return seeds
    return [int(getattr(cfg, "seed", 0))]


def _resolve_dataset_targets(
    *,
    explicit_dataset: Any,
    run_all: bool,
    tsv_path: str,
) -> List[str]:
    explicit = _normalize_list(explicit_dataset)
    if explicit and not run_all:
        return explicit

    if not run_all:
        return explicit

    names = _read_dataset_list(tsv_path)
    if explicit:
        allowed = {name.casefold() for name in explicit}
        names = [name for name in names if name.casefold() in allowed]
    return names


@contextmanager
def _temporary_argv(args: Sequence[str]):
    old = list(sys.argv)
    sys.argv = [old[0]] + list(args)
    try:
        yield
    finally:
        sys.argv = old


def _run_main_with_argv(main_func, args: Sequence[str]) -> int:
    with _temporary_argv(args):
        status = main_func()
    return int(status) if status is not None else 0


def run_bonsai_condensation_from_cfg(cfg) -> int:
    from .bonsai import condensation as module

    context_cfg = getattr(cfg, "context", None)
    bonsai_cfg = getattr(context_cfg, "bonsai", None)
    if bonsai_cfg is None:
        print("[Context generation][Bonsai] Missing context.bonsai config block.")
        return 1

    dataset_cfg = _cfg_block(getattr(bonsai_cfg, "dataset", None))
    run_all = _to_bool(getattr(bonsai_cfg, "run_all", False))
    dataset = _dataset_name(dataset_cfg)
    tsv_path = _dataset_list_path(dataset_cfg, "available_node_datasets", "data/available_node_datasets.tsv")

    targets = _resolve_dataset_targets(
        explicit_dataset=dataset,
        run_all=run_all,
        tsv_path=tsv_path,
    )
    if not targets:
        print("[Context generation][Bonsai] No datasets resolved.")
        return 1

    statuses: List[int] = []
    for dataset_name in targets:
        args: List[str] = ["--dataset", str(dataset_name)]

        args.extend(["--out_root", _bonsai_out_root(cfg, bonsai_cfg)])

        mapping = {
            "target_size_frac": "--target_size_frac",
            "budget_mb": "--budget_mb",
            "k": "--k",
            "neighbor_hops": "--neighbor_hops",
            "max_train_nodes": "--max_train_nodes",
            "knn_block_q": "--knn_block_q",
            "knn_block_c": "--knn_block_c",
            "knn_device": "--knn_device",
            "budget_mode": "--budget_mode",
            "min_nodes": "--min_nodes",
            "min_seed_nodes": "--min_seed_nodes",
            "csv_out": "--csv_out",
            "split_mode": "--split_mode",
            "mask_col": "--mask_col",
        }
        for key, flag in mapping.items():
            if not hasattr(bonsai_cfg, key):
                continue
            value = getattr(bonsai_cfg, key)
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            args.extend([flag, _coerce_value(value)])

        args.extend(["--dataset_root", _dataset_root(dataset_cfg)])
        seed_value = int(getattr(cfg, "seed", 0))
        args.extend(["--seed", str(seed_value)])

        args.extend(["--split_root", _dataset_split_root(dataset_cfg)])
        split_value = _split_arg(_dataset_fixed_split(dataset_cfg))
        args.extend(["--split", split_value])

        args.extend(["--feat_dim", str(_dataset_feat_dim(dataset_cfg))])
        args.append("--feat_reduction" if _dataset_feat_reduction(dataset_cfg) else "--no-feat_reduction")

        print(f"[Context generation][Bonsai][condense] dataset={dataset_name}")
        statuses.append(_run_main_with_argv(module.main, args))
    return 0 if all(status == 0 for status in statuses) else 1


def _bonsai_eval_single_dataset(cfg, dataset_name: str) -> int:
    from .bonsai import eval as module

    context_cfg = getattr(cfg, "context", None)
    bonsai_cfg = getattr(context_cfg, "bonsai", None)
    eval_cfg = getattr(bonsai_cfg, "eval", None) if bonsai_cfg is not None else None
    if eval_cfg is None:
        print("[Context generation][Bonsai][eval] Missing context.bonsai.eval config block.")
        return 1

    shared_dataset_cfg = _cfg_block(getattr(bonsai_cfg, "dataset", None))
    override_dataset_cfg = _cfg_block(getattr(eval_cfg, "dataset", None))
    args: List[str] = ["--dataset", str(dataset_name)]
    args.extend(["--condensed_root", _bonsai_condensed_root(cfg, eval_cfg)])

    mapping = [
        (("condensed_dir",), "--condensed_dir"),
        (("target_size_frac",), "--target_size_frac"),
        (("budget_mb",), "--budget_mb"),
        (("backend",), "--backend"),
        (("eval_mode", "mode"), "--eval_mode"),
        (("eval_batch_size", "batch_size"), "--eval_batch_size"),
        (("eval_num_workers", "num_workers"), "--eval_num_workers"),
        (("baseline_train_mode",), "--baseline_train_mode"),
        (("baseline_train_batch_size",), "--baseline_train_batch_size"),
        (("baseline_train_num_workers",), "--baseline_train_num_workers"),
        (("baseline_num_neighbors",), "--baseline_num_neighbors"),
        (("condensed_train_mode",), "--condensed_train_mode"),
        (("condensed_train_batch_size",), "--condensed_train_batch_size"),
        (("condensed_train_num_workers",), "--condensed_train_num_workers"),
        (("condensed_num_neighbors",), "--condensed_num_neighbors"),
        (("model",), "--model"),
        (("num_layers",), "--num_layers"),
        (("hidden_dim",), "--hidden_dim"),
        (("out_dim",), "--out_dim"),
        (("dropout",), "--dropout"),
        (("lr",), "--lr"),
        (("weight_decay",), "--weight_decay"),
        (("epochs",), "--epochs"),
        (("mask_col",), "--mask_col"),
        (("split_mode",), "--split_mode"),
        (("bonsai_runs", "runs"), "--bonsai_runs"),
        (("csv_out",), "--csv_out"),
    ]
    for aliases, flag in mapping:
        value = _cfg_alias_value(eval_cfg, *aliases)
        if value is None:
            continue
        args.extend([flag, _coerce_value(value)])

    args.extend(["--dataset_root", _override_dataset_root(override_dataset_cfg, shared_dataset_cfg)])
    args.extend(["--feat_dim", str(_override_dataset_feat_dim(override_dataset_cfg, shared_dataset_cfg))])
    args.append("--feat_reduction" if _override_dataset_feat_reduction(override_dataset_cfg, shared_dataset_cfg) else "--no-feat_reduction")
    args.extend(["--device", str(int(getattr(cfg, "device", 0)))])

    seed_value = int(getattr(cfg, "seed", 0))
    args.extend(["--seed", str(seed_value)])

    args.extend(["--split_root", _override_dataset_split_root(override_dataset_cfg, shared_dataset_cfg)])
    split_value = _split_arg(_override_fixed_split(override_dataset_cfg, shared_dataset_cfg))
    args.extend(["--split", split_value])

    if _to_bool(_cfg_alias_value(eval_cfg, "bonsai_preset", "preset") or False):
        args.append("--bonsai_preset")

    return _run_main_with_argv(module.main, args)


def run_bonsai_eval_from_cfg(cfg) -> int:
    context_cfg = getattr(cfg, "context", None)
    bonsai_cfg = getattr(context_cfg, "bonsai", None)
    eval_cfg = getattr(bonsai_cfg, "eval", None) if bonsai_cfg is not None else None

    if eval_cfg is None:
        print("[Context generation][Bonsai][eval] Missing context.bonsai.eval config block.")
        return 1

    run_all = (
        _to_bool(getattr(bonsai_cfg, "run_all", False))
        or _to_bool(getattr(eval_cfg, "run_all", False))
    )

    shared_dataset_cfg = _cfg_block(getattr(bonsai_cfg, "dataset", None))
    override_dataset_cfg = _cfg_block(getattr(eval_cfg, "dataset", None))
    explicit_dataset = _override_dataset_name(override_dataset_cfg, shared_dataset_cfg)
    tsv_path = _dataset_list_path(shared_dataset_cfg, "available_node_datasets", "data/available_node_datasets.tsv")

    targets = _resolve_dataset_targets(
        explicit_dataset=explicit_dataset,
        run_all=run_all,
        tsv_path=tsv_path,
    )

    if not targets:
        print("[Context generation][Bonsai][eval] No datasets resolved.")
        return 1

    statuses: List[int] = []
    if _to_bool(_cfg_alias_value(eval_cfg, "bonsai_preset", "preset") or False):
        seeds = [int(getattr(cfg, "seed", 0))]
    else:
        seeds = _resolve_run_seeds(cfg)
    total_runs = len(seeds)
    for dataset_name in targets:
        for index, seed in enumerate(seeds, start=1):
            run_cfg = cfg.clone()
            run_cfg.seed = int(seed)
            if total_runs > 1:
                print(
                    f"[Context generation][Bonsai][eval] "
                    f"dataset={dataset_name} run={index}/{total_runs} seed={seed}"
                )
            else:
                print(f"[Context generation][Bonsai][eval] Evaluating {dataset_name}")
            statuses.append(_bonsai_eval_single_dataset(run_cfg, dataset_name))
    return 0 if all(status == 0 for status in statuses) else 1


def _sgdc_targets(cfg, stage_cfg) -> List[str]:
    context_cfg = getattr(cfg, "context", None)
    sgdc_cfg = getattr(context_cfg, "sgdc", None)
    shared_dataset_cfg = _cfg_block(getattr(sgdc_cfg, "dataset", None))
    stage_dataset_cfg = _cfg_block(getattr(stage_cfg, "dataset", None))

    run_all = (
        _to_bool(getattr(sgdc_cfg, "run_all", False))
        or _to_bool(getattr(stage_cfg, "run_all", False))
    )

    explicit = _override_dataset_name(stage_dataset_cfg, shared_dataset_cfg)
    tsv_path = _dataset_list_path(shared_dataset_cfg, "available_graph_datasets", "data/available_graph_datasets.tsv")

    return _resolve_dataset_targets(
        explicit_dataset=explicit,
        run_all=run_all,
        tsv_path=tsv_path,
    )


def run_sgdc_pretrain_from_cfg(cfg) -> int:
    from .sgdc import pretrain as module

    context_cfg = getattr(cfg, "context", None)
    sgdc_cfg = getattr(context_cfg, "sgdc", None)
    pretrain_cfg = getattr(sgdc_cfg, "pretrain", None) if sgdc_cfg is not None else None
    if pretrain_cfg is None:
        print("[Context generation][SGDC][pretrain] Missing context.sgdc.pretrain config block.")
        return 1

    targets = _sgdc_targets(cfg, pretrain_cfg)
    if not targets:
        print("[Context generation][SGDC][pretrain] No datasets resolved.")
        return 1

    statuses: List[int] = []
    shared_dataset_cfg = _cfg_block(getattr(sgdc_cfg, "dataset", None))
    stage_dataset_cfg = _cfg_block(getattr(pretrain_cfg, "dataset", None))
    for dataset_name in targets:
        save_ckpt = str(getattr(pretrain_cfg, "save_ckpt", "") or "").strip()
        if not save_ckpt:
            model_name = _sgdc_teacher_model_name(pretrain_cfg)
            save_ckpt = str(
                Path(_sgdc_pretrain_root(cfg, pretrain_cfg))
                / _sgdc_pretrain_filename(dataset_name, model_name)
            )

        print(f"[Context generation][SGDC][pretrain] dataset={dataset_name}")
        statuses.append(
            int(
                module.run_sgdc_pretrain_dataset(
                    dataset_name=dataset_name,
                    pretrain_cfg=pretrain_cfg,
                    dataset_root=_override_dataset_root(stage_dataset_cfg, shared_dataset_cfg),
                    feat_reduction=_override_dataset_feat_reduction(stage_dataset_cfg, shared_dataset_cfg),
                    feat_dim=_override_dataset_feat_dim(stage_dataset_cfg, shared_dataset_cfg),
                    seed=int(getattr(cfg, "seed", 0)),
                    device_id=int(getattr(cfg, "device", 0)),
                    save_ckpt=save_ckpt,
                )
            )
        )

    return 0 if all(status == 0 for status in statuses) else 1


def _infer_sgdc_ckpt(cfg, dataset_name: str, condense_cfg, sgdc_cfg) -> str:
    explicit = str(getattr(condense_cfg, "pretrained_ckpt", "") or "").strip()
    if explicit:
        return explicit

    ckpt_dir = ""
    pretrain_cfg = getattr(sgdc_cfg, "pretrain", None)
    if pretrain_cfg is not None:
        ckpt_dir = _sgdc_pretrain_root(cfg, pretrain_cfg)
    if not ckpt_dir:
        ckpt_dir = str(_graph_condensation_root(cfg) / "sgdc" / "pretrain")

    model_name = _sgdc_teacher_model_name(condense_cfg)
    return str(Path(ckpt_dir) / _sgdc_pretrain_filename(dataset_name, model_name))


def run_sgdc_condensation_from_cfg(cfg) -> int:
    from .sgdc import condensation as module

    context_cfg = getattr(cfg, "context", None)
    sgdc_cfg = getattr(context_cfg, "sgdc", None)
    condense_cfg = getattr(sgdc_cfg, "condense", None) if sgdc_cfg is not None else None
    if condense_cfg is None:
        print("[Context generation][SGDC][condense] Missing context.sgdc.condense config block.")
        return 1

    targets = _sgdc_targets(cfg, condense_cfg)
    if not targets:
        print("[Context generation][SGDC][condense] No datasets resolved.")
        return 1

    statuses: List[int] = []
    shared_dataset_cfg = _cfg_block(getattr(sgdc_cfg, "dataset", None))
    stage_dataset_cfg = _cfg_block(getattr(condense_cfg, "dataset", None))
    for dataset_name in targets:
        pretrained_ckpt = _infer_sgdc_ckpt(cfg, dataset_name, condense_cfg, sgdc_cfg)
        if not pretrained_ckpt:
            print(f"[Context generation][SGDC][condense] Missing pretrained checkpoint for {dataset_name}.")
            statuses.append(1)
            continue

        print(f"[Context generation][SGDC][condense] dataset={dataset_name}")
        statuses.append(
            int(
                module.run_sgdc_condensation_dataset(
                    dataset_name=dataset_name,
                    condense_cfg=condense_cfg,
                    dataset_root=_override_dataset_root(stage_dataset_cfg, shared_dataset_cfg),
                    split_root=_override_dataset_split_root(stage_dataset_cfg, shared_dataset_cfg),
                    split=_override_fixed_split(stage_dataset_cfg, shared_dataset_cfg),
                    feat_reduction=_override_dataset_feat_reduction(stage_dataset_cfg, shared_dataset_cfg),
                    feat_dim=_override_dataset_feat_dim(stage_dataset_cfg, shared_dataset_cfg),
                    seed=int(getattr(cfg, "seed", 0)),
                    device_id=int(getattr(cfg, "device", 0)),
                    out_root=_sgdc_condense_root(cfg, condense_cfg),
                    pretrained_ckpt=pretrained_ckpt,
                )
            )
        )

    return 0 if all(status == 0 for status in statuses) else 1
