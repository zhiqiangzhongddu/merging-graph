"""
SGDC graph condensation for graph-level context generation.

This module is config-driven and invoked from `run_context_generation.py`
through `condensation_runner.py`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import torch

from code.data_loader.datasets import create_dataset, make_loaders
from code.utils import ensure_dir, format_split_for_name, resolve_project_path, set_seed

from .core import condense_sgdc
from .utils import save_graph_list_as_pyg_dataset

_ROCAUC_DATASETS = {
    "bace",
    "bbbp",
    "clintox",
    "hiv",
    "muv",
    "pcba",
    "sider",
    "tox21",
    "toxcast",
}
_RMSE_DATASETS = {
    "esol", 
    "freesolv", 
    "lipo"
}


def _default_metric(dataset_name: str, raw_metric: str) -> str:
    metric = str(raw_metric).strip().lower()
    if metric != "acc":
        return metric
    dataset_key = str(dataset_name).strip().lower()
    if dataset_key in _ROCAUC_DATASETS:
        return "rocauc"
    if dataset_key in _RMSE_DATASETS:
        return "rmse"
    return "acc"


def _build_sgdc_args(
    *,
    condense_cfg,
    dataset_name: str,
    feat_dim: int,
    pretrained_ckpt: str,
    save_dir: Path,
    seed: int,
    device: torch.device,
):
    model = str(getattr(condense_cfg, "model", "gin")).strip()
    return SimpleNamespace(
        dataset=str(dataset_name),
        seed=int(seed),
        device=device,
        feat_dim=int(feat_dim),
        save_dir=str(save_dir),
        model=model,
        budget=int(getattr(condense_cfg, "budget", 10)),
        budget_total=int(getattr(condense_cfg, "budget_total", 0)),
        initialize=str(getattr(condense_cfg, "initialize", "Herding")),
        pretrained_ckpt=str(pretrained_ckpt),
        nhid=int(getattr(condense_cfg, "nhid", 128)),
        layers=int(getattr(condense_cfg, "layers", 3)),
        drop_ratio=float(getattr(condense_cfg, "drop_ratio", 0.5)),
        outer_bs=int(getattr(condense_cfg, "outer_bs", 64)),
        outer_wd=float(getattr(condense_cfg, "outer_wd", 0.0)),
        outer_lr=float(getattr(condense_cfg, "outer_lr", 1e-4)),
        outer_epoch=int(getattr(condense_cfg, "outer_epoch", 50)),
        eval_every=int(getattr(condense_cfg, "eval_every", 1)),
        outer_grad_norm=float(getattr(condense_cfg, "outer_grad_norm", 0.1)),
        online_iteration=int(getattr(condense_cfg, "online_iteration", 50)),
        online_lr=float(getattr(condense_cfg, "online_lr", 0.01)),
        online_wd=float(getattr(condense_cfg, "online_wd", 0.0)),
        online_batch_size=int(getattr(condense_cfg, "online_batch_size", 128)),
        online_min_iteration=int(getattr(condense_cfg, "online_min_iteration", 20)),
        num_models=int(getattr(condense_cfg, "num_models", 1)),
        pre_epoch=int(getattr(condense_cfg, "pre_epoch", 100)),
        pre_lr=float(getattr(condense_cfg, "pre_lr", 0.01)),
        pre_wd=float(getattr(condense_cfg, "pre_wd", 2e-5)),
        test_epoch=int(getattr(condense_cfg, "test_epoch", 300)),
        test_bs=int(getattr(condense_cfg, "test_bs", 128)),
        test_lr=float(getattr(condense_cfg, "test_lr", 1e-3)),
        test_wd=float(getattr(condense_cfg, "test_wd", 2e-5)),
        metric=_default_metric(dataset_name, str(getattr(condense_cfg, "metric", "acc"))),
        scale=str(getattr(condense_cfg, "scale", "uniform")),
        reg_lambda=float(getattr(condense_cfg, "reg_lambda", 1e-6)),
        orth_reg=float(getattr(condense_cfg, "orth_reg", 1e-3)),
    )


def run_sgdc_condensation_dataset(
    *,
    dataset_name: str,
    condense_cfg,
    dataset_root: str,
    split_root: str,
    split: Sequence[float],
    feat_reduction: bool,
    feat_dim: int,
    seed: int,
    device_id: int,
    out_root: str,
    pretrained_ckpt: str,
) -> int:
    set_seed(int(seed))

    dataset = create_dataset(
        name=dataset_name,
        root=str(resolve_project_path(dataset_root)),
        task_level="graph",
        induced=False,
        feat_reduction=bool(feat_reduction),
        feat_reduction_dim=int(feat_dim),
    )

    batch_size = int(getattr(condense_cfg, "outer_bs", 64))
    split_tuple = tuple(float(value) for value in split)
    train_loader, val_loader, test_loader = make_loaders(
        dataset=dataset,
        dataset_name=dataset_name,
        task_level="graph",
        batch_size=batch_size,
        num_workers=0,
        split=split_tuple,
        seed=int(seed),
        induced=False,
        split_root=str(resolve_project_path(split_root)),
    )
    split_meta = split_tuple
    split_tag = format_split_for_name(split_tuple)

    if len(train_loader.dataset) == 0:
        raise RuntimeError(f"SGDC condensation has an empty train split for dataset '{dataset_name}'.")

    out_dir = resolve_project_path(out_root)
    ensure_dir(str(out_dir))
    save_dir = out_dir / dataset_name / (
        f"sgdc_budgetbase{int(getattr(condense_cfg, 'budget', 10))}"
        f"_hid{int(getattr(condense_cfg, 'nhid', 128))}"
        f"_lr{float(getattr(condense_cfg, 'outer_lr', 1e-4))}"
    ) / f"split-{split_tag}"

    device = torch.device(f"cuda:{int(device_id)}" if torch.cuda.is_available() else "cpu")
    args = _build_sgdc_args(
        condense_cfg=condense_cfg,
        dataset_name=dataset_name,
        feat_dim=int(feat_dim),
        pretrained_ckpt=str(resolve_project_path(pretrained_ckpt)),
        save_dir=save_dir,
        seed=int(seed),
        device=device,
    )

    result = condense_sgdc(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dataset_name=dataset_name,
        args=args,
        device=device,
    )

    meta = {
        "dataset": dataset_name,
        "split": split_meta,
        "split_root": str(resolve_project_path(split_root)),
        **result.meta,
        **result.scores,
    }
    save_graph_list_as_pyg_dataset(
        result.graphs,
        out_root=str(save_dir),
        meta=meta,
    )
    print(
        f"[SGDC][condense] dataset={dataset_name} saved={save_dir} "
        f"graphs={len(result.graphs)} score={result.scores['score_mean_last']}"
    )
    return 0
