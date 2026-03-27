"""Train-specific helper utilities and runtime orchestration."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, List

from code.utils.run_helpers import (
    aggregate_run_metrics,
    checkpoint_path_for_runner,
    collect_run_metrics,
    resolve_run_seeds,
    summarize_runs,
)
from code.utils.save_results import append_workflow_result

if TYPE_CHECKING:
    from .trainer import TrainRunner


def _shared_split_root(cfg) -> str:
    ds_cfg = getattr(getattr(cfg, "data_preparation", None), "dataset", None)
    return getattr(ds_cfg, "split_root", "data/splits")


def _shared_induced_root(cfg, fallback: str = "") -> str:
    ds_cfg = getattr(getattr(cfg, "data_preparation", None), "dataset", None)
    root = getattr(ds_cfg, "induced_root", "")
    return root or fallback


def _split_dataset_name(base_name: str, task_level: str, seed: int) -> str:
    name = str(base_name)
    if any(tag in name for tag in ("_node_seed", "_graph_seed", "_edge_seed")):
        return name
    return f"{name}_{task_level}_seed{int(seed)}"


def _collect_checkpoint_paths(runners: List["TrainRunner"]) -> list[str]:
    checkpoint_paths: list[str] = []
    seen_paths: set[str] = set()
    for runner in runners:
        path = checkpoint_path_for_runner(runner)
        if path and path not in seen_paths:
            seen_paths.add(path)
            checkpoint_paths.append(path)
    return checkpoint_paths


def _append_train_result(cfg, runners: List["TrainRunner"], run_metrics, started_at, ended_at, seeds) -> None:
    all_skipped = bool(runners) and all(bool(getattr(runner, "_skip_due_to_existing_checkpoint", False)) for runner in runners)
    if all_skipped and not bool(getattr(cfg.save_results, "save_skipped", False)):
        return

    summary = aggregate_run_metrics(run_metrics)
    append_workflow_result(
        cfg=cfg,
        workflow="train",
        started_at=started_at,
        ended_at=ended_at,
        checkpoint_save_paths=_collect_checkpoint_paths(runners),
        seeds=seeds,
        best_epochs=summary["epoch_values"].get("best_epoch"),
        metric_summary=summary["metric_stats"],
    )


def run_train_workflow(cfg) -> int:
    """Execute one or more training runs for the provided config."""
    from .trainer import TrainRunner

    requested_runs = int(getattr(getattr(cfg, "train", None), "num_runs", 0) or 0)
    seeds = resolve_run_seeds(
        base_seed=int(getattr(cfg, "seed", 0)),
        raw_seeds=getattr(cfg, "seeds", None),
        requested_runs=requested_runs,
        honor_base_seed_first=False,
    )
    if getattr(cfg, "seeds", None) != seeds:
        cfg.seeds = seeds
        print(f"[Train][Multi-run] Extended seeds list to {len(seeds)} runs.")

    started_at = datetime.now().astimezone()
    run_metrics = []
    runners: List[TrainRunner] = []
    total_runs = len(seeds)
    for index, seed in enumerate(seeds, start=1):
        run_cfg = cfg.clone()
        run_cfg.seed = int(seed)
        if total_runs > 1:
            print(f"[Train][Multi-run] Running {index}/{total_runs} with seed={seed}")

        runner = TrainRunner(cfg=run_cfg)
        runner.fit()
        runners.append(runner)
        run_metrics.append(collect_run_metrics(runner, log_prefix="[Train][Summary]"))

    summarize_runs(run_metrics, seeds, log_prefix="[Train][Summary]")
    ended_at = datetime.now().astimezone()
    _append_train_result(cfg, runners, run_metrics, started_at, ended_at, seeds)
    return 0
