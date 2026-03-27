"""Shared helpers for multi-run orchestration and run summaries."""

from __future__ import annotations

import os
import random
import statistics
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch

MetricDict = Dict[str, float]
_SUMMARY_BASE_EXCLUDED_METRICS = {"batch_size"}


def _normalize_seed_list(raw_seeds: object) -> List[int]:
    if isinstance(raw_seeds, (int, float)):
        return [int(raw_seeds)]
    if raw_seeds is None:
        return []
    return [int(seed) for seed in raw_seeds]


def resolve_run_seeds(
    *,
    base_seed: int,
    raw_seeds: object,
    requested_runs: int,
    honor_base_seed_first: bool = False,
) -> List[int]:
    """Resolve per-run seeds and append deterministic random seeds when needed."""
    base_seed = int(base_seed)
    configured = _normalize_seed_list(raw_seeds)
    if requested_runs <= 0:
        requested_runs = len(configured) or 1

    if honor_base_seed_first:
        seeds = [base_seed]
        existing = {base_seed}
        for seed in configured:
            if seed in existing:
                continue
            seeds.append(seed)
            existing.add(seed)
            if len(seeds) >= requested_runs:
                return seeds[:requested_runs]
    else:
        seeds = configured or [base_seed]
        existing = set(seeds)
        if len(seeds) >= requested_runs:
            return seeds[:requested_runs]

    rng = random.Random(base_seed)
    while len(seeds) < requested_runs:
        candidate = rng.randint(0, 2**31 - 1)
        if candidate in existing:
            continue
        seeds.append(candidate)
        existing.add(candidate)

    return seeds[:requested_runs]


def load_checkpoint_metrics(path: str, *, log_prefix: str) -> MetricDict:
    """Load persisted numeric metrics from a checkpoint file when present."""
    if not os.path.isfile(path):
        return {}
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"{log_prefix} Failed to load checkpoint metrics: {path} ({exc})")
        return {}

    metrics = payload.get("metrics", {})
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def checkpoint_path_for_runner(runner) -> str:
    """Resolve a runner checkpoint path even when helper methods are absent."""
    helper = getattr(runner, "get_checkpoint_path_for_metrics", None)
    if callable(helper):
        candidate = helper()
        if candidate:
            return str(candidate)
    return os.path.join(runner.run_dir, f"{runner.run_name}.pt")


def collect_run_metrics(runner, *, log_prefix: str) -> MetricDict:
    """Collect best metrics from memory, or fallback to persisted checkpoint."""
    in_memory = {
        key: float(value)
        for key, value in getattr(runner, "best_metrics", {}).items()
        if isinstance(value, (int, float))
    }
    if in_memory:
        return in_memory
    return load_checkpoint_metrics(checkpoint_path_for_runner(runner), log_prefix=log_prefix)


def should_include_summary_metric(
    metric_name: str,
    *,
    excluded_metrics: Optional[set[str]] = None,
) -> bool:
    """Return True when a metric should appear in summaries and saved tables."""
    name = str(metric_name or "").strip()
    if not name:
        return False
    if excluded_metrics and name in excluded_metrics:
        return False
    if name in _SUMMARY_BASE_EXCLUDED_METRICS:
        return False
    return "loss" not in name.lower()


def aggregate_run_metrics(
    run_metrics: Sequence[Mapping[str, float]],
    *,
    excluded_metrics: Optional[set[str]] = None,
) -> dict[str, dict[str, object]]:
    """Aggregate per-run metrics into epoch lists and mean/std/n summaries."""
    excluded = excluded_metrics or set()
    epoch_keys = {"best_epoch", "epoch"}
    values_by_key: Dict[str, List[float]] = {}
    for metrics in run_metrics:
        for key, value in metrics.items():
            if key not in epoch_keys and not should_include_summary_metric(key, excluded_metrics=excluded):
                continue
            if isinstance(value, (int, float)):
                values_by_key.setdefault(key, []).append(float(value))

    epoch_values: Dict[str, List[int]] = {}
    metric_stats: Dict[str, Dict[str, float | int]] = {}
    for key in sorted(values_by_key.keys()):
        values = values_by_key[key]
        if key in epoch_keys:
            epoch_values[key] = [int(round(value)) for value in values]
            continue

        metric_stats[key] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }

    return {
        "epoch_values": epoch_values,
        "metric_stats": metric_stats,
    }


def summarize_runs(
    run_metrics: Sequence[Mapping[str, float]],
    seeds: Iterable[int],
    *,
    log_prefix: str,
    excluded_metrics: Optional[set[str]] = None,
) -> None:
    """Print a compact multi-run summary."""
    seed_list = [int(seed) for seed in seeds]
    summary = aggregate_run_metrics(run_metrics, excluded_metrics=excluded_metrics)
    epoch_values = summary["epoch_values"]
    metric_stats = summary["metric_stats"]

    print(f"{log_prefix} Completed {len(run_metrics)} runs.")
    print(f"{log_prefix} Seeds: {seed_list}")

    if not epoch_values and not metric_stats:
        print(f"{log_prefix} No metrics available to summarize.")
        return

    for key in sorted(epoch_values.keys()):
        print(f"{log_prefix} {key}: {epoch_values[key]}")
    for key in sorted(metric_stats.keys()):
        stats = metric_stats[key]
        print(
            f"{log_prefix} {key}: mean={float(stats['mean']):.4f} "
            f"std={float(stats['std']):.4f} n={int(stats['n'])}"
        )


__all__ = [
    "aggregate_run_metrics",
    "MetricDict",
    "checkpoint_path_for_runner",
    "collect_run_metrics",
    "load_checkpoint_metrics",
    "resolve_run_seeds",
    "should_include_summary_metric",
    "summarize_runs",
]
