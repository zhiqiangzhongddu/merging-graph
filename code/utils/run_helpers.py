"""Shared helpers for multi-run orchestration and run summaries."""

from __future__ import annotations

import os
import random
import statistics
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch

MetricDict = Dict[str, float]


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


def summarize_runs(
    run_metrics: Sequence[Mapping[str, float]],
    seeds: Iterable[int],
    *,
    log_prefix: str,
    excluded_metrics: Optional[set[str]] = None,
) -> None:
    """Print a compact multi-run summary."""
    excluded = excluded_metrics or set()
    seed_list = [int(seed) for seed in seeds]

    print(f"{log_prefix} Completed {len(run_metrics)} runs.")
    print(f"{log_prefix} Seeds: {seed_list}")

    values_by_key: Dict[str, List[float]] = {}
    for metrics in run_metrics:
        for key, value in metrics.items():
            if key in excluded:
                continue
            if isinstance(value, (int, float)):
                values_by_key.setdefault(key, []).append(float(value))

    if not values_by_key:
        print(f"{log_prefix} No metrics available to summarize.")
        return

    epoch_keys = {"best_epoch", "epoch"}
    for key in sorted(values_by_key.keys()):
        values = values_by_key[key]
        if key in epoch_keys:
            epoch_values = [int(round(value)) for value in values]
            print(f"{log_prefix} {key}: {epoch_values}")
            continue

        mean_value = statistics.mean(values)
        std_value = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{log_prefix} {key}: mean={mean_value:.4f} std={std_value:.4f} n={len(values)}")


__all__ = [
    "MetricDict",
    "checkpoint_path_for_runner",
    "collect_run_metrics",
    "load_checkpoint_metrics",
    "resolve_run_seeds",
    "summarize_runs",
]
