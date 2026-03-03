"""Runtime orchestration for the `run_train.py` entrypoint."""

from __future__ import annotations

import os
import random
import statistics
from typing import Dict, Iterable, List

import torch

from code.config import cfg as base_cfg, update_cfg
from code.train import TrainRunner

MetricDict = Dict[str, float]


def _resolve_run_seeds(cfg) -> List[int]:
    """Resolve per-run seeds and append deterministic random seeds when needed."""
    raw_seeds = getattr(cfg, "seeds", None) or []
    if isinstance(raw_seeds, (int, float)):
        seeds = [int(raw_seeds)]
    else:
        seeds = [int(seed) for seed in raw_seeds]

    if not seeds:
        seeds = [int(getattr(cfg, "seed", 0))]

    train_cfg = getattr(cfg, "train", None)
    requested_runs = int(getattr(train_cfg, "num_runs", len(seeds) or 1) or 0)
    if requested_runs <= 0:
        requested_runs = len(seeds)

    rng = random.Random(int(getattr(cfg, "seed", 0)))
    existing = set(seeds)
    while len(seeds) < requested_runs:
        candidate = rng.randint(0, 2**31 - 1)
        if candidate in existing:
            continue
        seeds.append(candidate)
        existing.add(candidate)

    return seeds[:requested_runs]


def _load_checkpoint_metrics(path: str) -> MetricDict:
    if not os.path.isfile(path):
        return {}
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"[Train][Summary] Failed to load checkpoint metrics: {path} ({exc})")
        return {}

    metrics = payload.get("metrics", {})
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def _checkpoint_path_for_runner(runner: TrainRunner) -> str:
    """Resolve checkpoint path even when helper methods are not available."""
    helper = getattr(runner, "get_checkpoint_path_for_metrics", None)
    if callable(helper):
        candidate = helper()
        if candidate:
            return str(candidate)
    return os.path.join(runner.run_dir, f"{runner.run_name}.pt")


def _collect_run_metrics(runner: TrainRunner) -> MetricDict:
    """Collect best metrics from memory, or fallback to persisted checkpoint."""
    in_memory = {
        key: float(value)
        for key, value in getattr(runner, "best_metrics", {}).items()
        if isinstance(value, (int, float))
    }
    if in_memory:
        return in_memory
    return _load_checkpoint_metrics(_checkpoint_path_for_runner(runner))


def _summarize_runs(run_metrics: List[MetricDict], seeds: List[int]) -> None:
    print(f"[Train][Summary] Completed {len(run_metrics)} runs.")
    print(f"[Train][Summary] Seeds: {seeds}")

    values_by_key: Dict[str, List[float]] = {}
    for metrics in run_metrics:
        for key, value in metrics.items():
            if key == "batch_size":
                continue
            if isinstance(value, (int, float)):
                values_by_key.setdefault(key, []).append(float(value))

    if not values_by_key:
        print("[Train][Summary] No metrics available to summarize.")
        return

    epoch_keys = {"best_epoch", "epoch"}
    for key in sorted(values_by_key.keys()):
        values = values_by_key[key]
        if key in epoch_keys:
            epoch_values = [int(round(value)) for value in values]
            print(f"[Train][Summary] {key}: {epoch_values}")
            continue

        mean_value = statistics.mean(values)
        std_value = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"[Train][Summary] {key}: mean={mean_value:.4f} std={std_value:.4f} n={len(values)}")


def run_train(cfg) -> int:
    seeds = _resolve_run_seeds(cfg)
    if getattr(cfg, "seeds", None) != seeds:
        cfg.seeds = seeds
        print(f"[Train][Multi-run] Extended seeds list to {len(seeds)} runs.")

    run_metrics: List[MetricDict] = []
    total_runs = len(seeds)

    for index, seed in enumerate(seeds, start=1):
        run_cfg = cfg.clone()
        run_cfg.seed = int(seed)
        if total_runs > 1:
            print(f"[Train][Multi-run] Running {index}/{total_runs} with seed={seed}")

        runner = TrainRunner(cfg=run_cfg)
        runner.fit()
        run_metrics.append(_collect_run_metrics(runner))

    _summarize_runs(run_metrics, seeds)
    return 0


def run_train_from_cli(argv: Iterable[str]) -> int:
    cfg = update_cfg(base_cfg, " ".join(argv))
    return run_train(cfg)
