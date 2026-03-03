"""Runtime orchestration for the `run_finetune.py` entrypoint."""

from __future__ import annotations

import os
import random
import statistics
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from code.config import cfg as base_cfg, update_cfg
from code.utils import set_seed

from .finetuner import FinetuneRunner
from .utils import extract_few_shot, resolve_pretrained_checkpoint, run_finetune_tasks

MetricDict = Dict[str, float]
_SUMMARY_EXCLUDED_METRICS = {"batch_size", "test_loss"}


def _normalize_dataset_aliases(argv: List[str]) -> List[str]:
    """
    Support `dataset.*` CLI aliases by remapping to finetune and pretrain blocks.

    This keeps legacy CLI invocations working while still allowing explicit
    `finetune.dataset.*` and `pretrain.dataset.*` overrides.
    """
    normalized: List[str] = []
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token == "--config":
            normalized.append(token)
            if idx + 1 < len(argv):
                normalized.append(argv[idx + 1])
            idx += 2
            continue
        if token.startswith("--"):
            normalized.append(token)
            idx += 1
            continue
        if idx + 1 >= len(argv):
            normalized.append(token)
            break
        value = argv[idx + 1]
        if token.startswith("dataset."):
            normalized.extend([f"finetune.{token}", value, f"pretrain.{token}", value])
        else:
            normalized.extend([token, value])
        idx += 2
    return normalized


def _resolve_run_seeds(cfg) -> List[int]:
    """Resolve per-run seeds and append deterministic random seeds when needed."""
    base_seed = int(getattr(cfg, "seed", 0))
    raw_seeds = getattr(cfg, "seeds", None) or []
    if isinstance(raw_seeds, (int, float)):
        configured = [int(raw_seeds)]
    else:
        configured = [int(seed) for seed in raw_seeds]

    finetune_cfg = getattr(cfg, "finetune", None)
    requested_runs = int(getattr(finetune_cfg, "num_runs", len(configured) or 1) or 0)
    if requested_runs <= 0:
        requested_runs = len(configured) or 1

    # Always honor cfg.seed as the first run.
    seeds = [base_seed]
    existing = {base_seed}
    for seed in configured:
        if seed in existing:
            continue
        seeds.append(seed)
        existing.add(seed)
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


def _load_checkpoint_metrics(path: str) -> MetricDict:
    if not os.path.isfile(path):
        return {}
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"[Finetune][Summary] Failed to load checkpoint metrics: {path} ({exc})")
        return {}
    metrics = payload.get("metrics", {})
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def _checkpoint_path_for_runner(runner: FinetuneRunner) -> str:
    """Resolve checkpoint path even when helper methods are not available."""
    helper = getattr(runner, "get_checkpoint_path_for_metrics", None)
    if callable(helper):
        candidate = helper()
        if candidate:
            return str(candidate)
    return os.path.join(runner.run_dir, f"{runner.run_name}.pt")


def _collect_run_metrics(runner: FinetuneRunner) -> MetricDict:
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
    print(f"[Finetune][Summary] Completed {len(run_metrics)} runs.")
    print(f"[Finetune][Summary] Seeds: {seeds}")

    values_by_key: Dict[str, List[float]] = {}
    for metrics in run_metrics:
        for key, value in metrics.items():
            if key in _SUMMARY_EXCLUDED_METRICS:
                continue
            if isinstance(value, (int, float)):
                values_by_key.setdefault(key, []).append(float(value))

    if not values_by_key:
        print("[Finetune][Summary] No metrics available to summarize.")
        return

    epoch_keys = {"best_epoch", "epoch"}
    for key in sorted(values_by_key.keys()):
        values = values_by_key[key]
        if key in epoch_keys:
            epoch_values = [int(round(value)) for value in values]
            print(f"[Finetune][Summary] {key}: {epoch_values}")
            continue

        mean_value = statistics.mean(values)
        std_value = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"[Finetune][Summary] {key}: mean={mean_value:.4f} std={std_value:.4f} n={len(values)}")


def _resolve_checkpoint_once(cfg) -> Tuple[Optional[str], Optional[str], int]:
    explicit_ckpt = str(getattr(getattr(cfg, "finetune", None), "pretrained_checkpoint", "") or "").strip()
    if explicit_ckpt:
        if not os.path.isfile(explicit_ckpt):
            print(f"[Finetune] Explicit checkpoint does not exist: {explicit_ckpt}")
            return None, None, 1
        run_name = os.path.splitext(os.path.basename(explicit_ckpt))[0]
        return explicit_ckpt, run_name, 0

    ckpt_path, run_name = resolve_pretrained_checkpoint(cfg)
    if not ckpt_path:
        print("[Finetune] Unable to resolve pretrained checkpoint from pretrain/model config.")
        return None, None, 1
    return ckpt_path, run_name, 0


def _build_finetune_cfg(argv: Iterable[str]):
    """
    Parse CLI overrides for finetuning.

    `few_shot` is accepted as a compatibility alias for
    `finetune.dataset.fixed_split`.
    """
    raw_argv = list(argv)
    forwarded_argv, few_shot_split = extract_few_shot(raw_argv)
    normalized_argv = _normalize_dataset_aliases(forwarded_argv)
    cfg = update_cfg(base_cfg, " ".join(normalized_argv))

    if few_shot_split is not None:
        cfg.finetune.dataset.fixed_split = few_shot_split
    return cfg


def run_finetune(cfg) -> int:
    if getattr(cfg.finetune, "run_all", False):
        return run_finetune_tasks(cfg)

    seeds = _resolve_run_seeds(cfg)
    if getattr(cfg, "seeds", None) != seeds:
        cfg.seeds = seeds
        print(f"[Finetune][Multi-run] Extended seeds list to {len(seeds)} runs.")

    # Resolve checkpoint once so all seeds evaluate the same pretrained run.
    checkpoint_cfg = cfg.clone()
    checkpoint_cfg.seed = int(getattr(cfg, "seed", 0))
    ckpt_path, run_name, resolve_status = _resolve_checkpoint_once(checkpoint_cfg)
    if resolve_status != 0 or not ckpt_path or not run_name:
        _summarize_runs([], [int(seed) for seed in seeds])
        return resolve_status or 1

    run_metrics: List[MetricDict] = []
    total_runs = len(seeds)
    for index, seed in enumerate(seeds, start=1):
        run_cfg = cfg.clone()
        run_cfg.seed = int(seed)
        if total_runs > 1:
            print(f"[Finetune][Multi-run] Running {index}/{total_runs} with seed={seed}")

        set_seed(run_cfg.seed)
        print(f"[Finetune] Loaded checkpoint for run '{run_name}': {ckpt_path}")
        runner = FinetuneRunner(cfg=run_cfg, pretrained_checkpoint=ckpt_path, pretrained_run_name=run_name)
        runner.fit()
        run_metrics.append(_collect_run_metrics(runner))

    _summarize_runs(run_metrics, [int(seed) for seed in seeds])
    return 0


def run_finetune_from_cli(argv: Iterable[str]) -> int:
    warnings.filterwarnings("ignore")
    cfg = _build_finetune_cfg(argv)
    return run_finetune(cfg)
