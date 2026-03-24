"""Runtime orchestration for the `run_finetune.py` entrypoint."""

from __future__ import annotations

from typing import List

from code.utils.random import set_seed
from code.utils.run_helpers import collect_run_metrics, resolve_run_seeds, summarize_runs

from .finetuner import FinetuneRunner
from .utils import resolve_pretrained_checkpoint, run_finetune_tasks

_SUMMARY_EXCLUDED_METRICS = {"batch_size", "test_loss"}


def _resolve_checkpoint_once(cfg):
    explicit_ckpt = str(getattr(getattr(cfg, "finetune", None), "pretrained_checkpoint", "") or "").strip()
    if explicit_ckpt:
        import os

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


def run_finetune(cfg) -> int:
    """Execute one or more finetuning runs for the provided config."""
    if getattr(cfg.finetune, "run_all", False):
        return run_finetune_tasks(cfg)

    requested_runs = int(getattr(getattr(cfg, "finetune", None), "num_runs", 0) or 0)
    seeds = resolve_run_seeds(
        base_seed=int(getattr(cfg, "seed", 0)),
        raw_seeds=getattr(cfg, "seeds", None),
        requested_runs=requested_runs,
        honor_base_seed_first=True,
    )
    if getattr(cfg, "seeds", None) != seeds:
        cfg.seeds = seeds
        print(f"[Finetune][Multi-run] Extended seeds list to {len(seeds)} runs.")

    checkpoint_cfg = cfg.clone()
    checkpoint_cfg.seed = int(getattr(cfg, "seed", 0))
    ckpt_path, run_name, resolve_status = _resolve_checkpoint_once(checkpoint_cfg)
    if resolve_status != 0 or not ckpt_path or not run_name:
        summarize_runs([], seeds, log_prefix="[Finetune][Summary]", excluded_metrics=_SUMMARY_EXCLUDED_METRICS)
        return resolve_status or 1

    run_metrics: List[dict[str, float]] = []
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
        run_metrics.append(collect_run_metrics(runner, log_prefix="[Finetune][Summary]"))

    summarize_runs(run_metrics, seeds, log_prefix="[Finetune][Summary]", excluded_metrics=_SUMMARY_EXCLUDED_METRICS)
    return 0
