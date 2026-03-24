"""Runtime orchestration for the `run_train.py` entrypoint."""

from __future__ import annotations

from typing import List

from code.utils.run_helpers import collect_run_metrics, resolve_run_seeds, summarize_runs

from .trainer import TrainRunner


def run_train(cfg) -> int:
    """Execute one or more training runs for the provided config."""
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

    run_metrics: List[dict[str, float]] = []
    total_runs = len(seeds)
    for index, seed in enumerate(seeds, start=1):
        run_cfg = cfg.clone()
        run_cfg.seed = int(seed)
        if total_runs > 1:
            print(f"[Train][Multi-run] Running {index}/{total_runs} with seed={seed}")

        runner = TrainRunner(cfg=run_cfg)
        runner.fit()
        run_metrics.append(collect_run_metrics(runner, log_prefix="[Train][Summary]"))

    summarize_runs(run_metrics, seeds, log_prefix="[Train][Summary]", excluded_metrics={"batch_size"})
    return 0
