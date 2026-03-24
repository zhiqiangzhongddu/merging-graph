"""Runtime orchestration for the `run_pretrain.py` entrypoint."""

from __future__ import annotations

from .pretrainer import PretrainRunner
from .utils import run_pretrain_tasks


def run_pretrain(cfg) -> int:
    """Execute one pretraining workflow for the provided config."""
    if getattr(cfg.pretrain, "run_all", False):
        return run_pretrain_tasks(cfg)

    run_cfg = cfg.clone()
    run_cfg.seed = int(getattr(cfg, "seed", 0))
    runner = PretrainRunner(run_cfg)
    runner.fit()
    return 0
