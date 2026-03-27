"""Runtime orchestration for the `run_train.py` entrypoint."""

from __future__ import annotations

from .utils import run_train_workflow


def run_train(cfg) -> int:
    """Execute one or more training runs for the provided config."""
    return run_train_workflow(cfg)
