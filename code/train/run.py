"""Train entrypoint helpers and CLI wiring."""

from __future__ import annotations

from typing import Iterable

from code.config import cfg as base_cfg, update_cfg

from .runtime import run_train as _run_train


def build_train_cfg(argv: Iterable[str]):
    """Parse train CLI overrides into a config object."""
    return update_cfg(base_cfg, " ".join(argv))


def run_train(cfg) -> int:
    """Execute one or more training runs for the provided config."""
    return _run_train(cfg)


def run_train_from_cli(argv: Iterable[str]) -> int:
    """Parse CLI overrides and execute the training runtime."""
    cfg = build_train_cfg(argv)
    return run_train(cfg)
