"""Train entrypoint helpers and CLI wiring."""

from __future__ import annotations

from typing import Iterable

from code.config import cfg as base_cfg, update_cfg
from code.utils.save_results import extract_explicit_cfg_keys, set_explicit_cfg_keys

from .runtime import run_train as _run_train


def build_train_cfg(argv: Iterable[str]):
    """Parse train CLI overrides into a config object."""
    raw_argv = list(argv)
    cfg = update_cfg(base_cfg, " ".join(raw_argv))
    set_explicit_cfg_keys(cfg, extract_explicit_cfg_keys(raw_argv, flag_arity={"--config": 1}))
    return cfg


def run_train(cfg) -> int:
    """Execute one or more training runs for the provided config."""
    return _run_train(cfg)


def run_train_from_cli(argv: Iterable[str]) -> int:
    """Parse CLI overrides and execute the training runtime."""
    cfg = build_train_cfg(argv)
    return run_train(cfg)
