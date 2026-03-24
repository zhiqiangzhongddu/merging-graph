"""Finetune entrypoint helpers and CLI wiring."""

from __future__ import annotations

import warnings
from typing import Iterable

from code.config import cfg as base_cfg, update_cfg

from .runtime import run_finetune as _run_finetune
from .utils import extract_few_shot


def build_finetune_cfg(argv: Iterable[str]):
    """
    Parse CLI overrides for finetuning.

    `--fewshot` overrides `finetune.dataset.fixed_split`.
    """
    raw_argv = list(argv)
    forwarded_argv, few_shot_split = extract_few_shot(raw_argv)
    cfg = update_cfg(base_cfg, " ".join(forwarded_argv))

    if few_shot_split is not None:
        cfg.finetune.dataset.fixed_split = few_shot_split
    return cfg


def run_finetune(cfg) -> int:
    """Execute one or more finetuning runs for the provided config."""
    return _run_finetune(cfg)


def run_finetune_from_cli(argv: Iterable[str]) -> int:
    """Parse CLI overrides and execute the finetuning runtime."""
    warnings.filterwarnings("ignore")
    cfg = build_finetune_cfg(argv)
    return run_finetune(cfg)
