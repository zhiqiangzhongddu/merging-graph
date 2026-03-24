"""Pretrain entrypoint helpers and CLI wiring."""

from __future__ import annotations

from typing import Iterable

from code.config import cfg as base_cfg, update_cfg

from .runtime import run_pretrain as _run_pretrain


def build_pretrain_cfg(argv: Iterable[str]):
    """Parse pretrain CLI overrides and validate required dataset inputs."""
    raw_argv = list(argv)
    if not raw_argv:
        raise ValueError(
            "[Pretrain] No CLI overrides provided. "
            "Refusing to run with default dataset/model. "
            "Please specify at least pretrain.dataset.name/task_level."
        )

    cfg = update_cfg(base_cfg, " ".join(raw_argv))

    if not getattr(cfg.pretrain, "run_all", False):
        explicit_name = any(
            token == "pretrain.dataset.name" and idx + 1 < len(raw_argv)
            for idx, token in enumerate(raw_argv)
        )
        default_name = str(base_cfg.pretrain.dataset.name).strip().lower()
        configured_name = str(cfg.pretrain.dataset.name).strip().lower()
        if (not explicit_name) and configured_name == default_name:
            raise ValueError(
                "[Pretrain] Missing dataset override (pretrain.dataset.name). "
                "Refusing to use the default dataset."
            )

    return cfg


def run_pretrain(cfg) -> int:
    """Execute one pretraining workflow for the provided config."""
    return _run_pretrain(cfg)


def run_pretrain_from_cli(argv: Iterable[str]) -> int:
    """Parse CLI overrides and execute the pretraining runtime."""
    try:
        cfg = build_pretrain_cfg(argv)
    except ValueError as exc:
        print(str(exc))
        return 1
    return run_pretrain(cfg)
