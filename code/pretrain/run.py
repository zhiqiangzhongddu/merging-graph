"""Runtime orchestration for the `run_pretrain.py` entrypoint."""

from __future__ import annotations

from typing import Iterable, List

from code.config import cfg as base_cfg, update_cfg
from code.pretrain.pretrainer import PretrainRunner
from code.pretrain.utils import run_pretrain_tasks

def _has_dataset_override(argv: List[str]) -> bool:
    """Return True when pretrain dataset name is explicitly provided in CLI args."""
    for idx, token in enumerate(argv):
        if token == "pretrain.dataset.name":
            return idx + 1 < len(argv)
    return False


def run_pretrain(cfg) -> int:
    if getattr(cfg.pretrain, "run_all", False):
        return run_pretrain_tasks(cfg)

    run_cfg = cfg.clone()
    run_cfg.seed = int(getattr(cfg, "seed", 0))
    runner = PretrainRunner(run_cfg)
    runner.fit()
    return 0


def run_pretrain_from_cli(argv: Iterable[str]) -> int:
    raw_argv = list(argv)
    if not raw_argv:
        print(
            "[Pretrain] No CLI overrides provided. "
            "Refusing to run with default dataset/model. "
            "Please specify at least pretrain.dataset.name/task_level."
        )
        return 1

    cfg = update_cfg(base_cfg, " ".join(raw_argv))

    if not getattr(cfg.pretrain, "run_all", False):
        explicit_name = _has_dataset_override(raw_argv)
        default_name = str(base_cfg.pretrain.dataset.name).strip().lower()
        configured_name = str(cfg.pretrain.dataset.name).strip().lower()
        if (not explicit_name) and configured_name == default_name:
            print(
                "[Pretrain] Missing dataset override (pretrain.dataset.name). "
                "Refusing to fallback to default dataset."
            )
            return 1

    return run_pretrain(cfg)
