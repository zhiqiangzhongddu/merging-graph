"""Runtime orchestration for the `run_pretrain.py` entrypoint."""

from __future__ import annotations

from typing import Iterable, List

from code.config import cfg as base_cfg, update_cfg
from code.pretrain.pretrainer import PretrainRunner
from code.pretrain.utils import run_pretrain_tasks

def _normalize_dataset_aliases(argv: List[str]) -> List[str]:
    """Support `dataset.*` CLI aliases by remapping to `pretrain.dataset.*`."""
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
        key = f"pretrain.{token}" if token.startswith("dataset.") else token
        normalized.extend([key, value])
        idx += 2
    return normalized


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
            "Please specify at least dataset.name/task_level (or pretrain.dataset.*)."
        )
        return 1

    normalized_argv = _normalize_dataset_aliases(raw_argv)
    cfg = update_cfg(base_cfg, " ".join(normalized_argv))

    if not getattr(cfg.pretrain, "run_all", False):
        explicit_name = any(
            token == "pretrain.dataset.name" and idx + 1 < len(normalized_argv)
            for idx, token in enumerate(normalized_argv)
        )
        default_name = str(base_cfg.pretrain.dataset.name).strip().lower()
        configured_name = str(cfg.pretrain.dataset.name).strip().lower()
        if (not explicit_name) and configured_name == default_name:
            print(
                "[Pretrain] Missing dataset override (pretrain.dataset.name). "
                "Refusing to fallback to default dataset."
            )
            return 1

    return run_pretrain(cfg)
