"""Runtime orchestration for the `run_context_generation.py` entrypoint."""

from __future__ import annotations

import warnings
from typing import Iterable

from code.config import cfg as base_cfg, update_cfg
from code.utils import set_seed

from .condensation_runner import (
    run_bonsai_condensation_from_cfg,
    run_bonsai_eval_from_cfg,
    run_sgdc_condensation_from_cfg,
    run_sgdc_pretrain_from_cfg,
)
from .llm_runner import run_llm_dataset_from_cfg, run_llm_model_from_cfg


def _run_llm(cfg) -> int:
    llm_cfg = getattr(getattr(cfg, "context", None), "llm", None)
    target = str(getattr(llm_cfg, "target", "all") if llm_cfg is not None else "all").strip().lower()

    if target == "dataset":
        return run_llm_dataset_from_cfg(cfg)
    if target == "model":
        return run_llm_model_from_cfg(cfg)
    if target != "all":
        print(f"[Context generation] Unknown context.llm.target='{target}'. Expected dataset/model/all.")
        return 1

    statuses = [run_llm_dataset_from_cfg(cfg), run_llm_model_from_cfg(cfg)]
    return 0 if all(status == 0 for status in statuses) else 1


def _run_bonsai(cfg) -> int:
    bonsai_cfg = getattr(getattr(cfg, "context", None), "bonsai", None)
    stage = str(getattr(bonsai_cfg, "stage", "condense") if bonsai_cfg is not None else "condense").strip().lower()

    if stage == "condense":
        return run_bonsai_condensation_from_cfg(cfg)
    if stage == "eval":
        return run_bonsai_eval_from_cfg(cfg)
    if stage == "all":
        statuses = [run_bonsai_condensation_from_cfg(cfg), run_bonsai_eval_from_cfg(cfg)]
        return 0 if all(status == 0 for status in statuses) else 1

    print(f"[Context generation] Unknown context.bonsai.stage='{stage}'. Expected condense/eval/all.")
    return 1


def _run_sgdc(cfg) -> int:
    sgdc_cfg = getattr(getattr(cfg, "context", None), "sgdc", None)
    stage = str(getattr(sgdc_cfg, "stage", "condense") if sgdc_cfg is not None else "condense").strip().lower()

    if stage == "pretrain":
        return run_sgdc_pretrain_from_cfg(cfg)
    if stage == "condense":
        return run_sgdc_condensation_from_cfg(cfg)
    if stage == "all":
        statuses = [run_sgdc_pretrain_from_cfg(cfg), run_sgdc_condensation_from_cfg(cfg)]
        return 0 if all(status == 0 for status in statuses) else 1

    print(f"[Context generation] Unknown context.sgdc.stage='{stage}'. Expected pretrain/condense/all.")
    return 1


def run_context_generation(cfg) -> int:
    set_seed(int(getattr(cfg, "seed", 0)))

    context_cfg = getattr(cfg, "context", None)
    method = str(getattr(context_cfg, "method", "llm") if context_cfg is not None else "llm").strip().lower()

    dispatch = {
        "llm": _run_llm,
        "bonsai": _run_bonsai,
        "sgdc": _run_sgdc,
    }

    runner = dispatch.get(method)
    if runner is None:
        supported = ", ".join(sorted(dispatch.keys()))
        print(f"[Context generation] Unknown context.method='{method}'. Supported: {supported}")
        return 1
    return int(runner(cfg))


def run_context_generation_from_cli(argv: Iterable[str]) -> int:
    warnings.filterwarnings("ignore")
    cfg = update_cfg(base_cfg, " ".join(list(argv)))
    return run_context_generation(cfg)
