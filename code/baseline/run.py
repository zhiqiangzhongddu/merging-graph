"""Runtime orchestration for the `run_baseline.py` entrypoint."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List

from code.config import cfg as base_cfg, update_cfg
from code.utils import set_seed

from .anygraph import run_anygraph_baseline

RunnerFn = Callable[[object], int]
_BASELINE_RUNNERS: Dict[str, RunnerFn] = {
    "anygraph": run_anygraph_baseline,
}
_ANYGRAPH_STAGE_PREFIXES = (
    "baseline.anygraph.",
    "execution.",
    "paths.",
    "conversion.",
    "train.",
    "eval.",
    "prediction.",
    "output.",
)


def _normalize_baseline_aliases(argv: List[str]) -> List[str]:
    """Support stage-scoped CLI aliases in run_baseline.py."""
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
        key = token
        if token == "method":
            key = "baseline.method"
        elif token.startswith("execution."):
            key = f"baseline.anygraph.{token}"
        elif token.startswith("paths."):
            key = f"baseline.anygraph.{token}"
        elif token.startswith("conversion."):
            key = f"baseline.anygraph.{token}"
        elif token.startswith("train."):
            key = f"baseline.anygraph.{token}"
        elif token.startswith("eval."):
            key = f"baseline.anygraph.{token}"
        elif token.startswith("prediction."):
            key = f"baseline.anygraph.{token}"
        elif token.startswith("output."):
            key = f"baseline.anygraph.{token}"
        normalized.extend([key, value])
        idx += 2
    return normalized


def _build_baseline_cfg(argv: Iterable[str]):
    raw_argv = list(argv)
    if not raw_argv:
        print(
            "[Baseline] No CLI overrides provided. "
            "Refusing to run with default baseline config. "
            "Please specify at least baseline.method and stage-specific options."
        )
        return None

    normalized_argv = _normalize_baseline_aliases(raw_argv)
    has_method_override = any(
        token == "baseline.method" and idx + 1 < len(normalized_argv)
        for idx, token in enumerate(normalized_argv)
    )
    should_infer_anygraph_method = any(token.startswith(_ANYGRAPH_STAGE_PREFIXES) for token in normalized_argv)
    if not has_method_override and should_infer_anygraph_method:
        normalized_argv = ["baseline.method", "anygraph", *normalized_argv]
    return update_cfg(base_cfg, " ".join(normalized_argv))


def _resolve_method(cfg) -> str:
    baseline_cfg = getattr(cfg, "baseline", None)
    if baseline_cfg is None:
        return ""

    method = str(getattr(baseline_cfg, "method", "") or "").strip()
    if method:
        return method.lower()
    return ""


def run_baseline(cfg) -> int:
    method = _resolve_method(cfg)
    if not method:
        print("[Baseline] Missing baseline method. Set `baseline.method`.")
        return 1

    runner = _BASELINE_RUNNERS.get(method)
    if runner is None:
        available = ", ".join(sorted(_BASELINE_RUNNERS.keys()))
        print(f"[Baseline] Unsupported method '{method}'. Available methods: {available}")
        return 1

    set_seed(int(getattr(cfg, "seed", 0)))
    try:
        return int(runner(cfg))
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[Baseline][{method}] Failed: {exc}")
        return 1


def run_baseline_from_cli(argv: Iterable[str]) -> int:
    cfg = _build_baseline_cfg(argv)
    if cfg is None:
        return 1
    return run_baseline(cfg)
