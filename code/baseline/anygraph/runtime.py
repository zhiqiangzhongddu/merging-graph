"""Shared AnyGraph runtime paths and environment helpers."""

from __future__ import annotations

import os
from pathlib import Path

from code.utils import PROJECT_ROOT, project_path


REPO_ROOT = PROJECT_ROOT
ANYGRAPH_PACKAGE_ROOT = project_path("code", "baseline", "anygraph")
ANYGRAPH_SRC_ROOT = ANYGRAPH_PACKAGE_ROOT / "src"
ANYGRAPH_NODE_SRC_ROOT = ANYGRAPH_SRC_ROOT / "node_classification"
ANYGRAPH_LINK_MAIN = ANYGRAPH_SRC_ROOT / "main.py"
ANYGRAPH_NODE_MAIN = ANYGRAPH_NODE_SRC_ROOT / "main.py"
ANYGRAPH_RUNTIME_ROOT = project_path("trained_models", "anygraph")
ANYGRAPH_MODELS_DIR = ANYGRAPH_RUNTIME_ROOT / "Models"
ANYGRAPH_HISTORY_DIR = ANYGRAPH_RUNTIME_ROOT / "History"


def ensure_anygraph_runtime_files_exist(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "AnyGraph runtime sources are missing: " + ", ".join(missing)
        )


def build_anygraph_runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env["ANYGRAPH_RUNTIME_ROOT"] = str(ANYGRAPH_RUNTIME_ROOT)
    return env
