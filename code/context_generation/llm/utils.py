"""Utility helpers shared by LLM context generation modules."""

import json
import os
from typing import Dict, List


def load_json_if_exists(path: str) -> Dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"[Context generation] Skipped missing file: {path}")
    return {}


def list_architecture_paths(root: str) -> List[str]:
    """Collect all *_architecture.json files under a root directory."""
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith("_architecture.json"):
                paths.append(os.path.join(dirpath, fname))
    return sorted(paths)
