from typing import Dict, Type

from .task_base import FinetuneTask

REGISTRY: Dict[str, Type[FinetuneTask]] = {}
_METHODS_IMPORTED = False


def register(name: str):
    def decorator(cls: Type[FinetuneTask]):
        REGISTRY[name.lower()] = cls
        cls.name = name.lower()
        return cls
    return decorator


def build_finetune_task(name: str, cfg) -> FinetuneTask:
    global _METHODS_IMPORTED
    if not _METHODS_IMPORTED:
        # Import method modules lazily so decorators can populate REGISTRY.
        from . import methods as _methods  # noqa: F401

        _METHODS_IMPORTED = True

    key = name.lower()
    if key not in REGISTRY:
        raise ValueError(f"Unknown finetune method: {name}")
    return REGISTRY[key](cfg)
