from typing import Dict, Type

from .base import PretrainTask

REGISTRY: Dict[str, Type[PretrainTask]] = {}


def register(name: str):
    """Register a pretraining task class."""
    def decorator(cls: Type[PretrainTask]):
        """Decorator to register the class."""
        REGISTRY[name.lower()] = cls
        cls.name = name.lower()
        return cls

    return decorator


def build_pretrain_task(name: str, cfg) -> PretrainTask:
    """Build a pretraining task from the registry."""
    key = name.lower()
    if key not in REGISTRY:
        raise ValueError(f"Unknown pretraining method: {name}")
    return REGISTRY[key](cfg)
