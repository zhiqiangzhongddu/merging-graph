"""Runtime bridge for LLM-based context generation workflows."""

from .llm.runner import run_llm_dataset_from_cfg, run_llm_model_from_cfg

__all__ = [
    "run_llm_dataset_from_cfg",
    "run_llm_model_from_cfg",
]
