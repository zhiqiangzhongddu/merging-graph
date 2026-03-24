"""SGDC package for graph-level context generation."""

from .condensation import run_sgdc_condensation_dataset
from .pretrain import run_sgdc_pretrain_dataset

__all__ = [
    "run_sgdc_condensation_dataset",
    "run_sgdc_pretrain_dataset",
]
