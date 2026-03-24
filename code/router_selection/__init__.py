"""GraphRouter-style router expert selection."""
from .data import RouterGraphData, load_router_graph, build_router_splits
from .trainer import RouterSelectionTrainer

__all__ = [
    "RouterGraphData",
    "load_router_graph",
    "build_router_splits",
    "RouterSelectionTrainer",
]
