import torch
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool


POOL = {
    "mean": global_mean_pool,
    "add": global_add_pool,
    "max": global_max_pool,
}


def get_batch_vector(data) -> torch.Tensor:
    """Return batch vector; defaults to zeros for single-graph data."""

    if getattr(data, "batch", None) is not None:
        return data.batch
    return torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)


def pool_nodes(
    x: torch.Tensor,
    batch: torch.Tensor,
    mode: str = "mean",
    data=None,
) -> torch.Tensor:
    """
    Pool node embeddings into graph embeddings.

    Args:
        x: Node embeddings [num_nodes, dim]
        batch: Batch assignment [num_nodes]
        mode: Pooling mode - "mean", "add", "max", or "target"
        data: Optional data object with ptr and target_node for "target" mode

    For "target" mode (used in induced graphs for node classification):
        Extracts the embedding of the target node in each induced graph.
        Requires data.ptr (graph boundaries) and data.target_node (target indices).
    """
    if mode == "target":
        if data is None:
            raise ValueError("target pooling requires data object with ptr and target_node")
        ptr = getattr(data, "ptr", None)
        target_node = getattr(data, "target_node", None)
        if ptr is None or target_node is None:
            # Fallback to mean pooling if target info not available
            return global_mean_pool(x, batch)
        # Extract target node embedding for each graph
        # ptr[:-1] gives start index of each graph, target_node gives relative position
        target_indices = ptr[:-1] + target_node
        return x[target_indices]

    mode_key = str(mode).lower()
    if mode_key == "sum":
        mode_key = "add"
    return POOL.get(mode_key, global_mean_pool)(x, batch)
