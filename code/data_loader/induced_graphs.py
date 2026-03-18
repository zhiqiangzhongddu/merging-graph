from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph

from code.utils import ensure_dir

from .dataset_paths import _dataset_scoped_dir, _sanitize_name
from .utils import safe_torch_load


def _induced_cache_path(
    dataset_name: str,
    task_level: str,
    cache_root_path: Path,
    suffix: str,
):
    sanitized = _sanitize_name(dataset_name)
    dataset_dir = _dataset_scoped_dir(cache_root_path, sanitized)
    return dataset_dir / f"{sanitized}_induced_{task_level}_{suffix}.pt"


def _load_induced_cache(path: Path, expected_meta: Dict) -> Dict | None:
    if not path.is_file():
        return None
    try:
        payload = safe_torch_load(path)
        meta = payload.get("meta", {})
        for key, value in expected_meta.items():
            if meta.get(key) != value:
                return None
        return payload
    except Exception:
        return None


def _save_induced_cache(path: Path, payload: Dict):
    ensure_dir(path.parent)
    torch.save(payload, path)


class SingleGraphDataLoader:
    """Lightweight wrapper so node-level datasets match the DataLoader interface."""

    def __init__(self, data: Data):
        self.data = data

    def __iter__(self):
        yield self.data

    def __len__(self):
        return 1


def _build_reverse_csr(edge_index: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a compact inbound adjacency index for bounded node-induced expansion."""
    if edge_index is None or edge_index.numel() == 0 or num_nodes <= 0:
        return (
            torch.zeros(max(num_nodes, 0) + 1, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )

    reverse_edge_index = edge_index[[1, 0]].detach().cpu()
    values = torch.ones(reverse_edge_index.size(1), dtype=torch.uint8)
    reverse_adj = (
        torch.sparse_coo_tensor(
            reverse_edge_index,
            values,
            size=(num_nodes, num_nodes),
            device="cpu",
        )
        .coalesce()
        .to_sparse_csr()
    )
    return reverse_adj.crow_indices(), reverse_adj.col_indices()


def _sample_bounded_k_hop_subset(
    center_idx: int,
    reverse_crow: torch.Tensor,
    reverse_cols: torch.Tensor,
    smallest_size: int,
    largest_size: int | None,
    max_hops: int,
    start_hops: int,
) -> torch.Tensor:
    """Sample a bounded inbound k-hop neighborhood without materializing large frontiers."""
    min_size = max(1, int(smallest_size))
    max_size = int(largest_size) if largest_size is not None and int(largest_size) > 0 else None
    if max_size is not None:
        min_size = min(min_size, max_size)

    min_hops = max(0, int(start_hops))
    hop_limit = max(min_hops, int(max_hops))
    selected = [int(center_idx)]
    frontier = [int(center_idx)]
    visited = {int(center_idx)}
    hop = 0

    while frontier and hop < hop_limit:
        hop += 1
        if max_size is not None and len(selected) >= max_size:
            break

        remaining_slots = (max_size - len(selected)) if max_size is not None else max(min_size - len(selected), 1)
        if remaining_slots <= 0:
            break

        per_frontier_budget = max(1, (remaining_slots + len(frontier) - 1) // max(len(frontier), 1))
        sample_budget = max(per_frontier_budget * 2, min(remaining_slots, 32))
        next_frontier: List[int] = []

        for node_id in frontier:
            start = int(reverse_crow[node_id].item())
            end = int(reverse_crow[node_id + 1].item())
            degree = end - start
            if degree <= 0:
                continue

            neighbors = reverse_cols[start:end]
            if degree > sample_budget:
                choice = torch.randperm(degree)[:sample_budget]
                neighbors = neighbors[choice]

            for neighbor in neighbors.tolist():
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                next_frontier.append(int(neighbor))
                if len(selected) + len(next_frontier) >= (max_size or (len(selected) + remaining_slots)):
                    break
            if len(selected) + len(next_frontier) >= (max_size or (len(selected) + remaining_slots)):
                break

        if not next_frontier:
            break

        if max_size is not None and len(selected) + len(next_frontier) > max_size:
            next_frontier = next_frontier[: max_size - len(selected)]
        selected.extend(next_frontier)
        frontier = next_frontier

        if hop >= min_hops and len(selected) >= min_size:
            break

    return torch.tensor(selected, dtype=torch.long)


def build_induced_graphs(
    data: Data,
    smallest_size: int = 10,
    largest_size: int = 30,
    max_hops: int = 5,
    start_hops: int = 2,
):
    """Generate induced subgraphs centered at each node."""
    induced_graphs = []
    if data is None:
        return induced_graphs

    edge_index = getattr(data, "edge_index", None)
    num_nodes = getattr(data, "num_nodes", None)
    if num_nodes is None and getattr(data, "x", None) is not None:
        num_nodes = data.x.size(0)
    if num_nodes is None or num_nodes <= 0:
        return induced_graphs

    labels = getattr(data, "y", None)
    if labels is None:
        return induced_graphs
    labels = labels.view(-1)
    device = data.x.device if getattr(data, "x", None) is not None else torch.device("cpu")
    has_edges = edge_index is not None and edge_index.numel() > 0
    reverse_crow = reverse_cols = None
    use_bounded_expansion = has_edges and largest_size is not None and largest_size > 0
    if use_bounded_expansion:
        reverse_crow, reverse_cols = _build_reverse_csr(edge_index, int(num_nodes))

    for idx in range(num_nodes):
        label = int(labels[idx].item())
        if label < 0:
            continue
        hops = start_hops
        subset = torch.tensor([idx], device=device)

        if has_edges:
            if use_bounded_expansion and reverse_crow is not None and reverse_cols is not None:
                subset = _sample_bounded_k_hop_subset(
                    center_idx=idx,
                    reverse_crow=reverse_crow,
                    reverse_cols=reverse_cols,
                    smallest_size=smallest_size,
                    largest_size=largest_size,
                    max_hops=max_hops,
                    start_hops=start_hops,
                )
            else:
                subset, _, _, _ = k_hop_subgraph(
                    node_idx=idx,
                    num_hops=hops,
                    edge_index=edge_index,
                    relabel_nodes=True,
                    num_nodes=num_nodes,
                )
                while subset.numel() < smallest_size and hops < max_hops:
                    hops += 1
                    subset, _, _, _ = k_hop_subgraph(
                        node_idx=idx,
                        num_hops=hops,
                        edge_index=edge_index,
                        relabel_nodes=True,
                        num_nodes=num_nodes,
                    )

        subset_cpu = subset.cpu()
        if subset_cpu.numel() > largest_size:
            keep = subset_cpu[torch.randperm(subset_cpu.numel())[: largest_size - 1]]
            subset_cpu = torch.unique(torch.cat([torch.tensor([idx], dtype=torch.long), keep]))

        subset = subset_cpu.to(device)
        if has_edges:
            sub_edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True, num_nodes=num_nodes)
        else:
            sub_edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        x = data.x[subset] if getattr(data, "x", None) is not None else None

        graph = Data(x=x, edge_index=sub_edge_index, y=torch.tensor(label))
        graph.base_node_id = idx
        graph.index = idx
        induced_graphs.append(graph)
    return induced_graphs


def _edge_induced_subgraph(
    data: Data,
    endpoints: torch.Tensor,
    max_hops: int,
    max_size: int | None = None,
    label: int | None = None,
):
    """Build a single edge-centered induced subgraph given endpoints."""
    if endpoints.numel() != 2:
        raise ValueError("endpoints must contain exactly two node indices.")

    device = data.edge_index.device
    x = getattr(data, "x", None)
    num_nodes = getattr(data, "num_nodes", None)
    if num_nodes is None and x is not None:
        num_nodes = x.size(0)
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx=endpoints.to(device=device),
        edge_index=data.edge_index,
        num_hops=max_hops,
        relabel_nodes=True,
        num_nodes=num_nodes,
    )
    if max_size is not None and max_size > 0 and subset.numel() > max_size:
        target = max(max_size, 2)
        keep = torch.randperm(subset.numel(), device=subset.device)[: target - 2]
        keep = torch.unique(torch.cat([keep, torch.tensor([mapping[0], mapping[1]], device=subset.device)]))
        subset = subset[keep]
        sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
        map_u = torch.where(subset == endpoints[0])[0][0]
        map_v = torch.where(subset == endpoints[1])[0][0]
        mapping = torch.stack([map_u, map_v])
    sub_x = x[subset] if x is not None else None
    mapped_u, mapped_v = int(mapping[0]), int(mapping[1])
    target_edge = torch.tensor([[mapped_u], [mapped_v]], dtype=torch.long, device=device)
    graph = Data(
        x=sub_x,
        edge_index=sub_edge_index,
        edge_label_index=target_edge,
    )
    if label is not None:
        graph.y = torch.tensor(label, dtype=torch.long)
    return graph


def build_edge_induced_graphs(
    data: Data,
    edge_indices: torch.Tensor,
    max_hops: int = 2,
    max_size: int | None = None,
):
    """Generate edge-centered induced subgraphs (one per target edge)."""
    if data is None or edge_indices.numel() == 0:
        return []

    edge_pairs = data.edge_index[:, edge_indices.to(dtype=torch.long)]
    graphs = []

    for idx in range(edge_pairs.size(1)):
        endpoints = edge_pairs[:, idx]
        graphs.append(_edge_induced_subgraph(data, endpoints=endpoints, max_hops=max_hops, max_size=max_size))
    return graphs


def build_edge_induced_graphs_supervised(
    data: Data,
    pos_edge_pairs: torch.Tensor,
    neg_edge_pairs: torch.Tensor,
    max_hops: int = 2,
    max_size: int | None = None,
):
    """Generate edge-centered induced subgraphs with binary labels."""
    if data is None:
        return []

    graphs = []
    if pos_edge_pairs is not None and pos_edge_pairs.numel() > 0:
        for idx in range(pos_edge_pairs.size(1)):
            endpoints = pos_edge_pairs[:, idx]
            graphs.append(_edge_induced_subgraph(data, endpoints=endpoints, max_hops=max_hops, max_size=max_size, label=1))

    if neg_edge_pairs is not None and neg_edge_pairs.numel() > 0:
        for idx in range(neg_edge_pairs.size(1)):
            endpoints = neg_edge_pairs[:, idx]
            graphs.append(_edge_induced_subgraph(data, endpoints=endpoints, max_hops=max_hops, max_size=max_size, label=0))
    return graphs


class InducedGraphDataset(Dataset):
    """Dataset of induced subgraphs (one per original node)."""

    def __init__(
        self,
        graphs,
        base_num_nodes: int = None,
        base_num_edges: int = None,
        base_info: Dict = None,
        split_tags=None,
    ):
        self.graphs = graphs
        self.num_features = graphs[0].num_node_features if graphs else 0
        labels_list = []
        for graph in graphs:
            y = getattr(graph, "y", None)
            if y is not None:
                labels_list.append(y.view(-1)[0])
        labels = torch.stack(labels_list) if labels_list else torch.tensor([])
        self.num_classes = int(labels.max().item() + 1) if labels.numel() > 0 else None
        self.base_num_nodes = base_num_nodes
        self.base_num_edges = base_num_edges
        self.base_dataset_info = base_info or {}
        self.split_tags = split_tags or ["train"] * len(graphs)
        if self.base_dataset_info.get("name"):
            self.name = f"Induced({self.base_dataset_info['name']})"

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx]
        if not hasattr(data, "base_node_id"):
            data.base_node_id = idx
        if self.split_tags:
            data.split = self.split_tags[idx]
        return data
