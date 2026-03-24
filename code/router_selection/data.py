"""Data loading and split logic for router selection."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import torch


@dataclass
class RouterGraphData:
    graph: object
    application_id_map: Dict[str, Dict]
    subgraph_id_map: Dict[str, Dict]
    expert_id_map: Dict[str, Dict]
    app_to_sub_edge_index: torch.Tensor
    sub_to_expert_edge_index: torch.Tensor
    sub_to_expert_weight: torch.Tensor
    num_applications: int
    num_subgraphs: int
    num_experts: int
    subgraph_application_ids: List[int]
    subgraph_expert_ids: List[torch.Tensor]
    subgraph_expert_weights: List[torch.Tensor]


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_edge_store(graph: object, src: str, rel: str, dst: str):
    if isinstance(graph, dict):
        return graph[(src, rel, dst)]
    return graph[(src, rel, dst)]


def _get_num_nodes(graph: object, node_type: str) -> int:
    if isinstance(graph, dict):
        return int(graph.get("node_counts", {}).get(node_type, 0))
    return int(graph[node_type].num_nodes)


def _build_subgraph_application_map(
    edge_index: torch.Tensor,
    num_subgraphs: int,
) -> Tuple[List[int], int]:
    """Map each target_subgraph to a parent application id from app->sub edges."""
    subgraph_application_ids = [-1 for _ in range(num_subgraphs)]
    multi_parent = 0
    app_ids = edge_index[0].tolist() if edge_index.numel() else []
    sub_ids = edge_index[1].tolist() if edge_index.numel() else []
    for app_id, sub_id in zip(app_ids, sub_ids):
        current = subgraph_application_ids[sub_id]
        if current == -1:
            subgraph_application_ids[sub_id] = int(app_id)
        elif current != int(app_id):
            multi_parent += 1
    return subgraph_application_ids, multi_parent


def _build_subgraph_expert_lists(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_subgraphs: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Group subgraph->expert edges into per-subgraph lists without dense matrices."""
    if edge_index.numel() == 0:
        return [torch.empty(0, dtype=torch.long) for _ in range(num_subgraphs)], [
            torch.empty(0, dtype=torch.float)
            for _ in range(num_subgraphs)
        ]
    sub_ids = edge_index[0]
    order = torch.argsort(sub_ids)
    sub_sorted = sub_ids[order]
    expert_sorted = edge_index[1][order]
    weight_sorted = edge_weight[order]
    counts = torch.bincount(sub_sorted, minlength=num_subgraphs)
    subgraph_expert_ids: List[torch.Tensor] = []
    subgraph_expert_weights: List[torch.Tensor] = []
    offset = 0
    for count in counts.tolist():
        if count == 0:
            subgraph_expert_ids.append(torch.empty(0, dtype=torch.long))
            subgraph_expert_weights.append(torch.empty(0, dtype=torch.float))
            continue
        end = offset + count
        subgraph_expert_ids.append(expert_sorted[offset:end])
        subgraph_expert_weights.append(weight_sorted[offset:end])
        offset = end
    return subgraph_expert_ids, subgraph_expert_weights


def load_router_graph(graph_dir: str) -> RouterGraphData:
    """Load the router graph tensors and mappings from a saved graph directory."""
    graph_dir_path = Path(graph_dir)
    graph_path = graph_dir_path / "router_graph.pt"
    if not graph_path.exists():
        raise FileNotFoundError(f"Router graph not found at {graph_path}")

    graph = torch.load(graph_path, map_location="cpu")
    application_id_map = _load_json(graph_dir_path / "application_id_map.json")
    subgraph_id_map = _load_json(graph_dir_path / "subgraph_id_map.json")
    expert_id_map = _load_json(graph_dir_path / "expert_id_map.json")

    app_sub_store = _get_edge_store(graph, "application", "to", "target_subgraph")
    sub_expert_store = _get_edge_store(graph, "target_subgraph", "to", "expert")

    app_to_sub_edge_index = app_sub_store["edge_index"] if isinstance(app_sub_store, dict) else app_sub_store.edge_index
    sub_to_expert_edge_index = (
        sub_expert_store["edge_index"] if isinstance(sub_expert_store, dict) else sub_expert_store.edge_index
    )
    sub_to_expert_weight = (
        sub_expert_store["edge_weight"] if isinstance(sub_expert_store, dict) else sub_expert_store.edge_weight
    )

    num_applications = _get_num_nodes(graph, "application")
    num_subgraphs = _get_num_nodes(graph, "target_subgraph")
    num_experts = _get_num_nodes(graph, "expert")

    subgraph_application_ids, multi_parent = _build_subgraph_application_map(
        app_to_sub_edge_index,
        num_subgraphs,
    )
    if multi_parent > 0:
        print(f"[RouterSelect] Warning: {multi_parent} subgraphs map to multiple applications; using first seen.")

    subgraph_expert_ids, subgraph_expert_weights = _build_subgraph_expert_lists(
        sub_to_expert_edge_index,
        sub_to_expert_weight,
        num_subgraphs,
    )

    return RouterGraphData(
        graph=graph,
        application_id_map=application_id_map,
        subgraph_id_map=subgraph_id_map,
        expert_id_map=expert_id_map,
        app_to_sub_edge_index=app_to_sub_edge_index,
        sub_to_expert_edge_index=sub_to_expert_edge_index,
        sub_to_expert_weight=sub_to_expert_weight,
        num_applications=num_applications,
        num_subgraphs=num_subgraphs,
        num_experts=num_experts,
        subgraph_application_ids=subgraph_application_ids,
        subgraph_expert_ids=subgraph_expert_ids,
        subgraph_expert_weights=subgraph_expert_weights,
    )


def _dataset_from_application_meta(meta: Dict) -> Optional[str]:
    if not isinstance(meta, dict):
        return None
    dataset = meta.get("dataset")
    if dataset:
        return dataset
    key = meta.get("key") or {}
    if isinstance(key, dict):
        return key.get("dataset")
    return None


def _split_train_edges_for_validation(
    graph_data: RouterGraphData,
    train_subgraphs: List[int],
    valid_edge_ratio: float,
    seed: int,
) -> Tuple[
    Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    List[int],
]:
    """Split train subgraphs into train/valid sets and assign all edges by split."""
    train_edges: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    valid_edges: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    valid_edge_ratio = max(0.0, min(1.0, float(valid_edge_ratio)))
    if valid_edge_ratio <= 0 or not train_subgraphs:
        for sub_id in train_subgraphs:
            expert_ids = graph_data.subgraph_expert_ids[sub_id]
            weights = graph_data.subgraph_expert_weights[sub_id]
            if expert_ids.numel() == 0:
                continue
            train_edges[sub_id] = (expert_ids, weights)
        return train_edges, valid_edges, []

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(len(train_subgraphs), generator=generator)
    val_count = int(len(train_subgraphs) * valid_edge_ratio)
    if val_count <= 0:
        val_count = 1 if len(train_subgraphs) > 1 else 0
    val_ids = [int(train_subgraphs[i]) for i in perm[:val_count].tolist()]
    val_id_set = set(val_ids)

    for sub_id in train_subgraphs:
        expert_ids = graph_data.subgraph_expert_ids[sub_id]
        weights = graph_data.subgraph_expert_weights[sub_id]
        if expert_ids.numel() == 0:
            continue
        if sub_id in val_id_set:
            valid_edges[sub_id] = (expert_ids, weights)
        else:
            train_edges[sub_id] = (expert_ids, weights)

    return train_edges, valid_edges, val_ids


def build_router_splits(
    graph_data: RouterGraphData,
    train_datasets: List[str],
    test_datasets: List[str],
    valid_edge_ratio: float,
    seed: int,
) -> Tuple[Dict[str, List[int]], Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]]:
    """
    Build train/val/test splits based on application datasets.

    Train/test splits follow dataset membership, while validation edges are
    sampled from train subgraph->expert edges using valid_edge_ratio.
    """
    train_set = set(train_datasets)
    test_set = set(test_datasets)

    app_split: Dict[int, str] = {}
    ignored_apps = 0
    fallback_to_train = True
    for app_id_str, meta in graph_data.application_id_map.items():
        dataset = _dataset_from_application_meta(meta) or ""
        app_id = int(app_id_str)
        if dataset in train_set:
            app_split[app_id] = "train"
        elif dataset in test_set:
            app_split[app_id] = "test"
        else:
            if fallback_to_train:
                app_split[app_id] = "train"
            else:
                app_split[app_id] = "ignore"
                ignored_apps += 1

    splits = {"train": [], "valid": [], "test": []}
    ignored_subgraphs = 0
    for sub_id, app_id in enumerate(graph_data.subgraph_application_ids):
        if app_id == -1:
            ignored_subgraphs += 1
            continue
        split = app_split.get(app_id, "ignore")
        if split == "ignore":
            ignored_subgraphs += 1
            continue
        splits[split].append(sub_id)

    train_edge_map, valid_edge_map, valid_ids = _split_train_edges_for_validation(
        graph_data,
        splits["train"],
        valid_edge_ratio,
        seed,
    )
    if valid_edge_ratio > 0 and valid_ids:
        valid_set = set(valid_ids)
        splits["valid"] = sorted(valid_set)
        splits["train"] = sorted(sub_id for sub_id in splits["train"] if sub_id not in valid_set)
    else:
        splits["valid"] = []

    print(
        "[RouterSelect] Split counts: "
        f"train={len(splits['train'])} valid={len(splits['valid'])} test={len(splits['test'])} "
        f"ignored_apps={ignored_apps} ignored_subgraphs={ignored_subgraphs}"
    )
    edge_splits = {"train": train_edge_map, "valid": valid_edge_map}
    return splits, edge_splits
