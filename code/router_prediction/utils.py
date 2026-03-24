"""Shared utility helpers for router prediction and selection logging."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json


@dataclass
class RoutedExpert:
    expert_id: int
    score: Optional[float] = None
    expert_model: Dict[str, Any] | None = None


@dataclass
class RoutedQuery:
    query_id: int
    dataset: Optional[str]
    experts: List[RoutedExpert]


def load_router_output(path: str) -> Tuple[List[RoutedQuery], Dict[str, Any]]:
    if not path:
        raise FileNotFoundError("router output path is empty")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        predictions = payload.get("predictions", [])
        meta = payload
    else:
        predictions = payload
        meta = {}
    routed: List[RoutedQuery] = []
    for item in predictions or []:
        query_id = item.get("target_subgraph_id")
        if query_id is None:
            query_id = item.get("query_id")
        if query_id is None:
            continue
        dataset = item.get("dataset")
        experts: List[RoutedExpert] = []
        for expert in item.get("experts", []):
            expert_id = expert.get("expert_id")
            if expert_id is None:
                expert_id = expert.get("id")
            if expert_id is None:
                continue
            experts.append(
                RoutedExpert(
                    expert_id=int(expert_id),
                    score=expert.get("score"),
                    expert_model=expert.get("expert_model") or {},
                )
            )
        routed.append(RoutedQuery(query_id=int(query_id), dataset=dataset, experts=experts))
    return routed, meta


def load_weak_labels(path: str) -> Dict[int, float]:
    if not path:
        return {}
    path_obj = Path(path)
    if not path_obj.exists():
        return {}
    if path_obj.suffix.lower() in {".json", ".jsonl"}:
        payload = json.loads(path_obj.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return {int(k): float(v) for k, v in payload.items()}
        labels: Dict[int, float] = {}
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            qid = entry.get("query_id") or entry.get("id")
            label = entry.get("label")
            if qid is None or label is None:
                continue
            labels[int(qid)] = float(label)
        return labels
    labels: Dict[int, float] = {}
    for line in path_obj.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [p for p in stripped.replace(",", " ").split() if p]
        if len(parts) < 2:
            continue
        labels[int(parts[0])] = float(parts[1])
    return labels


def save_json(path: str, payload: Dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def graph_feature_dim(tensor) -> int:
    if tensor is None or not hasattr(tensor, "dim"):
        return 0
    if tensor.dim() == 2:
        return int(tensor.size(1))
    if tensor.dim() == 1:
        return 1
    return 0


def graph_edge_counts(graph_obj) -> dict:
    counts = {}
    if isinstance(graph_obj, dict):
        for key, value in graph_obj.items():
            if not isinstance(key, tuple):
                continue
            edge_index = value.get("edge_index") if isinstance(value, dict) else None
            counts[str(key)] = int(edge_index.size(1)) if edge_index is not None else 0
    else:
        for edge_type in graph_obj.edge_types:
            store = graph_obj[edge_type]
            edge_index = getattr(store, "edge_index", None)
            counts[str(edge_type)] = int(edge_index.size(1)) if edge_index is not None else 0
    return counts


def graph_feature_info(graph_obj) -> dict:
    node_dims = {}
    edge_dims = {}
    if isinstance(graph_obj, dict):
        for node_type, feats in graph_obj.get("node_features", {}).items():
            node_dims[str(node_type)] = graph_feature_dim(feats)
        for key, value in graph_obj.items():
            if not isinstance(key, tuple):
                continue
            edge_weight = value.get("edge_weight") if isinstance(value, dict) else None
            edge_dims[str(key)] = graph_feature_dim(edge_weight)
    else:
        for node_type in graph_obj.node_types:
            feats = getattr(graph_obj[node_type], "x", None)
            node_dims[str(node_type)] = graph_feature_dim(feats)
        for edge_type in graph_obj.edge_types:
            store = graph_obj[edge_type]
            edge_weight = getattr(store, "edge_weight", None)
            edge_dims[str(edge_type)] = graph_feature_dim(edge_weight)
    return {"node_feature_dims": node_dims, "edge_feature_dims": edge_dims}


def log_graph_info(label: str, graph_data, expected_node_dims: dict | None = None) -> None:
    node_counts = {
        "application": graph_data.num_applications,
        "target_subgraph": graph_data.num_subgraphs,
        "expert": graph_data.num_experts,
    }
    edge_counts = graph_edge_counts(graph_data.graph)
    feature_info = graph_feature_info(graph_data.graph)
    print(f"[RouterSelect]{label} Node counts: {node_counts}")
    print(f"[RouterSelect]{label} Edge counts: {edge_counts}")
    print(f"[RouterSelect]{label} Node feature dims: {feature_info['node_feature_dims']}")
    print(f"[RouterSelect]{label} Edge feature dims: {feature_info['edge_feature_dims']}")
    if expected_node_dims:
        print(f"[RouterSelect]{label} Router node feature dims: {expected_node_dims}")
