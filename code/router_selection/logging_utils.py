"""Logging helpers for router selection."""
from __future__ import annotations


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


def expected_text_dim(model_name: str) -> int:
    name = (model_name or "").lower()
    if "bert-large" in name:
        return 1024
    if "bert-base" in name:
        return 768
    if "roberta-large" in name:
        return 1024
    if "roberta-base" in name:
        return 768
    return 768


def expected_router_node_dims(cfg) -> dict:
    text_dim = expected_text_dim(getattr(cfg.router_selection, "text_embed_model", "bert-base-uncased"))
    feat_dim = int(getattr(cfg.dataset, "subgraph_svd_feat_dim", 0))
    struct_dim = int(getattr(cfg.dataset, "subgraph_svd_struct_dim", 0))
    sub_dim = feat_dim + struct_dim
    return {
        "application": text_dim,
        "target_subgraph": sub_dim,
        "expert": text_dim,
    }
