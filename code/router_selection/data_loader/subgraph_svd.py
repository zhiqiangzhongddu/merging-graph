"""Subgraph SVD features for router selection."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os

import torch
from torch_geometric.data import Data

from code.data_loader import create_dataset
from code.router_selection.utils import dataset_from_application_meta
from code.router_selection.data import RouterGraphData


def _normalize_key(value: object) -> str:
    return str(value).strip()


def _svd_singular_values(matrix: torch.Tensor, target_dim: int) -> torch.Tensor:
    if target_dim <= 0:
        return torch.empty((0,), dtype=torch.float32)
    if matrix.numel() == 0:
        return torch.zeros(target_dim, dtype=torch.float32)
    try:
        _, s, _ = torch.linalg.svd(matrix.float(), full_matrices=False)
        vec = s[:target_dim]
    except Exception:
        flat = matrix.flatten()
        vec = flat[:target_dim].float() if flat.numel() else torch.zeros(target_dim, dtype=torch.float32)
    if vec.numel() < target_dim:
        pad = torch.zeros(target_dim - vec.numel(), dtype=torch.float32)
        vec = torch.cat([vec, pad], dim=0)
    return vec.to(torch.float32)


def _feature_svd(data: Data, dim: int) -> torch.Tensor:
    x = getattr(data, "x", None)
    if x is None or x.numel() == 0:
        return torch.zeros(dim, dtype=torch.float32)
    return _svd_singular_values(x, dim)


def _structure_matrix(data: Data, matrix_type: str) -> torch.Tensor:
    num_nodes = int(getattr(data, "num_nodes", 0) or 0)
    if num_nodes <= 0:
        return torch.empty((0, 0), dtype=torch.float32)
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj = (adj + adj.t()).clamp(max=1.0)
    matrix_type = str(matrix_type or "adjacency").lower()
    if matrix_type == "laplacian":
        deg = torch.diag(adj.sum(dim=1))
        return deg - adj
    return adj


def _structure_svd(data: Data, dim: int, matrix_type: str) -> torch.Tensor:
    mat = _structure_matrix(data, matrix_type)
    return _svd_singular_values(mat, dim)


def _load_cache(path: Path) -> Optional[Dict[str, torch.Tensor]]:
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu")
        return payload.get("vectors")
    except Exception:
        return None


def _save_cache(path: Path, vectors: Dict[str, torch.Tensor], meta: Dict[str, object]) -> None:
    os.makedirs(path.parent, exist_ok=True)
    try:
        torch.save({"vectors": vectors, "meta": meta}, path)
    except Exception:
        pass


def _dataset_task_meta(cfg, graph_data: RouterGraphData) -> Dict[str, Tuple[str, bool]]:
    mapping: Dict[str, Tuple[str, bool]] = {}
    for meta in graph_data.application_id_map.values():
        dataset = dataset_from_application_meta(meta)
        if not dataset:
            continue
        task_meta = meta.get("task") or {}
        task_level = str(task_meta.get("raw_task_level") or task_meta.get("task_level") or cfg.dataset.task_level or "node")
        induced = bool(task_meta.get("induced")) if "induced" in task_meta else bool(cfg.dataset.induced)
        if induced and task_level == "graph":
            task_level = "node"
        mapping[dataset] = (task_level, induced)
    return mapping


def _build_dataset_subgraph_index(graph_data: RouterGraphData) -> Dict[str, List[Tuple[int, Dict]]]:
    by_dataset: Dict[str, List[Tuple[int, Dict]]] = {}
    for sub_id_str, meta in graph_data.subgraph_id_map.items():
        dataset = meta.get("dataset")
        if not dataset:
            continue
        by_dataset.setdefault(dataset, []).append((int(sub_id_str), meta))
    return by_dataset


def _subgraph_key(meta: Dict) -> str:
    node_id = meta.get("node_id")
    if node_id is not None:
        return _normalize_key(node_id)
    hash_key = meta.get("hash_key")
    if hash_key:
        return _normalize_key(hash_key)
    key = meta.get("key")
    return _normalize_key(key or "")


def _cache_path_from_cfg(cfg, dataset: str, task_level: str) -> Path:
    ds_cfg = cfg.dataset
    feat_dim = int(getattr(ds_cfg, "subgraph_svd_feat_dim", 0))
    struct_dim = int(getattr(ds_cfg, "subgraph_svd_struct_dim", 0))
    matrix_type = str(getattr(ds_cfg, "subgraph_svd_matrix", "adjacency")).lower()
    base = Path(getattr(ds_cfg, "subgraph_svd_dir", "data/subgraph_svd"))
    parts = [f"subgraph_svd_{dataset}_{task_level}"]
    if feat_dim and feat_dim > 0:
        parts.append(f"feat{feat_dim}")
    parts.append(f"struct{struct_dim}")
    parts.append(matrix_type)
    return base / f"{'_'.join(parts)}.pt"


def build_subgraph_svd_features(cfg, graph_data: RouterGraphData) -> torch.Tensor:
    """Load per-subgraph SVD features (node features + structure) from cache."""
    ds_cfg = cfg.dataset
    feat_dim = int(getattr(ds_cfg, "subgraph_svd_feat_dim", 0))
    struct_dim = int(getattr(ds_cfg, "subgraph_svd_struct_dim", 0))
    matrix_type = str(getattr(ds_cfg, "subgraph_svd_matrix", "adjacency")).lower()

    dataset_tasks = _dataset_task_meta(cfg, graph_data)
    by_dataset = _build_dataset_subgraph_index(graph_data)
    total_dim = feat_dim + struct_dim
    features = torch.zeros((graph_data.num_subgraphs, total_dim), dtype=torch.float32)
    if not by_dataset:
        return features

    for dataset, entries in by_dataset.items():
        task_level, induced = dataset_tasks.get(dataset, (cfg.dataset.task_level, cfg.dataset.induced))
        cache_path = _cache_path_from_cfg(cfg, dataset, task_level)
        payload = None
        if cache_path.exists():
            try:
                payload = torch.load(cache_path, map_location="cpu")
            except Exception:
                payload = None
        feat_svd = payload.get("feat_svd") if isinstance(payload, dict) else None
        struct_svd = payload.get("struct_svd") if isinstance(payload, dict) else None
        if payload is None and (feat_dim > 0 or struct_dim > 0):
            print(
                "[RouterSelect] Warning: missing subgraph SVD cache for "
                f"{dataset}:{task_level} at {cache_path}; using fallback features."
            )

        for sub_id, meta in entries:
            node_id = meta.get("node_id")
            vec = None
            if node_id is not None:
                idx = int(node_id)
                if feat_svd is not None and idx < feat_svd.size(0):
                    feat_vec = feat_svd[idx] if feat_dim > 0 else torch.empty((0,), dtype=torch.float32)
                else:
                    feat_vec = torch.zeros(feat_dim, dtype=torch.float32) if feat_dim > 0 else torch.empty((0,))
                if struct_svd is not None and idx < struct_svd.size(0):
                    struct_vec = struct_svd[idx] if struct_dim > 0 else torch.empty((0,), dtype=torch.float32)
                else:
                    struct_vec = torch.zeros(struct_dim, dtype=torch.float32) if struct_dim > 0 else torch.empty((0,))
                vec = torch.cat([feat_vec, struct_vec], dim=0)
            if vec is None:
                vec = torch.zeros(total_dim, dtype=torch.float32)
                if total_dim > 0:
                    num_nodes = float(meta.get("num_nodes") or 0.0)
                    vec[0] = num_nodes
                if total_dim > 1:
                    num_edges = float(meta.get("num_edges") or 0.0)
                    vec[1] = num_edges
            features[sub_id] = vec
    return features
