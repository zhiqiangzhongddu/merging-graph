"""
Run Bonsai-based graph condensation inside the context-generation pipeline.

This module is self-contained under `code/context_generation` and does not rely
on the removed `code/condensation` package.

Supported condensation modes:
- `budget_mode=bonsai`: official-style Bonsai size-proxy condensation.
- `budget_mode=node_frac`: node-fraction budget extension.
- `budget_mb>0` or `budget_mode=bytes`: byte-budgeted extension.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import torch
from scipy.sparse import coo_array
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import add_self_loops, from_networkx, subgraph
from torch_sparse import SparseTensor, matmul

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from code.data_loader.datasets import create_dataset, _get_or_create_split_indices  # type: ignore  # noqa: E402
from code.utils import (
    append_csv_row,
    ensure_dir,
    project_path,
    read_name_list_file,
    resolve_project_path,
    set_seed,
)  # noqa: E402


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_OFFICIAL_UTILS = None
_OFFICIAL_WL = None


def _official_utils():
    global _OFFICIAL_UTILS
    if _OFFICIAL_UTILS is None:
        _OFFICIAL_UTILS = _load_module("_bonsai_official_utils", project_path("baselines", "Bonsai", "utils.py"))
    return _OFFICIAL_UTILS


def _official_wl():
    global _OFFICIAL_WL
    if _OFFICIAL_WL is None:
        _OFFICIAL_WL = _load_module("_bonsai_official_wl", project_path("baselines", "Bonsai", "WL_Distance2.py"))
    return _OFFICIAL_WL


def save_condensed_as_pyg_dataset(condensed: Data, *, out_root: str, meta: Optional[Dict] = None) -> None:
    raw_dir = os.path.join(out_root, "raw")
    proc_dir = os.path.join(out_root, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    raw_path = os.path.join(raw_dir, "data.pt")
    raw_fd, raw_tmp = tempfile.mkstemp(prefix="data.pt.tmp.", dir=raw_dir)
    os.close(raw_fd)
    try:
        torch.save([condensed], raw_tmp)
        os.replace(raw_tmp, raw_path)
    finally:
        if os.path.exists(raw_tmp):
            os.remove(raw_tmp)

    data_obj, slices = InMemoryDataset.collate([condensed])
    proc_path = os.path.join(proc_dir, "data.pt")
    proc_fd, proc_tmp = tempfile.mkstemp(prefix="data.pt.tmp.", dir=proc_dir)
    os.close(proc_fd)
    try:
        torch.save((data_obj, slices), proc_tmp)
        os.replace(proc_tmp, proc_path)
    finally:
        if os.path.exists(proc_tmp):
            os.remove(proc_tmp)

    if meta is not None:
        meta_path = os.path.join(out_root, "meta.pt")
        meta_fd, meta_tmp = tempfile.mkstemp(prefix="meta.pt.tmp.", dir=out_root)
        os.close(meta_fd)
        try:
            torch.save(meta, meta_tmp)
            os.replace(meta_tmp, meta_path)
        finally:
            if os.path.exists(meta_tmp):
                os.remove(meta_tmp)


def _size_proxy(nnodes: int, nedges: int, feats: int, dtype: str = "int") -> int:
    mx = 1 if dtype == "int" else 2 if dtype == "float" else None
    if mx is None:
        raise ValueError(f"Unsupported dtype={dtype}")
    return (nnodes * feats * mx + nedges * 2) * 2


def _dataset_size_dtype(dataset_name: str, data: Data) -> str:
    name = str(dataset_name).strip().lower()
    if name in {"cora", "citeseer", "flickr"}:
        return "int"
    if name in {"ogbn-arxiv", "reddit", "pubmed"}:
        return "float"
    return "float" if torch.as_tensor(data.x).dtype.is_floating_point else "int"


def _bonsai_split_indices(num_nodes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_all, test = train_test_split(range(num_nodes), test_size=0.2, random_state=42)
    rng = np.random.RandomState(seed=0)
    train = rng.choice(train_all, size=int(0.7 * len(train_all)), replace=False)
    val = list(set(range(num_nodes)) - set(train).union(set(test)))
    return np.asarray(train), np.asarray(val), np.asarray(test)


def _resolve_train_indices(
    *,
    dataset_name: str,
    dataset,
    data: Data,
    split_mode: str,
    split_root: Path,
    split_tuple: Tuple[float, float, float],
    mask_col: int,
    seed: int,
) -> torch.Tensor:
    mode = str(split_mode).strip().lower()
    num_nodes = int(getattr(data, "num_nodes", None) or data.x.size(0))
    if mode == "bonsai":
        train, _, _ = _bonsai_split_indices(num_nodes)
        return torch.as_tensor(train, dtype=torch.long)
    if mode == "pretrain":
        train_i, _, _ = _get_or_create_split_indices(
            dataset_name=dataset_name,
            split=split_tuple,  # type: ignore[arg-type]
            seed=int(seed),
            split_root_path=split_root,
            total=num_nodes,
        )
        return torch.as_tensor(train_i, dtype=torch.long)

    train_idx = None
    if hasattr(dataset, "get_idx_split"):
        split = dataset.get_idx_split()
        if isinstance(split, dict) and "train" in split:
            train_idx = split["train"]
    if train_idx is None:
        train_mask = getattr(data, "train_mask", None)
        if train_mask is not None:
            t = torch.as_tensor(train_mask)
            if t.dtype == torch.bool:
                if t.dim() == 2 and t.size(0) == int(data.num_nodes):
                    t = t[:, int(mask_col)]
                if t.numel() == int(data.num_nodes) and int(t.sum().item()) > 0:
                    train_idx = t.nonzero(as_tuple=False).view(-1)
    if train_idx is None:
        train_i, _, _ = _get_or_create_split_indices(
            dataset_name=dataset_name,
            split=split_tuple,  # type: ignore[arg-type]
            seed=int(seed),
            split_root_path=split_root,
            total=num_nodes,
        )
        train_idx = torch.as_tensor(train_i, dtype=torch.long)
    return torch.as_tensor(train_idx, dtype=torch.long)


def _build_neighborhood_dict_sparse(adj) -> Dict[int, set[int]]:
    neighbors: Dict[int, set[int]] = {}
    for node in range(adj.shape[0]):
        node_nbrs = set(adj[[node]].tocoo().col.tolist())
        node_nbrs.add(int(node))
        neighbors[int(node)] = node_nbrs
    return neighbors


def _select_max_coverage(rknn_dict: Dict[int, Set[int]]) -> List[int]:
    utils = _official_utils()
    return utils.select_max_coverage_rknn_celf(dict(rknn_dict))


def _dist2rknn_sorting(wl_dist: np.ndarray, sampled_nodes: np.ndarray, k: int) -> List[int]:
    utils = _official_utils()
    rknn_result = utils.wl2rknn(wl_dist, sampled_nodes=sampled_nodes, k=int(k))
    return _select_max_coverage(dict(rknn_result["rknn"]))


def _prepare_feat_len(dataset_name: str, x: torch.Tensor):
    dataset_name_raw = str(dataset_name).strip().lower()
    nfeats = int(x.shape[1])
    if dataset_name_raw not in {"ogbn-arxiv", "reddit", "flickr", "pubmed"}:
        feat_multiplier = 1
        feat_len = [float(x[idx].sum().item()) for idx in range(x.shape[0])]
    elif dataset_name_raw in {"pubmed", "flickr"}:
        feat_multiplier = 2 if dataset_name_raw == "flickr" else 3
        feat_len = [int(torch.where(x[idx] == 0, 0, 1).sum().item()) for idx in range(x.shape[0])]
    else:
        feat_multiplier = 2
        feat_len = defaultdict(lambda: nfeats)
    return feat_len, feat_multiplier


def _rknn_sorted2budget_select_merged(
    *,
    sorted_nodes: Sequence[int],
    train: Sequence[int],
    target_size: float,
    neighbors: Dict[int, set[int]],
    feat_len,
    feat_multiplier: int,
) -> List[int]:
    size_selected_nodes: List[int] = []
    selected_nodes: set[int] = set()
    selected_edges: set[Tuple[int, int]] = set()
    ints_till_now_due_to_nodes = 0
    not_selected_for_n_consecutive_iters = 0

    for node in sorted_nodes:
        candidate_selected_nodes: set[int] = set()
        candidate_selected_edges: set[Tuple[int, int]] = set()
        train_node = int(train[int(node)])

        if train_node not in selected_nodes:
            candidate_selected_nodes.add(train_node)
        node_adj_set = neighbors[train_node]
        for nbr_node in node_adj_set:
            if nbr_node not in selected_nodes:
                candidate_selected_nodes.add(nbr_node)
            edge = tuple(sorted((int(nbr_node), train_node)))
            if edge not in selected_edges:
                candidate_selected_edges.add(edge)
            nbr_adj_set = neighbors[int(nbr_node)]
            ego_nodes = node_adj_set.intersection(nbr_adj_set)
            for nbr_nbr_node in ego_nodes:
                edge = tuple(sorted((int(nbr_node), int(nbr_nbr_node))))
                if edge not in selected_edges:
                    candidate_selected_edges.add(edge)

        num_edges = len(selected_edges) + len(candidate_selected_edges)
        ints_due_to_nodes = 0
        for added_node in candidate_selected_nodes:
            ints_due_to_nodes += int(feat_len[added_node])
        ints_due_to_nodes *= int(feat_multiplier)
        ints_due_to_nodes += ints_till_now_due_to_nodes
        ints_due_to_edges = num_edges * 2
        total_ints = ints_due_to_nodes + ints_due_to_edges
        size_till_now = total_ints * 2
        if size_till_now < float(target_size):
            size_selected_nodes.append(int(node))
            selected_nodes.update(candidate_selected_nodes)
            selected_edges.update(candidate_selected_edges)
            ints_till_now_due_to_nodes = ints_due_to_nodes
        else:
            not_selected_for_n_consecutive_iters += 1
            if not_selected_for_n_consecutive_iters > 100:
                break
    return size_selected_nodes


def _match_distribution(
    *,
    merged_graph: nx.Graph,
    data: Data,
    train: Sequence[int],
    ranked_nodes: Sequence[int],
) -> nx.Graph:
    num_trainable = sum(1 for _, attrs in merged_graph.nodes(data=True) if bool(attrs["target"]))
    labels_train = data.y[torch.as_tensor(list(train), dtype=torch.long)]
    class_counts = torch.bincount(labels_train).float()
    total_train = len(train)
    class_distribution_scaled = ((class_counts / total_train) * num_trainable).long()

    ranked_orig_nodes = [int(train[int(node)]) for node in ranked_nodes]
    class_dict = {idx: [] for idx in range(len(class_distribution_scaled))}
    for node in ranked_orig_nodes:
        class_dict[int(data.y[node].item())].append(node)

    all_nodes_to_add: set[int] = set()
    for class_label, scaled_count in enumerate(class_distribution_scaled):
        merged_class_nodes = [
            node
            for node, attrs in merged_graph.nodes(data=True)
            if int(data.y[node].item()) == class_label and bool(attrs["target"])
        ]
        lower_bound = 0.99 * scaled_count.item()
        upper_bound = 1.01 * scaled_count.item()
        current_count = len(merged_class_nodes)

        if current_count < lower_bound:
            cnt = 0
            for node in class_dict[class_label]:
                if node not in merged_graph:
                    merged_graph.add_node(node)
                    all_nodes_to_add.add(node)
                    cnt += 1
                if cnt >= lower_bound - current_count:
                    break
        elif current_count > upper_bound:

            def sort_key(node_id: int) -> int:
                try:
                    return ranked_orig_nodes.index(node_id)
                except ValueError:
                    return 10**12

            merged_class_nodes_sorted = sorted(merged_class_nodes, key=sort_key)
            for node in merged_class_nodes_sorted[-(current_count - int(upper_bound)) :]:
                if node in merged_graph:
                    merged_graph.remove_node(node)

    def _p1(u: int, v: int) -> bool:
        return u in all_nodes_to_add and v in all_nodes_to_add

    def _p2(u: int, v: int) -> bool:
        return u in all_nodes_to_add and v in merged_graph.nodes

    def _p3(u: int, v: int) -> bool:
        return u in merged_graph.nodes and v in all_nodes_to_add

    for idx in range(data.edge_index.shape[1]):
        u = int(data.edge_index[0, idx].item())
        v = int(data.edge_index[1, idx].item())
        if _p1(u, v) or _p2(u, v) or _p3(u, v):
            if not merged_graph.has_edge(u, v):
                merged_graph.add_edge(u, v)
    return merged_graph


def _merged_nodes_from_seed_positions(
    *,
    seed_positions: Sequence[int],
    train: Sequence[int],
    neighbors: Dict[int, set[int]],
) -> set[int]:
    merged_nodes: set[int] = set()
    for orig_idx in seed_positions:
        train_node = int(train[int(orig_idx)])
        merged_nodes.update(neighbors[train_node])
        merged_nodes.add(train_node)
    return merged_nodes


def _candidate_merge_additions(
    *,
    train_node: int,
    neighbors: Dict[int, set[int]],
    selected_nodes: set[int],
    selected_edges: set[Tuple[int, int]],
) -> Tuple[set[int], set[Tuple[int, int]]]:
    candidate_selected_nodes: set[int] = set()
    candidate_selected_edges: set[Tuple[int, int]] = set()

    if train_node not in selected_nodes:
        candidate_selected_nodes.add(train_node)
    node_adj_set = neighbors[train_node]
    for nbr_node in node_adj_set:
        if nbr_node not in selected_nodes:
            candidate_selected_nodes.add(nbr_node)
        edge = tuple(sorted((int(nbr_node), train_node)))
        if edge not in selected_edges:
            candidate_selected_edges.add(edge)
        nbr_adj_set = neighbors[int(nbr_node)]
        ego_nodes = node_adj_set.intersection(nbr_adj_set)
        for nbr_nbr_node in ego_nodes:
            edge = tuple(sorted((int(nbr_node), int(nbr_nbr_node))))
            if edge not in selected_edges:
                candidate_selected_edges.add(edge)
    return candidate_selected_nodes, candidate_selected_edges


def _prepare_bonsai_inputs(
    *,
    dataset_name: str,
    data: Data,
    train_idx: torch.Tensor,
    k: int,
    max_train_nodes: Optional[int] = None,
    seed: int = 0,
) -> Dict:
    if max_train_nodes is not None and int(train_idx.numel()) > int(max_train_nodes):
        g = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(int(train_idx.numel()), generator=g)
        train_idx = train_idx[perm[: int(max_train_nodes)]]

    nnodes = int(getattr(data, "num_nodes", None) or data.x.size(0))
    nedges = int(data.edge_index.shape[1])
    nfeats = int(data.x.shape[1])
    row = data.edge_index[0].cpu().numpy()
    col = data.edge_index[1].cpu().numpy()
    weights = np.ones(len(row), dtype=np.float32)
    adj = coo_array((weights, (row, col)), shape=(nnodes, nnodes)).tocsr()

    train = train_idx.cpu().numpy()
    wl_repr = _official_wl().compute_wl_representations(data.x, adj)
    wl_repr = wl_repr[train]

    wl_repr_input = wl_repr
    wl_repr_filtered, features_used = _official_utils().transform_features_with_tree(data, wl_repr_input, train)
    features_used = [int(x) for x in features_used]
    if not features_used:
        features_used = list(range(int(data.x.shape[1])))
        wl_repr_filtered = [row for row in wl_repr_input]

    data = data.clone()
    data.x = data.x[:, features_used]
    feat_len, feat_multiplier = _prepare_feat_len(dataset_name, data.x)
    neighbors = _build_neighborhood_dict_sparse(adj)

    wl_repr_np = np.stack(
        [np.asarray(torch.as_tensor(row).detach().cpu().numpy(), dtype=np.float32) for row in wl_repr_filtered],
        axis=0,
    )
    wl_dist = pairwise_distances(wl_repr_np, n_jobs=20)
    sampled_nodes = np.arange(len(wl_repr_np))
    sorted_nodes = _dist2rknn_sorting(wl_dist, sampled_nodes, int(k))

    return {
        "data": data,
        "adj": adj,
        "train": train,
        "sorted_nodes": sorted_nodes,
        "neighbors": neighbors,
        "feat_len": feat_len,
        "feat_multiplier": feat_multiplier,
        "features_used": features_used,
        "num_nodes_orig": nnodes,
        "num_edges_orig": nedges,
        "num_feats_orig": nfeats,
        "size_full": _size_proxy(nnodes, nedges, nfeats, _dataset_size_dtype(dataset_name, data)),
        "bonsai_dtype": _dataset_size_dtype(dataset_name, data),
    }


def _select_seed_positions_for_budget(
    *,
    sorted_nodes: Sequence[int],
    train: Sequence[int],
    neighbors: Dict[int, set[int]],
    budget_mode: str,
    target_size: Optional[float] = None,
    target_nodes: Optional[int] = None,
    target_bytes: Optional[int] = None,
    feat_len=None,
    feat_multiplier: int = 1,
    feat_dim: Optional[int] = None,
    feat_bytes: Optional[int] = None,
    index_bytes: Optional[int] = None,
) -> Tuple[List[int], set[int], set[Tuple[int, int]]]:
    seed_positions: List[int] = []
    selected_nodes: set[int] = set()
    selected_edges: set[Tuple[int, int]] = set()
    ints_till_now_due_to_nodes = 0
    rejected = 0

    for node in sorted_nodes:
        train_node = int(train[int(node)])
        candidate_selected_nodes, candidate_selected_edges = _candidate_merge_additions(
            train_node=train_node,
            neighbors=neighbors,
            selected_nodes=selected_nodes,
            selected_edges=selected_edges,
        )

        accept = False
        if budget_mode == "bonsai":
            if target_size is None:
                raise ValueError("target_size is required for budget_mode='bonsai'.")
            num_edges = len(selected_edges) + len(candidate_selected_edges)
            ints_due_to_nodes = 0
            for added_node in candidate_selected_nodes:
                ints_due_to_nodes += int(feat_len[added_node])
            ints_due_to_nodes *= int(feat_multiplier)
            ints_due_to_nodes += ints_till_now_due_to_nodes
            ints_due_to_edges = num_edges * 2
            size_till_now = (ints_due_to_nodes + ints_due_to_edges) * 2
            accept = size_till_now < float(target_size)
        elif budget_mode == "node_frac":
            if target_nodes is None:
                raise ValueError("target_nodes is required for budget_mode='node_frac'.")
            accept = (len(selected_nodes) + len(candidate_selected_nodes)) <= int(target_nodes)
        elif budget_mode == "bytes":
            if target_bytes is None or feat_dim is None or feat_bytes is None or index_bytes is None:
                raise ValueError("byte-budget selection requires target_bytes/feat_dim/feat_bytes/index_bytes.")
            accept = _estimate_bytes(
                num_nodes=len(selected_nodes) + len(candidate_selected_nodes),
                num_edges=len(selected_edges) + len(candidate_selected_edges),
                feat_dim=int(feat_dim),
                feat_bytes=int(feat_bytes),
                index_bytes=int(index_bytes),
            ) <= int(target_bytes)
        else:
            raise ValueError(f"Unsupported budget_mode={budget_mode}.")

        if accept:
            seed_positions.append(int(node))
            selected_nodes.update(candidate_selected_nodes)
            selected_edges.update(candidate_selected_edges)
            if budget_mode == "bonsai":
                ints_till_now_due_to_nodes = ints_due_to_nodes
            rejected = 0
        else:
            rejected += 1
            if rejected > 100:
                break

    return seed_positions, selected_nodes, selected_edges


def _build_merged_graph_from_seed_positions(
    *,
    seed_positions: Sequence[int],
    train: Sequence[int],
    neighbors: Dict[int, set[int]],
    adj,
) -> nx.Graph:
    merged_graph = nx.Graph()
    for orig_idx in seed_positions:
        train_node = int(train[int(orig_idx)])
        node_neighbors = neighbors[train_node]
        merged_graph.add_node(train_node)
        row_center = adj[[train_node]]
        for nbr in node_neighbors:
            merged_graph.add_node(int(nbr))
            row_nbr = adj[[int(nbr)]]
            ego = row_center.multiply(row_nbr)
            for nbrnbr in ego.tocoo().col:
                merged_graph.add_edge(*sorted((int(nbr), int(nbrnbr))))
            merged_graph.add_edge(*sorted((train_node, int(nbr))))
    return merged_graph


def _annotate_merged_graph(merged_graph: nx.Graph, *, data: Data, train: Sequence[int]) -> None:
    train_set = set(int(x) for x in train)
    for node in merged_graph.nodes:
        merged_graph.nodes[node]["target"] = node in train_set
        merged_graph.nodes[node]["x"] = data.x[node].cpu().numpy().tolist()
        merged_graph.nodes[node]["y"] = data.y[node]


def _to_condensed_data(merged_graph: nx.Graph) -> Data:
    node_order = list(merged_graph.nodes())
    condensed = from_networkx(merged_graph, group_node_attrs=["x"])
    condensed.orig_nid = torch.as_tensor(node_order, dtype=torch.long)
    if getattr(condensed, "target", None) is not None:
        condensed.target = condensed.target.to(dtype=torch.bool)
    if getattr(condensed, "y", None) is not None:
        condensed.y = torch.as_tensor(condensed.y)
        if condensed.y.dim() > 1 and condensed.y.size(-1) == 1:
            condensed.y = condensed.y.view(-1)
    return condensed


def _condense_bonsai_ranked(
    *,
    dataset_name: str,
    data: Data,
    train_idx: torch.Tensor,
    target_size_frac: float,
    k: int,
    budget_mode: str,
    min_target_nodes: int = 0,
    budget_bytes: Optional[int] = None,
    max_train_nodes: Optional[int] = None,
    seed: int = 0,
) -> Tuple[Data, Dict]:
    prep = _prepare_bonsai_inputs(
        dataset_name=dataset_name,
        data=data,
        train_idx=train_idx,
        k=k,
        max_train_nodes=max_train_nodes,
        seed=seed,
    )
    data = prep["data"]
    adj = prep["adj"]
    train = prep["train"]
    sorted_nodes = prep["sorted_nodes"]
    neighbors = prep["neighbors"]
    feat_len = prep["feat_len"]
    feat_multiplier = prep["feat_multiplier"]
    features_used = prep["features_used"]
    nnodes = prep["num_nodes_orig"]
    nedges = prep["num_edges_orig"]
    size_full = prep["size_full"]
    bonsai_dtype = prep["bonsai_dtype"]

    if len(sorted_nodes) == 0:
        raise ValueError("No ranked train nodes available for Bonsai condensation.")

    m = 0.9
    upscale = 1 + m / (1 - m)

    if budget_mode == "bonsai":
        target_size = float(f"{float(target_size_frac) * size_full:.2f}")
        base_seed_positions, base_nodes, _ = _select_seed_positions_for_budget(
            sorted_nodes=sorted_nodes,
            train=train,
            neighbors=neighbors,
            budget_mode="bonsai",
            target_size=target_size,
            feat_len=feat_len,
            feat_multiplier=feat_multiplier,
        )
        oversampled_seed_positions, _, _ = _select_seed_positions_for_budget(
            sorted_nodes=sorted_nodes,
            train=train,
            neighbors=neighbors,
            budget_mode="bonsai",
            target_size=target_size * upscale,
            feat_len=feat_len,
            feat_multiplier=feat_multiplier,
        )
        target_nodes_budget = None
    elif budget_mode == "node_frac":
        target_nodes_budget = max(int(min_target_nodes), max(1, int(math.ceil(float(target_size_frac) * float(nnodes)))))
        base_seed_positions, base_nodes, _ = _select_seed_positions_for_budget(
            sorted_nodes=sorted_nodes,
            train=train,
            neighbors=neighbors,
            budget_mode="node_frac",
            target_nodes=target_nodes_budget,
        )
        oversampled_seed_positions, _, _ = _select_seed_positions_for_budget(
            sorted_nodes=sorted_nodes,
            train=train,
            neighbors=neighbors,
            budget_mode="node_frac",
            target_nodes=min(int(nnodes), max(int(target_nodes_budget), int(math.ceil(float(target_nodes_budget) * upscale)))),
        )
        target_size = None
    elif budget_mode == "bytes":
        if budget_bytes is None:
            raise ValueError("budget_mode='bytes' requires budget_bytes.")
        feat_dim = int(data.x.size(1))
        feat_bytes = int(data.x.element_size())
        index_bytes = int(data.edge_index.element_size())
        base_seed_positions, base_nodes, _ = _select_seed_positions_for_budget(
            sorted_nodes=sorted_nodes,
            train=train,
            neighbors=neighbors,
            budget_mode="bytes",
            target_bytes=int(budget_bytes),
            feat_dim=feat_dim,
            feat_bytes=feat_bytes,
            index_bytes=index_bytes,
        )
        oversampled_seed_positions, _, _ = _select_seed_positions_for_budget(
            sorted_nodes=sorted_nodes,
            train=train,
            neighbors=neighbors,
            budget_mode="bytes",
            target_bytes=int(math.ceil(float(budget_bytes) * upscale)),
            feat_dim=feat_dim,
            feat_bytes=feat_bytes,
            index_bytes=index_bytes,
        )
        target_nodes_budget = None
        target_size = None
    else:
        raise ValueError(f"Unsupported budget_mode={budget_mode}.")

    if not base_seed_positions:
        base_seed_positions = [int(sorted_nodes[0])]
        base_nodes = _merged_nodes_from_seed_positions(
            seed_positions=base_seed_positions,
            train=train,
            neighbors=neighbors,
        )
    if not oversampled_seed_positions:
        oversampled_seed_positions = list(base_seed_positions)

    ogsize = len(base_nodes)
    merged_graph = _build_merged_graph_from_seed_positions(
        seed_positions=oversampled_seed_positions,
        train=train,
        neighbors=neighbors,
        adj=adj,
    )
    _annotate_merged_graph(merged_graph, data=data, train=train)

    personalization = {int(node): 0.0 for node in merged_graph.nodes()}
    if oversampled_seed_positions:
        weight = 1.0 / len(oversampled_seed_positions)
        for node in oversampled_seed_positions:
            personalization[int(train[int(node)])] = weight
    ppr = sorted(nx.pagerank(merged_graph, personalization=personalization).items(), key=lambda item: item[1])
    todel = max(0, len(ppr) - ogsize)
    for node, _ in ppr[:todel]:
        merged_graph.remove_node(int(node))

    merged_graph = _match_distribution(
        merged_graph=merged_graph,
        data=data,
        train=train,
        ranked_nodes=sorted_nodes,
    )
    _annotate_merged_graph(merged_graph, data=data, train=train)
    condensed = _to_condensed_data(merged_graph)

    meta = {
        "method": "bonsai_official" if budget_mode == "bonsai" else "bonsai_ranked",
        "budget_mode": str(budget_mode),
        "target_size_frac": float(target_size_frac),
        "k": int(k),
        "size_full": int(size_full),
        "target_size": float(target_size) if target_size is not None else None,
        "target_nodes_budget": int(target_nodes_budget) if target_nodes_budget is not None else None,
        "budget_bytes": int(budget_bytes) if budget_bytes is not None else None,
        "bonsai_dtype": str(bonsai_dtype),
        "merge_prune_m": float(m),
        "num_nodes_orig": int(nnodes),
        "num_edges_orig": int(nedges),
        "num_nodes_condensed": int(condensed.num_nodes),
        "num_edges_condensed": int(condensed.edge_index.size(1)),
        "num_train_orig": int(len(train)),
        "num_train_condensed": int(condensed.target.sum().item()) if getattr(condensed, "target", None) is not None else None,
        "features_used_dim": int(len(features_used)),
        "features_used": [int(x) for x in features_used],
        "num_seed_nodes": int(len(base_seed_positions)),
        "num_seed_nodes_oversampled": int(len(oversampled_seed_positions)),
        "neighbor_hops": 1,
        "max_train_nodes": int(max_train_nodes) if max_train_nodes is not None else None,
    }
    return condensed, meta


def _condense_bonsai_official(
    *,
    dataset_name: str,
    data: Data,
    train_idx: torch.Tensor,
    target_size_frac: float,
    k: int,
) -> Tuple[Data, Dict]:
    return _condense_bonsai_ranked(
        dataset_name=dataset_name,
        data=data,
        train_idx=train_idx,
        target_size_frac=target_size_frac,
        k=k,
        budget_mode="bonsai",
    )


def _as_bool_mask(mask, n: int) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    m = torch.as_tensor(mask)
    if m.dtype == torch.bool:
        if m.numel() == n:
            return m.view(-1)
        if m.dim() == 2 and m.size(0) == n:
            return m[:, 0].contiguous().view(-1)
    return None


def _build_out_neighbors(edge_index: torch.Tensor, num_nodes: int) -> List[Set[int]]:
    neighbors: List[Set[int]] = [set([i]) for i in range(num_nodes)]
    row = edge_index[0].tolist()
    col = edge_index[1].tolist()
    for u, v in zip(row, col):
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            neighbors[u].add(v)
    return neighbors


def _estimate_bytes(*, num_nodes: int, num_edges: int, feat_dim: int, feat_bytes: int, index_bytes: int) -> int:
    return int(num_nodes) * int(feat_dim) * int(feat_bytes) + 2 * int(num_edges) * int(index_bytes)


def _k_hop_closure(seeds: Set[int], out_neighbors: List[Set[int]], *, hops: int) -> Set[int]:
    kept = set(seeds)
    frontier = set(seeds)
    for _ in range(int(max(0, hops))):
        nxt: Set[int] = set()
        for u in frontier:
            nxt |= out_neighbors[u]
        nxt -= kept
        kept |= nxt
        frontier = nxt
        if not frontier:
            break
    return kept


@torch.no_grad()
def _wl_one_step(x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edge_index2, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index2[0], edge_index2[1]
    deg = torch.bincount(row, minlength=num_nodes).to(dtype=torch.float32, device=x.device)
    deg = torch.clamp(deg, min=1.0)
    vals = (1.0 / deg[row]) * (1.0 / deg[col])
    adj = SparseTensor(row=row, col=col, value=vals, sparse_sizes=(num_nodes, num_nodes))
    return matmul(adj, x)


@torch.no_grad()
def _exact_knn_blockwise(
    x: torch.Tensor,
    *,
    k: int,
    block_q: int = 1024,
    block_c: int = 8192,
    device: str = "cpu",
) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError("x must be 2D [n,d].")
    n = int(x.size(0))
    if n == 0:
        return torch.empty((0, 0), dtype=torch.long)
    k = max(1, min(int(k), n - 1))
    dev = torch.device(device)
    x = x.detach().to(dev, dtype=torch.float32)
    norms = (x * x).sum(dim=1)
    knn_idx = torch.empty((n, k), dtype=torch.long, device="cpu")
    block_q = max(1, int(block_q))
    block_c = max(1, int(block_c))
    for i0 in range(0, n, block_q):
        i1 = min(n, i0 + block_q)
        q = x[i0:i1]
        qn = norms[i0:i1].view(-1, 1)
        bq = int(i1 - i0)
        best_d = torch.full((bq, k), float("inf"), dtype=torch.float32, device=dev)
        best_i = torch.full((bq, k), -1, dtype=torch.long, device=dev)
        for j0 in range(0, n, block_c):
            j1 = min(n, j0 + block_c)
            c = x[j0:j1]
            cn = norms[j0:j1].view(1, -1)
            dist2 = qn + cn - 2.0 * (q @ c.t())
            dist2 = torch.clamp(dist2, min=0.0)
            rows = torch.arange(i0, i1, device=dev)
            mask = (rows >= j0) & (rows < j1)
            if bool(mask.any().item()):
                rr = (rows[mask] - i0).to(torch.long)
                cc = (rows[mask] - j0).to(torch.long)
                dist2[rr, cc] = float("inf")
            kk = min(k, int(j1 - j0))
            d_block, idx_block = torch.topk(dist2, k=kk, largest=False, dim=1)
            idx_block = idx_block.to(torch.long) + int(j0)
            d_cat = torch.cat([best_d, d_block], dim=1)
            i_cat = torch.cat([best_i, idx_block], dim=1)
            best_d, sel = torch.topk(d_cat, k=k, largest=False, dim=1)
            best_i = torch.gather(i_cat, 1, sel)
        knn_idx[i0:i1] = best_i.detach().to("cpu")
    return knn_idx


def _knn_idx_to_rknn(knn_idx: torch.Tensor) -> Dict[int, Set[int]]:
    n = int(knn_idx.size(0))
    rknn: Dict[int, Set[int]] = {}
    knn_idx = knn_idx.to("cpu")
    for i in range(n):
        for q in knn_idx[i].tolist():
            if q >= 0:
                rknn.setdefault(int(q), set()).add(int(i))
    return rknn


def _condense_bonsai_extended(
    data: Data,
    *,
    dataset_name: str,
    target_size_frac: float,
    k: int,
    seed: int,
    neighbor_hops: int,
    train_idx: torch.Tensor,
    max_train_nodes: Optional[int],
    budget_bytes: Optional[int],
    budget_mode: str,
    min_target_nodes: int,
    min_seed_nodes: int,
    knn_block_q: int,
    knn_block_c: int,
    knn_device: str,
) -> Tuple[Data, Dict]:
    num_nodes = int(getattr(data, "num_nodes", None) or data.x.size(0))
    edge_index = data.edge_index
    if edge_index.numel() == 0:
        raise ValueError("edge_index is empty; cannot condense.")

    if max_train_nodes is not None and int(train_idx.numel()) > int(max_train_nodes):
        g = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(int(train_idx.numel()), generator=g)
        train_idx = train_idx[perm[: int(max_train_nodes)]]

    wl = _wl_one_step(data.x, edge_index, num_nodes)
    wl_train = wl[train_idx]
    knn_idx = _exact_knn_blockwise(
        wl_train,
        k=int(k),
        block_q=int(knn_block_q),
        block_c=int(knn_block_c),
        device=str(knn_device),
    )
    ranked_train_positions = _select_max_coverage(_knn_idx_to_rknn(knn_idx)) or list(range(int(train_idx.numel())))

    out_neighbors = _build_out_neighbors(edge_index, num_nodes=num_nodes)
    feat_dim = int(data.x.size(1))
    feat_bytes = int(data.x.element_size())
    index_bytes = int(edge_index.element_size())

    budget_mode = str(budget_mode).lower().strip()
    if budget_mode not in ("node_frac", "bonsai", "bytes"):
        raise ValueError(f"Unknown budget_mode={budget_mode}.")
    if budget_bytes is not None and int(budget_bytes) > 0:
        budget_mode_effective = "bytes"
    else:
        budget_mode_effective = budget_mode
        budget_bytes = None
    if budget_mode_effective == "bytes" and budget_bytes is None:
        raise ValueError("budget_mode=bytes requires budget_mb > 0.")

    target_nodes_budget: Optional[int] = None
    if budget_mode_effective == "node_frac":
        target_nodes_budget = max(int(min_target_nodes), max(1, int(math.ceil(float(target_size_frac) * float(num_nodes)))))

    rank_lookup = {
        int(train_idx[int(pos)].item()): order
        for order, pos in enumerate(ranked_train_positions)
    }
    default_rank = len(rank_lookup) + 1_000_000

    def removal_key(node: int) -> Tuple[int, int]:
        return (len(out_neighbors[node]), rank_lookup.get(node, default_rank))

    def _subset_num_edges(nodes: Set[int]) -> int:
        if not nodes:
            return 0
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
        mask[torch.as_tensor(list(nodes), device=edge_index.device)] = True
        edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
        return int(edge_mask.sum().item())

    def _subset_bytes(nodes: Set[int]) -> int:
        return _estimate_bytes(
            num_nodes=len(nodes),
            num_edges=_subset_num_edges(nodes),
            feat_dim=feat_dim,
            feat_bytes=feat_bytes,
            index_bytes=index_bytes,
        )

    selected_seed_nodes: List[int] = []
    subset: Set[int] = set()

    def add_seed(seed_node: int) -> None:
        if seed_node in selected_seed_nodes:
            return
        selected_seed_nodes.append(seed_node)
        subset.update(_k_hop_closure({seed_node}, out_neighbors, hops=int(neighbor_hops)))

    initial_seeds = max(1, int(min_seed_nodes))
    for pos in ranked_train_positions:
        if len(selected_seed_nodes) >= initial_seeds:
            break
        add_seed(int(train_idx[int(pos)].item()))

    remaining_positions = [p for p in ranked_train_positions if int(train_idx[int(p)].item()) not in set(selected_seed_nodes)]

    def prune_to_budget() -> None:
        seed_set = set(selected_seed_nodes)
        if budget_mode_effective == "bytes":
            if budget_bytes is None:
                return
            used_bytes = _subset_bytes(subset)
            if used_bytes <= int(budget_bytes):
                return
            non_seed = sorted(subset - seed_set, key=removal_key)
            for node in non_seed:
                if used_bytes <= int(budget_bytes):
                    break
                subset.remove(node)
                used_bytes = _subset_bytes(subset)
            return

        if target_nodes_budget is None or len(subset) <= int(target_nodes_budget):
            return
        if len(seed_set) >= int(target_nodes_budget):
            subset.clear()
            subset.update(seed_set)
            return
        keep_non = sorted(subset - seed_set, key=removal_key, reverse=True)[: max(0, int(target_nodes_budget) - len(seed_set))]
        subset.clear()
        subset.update(seed_set | set(keep_non))

    prune_to_budget()

    def under_budget() -> bool:
        if budget_mode_effective == "bytes":
            return budget_bytes is not None and _subset_bytes(subset) < int(budget_bytes)
        if target_nodes_budget is None:
            return False
        return len(subset) < int(target_nodes_budget)

    for pos in remaining_positions:
        if not under_budget():
            break
        add_seed(int(train_idx[int(pos)].item()))
        prune_to_budget()

    if not selected_seed_nodes:
        selected_seed_nodes = [int(train_idx[int(ranked_train_positions[0])].item())]
        subset = _k_hop_closure(set(selected_seed_nodes), out_neighbors, hops=int(neighbor_hops))

    subset_tensor = torch.as_tensor(sorted(subset), dtype=torch.long)
    sub_edge_index, _ = subgraph(subset=subset_tensor, edge_index=edge_index, relabel_nodes=True, num_nodes=num_nodes)
    condensed = Data(
        x=data.x[subset_tensor],
        edge_index=sub_edge_index,
        y=data.y[subset_tensor].view(-1) if data.y.dim() == 1 or (data.y.dim() > 1 and data.y.size(-1) == 1) else data.y[subset_tensor],
    )
    train_set = set(int(i) for i in train_idx.tolist())
    condensed.target = torch.as_tensor([int(node) in train_set for node in subset_tensor.tolist()], dtype=torch.bool)
    condensed.orig_nid = subset_tensor.clone()

    meta = {
        "method": "bonsai_lite",
        "budget_mode": str(budget_mode_effective),
        "target_size_frac": float(target_size_frac),
        "k": int(k),
        "seed": int(seed),
        "neighbor_hops": int(neighbor_hops),
        "max_train_nodes": int(max_train_nodes) if max_train_nodes is not None else None,
        "budget_bytes": int(budget_bytes) if budget_bytes is not None else None,
        "target_nodes_budget": int(target_nodes_budget) if target_nodes_budget is not None else None,
        "num_nodes_orig": int(num_nodes),
        "num_edges_orig": int(edge_index.size(1)),
        "num_nodes_condensed": int(condensed.num_nodes),
        "num_edges_condensed": int(condensed.edge_index.size(1)),
        "num_train_orig": int(train_idx.numel()),
        "num_train_condensed": int(condensed.target.sum().item()),
        "num_seed_nodes": int(len(selected_seed_nodes)),
    }
    return condensed, meta


def _condense_bonsai_lite(
    data: Data,
    *,
    dataset_name: str,
    target_size_frac: float,
    k: int,
    seed: int,
    neighbor_hops: int,
    train_idx: torch.Tensor,
    max_train_nodes: Optional[int],
    budget_bytes: Optional[int],
    budget_mode: str,
    min_target_nodes: int,
    min_seed_nodes: int,
    knn_block_q: int,
    knn_block_c: int,
    knn_device: str,
) -> Tuple[Data, Dict]:
    # Route explicit non-official extensions through the maintained extended path.
    if int(neighbor_hops) != 1 or int(min_seed_nodes) > 0:
        return _condense_bonsai_extended(
            data,
            dataset_name=dataset_name,
            target_size_frac=target_size_frac,
            k=k,
            seed=seed,
            neighbor_hops=neighbor_hops,
            train_idx=train_idx,
            max_train_nodes=max_train_nodes,
            budget_bytes=budget_bytes,
            budget_mode=budget_mode,
            min_target_nodes=min_target_nodes,
            min_seed_nodes=min_seed_nodes,
            knn_block_q=knn_block_q,
            knn_block_c=knn_block_c,
            knn_device=knn_device,
        )

    budget_mode_effective = str(budget_mode).lower().strip()
    if budget_bytes is not None and int(budget_bytes) > 0:
        budget_mode_effective = "bytes"

    if budget_mode_effective in {"node_frac", "bytes"}:
        return _condense_bonsai_ranked(
            dataset_name=dataset_name,
            data=data,
            train_idx=train_idx,
            target_size_frac=target_size_frac,
            k=k,
            budget_mode=budget_mode_effective,
            min_target_nodes=min_target_nodes,
            budget_bytes=budget_bytes,
            max_train_nodes=max_train_nodes,
            seed=seed,
        )

    return _condense_bonsai_extended(
        data,
        dataset_name=dataset_name,
        target_size_frac=target_size_frac,
        k=k,
        seed=seed,
        neighbor_hops=neighbor_hops,
        train_idx=train_idx,
        max_train_nodes=max_train_nodes,
        budget_bytes=budget_bytes,
        budget_mode=budget_mode,
        min_target_nodes=min_target_nodes,
        min_seed_nodes=min_seed_nodes,
        knn_block_q=knn_block_q,
        knn_block_c=knn_block_c,
        knn_device=knn_device,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="", help="Single dataset name (overrides --tsv).")
    p.add_argument("--tsv", default="data/available_datasets.tsv", help="TSV with dataset names (one per line).")
    p.add_argument("--dataset_root", default="data/datasets", help="Root where original datasets are stored.")
    p.add_argument("--out_root", default="data/datasets/bonsai", help="Root to write condensed datasets.")
    p.add_argument("--target_size_frac", type=float, default=0.01)
    p.add_argument("--budget_mb", type=int, default=0, help="Per-dataset byte budget in MB for condensed graph (0 disables; overrides target_size_frac).")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--neighbor_hops", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_train_nodes", type=int, default=0, help="Cap train nodes for pairwise distance (set 0 to disable).")
    p.add_argument("--knn_block_q", type=int, default=1024)
    p.add_argument("--knn_block_c", type=int, default=8192)
    p.add_argument("--knn_device", default="cpu", choices=["cpu", "cuda"], help="Where to run blockwise kNN.")
    p.add_argument("--feat_reduction", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--feat_dim", type=int, default=100)
    p.add_argument(
        "--budget_mode",
        default="node_frac",
        choices=["node_frac", "bonsai", "bytes"],
        help="How to interpret target_size_frac when budget_mb=0.",
    )
    p.add_argument("--min_nodes", type=int, default=0, help="Minimum nodes to keep when using node_frac mode.")
    p.add_argument("--min_seed_nodes", type=int, default=0, help="Minimum seed nodes to select before expansion.")
    p.add_argument("--csv_out", default="", help="Append per-dataset results to this CSV after each run.")
    p.add_argument("--split_mode", default="pretrain", choices=["pretrain", "auto", "bonsai"], help="Split logic for the train nodes used by Bonsai.")
    p.add_argument("--split_root", default="data/splits", help="Where to save/load fixed splits.")
    p.add_argument("--split", default="0.8,0.1,0.1", help="Split ratios when masks are missing.")
    p.add_argument("--mask_col", type=int, default=0, help="For 2D masks [N,S], which split column to use.")
    args = p.parse_args()

    set_seed(int(args.seed))

    if args.dataset:
        datasets = [args.dataset]
    else:
        tsv_path = resolve_project_path(args.tsv)
        if not tsv_path.is_file():
            raise FileNotFoundError(f"TSV not found: {tsv_path}")
        datasets = read_name_list_file(tsv_path)

    out_root = resolve_project_path(args.out_root)
    ds_root = resolve_project_path(args.dataset_root)
    out_root.mkdir(parents=True, exist_ok=True)

    split_tuple = tuple(float(x) for x in str(args.split).split(","))
    split_tuple = split_tuple if len(split_tuple) == 3 else (0.8, 0.1, 0.1)
    split_root = resolve_project_path(args.split_root)
    ensure_dir(split_root)

    csv_path = resolve_project_path(args.csv_out) if args.csv_out else None
    csv_fields = [
        "dataset",
        "status",
        "message",
        "save_dir",
        "target_size_frac",
        "budget_mb",
        "budget_mode",
        "nodes_condensed",
        "edges_condensed",
        "num_seed_nodes",
        "neighbor_hops",
    ]
    ok = 0
    for idx, name in enumerate(datasets, 1):
        print(f"[{idx}/{len(datasets)}] Condensing {name}")
        try:
            ds = create_dataset(
                name=name,
                root=str(ds_root),
                task_level="node",
                induced=False,
                feat_reduction=bool(args.feat_reduction),
                feat_reduction_dim=int(args.feat_dim),
            )
            data = ds[0]
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[SKIP] {name}: not a supported node-level single-graph dataset ({exc})")
            if csv_path is not None:
                append_csv_row(
                    csv_path,
                    fieldnames=csv_fields,
                    row={
                        "dataset": name,
                        "status": "skip",
                        "message": str(exc),
                        "save_dir": "",
                        "target_size_frac": float(args.target_size_frac),
                        "budget_mb": int(args.budget_mb),
                        "budget_mode": str(args.budget_mode),
                        "nodes_condensed": "",
                        "edges_condensed": "",
                        "num_seed_nodes": "",
                        "neighbor_hops": int(args.neighbor_hops),
                    },
                )
            continue

        try:
            train_idx = _resolve_train_indices(
                dataset_name=name,
                dataset=ds,
                data=data,
                split_mode=str(args.split_mode),
                split_root=split_root,
                split_tuple=split_tuple,  # type: ignore[arg-type]
                mask_col=int(args.mask_col),
                seed=int(args.seed),
            )
            budget_mode = str(args.budget_mode).lower().strip()
            if int(args.budget_mb) <= 0 and budget_mode == "bonsai":
                condensed, meta = _condense_bonsai_official(
                    dataset_name=name,
                    data=data,
                    train_idx=train_idx,
                    target_size_frac=float(args.target_size_frac),
                    k=int(args.k),
                )
            else:
                condensed, meta = _condense_bonsai_lite(
                    data,
                    dataset_name=name,
                    target_size_frac=float(args.target_size_frac),
                    k=int(args.k),
                    seed=int(args.seed),
                    neighbor_hops=int(args.neighbor_hops),
                    train_idx=train_idx,
                    max_train_nodes=None if int(args.max_train_nodes) <= 0 else int(args.max_train_nodes),
                    budget_bytes=None if int(args.budget_mb) <= 0 else int(args.budget_mb) * 1024 * 1024,
                    budget_mode=budget_mode,
                    min_target_nodes=int(args.min_nodes),
                    min_seed_nodes=int(args.min_seed_nodes),
                    knn_block_q=int(args.knn_block_q),
                    knn_block_c=int(args.knn_block_c),
                    knn_device=str(args.knn_device),
                )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[FAIL] {name}: condensation failed ({exc})")
            if csv_path is not None:
                append_csv_row(
                    csv_path,
                    fieldnames=csv_fields,
                    row={
                        "dataset": name,
                        "status": "fail",
                        "message": str(exc),
                        "save_dir": "",
                        "target_size_frac": float(args.target_size_frac),
                        "budget_mb": int(args.budget_mb),
                        "budget_mode": str(args.budget_mode),
                        "nodes_condensed": "",
                        "edges_condensed": "",
                        "num_seed_nodes": "",
                        "neighbor_hops": int(args.neighbor_hops),
                    },
                )
            continue

        tag = (
            f"budget{int(args.budget_mb)}mb_frac{args.target_size_frac}"
            if int(args.budget_mb) > 0
            else (f"frac-{args.target_size_frac}" if budget_mode == "node_frac" else f"frac-{args.target_size_frac}_mode-{budget_mode}")
        )
        save_dir = out_root / name / tag
        save_condensed_as_pyg_dataset(condensed, out_root=str(save_dir), meta=meta)
        print(f"[OK] {name}: saved to {save_dir} (nodes={meta['num_nodes_condensed']}, edges={meta['num_edges_condensed']})")
        ok += 1
        if csv_path is not None:
            append_csv_row(
                csv_path,
                fieldnames=csv_fields,
                row={
                    "dataset": name,
                    "status": "ok",
                    "message": "",
                    "save_dir": str(save_dir),
                    "target_size_frac": float(args.target_size_frac),
                    "budget_mb": int(args.budget_mb),
                    "budget_mode": str(args.budget_mode),
                    "nodes_condensed": int(meta.get("num_nodes_condensed", 0)),
                    "edges_condensed": int(meta.get("num_edges_condensed", 0)),
                    "num_seed_nodes": int(meta.get("num_seed_nodes", 0) or 0),
                    "neighbor_hops": int(meta.get("neighbor_hops", int(args.neighbor_hops))),
                },
            )

    print(f"[SUMMARY] ok={ok}, total_requested={len(datasets)}")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
