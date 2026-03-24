"""
Graph augmentations used by the SGDC pretrain path.

This is kept inside `code.context_generation.sgdc` so SGDC stays self-contained
under the context-generation package.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce


def _reindex_compact(data: Data) -> Data:
    """Compact node ids after aggressive edge-dropping style augmentations."""
    if getattr(data, "edge_index", None) is None or data.edge_index.numel() == 0:
        return data
    edge_index = data.edge_index
    active = torch.unique(edge_index)
    if active.numel() == 0:
        return data
    mapping = torch.full((int(active.max()) + 1,), -1, dtype=torch.long, device=active.device)
    mapping[active] = torch.arange(active.numel(), device=active.device)
    remapped = mapping[edge_index]
    data.edge_index = coalesce(remapped)
    if getattr(data, "x", None) is not None:
        data.x = data.x[active]
    if getattr(data, "batch", None) is not None:
        data.batch = data.batch[active]
    return data


def aug(data: Data, aug: str, is_syn: bool = False) -> Tuple[Data, Data]:
    if aug == "dnodes":
        data_aug = drop_nodes(deepcopy(data))
        data_aug = process(data, data_aug, is_syn)
    elif aug == "pedges":
        data_aug = permute_edges(deepcopy(data))
    elif aug == "subgraph":
        data_aug = subgraph(deepcopy(data))
        data_aug = process(deepcopy(data), data_aug)
    elif aug == "mask_nodes":
        data_aug = mask_nodes(deepcopy(data))
    elif aug == "none":
        data_aug = deepcopy(data)
        data_aug.x = torch.ones((data.edge_index.max() + 1, 1))
    elif aug == "random2":
        data_aug = drop_nodes(deepcopy(data)) if np.random.randint(2) == 0 else subgraph(deepcopy(data))
    elif aug == "random3":
        choice = np.random.randint(3)
        if choice == 0:
            data_aug = drop_nodes(deepcopy(data))
        elif choice == 1:
            data_aug = permute_edges(deepcopy(data))
        else:
            data_aug = subgraph(deepcopy(data))
    elif aug == "random4":
        choice = np.random.randint(4)
        if choice == 0:
            data_aug = drop_nodes(deepcopy(data))
        elif choice == 1:
            data_aug = permute_edges(deepcopy(data))
        elif choice == 2:
            data_aug = subgraph(deepcopy(data))
        else:
            data_aug = mask_nodes(deepcopy(data))
    else:
        raise ValueError(f"augmentation error: {aug}")

    data = _reindex_compact(data)
    data_aug = _reindex_compact(data_aug)
    return data, data_aug


def drop_nodes(data: Data) -> Data:
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num / 10)

    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_nondrop = [n for n in range(node_num) if n not in idx_drop]

    edge_index = data.edge_index.numpy()
    if data.edge_attr is not None:
        idx_edge = [
            n
            for n in range(edge_num)
            if edge_index[0, n] in idx_nondrop and edge_index[1, n] in idx_nondrop
        ]
        data.edge_attr = data.edge_attr[idx_edge]
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    data.edge_index = adj.nonzero().t()
    return data


def permute_edges(data: Data) -> Data:
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num / 10)
    edge_index = data.edge_index.transpose(0, 1).numpy()
    edge_index = edge_index[np.random.choice(edge_num, edge_num - permute_num, replace=False)]
    data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
    return data


def subgraph(data: Data) -> Data:
    node_num, _ = data.x.size()
    sub_num = int(node_num * 0.2)
    edge_index = data.edge_index.numpy()

    idx_sub = [int(np.random.randint(node_num, size=1)[0])]
    idx_neigh = set(edge_index[1][edge_index[0] == idx_sub[0]])

    count = 0
    while len(idx_sub) <= sub_num:
        count += 1
        if count > node_num or len(idx_neigh) == 0:
            break
        sample_node = int(np.random.choice(list(idx_neigh)))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh = idx_neigh.union(set(edge_index[1][edge_index[0] == idx_sub[-1]]))

    idx_drop = [n for n in range(node_num) if n not in idx_sub]
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    data.edge_index = adj.nonzero().t()
    return data


def mask_nodes(data: Data) -> Data:
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num / 10)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(
        np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)),
        dtype=torch.float32,
    )
    return data


def process(data: Data, data_aug: Data, is_syn: bool = False) -> Data:
    node_num, _ = data.x.size()
    edge_idx = data_aug.edge_index.numpy()
    _, edge_num = edge_idx.shape
    idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

    node_num_aug = len(idx_not_missing)
    data_aug.x = data_aug.x[idx_not_missing]
    data_aug.batch = data.batch[idx_not_missing]
    idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
    if is_syn:
        data_aug.edge_index = torch.eye(node_num_aug).nonzero().t()
    else:
        edge_idx_aug = [
            [idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]]
            for n in range(edge_num)
            if edge_idx[0, n] != edge_idx[1, n]
        ]
        data_aug.edge_index = torch.tensor(edge_idx_aug).transpose_(0, 1)
    return data_aug
