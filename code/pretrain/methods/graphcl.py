import random
from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from ..base import PretrainTask
from ..registry import register
from .utils import get_batch_vector, pool_nodes


def _apply_edge_mask(data, edge_mask: torch.Tensor) -> None:
    """Apply an edge mask to edge attributes when available."""
    edge_attr = getattr(data, "edge_attr", None)
    if edge_attr is None:
        return
    if edge_attr.size(0) == edge_mask.numel():
        data.edge_attr = edge_attr[edge_mask]


def edge_perturbation(data, p: float, add_random_edges: bool = False):
    """
    Perturb edges with ratio p.

    For plain edge-index graphs, `permE` can be either delete-only (TU-style)
    or delete-and-add (chem GraphCL style). When edge attributes exist,
    keep a delete-only variant to avoid fabricating invalid edge attributes.
    """
    aug = data.clone()
    edge_index = getattr(aug, "edge_index", None)
    if edge_index is None:
        return aug
    if p <= 0:
        return aug

    edge_num = int(edge_index.size(1))
    if edge_num == 0:
        return aug
    permute_num = int(edge_num * float(p))
    if permute_num <= 0:
        return aug

    edge_attr = getattr(aug, "edge_attr", None)
    # With edge_attr, prefer deletion only (official chemistry loader behavior).
    if edge_attr is not None:
        keep_num = max(1, edge_num - permute_num)
        keep_idx = torch.randperm(edge_num, device=edge_index.device)[:keep_num]
        aug.edge_index = edge_index[:, keep_idx]
        if edge_attr.size(0) == edge_num:
            aug.edge_attr = edge_attr[keep_idx]
        return aug

    # permE for plain graphs.
    keep_num = max(1, edge_num - permute_num)
    keep_idx = torch.randperm(edge_num, device=edge_index.device)[:keep_num]
    kept_edge_index = edge_index[:, keep_idx]
    if not add_random_edges:
        aug.edge_index = kept_edge_index
        return aug

    num_nodes = int(getattr(aug, "num_nodes", 0) or 0)
    if num_nodes <= 0 and getattr(aug, "x", None) is not None:
        num_nodes = int(aug.x.size(0))
    if num_nodes <= 0 and edge_index.numel() > 0:
        num_nodes = int(edge_index.max().item()) + 1
    if num_nodes <= 0:
        aug.edge_index = kept_edge_index
        return aug

    added = torch.randint(
        low=0,
        high=num_nodes,
        size=(2, permute_num),
        device=edge_index.device,
        dtype=edge_index.dtype,
    )
    aug.edge_index = torch.cat([kept_edge_index, added], dim=1)
    return aug


def _subset_nodes(data, keep_mask: torch.Tensor):
    """Filter graph nodes and relabel edges accordingly."""
    aug = data.clone()
    num_nodes = keep_mask.size(0)
    keep_idx = keep_mask.nonzero(as_tuple=True)[0]
    if keep_idx.numel() == 0:
        keep_idx = torch.zeros(1, dtype=torch.long, device=keep_mask.device)
        keep_mask = torch.zeros_like(keep_mask)
        keep_mask[0] = True

    if getattr(aug, "x", None) is not None:
        aug.x = aug.x[keep_idx]

    if getattr(aug, "batch", None) is not None:
        aug.batch = aug.batch[keep_idx]

    if getattr(aug, "pos", None) is not None and torch.is_tensor(aug.pos) and aug.pos.size(0) == num_nodes:
        aug.pos = aug.pos[keep_idx]
    # Keep num_nodes explicit for downstream components that rely on it.
    aug.num_nodes = int(keep_idx.numel())

    edge_index = getattr(aug, "edge_index", None)
    if edge_index is not None:
        edge_mask = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
        filtered_edge_index = edge_index[:, edge_mask]
        node_map = torch.full((num_nodes,), -1, dtype=torch.long, device=edge_index.device)
        node_map[keep_idx] = torch.arange(keep_idx.numel(), dtype=torch.long, device=edge_index.device)
        aug.edge_index = node_map[filtered_edge_index]
        _apply_edge_mask(aug, edge_mask)

    return aug


def subgraph_sampling(data, p: float):
    """GraphCL-style subgraph augmentation: keep ~p fraction of nodes via neighborhood expansion."""
    aug = data.clone()
    if getattr(aug, "x", None) is None or getattr(aug, "edge_index", None) is None:
        return aug

    node_num = int(aug.x.size(0))
    if node_num <= 1:
        return aug

    keep_target = max(1, int(node_num * float(p)))
    edge_index = aug.edge_index
    src = edge_index[0]
    dst = edge_index[1]

    seed = int(torch.randint(node_num, (1,), device=src.device).item())
    selected = [seed]
    selected_set = {seed}
    neighbors = set(dst[src == seed].tolist())

    count = 0
    while len(selected) <= keep_target:
        count += 1
        if count > node_num:
            break
        if not neighbors:
            break
        nxt = random.choice(list(neighbors))
        if nxt in selected_set:
            continue
        selected.append(nxt)
        selected_set.add(nxt)
        neighbors.update(dst[src == nxt].tolist())

    keep_mask = torch.zeros(node_num, dtype=torch.bool, device=aug.x.device)
    keep_mask[torch.as_tensor(selected, dtype=torch.long, device=aug.x.device)] = True
    return _subset_nodes(aug, keep_mask)


def node_dropping(data, p: float):
    """Drop an exact p-ratio of nodes (GraphCL-style dropN)."""
    if getattr(data, "x", None) is None:
        return data.clone()
    num_nodes = data.x.size(0)
    drop_num = int(num_nodes * float(p))
    if drop_num <= 0:
        return data.clone()

    idx_perm = torch.randperm(num_nodes, device=data.x.device)
    keep_idx = idx_perm[drop_num:]
    if keep_idx.numel() == 0:
        keep_idx = idx_perm[:1]
    keep_idx, _ = torch.sort(keep_idx)
    keep_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
    keep_mask[keep_idx] = True
    return _subset_nodes(data, keep_mask)


def feature_masking(data, p: float):
    """Mask node features with probability p (node-wise token masking)."""
    aug = data.clone()
    if getattr(aug, "x", None) is not None:
        node_num = aug.x.size(0)
        mask_num = int(node_num * p)
        if mask_num > 0:
            idx_mask = torch.randperm(node_num, device=aug.x.device)[:mask_num]
            token = aug.x.float().mean(dim=0, keepdim=True)
            if torch.is_floating_point(aug.x):
                token = token.to(dtype=aug.x.dtype)
            else:
                token = token.round().to(dtype=aug.x.dtype)
            aug.x[idx_mask] = token.expand_as(aug.x[idx_mask])
    return aug


def rw_subgraph(data, restart_prob: float = 0.15, num_seeds_ratio: float = 0.3):
    """Random walk subgraph sampling.

    Args:
        data: Input graph data
        restart_prob: Probability of restarting the random walk
        num_seeds_ratio: Ratio of nodes to use as seeds
    """
    aug = data.clone()
    if aug.x is None or getattr(aug, "edge_index", None) is None:
        return aug

    num_nodes = aug.x.size(0)
    num_seeds = max(1, int(num_nodes * num_seeds_ratio))

    seed_indices = torch.randperm(num_nodes, device=aug.x.device)[:num_seeds]
    visited_mask = torch.zeros(num_nodes, dtype=torch.bool, device=aug.x.device)
    visited_mask[seed_indices] = True
    edge_index = aug.edge_index

    src = edge_index[0]
    dst = edge_index[1]
    for seed in seed_indices:
        current = seed.item()
        walk_length = random.randint(5, 15)

        for _ in range(walk_length):
            if random.random() < restart_prob:
                current = seed.item()
                continue

            neighbors = dst[src == current]
            if neighbors.size(0) == 0:
                break

            next_node = neighbors[torch.randint(neighbors.size(0), (1,), device=neighbors.device)].item()
            visited_mask[next_node] = True
            current = next_node

    if int(visited_mask.sum().item()) < max(3, int(num_nodes * 0.2)):
        return aug

    return _subset_nodes(aug, visited_mask)


class _Projector(nn.Module):
    """Projection head for contrastive learning."""
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


@register("graphcl")
class GraphCL(PretrainTask):
    """GraphCL: Graph Contrastive Learning with multiple augmentation strategies.

    Reference: You et al. "Graph Contrastive Learning with Augmentations" NeurIPS 2020.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.tau = cfg.pretrain.graphcl.temperature
        self.edge_remove_prob = cfg.pretrain.graphcl.edge_remove_prob
        self.node_drop_prob = cfg.pretrain.graphcl.node_drop_prob
        self.feature_mask_prob = cfg.pretrain.graphcl.feature_mask_prob
        self.permE_add_edges = bool(getattr(cfg.pretrain.graphcl, "permE_add_edges", False))
        self.use_subgraph_aug = bool(getattr(cfg.pretrain.graphcl, "use_subgraph_aug", True))
        # Backward compatibility: prefer explicit subgraph_ratio; fall back to legacy rw ratio field.
        self.subgraph_ratio = float(
            getattr(
                cfg.pretrain.graphcl,
                "subgraph_ratio",
                getattr(cfg.pretrain.graphcl, "rw_num_seeds_ratio", 0.3),
            )
        )

        # Augmentations from GraphCL family.
        self.augmentations: List[Callable] = [
            lambda data: edge_perturbation(
                data,
                self.edge_remove_prob,
                add_random_edges=self.permE_add_edges,
            ),
            lambda data: node_dropping(data, self.node_drop_prob),
            lambda data: feature_masking(data, self.feature_mask_prob),
        ]
        if self.use_subgraph_aug and self.subgraph_ratio > 0:
            self.augmentations.append(lambda data: subgraph_sampling(data, self.subgraph_ratio))

        hidden = cfg.pretrain.graphcl.proj_hidden
        self.projector = _Projector(cfg.model.out_dim, hidden)

    def _readout(self, node_repr, data):
        """Pool node representations to graph level."""
        batch = get_batch_vector(data)
        if getattr(data, "batch", None) is None:
            return node_repr.mean(dim=0, keepdim=True)
        return pool_nodes(node_repr, batch, mode=self.cfg.model.graph_pooling)

    @staticmethod
    def _ensure_num_nodes_attr(data) -> None:
        """Normalize `num_nodes` so all items can be collated by PyG Batch."""
        x = getattr(data, "x", None)
        edge_index = getattr(data, "edge_index", None)
        num_nodes = getattr(data, "num_nodes", None)
        if x is not None:
            num_nodes = int(x.size(0))
        elif edge_index is not None and edge_index.numel() > 0:
            num_nodes = int(edge_index.max().item()) + 1
        elif num_nodes is None:
            num_nodes = 0
        # Keep explicit key on every graph so Batch.from_data_list sees consistent fields.
        data.num_nodes = int(num_nodes)

    @staticmethod
    def _apply_per_graph_augment(data, augment: Callable):
        """
        Apply an augmentation per graph to preserve positive-pair alignment.
        """
        if getattr(data, "batch", None) is None:
            aug = augment(data)
            GraphCL._ensure_num_nodes_attr(aug)
            return aug
        graphs = data.to_data_list()
        aug_graphs = [augment(g) for g in graphs]
        for graph in aug_graphs:
            GraphCL._ensure_num_nodes_attr(graph)
        return Batch.from_data_list(aug_graphs)

    def _contrastive_loss(self, z1, z2):
        """
        Official GraphCL contrastive objective:
        log(pos / sum(negatives)).
        """
        sim12 = torch.exp((z1 @ z2.t()) / self.tau)
        sim21 = torch.exp((z2 @ z1.t()) / self.tau)
        pos12 = torch.diag(sim12)
        pos21 = torch.diag(sim21)
        denom12 = (sim12.sum(dim=1) - pos12).clamp(min=1e-12)
        denom21 = (sim21.sum(dim=1) - pos21).clamp(min=1e-12)

        loss12 = -torch.log((pos12 / denom12).clamp(min=1e-12)).mean()
        loss21 = -torch.log((pos21 / denom21).clamp(min=1e-12)).mean()
        loss = 0.5 * (loss12 + loss21)

        with torch.no_grad():
            diag_sim = torch.diag(z1 @ z2.t()).mean().item()
        return loss, diag_sim

    def step(self, model, data, device):
        """Perform one pretraining step.

        Args:
            model: GNN encoder
            data: Batch of graphs
            device: torch device

        Returns:
            loss: scalar tensor
            logs: dict with logging info
        """
        data = data.to(device)

        # Sample two different augmentations.
        if len(self.augmentations) >= 2:
            aug_idx1, aug_idx2 = random.sample(range(len(self.augmentations)), 2)
        else:
            aug_idx1 = aug_idx2 = 0
        aug1 = self._apply_per_graph_augment(data, self.augmentations[aug_idx1])
        aug2 = self._apply_per_graph_augment(data, self.augmentations[aug_idx2])

        # Encode augmented views
        z1_nodes, g1 = model(aug1)
        z2_nodes, g2 = model(aug2)

        # Graph-level representations (pool if model doesn't return graph embedding)
        if g1 is None:
            g1 = self._readout(z1_nodes, aug1)
            g2 = self._readout(z2_nodes, aug2)

        # Pair alignment should stay exact because augmentations are per-graph.
        if g1.size(0) != g2.size(0):
            raise RuntimeError(
                "GraphCL positive-pair alignment failed (view batch sizes differ)."
            )

        # Project and normalize
        p1 = F.normalize(self.projector(g1), dim=-1)
        p2 = F.normalize(self.projector(g2), dim=-1)

        if p1.size(0) < 2:
            raise RuntimeError(
                "GraphCL requires at least 2 graphs per training batch. "
                "Use induced subgraphs for node datasets or increase effective batch size."
            )

        # Compute contrastive loss
        loss, sim_stat = self._contrastive_loss(p1, p2)

        return loss, {"sim": sim_stat, "batch_size": p1.size(0)}
