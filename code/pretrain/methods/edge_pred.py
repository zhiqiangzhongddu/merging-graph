import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import negative_sampling

from ..base import PretrainTask
from ..registry import register


@register("edge_pred")
class EdgePrediction(PretrainTask):
    """Edge Prediction pretraining task.
    
    Reference: Hu et al. "Strategies for Pre-training Graph Neural Networks" ICLR 2020.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        edge_cfg = getattr(getattr(cfg, "pretrain", None), "edge_pred", None)
        self.use_mlp_scorer = bool(getattr(edge_cfg, "use_mlp_scorer", False)) if edge_cfg is not None else False
        if self.use_mlp_scorer:
            hidden = cfg.model.out_dim
            self.scorer = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        else:
            self.scorer = None

    def score_edges(
        self, node_repr: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        src, dst = edge_index
        if self.scorer is None:
            # Original pretrain-gnns formulation: dot-product edge score.
            return torch.sum(node_repr[src] * node_repr[dst], dim=-1)
        pairs = torch.cat([node_repr[src], node_repr[dst]], dim=-1)
        return self.scorer(pairs).view(-1)

    def step(
        self, model: nn.Module, data, device
    ) -> tuple[torch.Tensor, dict]:
        data = data.to(device)
        data, forward_edge = self._sample_forward_edges(data)
        node_repr, _ = model(data)
        pos_edge, neg_edge = self._extract_pos_neg_edges(data, forward_edge)
        pos_edge = self._sample_edges(pos_edge)
        if pos_edge is None or pos_edge.numel() == 0 or pos_edge.size(1) == 0:
            zero = node_repr.sum() * 0.0
            return zero, {
                "pos_mean": 0.0,
                "neg_mean": 0.0,
                "num_edges": 0,
                "num_nodes": data.num_nodes,
                "batch_size": int(getattr(data, "num_graphs", 1)),
            }
        if neg_edge is None or neg_edge.numel() == 0:
            neg_edge = negative_sampling(
                pos_edge,
                num_nodes=data.num_nodes,
                num_neg_samples=self._num_neg_samples(pos_edge),
            )
        elif self._num_neg_samples(pos_edge) < neg_edge.size(1):
            neg_edge = self._sample_edges(neg_edge)
        pos_logits = self.score_edges(node_repr, pos_edge)
        neg_logits = self.score_edges(node_repr, neg_edge)
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat(
            [
                torch.ones_like(pos_logits, device=device),
                torch.zeros_like(neg_logits, device=device),
            ],
            dim=0,
        )
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        preds = torch.sigmoid(logits).detach()
        return loss, {
            "pos_mean": preds[: pos_logits.size(0)].mean().item(),
            "neg_mean": preds[pos_logits.size(0) :].mean().item(),
            "num_edges": pos_edge.size(1),
            "num_nodes": data.num_nodes,
            "batch_size": int(getattr(data, "num_graphs", 1)),
        }

    @staticmethod
    def _every_other_directed_edge(edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index is None or edge_index.numel() == 0:
            return edge_index
        if edge_index.size(1) <= 1:
            return edge_index
        return edge_index[:, ::2]

    def _extract_pos_neg_edges(self, data, forward_edge: torch.Tensor | None):
        edge_label_index = getattr(data, "edge_label_index", None)
        edge_label = getattr(data, "edge_label", None)

        if edge_label_index is not None and edge_label is not None:
            labels = torch.as_tensor(edge_label, device=edge_label_index.device).view(-1)
            # Support float/int labels where positives are >0.
            pos_mask = labels > 0
            neg_mask = ~pos_mask
            pos_edge = edge_label_index[:, pos_mask]
            neg_edge = edge_label_index[:, neg_mask]
            if pos_edge.numel() == 0:
                fallback = forward_edge if forward_edge is not None else data.edge_index
                pos_edge = self._every_other_directed_edge(fallback)
            return pos_edge, neg_edge

        if edge_label_index is not None:
            # Induced edge graphs may only provide edge_label_index and graph-level
            # binary labels in `y`. In that case, route labels to pos/neg edges.
            y = getattr(data, "y", None)
            if y is not None:
                inferred = self._split_edges_from_graph_labels(data)
                if inferred is not None:
                    return inferred
            return edge_label_index, None

        pos_edge = forward_edge if forward_edge is not None else data.edge_index
        pos_edge = self._every_other_directed_edge(pos_edge)
        return pos_edge, None

    @staticmethod
    def _split_edges_from_graph_labels(data):
        edge_label_index = getattr(data, "edge_label_index", None)
        if edge_label_index is None or edge_label_index.numel() == 0:
            return None

        y = torch.as_tensor(getattr(data, "y"), device=edge_label_index.device)
        if y.dim() > 1:
            y = y.view(y.size(0), -1)[:, 0]
        y = y.view(-1)
        if y.numel() == 0:
            return None

        batch = getattr(data, "batch", None)
        if batch is None or y.numel() == 1:
            edge_labels = y[0].expand(edge_label_index.size(1))
        else:
            edge_batch = batch[edge_label_index[0]]
            if int(edge_batch.max().item()) >= y.numel():
                return None
            edge_labels = y[edge_batch]

        pos_mask = edge_labels > 0
        neg_mask = ~pos_mask
        return edge_label_index[:, pos_mask], edge_label_index[:, neg_mask]

    def _sample_edges(self, edge_index: torch.Tensor) -> torch.Tensor:
        cfg = getattr(getattr(self.cfg, "pretrain", None), "edge_pred", None)
        if cfg is None:
            return edge_index
        ratio = float(getattr(cfg, "pos_edge_ratio", 1.0) or 1.0)
        max_edges = int(getattr(cfg, "pos_edge_max", 0) or 0)
        if ratio >= 1.0 and max_edges <= 0:
            return edge_index

        total = edge_index.size(1)
        target = total
        if ratio < 1.0:
            target = int(total * ratio)
        if max_edges > 0:
            target = min(target, max_edges)
        target = max(1, target)
        if target >= total:
            return edge_index

        # Keep stochastic edge sampling across steps; global seed controls reproducibility.
        perm = torch.randperm(total, device=edge_index.device)
        idx = perm[:target]
        return edge_index[:, idx]

    def _num_neg_samples(self, pos_edge: torch.Tensor) -> int:
        cfg = getattr(getattr(self.cfg, "pretrain", None), "edge_pred", None)
        ratio = float(getattr(cfg, "neg_ratio", 1.0) or 1.0) if cfg is not None else 1.0
        count = int(pos_edge.size(1) * ratio)
        if pos_edge.numel() > 0 and count == 0:
            count = 1
        return count

    def _sample_forward_edges(self, data):
        cfg = getattr(getattr(self.cfg, "pretrain", None), "edge_pred", None)
        if cfg is None:
            return data, None
        ratio = float(getattr(cfg, "forward_edge_ratio", 1.0) or 1.0)
        max_edges = int(getattr(cfg, "forward_edge_max", 0) or 0)
        if ratio >= 1.0 and max_edges <= 0:
            return data, None

        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            return data, None

        total = edge_index.size(1)
        target = total
        if ratio < 1.0:
            target = int(total * ratio)
        if max_edges > 0:
            target = min(target, max_edges)
        target = max(1, target)
        if target >= total:
            return data, None

        perm = torch.randperm(total, device=edge_index.device)
        idx = perm[:target]
        sampled = edge_index[:, idx]

        data = data.clone()
        data.edge_index = sampled
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is not None and edge_attr.size(0) == total:
            data.edge_attr = edge_attr[idx]
        return data, sampled
