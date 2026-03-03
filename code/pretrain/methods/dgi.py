import torch
from torch import nn
from typing import Tuple

from ..base import PretrainTask
from ..registry import register
from .utils import get_batch_vector, pool_nodes


class _Discriminator(nn.Module):
    """Bilinear discriminator used in the original DGI objective."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.scorer = nn.Bilinear(hidden_dim, hidden_dim, 1)
        nn.init.xavier_uniform_(self.scorer.weight)
        if self.scorer.bias is not None:
            nn.init.zeros_(self.scorer.bias)

    def forward(
        self,
        summary: torch.Tensor,
        pos_repr: torch.Tensor,
        neg_repr: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context = summary[batch]
        pos_logits = self.scorer(pos_repr, context).view(-1)
        neg_logits = self.scorer(neg_repr, context).view(-1)
        return pos_logits, neg_logits


@register("dgi")
class DGI(PretrainTask):
    """DGI pretraining task.

    Reference: Veličković et al. "Deep Graph Infomax" ICLR 2019.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.readout_activation = nn.Sigmoid()
        self.discriminator = _Discriminator(cfg.model.out_dim)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def step(self, model, data, device):
        if getattr(data, "x", None) is None:
            raise ValueError("DGI requires node features `x` for corruption.")

        data = data.to(device)
        node_repr, _ = model(data)
        batch = get_batch_vector(data)
        # Original DGI uses average readout before sigmoid.
        summary = self.readout_activation(pool_nodes(node_repr, batch, mode="mean"))

        corrupted = data.clone()
        perm = torch.randperm(data.num_nodes, device=device)
        corrupted.x = data.x[perm]

        neg_repr, _ = model(corrupted)
        pos_logits, neg_logits = self.discriminator(summary, node_repr, neg_repr, batch)

        logits = torch.cat((pos_logits, neg_logits), dim=0)
        labels = torch.cat((torch.ones_like(pos_logits), torch.zeros_like(neg_logits)), dim=0)
        loss = self.loss_fn(logits, labels)

        return loss, {
            "summary_norm": summary.norm(dim=-1).mean().item(),
            "pos_prob": torch.sigmoid(pos_logits).mean().item(),
            "neg_prob": torch.sigmoid(neg_logits).mean().item(),
        }
