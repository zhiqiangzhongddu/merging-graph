"""GNN backbone and scoring head for router selection."""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
from torch import nn
from torch_geometric.nn import GeneralConv, HeteroConv


class RouterGNN(nn.Module):
    def __init__(
        self,
        node_input_dims: Dict[str, int],
        embed_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        edge_types: Optional[Iterable[tuple]] = None,
        use_edge_attr: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.dropout = float(dropout)
        self.use_edge_attr = bool(use_edge_attr)

        self.node_encoders = nn.ModuleDict()
        for node_type, in_dim in node_input_dims.items():
            self.node_encoders[node_type] = nn.Sequential(
                nn.Linear(int(in_dim), self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )

        if edge_types is None:
            edge_types = []
        self.edge_types = list(edge_types)
        self.convs = nn.ModuleList()
        for _ in range(max(1, int(num_layers))):
            convs = {}
            for edge_type in self.edge_types:
                convs[edge_type] = GeneralConv(
                    in_channels=self.embed_dim,
                    out_channels=self.embed_dim,
                    in_edge_channels=1 if self.use_edge_attr else None,
                )
            self.convs.append(HeteroConv(convs, aggr="sum"))

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[tuple, torch.Tensor],
        edge_attr_dict: Optional[Dict[tuple, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        encoded = {}
        for node_type, x in x_dict.items():
            if node_type in self.node_encoders:
                encoded[node_type] = self.node_encoders[node_type](x)
            else:
                encoded[node_type] = x

        for conv in self.convs:
            if self.use_edge_attr and edge_attr_dict is not None:
                encoded = conv(encoded, edge_index_dict, edge_attr_dict)
            else:
                encoded = conv(encoded, edge_index_dict)
            for node_type, x in encoded.items():
                encoded[node_type] = torch.relu(x)
                if self.dropout > 0:
                    encoded[node_type] = nn.functional.dropout(
                        encoded[node_type],
                        p=self.dropout,
                        training=self.training,
                    )
        return encoded


class RouterScorer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        scorer: str = "dot",
        mlp_hidden_dim: int = 128,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.scorer = scorer
        self.temperature = float(temperature) if temperature else 1.0

        self.mlp: Optional[nn.Module] = None
        if scorer == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(self.embed_dim * 2, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, 1),
            )

    def score_all(self, sub_emb: torch.Tensor, expert_emb: torch.Tensor) -> torch.Tensor:
        if self.mlp is None:
            scores = sub_emb @ expert_emb.t()
        else:
            batch = sub_emb.shape[0]
            experts = expert_emb.shape[0]
            sub_expanded = sub_emb[:, None, :].expand(batch, experts, -1)
            exp_expanded = expert_emb[None, :, :].expand(batch, experts, -1)
            pair = torch.cat([sub_expanded, exp_expanded], dim=-1)
            scores = self.mlp(pair).squeeze(-1)
        if self.temperature != 1.0:
            scores = scores / self.temperature
        return scores

    def score_pairs(self, sub_emb: torch.Tensor, expert_emb: torch.Tensor) -> torch.Tensor:
        if self.mlp is None:
            scores = (sub_emb * expert_emb).sum(dim=-1)
        else:
            scores = self.mlp(torch.cat([sub_emb, expert_emb], dim=-1)).squeeze(-1)
        if self.temperature != 1.0:
            scores = scores / self.temperature
        return scores
