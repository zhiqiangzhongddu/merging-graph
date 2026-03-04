from typing import Optional

import torch
from torch import nn
from torch_geometric.nn import (
    GPSConv,
    GCNConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from .activations import get_activation


class GPSEncoder(nn.Module):
    """GPSConv-based encoder with a simple GCN local message passing component."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        act: str = "relu",
        graph_pooling: str = "mean",
        attn_type: str = "multihead",
    ):
        super().__init__()
        self.act = get_activation(act)
        self.dropout = nn.Dropout(dropout)
        poolers = {
            "mean": global_mean_pool,
            "add": global_add_pool,
            "max": global_max_pool,
        }
        self.pool = poolers.get(graph_pooling, global_mean_pool)

        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            conv = GCNConv(hidden_dim, hidden_dim)
            self.layers.append(
                GPSConv(
                    channels=hidden_dim,
                    conv=conv,
                    heads=heads,
                    dropout=dropout,
                    attn_type=attn_type,
                )
            )
        self.out_lin = nn.Linear(hidden_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_in.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.out_lin.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)

        x = self.lin_in(x)
        for layer in self.layers:
            x = layer(x, edge_index, batch=batch)
            x = self.act(x)
            x = self.dropout(x)

        x = self.out_lin(x)
        node_repr = x
        graph_repr: Optional[torch.Tensor] = None
        if batch is not None:
            graph_repr = self.pool(node_repr, batch)
        return node_repr, graph_repr
