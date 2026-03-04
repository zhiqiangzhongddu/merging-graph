from typing import Optional

import torch
from torch import nn
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from .activations import get_activation
from .h2gcn import H2GCNEncoder
from .fagcn import FAGCNEncoder
from .transformer import TransformerEncoder
from .gps import GPSEncoder
from .nodeformer import NodeFormerEncoder


POOLERS = {
    "mean": global_mean_pool,
    "add": global_add_pool,
    "max": global_max_pool,
}


def _build_gin_mlp(
    in_channels: int, 
    out_channels: int, 
    act: nn.Module
) -> nn.Sequential:
    hidden = max(out_channels, in_channels)
    return nn.Sequential(
        nn.Linear(in_channels, hidden),
        act,
        nn.Linear(hidden, out_channels),
    )


def build_conv(
    model_type: str,
    in_channels: int,
    out_channels: int,
    act: nn.Module,
    gat_heads: int = 2,
) -> nn.Module:
    mtype = model_type.lower()
    if mtype == "gcn":
        return GCNConv(
            in_channels=in_channels, 
            out_channels=out_channels
        )
    if mtype == "gin":
        return GINConv(
            _build_gin_mlp(in_channels, out_channels, act)
        )
    if mtype == "gat":
        return GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=gat_heads,
            concat=False,
        )
    if mtype == "mlp":
        return nn.Linear(
            in_features=in_channels, 
            out_features=out_channels
        )
    raise ValueError(f"Unknown model type: {model_type}")


class GNNEncoder(nn.Module):
    """
    Flexible encoder that supports MLP, GCN, GIN, and GAT backbones.
    Returns both node-level and graph-level representations.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        model_type: str = "gcn",
        act: str = "relu",
        dropout: float = 0.1,
        graph_pooling: str = "mean",
        gat_heads: int = 2,
        use_batchnorm: bool = False,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        self.model_type = model_type.lower()
        self.act = get_activation(act)
        self.dropout = nn.Dropout(dropout)
        pool_key = "add" if str(graph_pooling).lower() == "sum" else str(graph_pooling).lower()
        self.pool = POOLERS.get(pool_key, global_mean_pool)
        self.use_batchnorm = bool(use_batchnorm)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            in_c = in_dim if i == 0 else hidden_dim
            out_c = out_dim if i == num_layers - 1 else hidden_dim
            conv = build_conv(self.model_type, in_c, out_c, self.act, gat_heads)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(out_c))

        self.out_dim = out_dim
        self.cached_layer_node_reprs = []
        self.cached_layer_graph_reprs = []

    def forward(self, data):
        x, edge_index = data.x, getattr(data, "edge_index", None)
        batch = getattr(data, "batch", None)
        layer_node_reprs = []

        for idx, conv in enumerate(self.convs):
            if self.model_type == "mlp":
                x = conv(x)
            else:
                if edge_index is None:
                    raise ValueError("edge_index is required for GNN models")
                x = conv(x, edge_index)
            if idx != len(self.convs) - 1:
                x = self.act(x)
                if self.use_batchnorm:
                    x = self.bns[idx](x)
                layer_node_reprs.append(x)
                x = self.dropout(x)
                continue
            layer_last = x
            if self.use_batchnorm:
                layer_last = self.act(layer_last)
                layer_last = self.bns[idx](layer_last)
            layer_node_reprs.append(layer_last)

        node_repr = x
        graph_repr: Optional[torch.Tensor] = None
        if batch is not None:
            graph_repr = self.pool(node_repr, batch)

        self.cached_layer_node_reprs = layer_node_reprs
        if batch is not None:
            self.cached_layer_graph_reprs = [self.pool(layer_x, batch) for layer_x in layer_node_reprs]
        else:
            self.cached_layer_graph_reprs = []
        return node_repr, graph_repr


def build_encoder_from_cfg(cfg, in_dim: int) -> GNNEncoder:
    model_name = getattr(cfg.model, "name", "").lower()
    if model_name == "h2gcn":
        return H2GCNEncoder(
            in_dim=in_dim,
            hidden_dim=cfg.model.hidden_dim,
            out_dim=cfg.model.out_dim,
            act=cfg.model.activation,
            dropout=cfg.model.dropout,
            use_batchnorm=getattr(cfg.model, "use_batchnorm", False),
            graph_pooling=cfg.model.graph_pooling,
        )
    if model_name == "fagcn":
        fagcn_cfg = getattr(cfg.model, "fagcn", None)
        fagcn_eps = getattr(fagcn_cfg, "eps", getattr(cfg.model, "fagcn_eps", 0.1))
        return FAGCNEncoder(
            in_dim=in_dim,
            hidden_dim=cfg.model.hidden_dim,
            out_dim=cfg.model.out_dim,
            num_layers=cfg.model.num_layers,
            act=cfg.model.activation,
            dropout=cfg.model.dropout,
            eps=fagcn_eps,
            graph_pooling=cfg.model.graph_pooling,
        )
    if model_name == "transformer":
        heads = getattr(cfg.model.gat, "heads", 4)
        return TransformerEncoder(
            in_dim=in_dim,
            hidden_dim=cfg.model.hidden_dim,
            out_dim=cfg.model.out_dim,
            num_layers=cfg.model.num_layers,
            heads=heads,
            dropout=cfg.model.dropout,
            act=cfg.model.activation,
            graph_pooling=cfg.model.graph_pooling,
        )
    if model_name == "gps":
        gps_cfg = getattr(cfg.model, "gps", None)
        heads = getattr(gps_cfg, "heads", 4)
        dropout = getattr(gps_cfg, "dropout", cfg.model.dropout)
        attn_type = getattr(gps_cfg, "attn_type", "multihead")
        return GPSEncoder(
            in_dim=in_dim,
            hidden_dim=cfg.model.hidden_dim,
            out_dim=cfg.model.out_dim,
            num_layers=cfg.model.num_layers,
            heads=heads,
            dropout=dropout,
            act=cfg.model.activation,
            graph_pooling=cfg.model.graph_pooling,
            attn_type=attn_type,
        )
    if model_name == "nodeformer":
        return NodeFormerEncoder(
            cfg=cfg,
            in_dim=in_dim,
        )
    return GNNEncoder(
        in_dim=in_dim,
        hidden_dim=cfg.model.hidden_dim,
        out_dim=cfg.model.out_dim,
        num_layers=cfg.model.num_layers,
        model_type=getattr(cfg.model, "name", model_name),
        act=cfg.model.activation,
        dropout=cfg.model.dropout,
        graph_pooling=cfg.model.graph_pooling,
        gat_heads=cfg.model.gat.heads,
        use_batchnorm=bool(getattr(cfg.model, "use_batchnorm", False)),
    )
