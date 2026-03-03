import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import PretrainTask
from ..registry import register
from .utils import get_batch_vector, pool_nodes


def get_positive_expectation(
    p_samples: torch.Tensor,
    measure: str = "JSD",
    average: bool = True,
) -> torch.Tensor:
    if measure != "JSD":
        raise ValueError(f"Unsupported InfoGraph measure: {measure}")
    ep = math.log(2.0) - F.softplus(-p_samples)
    return ep.mean() if average else ep


def get_negative_expectation(
    q_samples: torch.Tensor,
    measure: str = "JSD",
    average: bool = True,
) -> torch.Tensor:
    if measure != "JSD":
        raise ValueError(f"Unsupported InfoGraph measure: {measure}")
    eq = F.softplus(-q_samples) + q_samples - math.log(2.0)
    return eq.mean() if average else eq


def local_global_loss(
    l_enc: torch.Tensor,
    g_enc: torch.Tensor,
    batch: torch.Tensor,
    measure: str = "JSD",
) -> torch.Tensor:
    num_nodes = l_enc.size(0)
    num_graphs = g_enc.size(0)
    if num_nodes == 0 or num_graphs <= 1:
        return l_enc.sum() * 0.0

    batch = batch.view(-1).long()
    pos_mask = F.one_hot(batch, num_classes=num_graphs).to(l_enc.dtype)
    neg_mask = 1.0 - pos_mask

    scores = torch.mm(l_enc, g_enc.t())

    e_pos = get_positive_expectation(scores * pos_mask, measure=measure, average=False)
    e_pos = (e_pos * pos_mask).sum() / pos_mask.sum().clamp(min=1.0)

    e_neg = get_negative_expectation(scores * neg_mask, measure=measure, average=False)
    e_neg = (e_neg * neg_mask).sum() / neg_mask.sum().clamp(min=1.0)
    return e_neg - e_pos


class FF(nn.Module):
    """Residual MLP discriminator used by InfoGraph."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.linear_shortcut(x)


class PriorDiscriminator(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.l0 = nn.Linear(input_dim, input_dim)
        self.l1 = nn.Linear(input_dim, input_dim)
        self.l2 = nn.Linear(input_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))


def prior_loss(y: torch.Tensor, prior_d: nn.Module, gamma: float) -> torch.Tensor:
    prior = torch.rand_like(y)
    eps = 1e-12
    term_a = torch.log(prior_d(prior).clamp(min=eps, max=1.0 - eps)).mean()
    term_b = torch.log((1.0 - prior_d(y)).clamp(min=eps, max=1.0)).mean()
    return -(term_a + term_b) * float(gamma)


@register("infograph")
class InfoGraph(PretrainTask):
    """
    InfoGraph objective from Sun et al. (ICLR 2020):
    - local-global MI with JSD (Fenchel-dual) estimator
    - optional prior matching regularization
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        ig_cfg = cfg.pretrain.infograph
        self.measure = str(getattr(ig_cfg, "measure", "JSD")).upper()
        pooling = str(getattr(ig_cfg, "graph_pooling", "add")).lower()
        self.graph_pooling = "add" if pooling == "sum" else pooling
        self.use_layerwise = bool(getattr(ig_cfg, "use_layerwise", True))
        self.use_prior = bool(getattr(ig_cfg, "prior", False))
        self.gamma = float(getattr(ig_cfg, "gamma", 0.1))
        self._supports_layer_cache = str(getattr(cfg.model, "name", "")).lower() in {"gin", "gcn", "gat", "mlp"}

        num_layers = max(1, int(getattr(cfg.model, "num_layers", 1) or 1))
        hidden_dim = int(getattr(cfg.model, "hidden_dim", cfg.model.out_dim) or cfg.model.out_dim)
        out_dim = int(getattr(cfg.model, "out_dim", hidden_dim) or hidden_dim)
        if self.use_layerwise and self._supports_layer_cache:
            self.repr_dim = hidden_dim * max(0, num_layers - 1) + out_dim
        else:
            self.repr_dim = out_dim

        self.local_d = FF(self.repr_dim)
        self.global_d = FF(self.repr_dim)
        self.prior_d = PriorDiscriminator(self.repr_dim) if self.use_prior else None

    @staticmethod
    def _align_last_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
        current_dim = int(x.size(-1))
        if current_dim == target_dim:
            return x
        if current_dim > target_dim:
            return x[:, :target_dim]
        pad = x.new_zeros((x.size(0), target_dim - current_dim))
        return torch.cat([x, pad], dim=-1)

    def _resolve_representations(self, model, node_repr, graph_repr, batch, data):
        local_repr = node_repr
        global_repr = graph_repr
        used_layerwise = False

        if self.use_layerwise and self._supports_layer_cache:
            layer_nodes = getattr(model, "cached_layer_node_reprs", None)
            if isinstance(layer_nodes, list) and len(layer_nodes) > 0:
                local_repr = torch.cat(layer_nodes, dim=-1)
                layer_graphs = getattr(model, "cached_layer_graph_reprs", None)
                if isinstance(layer_graphs, list) and len(layer_graphs) == len(layer_nodes) and len(layer_graphs) > 0:
                    global_repr = torch.cat(layer_graphs, dim=-1)
                else:
                    pooled = [pool_nodes(x=h, batch=batch, mode=self.graph_pooling, data=data) for h in layer_nodes]
                    global_repr = torch.cat(pooled, dim=-1)
                used_layerwise = True

        if global_repr is None:
            base_nodes = local_repr if used_layerwise else node_repr
            global_repr = pool_nodes(x=base_nodes, batch=batch, mode=self.graph_pooling, data=data)
        elif (not used_layerwise) and (self.graph_pooling != self.cfg.model.graph_pooling):
            global_repr = pool_nodes(x=node_repr, batch=batch, mode=self.graph_pooling, data=data)

        local_repr = self._align_last_dim(local_repr, self.repr_dim)
        global_repr = self._align_last_dim(global_repr, self.repr_dim)
        return local_repr, global_repr, used_layerwise

    def step(self, model, data, device):
        data = data.to(device)
        node_repr, graph_repr = model(data)
        batch = get_batch_vector(data)
        local_repr, global_repr, _ = self._resolve_representations(
            model=model,
            node_repr=node_repr,
            graph_repr=graph_repr,
            batch=batch,
            data=data,
        )

        g_enc = self.global_d(global_repr)
        l_enc = self.local_d(local_repr)
        mi_loss = local_global_loss(l_enc=l_enc, g_enc=g_enc, batch=batch, measure=self.measure)

        prior_reg = global_repr.sum() * 0.0
        if self.prior_d is not None:
            prior_reg = prior_loss(y=global_repr, prior_d=self.prior_d, gamma=self.gamma)

        loss = mi_loss + prior_reg
        logs = {
            "mi_loss": float(mi_loss.detach().item()),
        }
        if self.prior_d is not None:
            logs["prior_loss"] = float(prior_reg.detach().item())
        return loss, logs
