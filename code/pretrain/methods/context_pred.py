import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import k_hop_subgraph, subgraph

from code.model import build_encoder_from_cfg
from ..base import PretrainTask
from ..registry import register
from .utils import get_batch_vector


@register("context_pred")
class ContextPred(PretrainTask):
    """Substructure-context prediction (CBOW/skipgram), aligned with pretrain-gnns."""

    def __init__(self, cfg):
        super().__init__(cfg)
        cp_cfg = cfg.pretrain.context_pred
        self.mode = str(getattr(cp_cfg, "mode", "cbow") or "cbow").lower()
        self.context_pooling = str(getattr(cp_cfg, "context_pooling", "mean") or "mean").lower()
        self.neg_samples = int(getattr(cp_cfg, "neg_samples", 1) or 1)
        self.context_size = int(getattr(cp_cfg, "context_size", 3) or 3)
        self.k_hops = int(getattr(cp_cfg, "substruct_hops", 0) or 0)
        if self.k_hops <= 0:
            # pretrain-gnns default: k = num_layer
            self.k_hops = max(1, int(getattr(cfg.model, "num_layers", 1) or 1))
        # pretrain-gnns default: l1 = k-1, l2 = l1 + csize
        self.l1 = max(0, self.k_hops - 1)
        self.l2 = self.l1 + max(1, self.context_size)
        self.criterion = nn.BCEWithLogitsLoss()
        # Official implementation uses a separate context encoder.
        context_cfg = cfg.clone()
        context_cfg.model.num_layers = max(1, int(self.l2 - self.l1))
        self.context_encoder = build_encoder_from_cfg(cfg=context_cfg, in_dim=cfg.model.in_dim)

    def step(self, model, data, device):
        batch = get_batch_vector(data)
        edge_index = getattr(data, "edge_index", None)
        edge_attr = getattr(data, "edge_attr", None)
        if edge_index is None or edge_index.numel() == 0:
            return self._zero_output(model=model, device=device)

        x = data.x
        ptr = getattr(data, "ptr", None)
        if ptr is None or ptr.numel() <= 1:
            # Fallback for single-graph data objects.
            spans = [(0, int(getattr(data, "num_nodes", x.size(0))))]
        else:
            spans = [(int(ptr[i].item()), int(ptr[i + 1].item())) for i in range(int(ptr.numel()) - 1)]

        sub_data_list = []
        context_data_list = []
        sub_center_local_idx = []
        overlap_local_idx_list = []

        for start, end in spans:
            num_graph_nodes = int(end - start)
            if num_graph_nodes <= 1:
                continue

            in_graph = (
                (edge_index[0] >= start)
                & (edge_index[0] < end)
                & (edge_index[1] >= start)
                & (edge_index[1] < end)
            )
            graph_edge_index = edge_index[:, in_graph] - start
            graph_edge_attr = edge_attr[in_graph] if edge_attr is not None else None
            if graph_edge_index is None or graph_edge_index.numel() == 0:
                continue

            x_graph = x[start:end]
            root_idx = int(torch.randint(0, num_graph_nodes, (1,)).item())

            # k-hop rooted substructure
            sub_nodes, sub_edge_index, mapping, sub_edge_mask = k_hop_subgraph(
                node_idx=root_idx,
                num_hops=self.k_hops,
                edge_index=graph_edge_index,
                relabel_nodes=True,
                num_nodes=num_graph_nodes,
            )
            if sub_nodes.numel() == 0:
                continue
            sub_edge_attr = graph_edge_attr[sub_edge_mask] if graph_edge_attr is not None else None

            # Context nodes: symmetric difference between <=l1 and <=l2 neighborhoods.
            l1_nodes, _, _, _ = k_hop_subgraph(
                node_idx=root_idx,
                num_hops=self.l1,
                edge_index=graph_edge_index,
                relabel_nodes=False,
                num_nodes=num_graph_nodes,
            )
            l2_nodes, _, _, _ = k_hop_subgraph(
                node_idx=root_idx,
                num_hops=self.l2,
                edge_index=graph_edge_index,
                relabel_nodes=False,
                num_nodes=num_graph_nodes,
            )
            if l2_nodes.numel() == 0:
                continue
            if l1_nodes.numel() == 0:
                context_nodes = l2_nodes
            else:
                context_nodes = l2_nodes[~torch.isin(l2_nodes, l1_nodes)]
            if context_nodes.numel() == 0:
                continue
            context_nodes = torch.unique(context_nodes, sorted=True)

            context_edge_index, context_edge_attr = subgraph(
                subset=context_nodes,
                edge_index=graph_edge_index,
                edge_attr=graph_edge_attr,
                relabel_nodes=True,
                num_nodes=num_graph_nodes,
            )

            overlap_idx = torch.nonzero(torch.isin(context_nodes, sub_nodes), as_tuple=False).view(-1)
            if overlap_idx.numel() == 0:
                continue

            sub_data_kwargs = {
                "x": x_graph[sub_nodes],
                "edge_index": sub_edge_index,
            }
            if sub_edge_attr is not None:
                sub_data_kwargs["edge_attr"] = sub_edge_attr
            sub_data_list.append(Data(**sub_data_kwargs))
            sub_center_local_idx.append(int(mapping.view(-1)[0].item()))

            context_data_kwargs = {
                "x": x_graph[context_nodes],
                "edge_index": context_edge_index,
            }
            if context_edge_attr is not None:
                context_data_kwargs["edge_attr"] = context_edge_attr
            context_data_list.append(Data(**context_data_kwargs))
            overlap_local_idx_list.append(overlap_idx)

        if not sub_data_list:
            return self._zero_output(model=model, device=device)

        # Batch substructure/context graphs to avoid O(num_pairs) tiny forward passes.
        sub_batch = Batch.from_data_list(sub_data_list).to(device)
        context_batch = Batch.from_data_list(context_data_list).to(device)

        sub_node_rep, _ = model(sub_batch)
        context_node_rep, _ = self.context_encoder(context_batch)

        sub_ptr = getattr(sub_batch, "ptr", None)
        context_ptr = getattr(context_batch, "ptr", None)
        if sub_ptr is None or context_ptr is None:
            return self._zero_output(model=model, device=device)

        center_local = torch.as_tensor(sub_center_local_idx, dtype=torch.long, device=device)
        center_global = sub_ptr[:-1] + center_local
        substruct_rep = sub_node_rep[center_global]

        overlap_sizes = torch.as_tensor(
            [int(idx.numel()) for idx in overlap_local_idx_list],
            dtype=torch.long,
            device=device,
        )
        pair_ids = torch.arange(len(overlap_local_idx_list), device=device).repeat_interleave(overlap_sizes)
        context_offsets = context_ptr[:-1]
        overlap_global = torch.cat(
            [
                overlap_local_idx_list[i].to(device=device) + context_offsets[i]
                for i in range(len(overlap_local_idx_list))
            ],
            dim=0,
        )
        overlapped_cat = context_node_rep[overlap_global]

        if self.mode == "cbow":
            context_rep = self._pool_by_pair(
                x=overlapped_cat,
                pair_ids=pair_ids,
                mode=self.context_pooling,
            )
            neg_context_rep = torch.cat(
                [
                    context_rep[self._cycle_index(len(context_rep), i + 1, context_rep.device)]
                    for i in range(self.neg_samples)
                ],
                dim=0,
            )
            pred_pos = torch.sum(substruct_rep * context_rep, dim=1)
            pred_neg = torch.sum(substruct_rep.repeat((self.neg_samples, 1)) * neg_context_rep, dim=1)
        elif self.mode == "skipgram":
            expanded_substruct = substruct_rep[pair_ids]
            pred_pos = torch.sum(expanded_substruct * overlapped_cat, dim=1)

            shifted_expanded = []
            for i in range(self.neg_samples):
                shifted = substruct_rep[self._cycle_index(len(substruct_rep), i + 1, substruct_rep.device)]
                shifted_expanded.append(shifted[pair_ids])
            shifted_expanded = torch.cat(shifted_expanded, dim=0)
            pred_neg = torch.sum(
                shifted_expanded * overlapped_cat.repeat((self.neg_samples, 1)),
                dim=1,
            )
        else:
            raise ValueError(f"Invalid context_pred mode: {self.mode}")

        ones = torch.ones_like(pred_pos, device=pred_pos.device)
        zeros = torch.zeros_like(pred_neg, device=pred_neg.device)
        loss_pos = self.criterion(pred_pos, ones)
        loss_neg = self.criterion(pred_neg, zeros)
        loss = loss_pos + float(self.neg_samples) * loss_neg

        acc_pos = float((pred_pos > 0).float().mean().item()) if pred_pos.numel() > 0 else 0.0
        acc_neg = float((pred_neg < 0).float().mean().item()) if pred_neg.numel() > 0 else 0.0
        acc = 0.5 * (acc_pos + acc_neg)

        return loss, {
            "balanced_loss": float((loss_pos.detach() + loss_neg.detach()).item()),
            "train_acc": acc,
            "num_pairs": float(substruct_rep.size(0)),
        }

    @staticmethod
    def _cycle_index(num: int, shift: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(num, device=device)
        if num <= 0:
            return idx
        shift = int(shift) % int(num)
        if shift == 0:
            return idx
        return torch.cat([idx[shift:], idx[:shift]], dim=0)

    @staticmethod
    def _pool_context(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        if mode == "sum":
            return global_add_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))[0]
        if mode == "max":
            return global_max_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))[0]
        return global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))[0]

    @staticmethod
    def _pool_by_pair(x: torch.Tensor, pair_ids: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        if mode == "sum":
            return global_add_pool(x, pair_ids)
        if mode == "max":
            return global_max_pool(x, pair_ids)
        return global_mean_pool(x, pair_ids)

    def _zero_output(self, model, device):
        model_param = next(model.parameters(), None)
        if model_param is None:
            model_param = next(self.context_encoder.parameters(), None)
        zero = (
            model_param.sum() * 0.0
            if model_param is not None
            else torch.zeros((), device=device, requires_grad=True)
        )
        return zero, {"balanced_loss": 0.0, "train_acc": 0.0, "num_pairs": 0.0}
