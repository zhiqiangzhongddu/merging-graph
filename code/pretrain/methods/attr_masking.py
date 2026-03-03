import torch
import torch.nn.functional as F
from torch import nn

from ..base import PretrainTask
from ..registry import register


@register("attr_masking")
class AttrMasking(PretrainTask):
    """Attribute Masking pretraining task.

    Reference: Hu et al. "Strategies for Pre-training Graph Neural Networks" ICLR 2020.
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        am_cfg = cfg.pretrain.attr_masking
        self.mask_ratio = float(getattr(am_cfg, "mask_ratio", 0.15) or 0.15)
        self.mask_edge = bool(getattr(am_cfg, "mask_edge", False))
        self.edge_loss_weight = float(getattr(am_cfg, "edge_loss_weight", 1.0) or 1.0)
        self.node_vocab_size = int(getattr(am_cfg, "node_vocab_size", 0) or 0)
        self.edge_vocab_size = int(getattr(am_cfg, "edge_vocab_size", 0) or 0)

        self.node_reg_head = nn.Linear(cfg.model.out_dim, cfg.model.in_dim)
        self.node_cls_head = (
            nn.Linear(cfg.model.out_dim, self.node_vocab_size)
            if self.node_vocab_size > 0
            else None
        )
        self.edge_cls_head = (
            nn.Linear(cfg.model.out_dim, self.edge_vocab_size)
            if self.mask_edge and self.edge_vocab_size > 0
            else None
        )

    @staticmethod
    def _extract_class_targets(values: torch.Tensor, vocab_size: int):
        if values.dim() > 1:
            values = values[:, 0]
        raw = values.float()
        target = torch.round(raw).long()
        valid = torch.isfinite(raw)
        valid = valid & (target >= 0) & (target < int(vocab_size))
        return target, valid

    def step(self, model: nn.Module, data, device):
        data = data.to(device)

        num_nodes = data.num_nodes
        if num_nodes <= 0:
            model_param = next(model.parameters(), None)
            zero = (model_param.sum() * 0.0) if model_param is not None else torch.zeros((), device=device, requires_grad=True)
            return zero, {"masked": 0.0, "feature_dim": float(data.x.size(-1))}
        use_precomputed_nodes = (
            getattr(data, "masked_atom_indices", None) is not None
            and getattr(data, "mask_node_label", None) is not None
        )

        if use_precomputed_nodes:
            perm_all = torch.as_tensor(data.masked_atom_indices, dtype=torch.long, device=device).view(-1)
            valid_perm = (perm_all >= 0) & (perm_all < num_nodes)
            perm = perm_all[valid_perm]
            target_all = torch.as_tensor(data.mask_node_label, device=device)
            if target_all.size(0) == perm_all.numel():
                target = target_all[valid_perm]
            else:
                target = target_all[: perm.size(0)]
            if perm.numel() == 0:
                use_precomputed_nodes = False

        if not use_precomputed_nodes:
            num_mask = max(1, int(self.mask_ratio * num_nodes))
            perm = torch.randperm(num_nodes, device=device)[:num_mask]
            target = data.x[perm]
            corrupted = data.clone()
            corrupted.x = data.x.clone()
            corrupted.x[perm] = 0
            forward_data = corrupted
        else:
            forward_data = data

        num_mask = int(perm.numel())
        node_repr, _ = model(forward_data)
        masked_repr = node_repr[perm]

        node_loss = None
        node_acc = None

        if self.node_cls_head is not None:
            node_target, node_valid = self._extract_class_targets(target, self.node_vocab_size)
            if bool(node_valid.any().item()):
                logits = self.node_cls_head(masked_repr[node_valid])
                node_loss = F.cross_entropy(logits, node_target[node_valid])
                pred = logits.argmax(dim=-1)
                node_acc = float((pred == node_target[node_valid]).float().mean().item())

        if node_loss is None:
            pred = self.node_reg_head(masked_repr)
            node_loss = F.mse_loss(pred, target.float())

        loss = node_loss

        edge_acc = None
        if self.edge_cls_head is not None and getattr(data, "edge_index", None) is not None and getattr(data, "edge_attr", None) is not None:
            edge_index = data.edge_index
            src, dst = edge_index[0], edge_index[1]
            pre_connected = getattr(data, "connected_edge_indices", None)
            pre_edge_label = getattr(data, "mask_edge_label", None)

            if pre_connected is not None and pre_edge_label is not None:
                connected_all = torch.as_tensor(pre_connected, dtype=torch.long, device=device).view(-1)
                valid_connected = (connected_all >= 0) & (connected_all < edge_index.size(1))
                connected = connected_all[valid_connected]
                edge_target_raw_all = torch.as_tensor(pre_edge_label, device=device)
                if edge_target_raw_all.size(0) == connected_all.size(0):
                    edge_target_raw = edge_target_raw_all[valid_connected]
                else:
                    edge_target_raw = edge_target_raw_all[: connected.size(0)]
            else:
                node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
                node_mask[perm] = True
                connected = torch.nonzero(node_mask[src] | node_mask[dst], as_tuple=False).view(-1)
                if connected.numel() > 1:
                    # Most undirected PyG graphs duplicate each edge in two directions.
                    connected = connected[::2]
                edge_target_raw = data.edge_attr[connected]

            if connected.numel() > 0 and edge_target_raw.size(0) > 0:
                if edge_target_raw.size(0) != connected.size(0):
                    n = min(int(edge_target_raw.size(0)), int(connected.size(0)))
                    connected = connected[:n]
                    edge_target_raw = edge_target_raw[:n]
                edge_rep = node_repr[src[connected]] + node_repr[dst[connected]]
                edge_target, edge_valid = self._extract_class_targets(edge_target_raw, self.edge_vocab_size)
                if bool(edge_valid.any().item()):
                    edge_logits = self.edge_cls_head(edge_rep[edge_valid])
                    edge_loss = F.cross_entropy(edge_logits, edge_target[edge_valid])
                    loss = loss + self.edge_loss_weight * edge_loss
                    edge_pred = edge_logits.argmax(dim=-1)
                    edge_acc = float((edge_pred == edge_target[edge_valid]).float().mean().item())
        logs = {
            "masked": float(num_mask),
            "feature_dim": float(data.x.size(-1)),
        }
        if node_acc is not None:
            logs["node_acc"] = float(node_acc)
        if edge_acc is not None:
            logs["edge_acc"] = float(edge_acc)
        return loss, logs
