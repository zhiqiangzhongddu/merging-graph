"""
SGDC teacher pretraining for graph-level context generation.

This module is config-driven and intended to be invoked from
`run_context_generation.py` via `condensation_runner.py`.
"""

from __future__ import annotations

import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from code.data_loader.datasets import create_dataset
from code.utils import resolve_project_path, set_seed

from .aug import aug
from .src.model.sgdc_gin import G_GIN
from .utils import ensure_x_feat_dim


def _sanitize_graph(data):
    if data is None:
        return None
    x = getattr(data, "x", None)
    edge_index = getattr(data, "edge_index", None)
    batch = getattr(data, "batch", None)
    if x is None or edge_index is None:
        return None
    if x.numel() == 0 or edge_index.numel() == 0:
        return None
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        return None
    num_nodes = x.size(0)
    if num_nodes <= 0:
        return None
    if edge_index.min().item() < 0 or edge_index.max().item() >= num_nodes:
        mask = (
            (edge_index[0] >= 0)
            & (edge_index[0] < num_nodes)
            & (edge_index[1] >= 0)
            & (edge_index[1] < num_nodes)
        )
        if int(mask.sum()) == 0:
            return None
        data = data.clone()
        data.edge_index = edge_index[:, mask]
    if batch is None or batch.numel() != num_nodes:
        return None
    return data


def _subset_graphs(dataset, feat_dim: int) -> list:
    indices = getattr(dataset, "indices", None)
    parent = getattr(dataset, "dataset", None)
    if indices is not None and parent is not None:
        source = (ensure_x_feat_dim(parent[idx], feat_dim) for idx in indices)
    else:
        source = (ensure_x_feat_dim(dataset[idx], feat_dim) for idx in range(len(dataset)))
    return [graph for graph in source if graph is not None]


def run_sgdc_pretrain_dataset(
    *,
    dataset_name: str,
    pretrain_cfg,
    dataset_root: str,
    feat_reduction: bool,
    feat_dim: int,
    seed: int,
    device_id: int,
    save_ckpt: str,
) -> int:
    set_seed(int(seed))

    dataset = create_dataset(
        name=dataset_name,
        root=str(resolve_project_path(dataset_root)),
        task_level="graph",
        induced=False,
        feat_reduction=bool(feat_reduction),
        feat_reduction_dim=int(feat_dim),
    )
    train_graphs = _subset_graphs(dataset, int(feat_dim))
    if not train_graphs:
        raise RuntimeError(f"SGDC pretrain has no valid graphs for dataset '{dataset_name}'.")

    print(f"[SGDC][pretrain] dataset={dataset_name} using full dataset for self-supervised teacher pretrain")

    loader = DataLoader(
        train_graphs,
        batch_size=int(getattr(pretrain_cfg, "batch_size", 128)),
        shuffle=True,
    )

    device = torch.device(f"cuda:{int(device_id)}" if torch.cuda.is_available() else "cpu")
    train_model = str(getattr(pretrain_cfg, "train_model", "gin")).strip().lower()
    if train_model != "gin":
        raise NotImplementedError(
            f"SGDC pretrain currently supports only train_model='gin', got '{train_model}'."
        )
    input_dim = int(train_graphs[0].num_node_features)
    model = G_GIN(
        input_dim=input_dim,
        hidden_dim=int(getattr(pretrain_cfg, "nhid", 128)),
        output_dim=int(getattr(pretrain_cfg, "nhid", 128)),
        nconvs=int(getattr(pretrain_cfg, "layers", 3)),
        pooling=str(getattr(pretrain_cfg, "pooling", "mean")),
    ).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=float(getattr(pretrain_cfg, "lr", 1e-3)),
        weight_decay=float(getattr(pretrain_cfg, "weight_decay", 2e-5)),
    )

    model.train()
    epochs = int(getattr(pretrain_cfg, "epochs", 50))
    aug_mode = str(getattr(pretrain_cfg, "aug", "dnodes"))
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        steps = 0
        for batch in loader:
            if not torch.is_floating_point(batch.x):
                batch.x = batch.x.to(torch.float32)
            data, data_aug = aug(batch, aug_mode, is_syn=False)
            data = _sanitize_graph(data)
            data_aug = _sanitize_graph(data_aug)
            if data is None or data_aug is None:
                continue

            data = data.to(device)
            data_aug = data_aug.to(device)
            z = model(data.edge_index, data.x, data.batch)
            z_aug = model(data_aug.edge_index, data_aug.x, data_aug.batch)
            if data_aug.batch is None or data_aug.batch.numel() == 0:
                continue
            keep = torch.unique(data_aug.batch).sort().values
            if keep.numel() == 0:
                continue
            z = z[keep]
            z_aug = z_aug[keep]
            loss = model.loss_cal(z, z_aug)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            steps += 1

        avg_loss = total_loss / max(1, steps)
        print(f"[SGDC][pretrain] dataset={dataset_name} epoch={epoch}/{epochs} loss={avg_loss:.4f}")

    save_path = resolve_project_path(save_ckpt)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[SGDC][pretrain] saved checkpoint -> {save_path}")
    return 0
