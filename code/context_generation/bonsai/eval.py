"""
Evaluate condensation quality (Bonsai outputs) via a simple protocol:

- Load full single-graph node dataset (PyG Data).
- Load condensed graph (PyG Data) produced by the unified context-generation runner.
- Train the same encoder+linear head on:
    (A) condensed graph (supervision only on condensed.target if present, else condensed.train_mask)
    (B) full graph baseline (supervision on full train split)
- Evaluate both models on the full graph test split.

This mirrors the typical "train on condensed, test on full" evaluation used by condensation works.

Note: this script is for validation only; it is not part of the main training pipeline.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from ogb.nodeproppred import Evaluator as OGBEvaluator

BONSAI_LARGE_HIDDEN_DATASETS = {"flickr", "ogbn-arxiv", "reddit"}

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from code.utils import append_csv_row, ensure_dir, format_split_for_name, project_path, resolve_project_path, set_seed  # noqa: E402
from code.data_loader.datasets import create_dataset  # noqa: E402
from code.data_loader.datasets import _get_or_create_split_indices  # type: ignore  # noqa: E402
from code.model import build_encoder_from_cfg  # noqa: E402
def _try_import_torch_sparse():
    try:
        from torch_sparse import SparseTensor, matmul  # type: ignore
        return SparseTensor, matmul
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(
            "torch-sparse is required for backend=sparse_gcn. "
            "If you recently installed a broken pyg-lib, uninstall it."
        ) from exc


def _bool_mask_from_idx(idx: torch.Tensor, n: int) -> torch.Tensor:
    mask = torch.zeros(n, dtype=torch.bool)
    mask[idx.to(dtype=torch.long)] = True
    return mask


def _select_mask_col(mask: torch.Tensor, col: int) -> torch.Tensor:
    """Handle datasets that provide multiple split columns, e.g. [N, S]."""
    m = torch.as_tensor(mask).to(dtype=torch.bool)
    if m.dim() == 1:
        return m
    if m.dim() == 2:
        c = int(col)
        c = max(0, min(c, m.size(1) - 1))
        return m[:, c]
    raise ValueError(f"Unsupported mask shape: {tuple(m.shape)}")


def _get_splits(
    dataset,
    data: Data,
    *,
    dataset_name: str,
    seed: int,
    split_root: str = "",
    split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    mask_col: int = 0,
    split_mode: str = "pretrain",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return boolean masks (train_mask, val_mask, test_mask) for a single-graph node dataset.
    split_mode:
      - pretrain: match PretrainRunner(make_loaders) for node-level single-graph datasets:
        use fixed split indices persisted under split_root keyed by dataset_name only.
      - auto: prefer official splits/masks; fallback to fixed split indices.
      - bonsai: match original Bonsai evaluation split (80/20 train-test, then 70/30 split of train into train/val).
    """
    n = int(getattr(data, "num_nodes", None) or data.x.size(0))
    mode = str(split_mode).lower().strip()
    if mode == "bonsai":
        from sklearn.model_selection import train_test_split

        train_all, test = train_test_split(range(n), test_size=0.2, random_state=42)
        rng = np.random.RandomState(seed=0)
        train = rng.choice(train_all, size=int(0.7 * len(train_all)), replace=False)
        val = list(set(range(n)) - set(train).union(set(test)))
        return (
            _bool_mask_from_idx(torch.as_tensor(train), n),
            _bool_mask_from_idx(torch.as_tensor(val), n),
            _bool_mask_from_idx(torch.as_tensor(test), n),
        )

    if mode == "pretrain":
        if not split_root:
            raise ValueError("split_root is required for split_mode=pretrain (use data/splits).")
        split_root_path = Path(split_root)
        ensure_dir(str(split_root_path))
        train_idx, val_idx, test_idx = _get_or_create_split_indices(
            dataset_name=dataset_name,
            split=split,
            seed=int(seed),
            split_root_path=split_root_path,
            total=n,
        )
        return (
            _bool_mask_from_idx(torch.as_tensor(train_idx), n),
            _bool_mask_from_idx(torch.as_tensor(val_idx), n),
            _bool_mask_from_idx(torch.as_tensor(test_idx), n),
        )

    # auto: prefer official/built-in splits first
    if hasattr(dataset, "get_idx_split"):
        split_dict = dataset.get_idx_split()
        if isinstance(split_dict, dict) and all(k in split_dict for k in ("train", "valid", "test")):
            train_idx = split_dict["train"]
            val_idx = split_dict["valid"]
            test_idx = split_dict["test"]
            return (
                _bool_mask_from_idx(train_idx, n),
                _bool_mask_from_idx(val_idx, n),
                _bool_mask_from_idx(test_idx, n),
            )

    # Fall back to masks on data (some datasets store multiple splits as [N, S]).
    has_all = all(hasattr(data, name) for name in ("train_mask", "val_mask", "test_mask"))
    if has_all:
        return (
            _select_mask_col(getattr(data, "train_mask"), mask_col),
            _select_mask_col(getattr(data, "val_mask"), mask_col),
            _select_mask_col(getattr(data, "test_mask"), mask_col),
        )

    # Last resort: create/persist a fixed split.
    if not split_root:
        # Deterministic in-memory split (not persisted).
        g = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(n, generator=g).tolist()
        tr = int(split[0] * n)
        va = int(split[1] * n)
        train_idx = perm[:tr]
        val_idx = perm[tr : tr + va]
        test_idx = perm[tr + va :]
    else:
        split_root_path = Path(split_root)
        ensure_dir(str(split_root_path))
        train_idx, val_idx, test_idx = _get_or_create_split_indices(
            dataset_name=dataset_name,
            split=split,
            seed=int(seed),
            split_root_path=split_root_path,
            total=n,
        )
    return (
        _bool_mask_from_idx(torch.as_tensor(train_idx), n),
        _bool_mask_from_idx(torch.as_tensor(val_idx), n),
        _bool_mask_from_idx(torch.as_tensor(test_idx), n),
    )

def _is_multilabel(y: torch.Tensor) -> bool:
    y = torch.as_tensor(y)
    return y.dim() == 2 and y.size(1) > 1

def _num_outputs_from_y(y: torch.Tensor) -> int:
    y = torch.as_tensor(y)
    if _is_multilabel(y):
        return int(y.size(1))
    # Single-label: ignore negative/unlabeled entries when inferring num_classes.
    y1 = y.reshape(-1)
    if y1.numel() == 0:
        raise ValueError("Empty y; cannot infer num_outputs.")
    if y1.dtype.is_floating_point:
        # Some datasets may store labels as float; cast safely.
        y1 = y1.to(torch.long)
    valid = y1 >= 0
    if not bool(valid.any().item()):
        raise ValueError("No non-negative labels found; cannot infer num_outputs.")
    return int(y1[valid].max().item() + 1)


def _valid_singlelabel_mask(y: torch.Tensor, num_outputs: int) -> torch.Tensor:
    """
    Return a boolean mask for entries that are valid class ids in [0, num_outputs).
    Handles y stored as float/long; flattens to [N].
    """
    y1 = torch.as_tensor(y).reshape(-1)
    if y1.dtype.is_floating_point:
        # Treat non-integer floats as invalid.
        y_int = y1.to(torch.long)
        valid_int = (y1 == y_int.to(y1.dtype))
    else:
        y_int = y1.to(torch.long)
        valid_int = torch.ones_like(y_int, dtype=torch.bool)
    return valid_int & (y_int >= 0) & (y_int < int(num_outputs))

def _ogb_metric_name(dataset_name: str) -> Optional[str]:
    if not dataset_name.startswith("ogbn-"):
        return None
    if dataset_name == "ogbn-proteins":
        return "rocauc"
    return "acc"

def _compute_metric(
    dataset_name: str,
    *,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> Tuple[str, float]:
    """
    Compute dataset-appropriate metric.
    - ogbn-proteins: ROC-AUC (OGB evaluator)
    - other ogbn-*: OGB acc evaluator
    - default: accuracy
    """
    y_true = y_true.detach().cpu()
    y_pred = y_pred.detach().cpu()

    metric = _ogb_metric_name(dataset_name)
    if metric and OGBEvaluator is not None:
        evaluator = OGBEvaluator(name=dataset_name)
        # OGB expects dict inputs; y_pred should be probabilities for rocauc, logits OK for acc (argmax).
        if metric == "rocauc":
            y_score = torch.sigmoid(y_pred)
            res = evaluator.eval({"y_true": y_true, "y_pred": y_score})
        else:
            # acc expects class indices shaped [N,1]
            y_hat = y_pred.argmax(dim=-1, keepdim=True)
            res = evaluator.eval({"y_true": y_true.view(-1, 1), "y_pred": y_hat})
        return metric, float(res.get(metric, list(res.values())[0]))

    # fallback: plain accuracy
    y_hat = y_pred.argmax(dim=-1)
    y = y_true.reshape(-1)
    return "acc", float((y_hat == y).float().mean().item())


def _metric_on_mask(
    dataset_name: str,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mask: torch.Tensor,
    num_outputs: int,
) -> float:
    m = mask.to(dtype=torch.bool)
    return _compute_metric(dataset_name, y_true=y_true[m], y_pred=y_pred[m])[1]

def _bce_with_nan_mask(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    BCEWithLogits for multi-label targets that may contain NaNs (e.g., ogbn-proteins).
    We ignore NaN entries in the loss.
    """
    targets = targets.to(torch.float32)
    valid = torch.isfinite(targets)
    if not bool(valid.any().item()):
        # No valid labels in this batch; return zero loss to avoid NaNs.
        return logits.sum() * 0.0
    # Replace NaNs with 0 just to satisfy BCE input requirements; mask them out in reduction.
    t = torch.nan_to_num(targets, nan=0.0)
    per_entry = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    per_entry = per_entry * valid.to(per_entry.dtype)
    return per_entry.sum() / valid.sum().to(per_entry.dtype)


def _load_condensed(condensed_root: Path, dataset_name: str, frac: float) -> Data:
    base = condensed_root / dataset_name / f"frac-{frac}"
    # Prefer processed (collated) if present, else raw list
    proc = base / "processed" / "data.pt"
    raw = base / "raw" / "data.pt"
    if proc.is_file():
        payload = torch.load(proc, map_location="cpu", weights_only=False)  # type: ignore[arg-type]
        # We save either (data, slices) or (data, None) depending on PyG version.
        if isinstance(payload, tuple) and len(payload) == 2:
            data_obj, slices = payload
            if slices is None:
                return data_obj
            out = Data()
            for key, item in data_obj.items():
                if torch.is_tensor(item):
                    s = slices[key]
                    if item.dim() == 0:
                        setattr(out, key, item)
                    else:
                        setattr(out, key, item[s[0] : s[1]])
                else:
                    setattr(out, key, item)
            return out
        if isinstance(payload, Data):
            return payload
        raise ValueError(f"Unexpected processed payload format at {proc}: {type(payload)}")
    if raw.is_file():
        lst = torch.load(raw, map_location="cpu", weights_only=False)  # type: ignore[arg-type]
        if not isinstance(lst, list) or not lst:
            raise ValueError(f"raw/data.pt does not contain a non-empty list: {raw}")
        return lst[0]
    raise FileNotFoundError(
        f"Condensed data not found under {base} (looked for processed/data.pt or raw/data.pt).\n"
        f"Tip: run condensation first, e.g.\n"
        f"  python run_context_generation.py context.method bonsai "
        f"context.bonsai.stage condense context.bonsai.dataset.name {dataset_name} "
        f"context.bonsai.out_root {condensed_root} context.bonsai.target_size_frac {frac}\n"
    )


def _resolve_condensed_dir(condensed_root: Path, dataset_name: str, frac: float, budget_mb: int) -> Path:
    """
    Resolve the exact condensed directory to evaluate.
    We prefer the new naming when budget_mb>0: budget{mb}mb_frac{frac}
    Falls back to frac-{frac}.
    """
    if int(budget_mb) > 0:
        cand = condensed_root / dataset_name / f"budget{int(budget_mb)}mb_frac{frac}"
        if cand.exists():
            return cand
    frac_dir = condensed_root / dataset_name / f"frac-{frac}"
    if frac_dir.exists():
        return frac_dir
    # last resort: try to find any matching dirs
    ds_dir = condensed_root / dataset_name
    if ds_dir.exists():
        for sub in sorted(ds_dir.iterdir()):
            if sub.is_dir() and f"frac{frac}" in sub.name.replace("-", ""):
                return sub
    return frac_dir


def _load_condensed_from_dir(base: Path) -> Data:
    proc = base / "processed" / "data.pt"
    raw = base / "raw" / "data.pt"
    if proc.is_file():
        payload = torch.load(proc, map_location="cpu", weights_only=False)  # type: ignore[arg-type]
        if isinstance(payload, tuple) and len(payload) == 2:
            data_obj, slices = payload
            if slices is None:
                return data_obj
            out = Data()
            for key, item in data_obj.items():
                if torch.is_tensor(item):
                    s = slices[key]
                    if item.dim() == 0:
                        setattr(out, key, item)
                    else:
                        setattr(out, key, item[s[0] : s[1]])
                else:
                    setattr(out, key, item)
            return out
        if isinstance(payload, Data):
            return payload
        raise ValueError(f"Unexpected processed payload format at {proc}: {type(payload)}")
    if raw.is_file():
        lst = torch.load(raw, map_location="cpu", weights_only=False)  # type: ignore[arg-type]
        if not isinstance(lst, list) or not lst:
            raise ValueError(f"raw/data.pt does not contain a non-empty list: {raw}")
        return lst[0]
    raise FileNotFoundError(f"Condensed data not found under {base} (processed/data.pt or raw/data.pt).")


def _train_on_graph(
    model: nn.Module,
    head: nn.Module,
    data: Data,
    train_mask: torch.Tensor,
    *,
    device: torch.device,
    lr: float,
    weight_decay: float,
    epochs: int,
) -> None:
    model.train()
    head.train()
    params = list(model.parameters()) + list(head.parameters())
    opt = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    data = data.to(device)
    train_mask = train_mask.to(device)
    y = data.y.reshape(-1).to(device)
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        node_repr, _ = model(data)
        logits = head(node_repr)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        opt.step()

def _train_on_graph_neighbor_exact(
    model: nn.Module,
    head: nn.Module,
    full: Data,
    train_mask: torch.Tensor,
    *,
    device: torch.device,
    lr: float,
    weight_decay: float,
    epochs: int,
    num_layers: int,
    batch_size: int,
    num_workers: int = 0,
) -> None:
    """
    Mini-batch training on a full graph using NeighborLoader with num_neighbors=-1 for each layer.
    This computes an exact receptive field for each seed node in the batch (no neighbor sampling).
    """
    model.train()
    head.train()
    params = list(model.parameters()) + list(head.parameters())
    opt = optim.Adam(params, lr=lr, weight_decay=weight_decay)

    full = full.to("cpu")  # NeighborLoader expects CPU graph; it will move batches to device.
    train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    loader = NeighborLoader(
        full,
        input_nodes=train_idx,
        num_neighbors=[-1] * int(num_layers),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=bool(device.type == "cuda"),
        persistent_workers=bool(int(num_workers) > 0),
    )
    for _ in range(int(epochs)):
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            batch = batch.to(device)
            node_repr, _ = model(batch)
            logits = head(node_repr)
            seed = int(getattr(batch, "batch_size", batch.num_nodes))
            y = batch.y[:seed].reshape(-1)
            loss = F.cross_entropy(logits[:seed], y)
            loss.backward()
            opt.step()

def _train_on_graph_neighbor_sampled(
    model: nn.Module,
    head: nn.Module,
    full: Data,
    train_mask: torch.Tensor,
    *,
    device: torch.device,
    lr: float,
    weight_decay: float,
    epochs: int,
    num_neighbors: list[int],
    batch_size: int,
    num_workers: int = 0,
) -> None:
    """
    Mini-batch training with neighbor sampling (fast, not exact full-batch).
    This is the standard scalable training baseline for large graphs.
    """
    model.train()
    head.train()
    params = list(model.parameters()) + list(head.parameters())
    opt = optim.Adam(params, lr=lr, weight_decay=weight_decay)

    full = full.to("cpu")
    train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    loader = NeighborLoader(
        full,
        input_nodes=train_idx,
        num_neighbors=[int(x) for x in num_neighbors],
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=bool(device.type == "cuda"),
        persistent_workers=bool(int(num_workers) > 0),
    )
    for _ in range(int(epochs)):
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            batch = batch.to(device)
            node_repr, _ = model(batch)
            logits = head(node_repr)
            seed = int(getattr(batch, "batch_size", batch.num_nodes))
            y = batch.y[:seed].reshape(-1)
            loss = F.cross_entropy(logits[:seed], y)
            loss.backward()
            opt.step()


def _train_fullbatch_bonsai(
    model: nn.Module,
    head: nn.Module,
    *,
    train_data: Data,
    val_data: Data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    dataset_name: str,
    num_outputs: int,
    device: torch.device,
    lr: float,
    weight_decay: float,
    epochs: int,
):
    """
    Bonsai-style full-batch training with validation on the full graph.
    Tracks the best validation metric and restores best weights.
    """
    model.train()
    head.train()
    params = list(model.parameters()) + list(head.parameters())
    opt = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    best_metric = float("-inf")
    best_state = None
    is_multilabel = _is_multilabel(val_data.y)
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        node_repr, _ = model(train_data)
        logits = head(node_repr)
        if is_multilabel:
            loss = _bce_with_nan_mask(logits[train_mask], train_data.y[train_mask])
        else:
            loss = F.cross_entropy(logits[train_mask], train_data.y.reshape(-1)[train_mask])
        loss.backward()
        opt.step()
        with torch.no_grad():
            model.eval()
            head.eval()
            node_repr_val, _ = model(val_data)
            logits_val = head(node_repr_val)
            metric_val = _metric_on_mask(
                dataset_name=dataset_name,
                y_true=val_data.y.reshape(-1) if not is_multilabel else val_data.y,
                y_pred=logits_val,
                mask=val_mask,
                num_outputs=num_outputs,
            )
            if metric_val > best_metric:
                best_metric = metric_val
                best_state = copy.deepcopy({"model": model.state_dict(), "head": head.state_dict()})
            model.train()
            head.train()
    if best_state:
        model.load_state_dict(best_state["model"])
        head.load_state_dict(best_state["head"])


@torch.no_grad()
def _eval_on_full(
    model: nn.Module,
    head: nn.Module,
    full: Data,
    test_mask: torch.Tensor,
    *,
    device: torch.device,
) -> float:
    model.eval()
    head.eval()
    d = full.to(device)
    node_repr, _ = model(d)
    logits = head(node_repr)
    pred = logits.argmax(dim=-1)
    m = test_mask.to(device)
    y = d.y.reshape(-1)
    return float((pred[m] == y[m]).float().mean().item())

@torch.no_grad()
def _eval_on_full_neighbor_exact(
    model: nn.Module,
    head: nn.Module,
    full: Data,
    test_mask: torch.Tensor,
    *,
    device: torch.device,
    num_layers: int,
    batch_size: int,
    num_workers: int = 0,
) -> float:
    """
    Exact test accuracy on the full graph, computed in mini-batches over test nodes.
    Uses NeighborLoader with num_neighbors=-1 at each layer (no sampling), so predictions
    for seed nodes are exact while avoiding full-batch GPU OOM.
    """
    model.eval()
    head.eval()

    full = full.to("cpu")
    test_idx = test_mask.nonzero(as_tuple=False).view(-1)
    loader = NeighborLoader(
        full,
        input_nodes=test_idx,
        num_neighbors=[-1] * int(num_layers),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(device.type == "cuda"),
        persistent_workers=bool(int(num_workers) > 0),
    )
    correct = 0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        node_repr, _ = model(batch)
        logits = head(node_repr)
        seed = int(getattr(batch, "batch_size", batch.num_nodes))
        pred = logits[:seed].argmax(dim=-1)
        y = batch.y[:seed].reshape(-1)
        correct += int((pred == y).sum().item())
        total += int(seed)
    return float(correct / max(1, total))


@torch.no_grad()
def _eval_on_full_neighbor_exact_autobatch(
    model: nn.Module,
    head: nn.Module,
    full: Data,
    test_mask: torch.Tensor,
    *,
    device: torch.device,
    num_layers: int,
    batch_size: int,
    num_workers: int = 0,
    min_batch_size: int = 64,
) -> float:
    """
    Try exact NeighborLoader-based evaluation with automatic batch size backoff on CUDA OOM.
    This keeps correctness (still exact for seed nodes) while improving robustness for large graphs.
    """
    bs = int(batch_size)
    bs = max(int(min_batch_size), bs)
    while True:
        try:
            return _eval_on_full_neighbor_exact(
                model,
                head,
                full,
                test_mask,
                device=device,
                num_layers=int(num_layers),
                batch_size=int(bs),
                num_workers=int(num_workers),
            )
        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda":
                raise
            torch.cuda.empty_cache()
            if bs <= int(min_batch_size):
                raise
            bs = max(int(min_batch_size), bs // 2)

def _gcn_norm_sparse_tensor(edge_index: torch.Tensor, num_nodes: int, device: torch.device):
    """Return normalized adjacency as torch_sparse.SparseTensor (with self-loops)."""
    SparseTensor, _ = _try_import_torch_sparse()
    row, col = edge_index[0].to(device), edge_index[1].to(device)
    # add self loops
    self_idx = torch.arange(num_nodes, device=device)
    row = torch.cat([row, self_idx], dim=0)
    col = torch.cat([col, self_idx], dim=0)
    deg = torch.bincount(row, minlength=num_nodes).to(torch.float32)
    deg = torch.clamp(deg, min=1.0)
    deg_inv_sqrt = deg.pow(-0.5)
    val = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return SparseTensor(row=row, col=col, value=val, sparse_sizes=(num_nodes, num_nodes))


class SparseGCN(nn.Module):
    """GCN using sparse SpMM (no message passing materialization)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj) -> torch.Tensor:
        _, matmul = _try_import_torch_sparse()
        h = matmul(adj, self.lin1(x))
        h = F.relu(h)
        h = self.dropout(h)
        out = matmul(adj, self.lin2(h))
        return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--dataset_root", default="data/datasets")
    p.add_argument("--condensed_root", default="data/datasets/bonsai")
    p.add_argument("--condensed_dir", default="", help="Explicit condensed directory to evaluate (overrides condensed_root/frac/budget resolution).")
    p.add_argument("--target_size_frac", type=float, default=0.01)
    p.add_argument("--budget_mb", type=int, default=0, help="Budget tag to select the correct condensed directory (must match condensation run).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--feat_dim", type=int, default=100)
    p.add_argument("--feat_reduction", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--backend", default="sparse_gcn", choices=["gcnconv", "sparse_gcn"])
    p.add_argument("--eval_mode", default="neighbor_exact", choices=["full", "neighbor_exact"])
    p.add_argument("--eval_batch_size", type=int, default=4096)
    p.add_argument("--eval_num_workers", type=int, default=0)
    p.add_argument("--baseline_train_mode", default="neighbor_sampled", choices=["full", "neighbor_exact", "neighbor_sampled"])
    p.add_argument("--baseline_train_batch_size", type=int, default=4096)
    p.add_argument("--baseline_train_num_workers", type=int, default=0)
    p.add_argument("--baseline_num_neighbors", default="15,10", help="Comma-separated fanouts for neighbor_sampled, e.g. '15,10' for 2-layer GCN.")
    # Condensed training mode (gcnconv backend). Large condensed graphs can OOM in full-batch GCNConv.
    p.add_argument(
        "--condensed_train_mode",
        default="auto",
        choices=["auto", "full", "neighbor_exact", "neighbor_sampled"],
        help="How to train on the condensed graph for backend=gcnconv. auto picks neighbor_sampled for large graphs.",
    )
    p.add_argument("--condensed_train_batch_size", type=int, default=4096)
    p.add_argument("--condensed_train_num_workers", type=int, default=0)
    p.add_argument("--condensed_num_neighbors", default="15,10", help="Comma-separated fanouts for condensed neighbor_sampled.")

    # model/training hyperparams (keep simple + explicit)
    p.add_argument("--model", default="gcn", choices=["gcn", "gat", "gin", "mlp"])
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--out_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=200)
    # Split handling for datasets without built-in masks (or with multiple split columns)
    p.add_argument("--split_root", default="", help="Where to persist fixed splits when train/val/test masks are missing.")
    p.add_argument("--split", default="0.8,0.1,0.1", help="Split ratios when masks are missing, e.g. 0.8,0.1,0.1")
    p.add_argument("--mask_col", type=int, default=0, help="For 2D masks [N,S], which split column to use.")
    p.add_argument("--split_mode", default="pretrain", choices=["pretrain", "auto", "bonsai"], help="Split logic: fixed split in split_root, auto (prefer official/built-in), or bonsai (80/20 then 70/30).")
    # Bonsai alignment preset
    p.add_argument("--bonsai_preset", action="store_true", help="Match Bonsai evaluation config (GCN, full-batch, 100 epochs, Bonsai split).")
    p.add_argument("--bonsai_runs", type=int, default=5, help="Number of repeated runs when bonsai_preset is enabled.")
    p.add_argument("--csv_out", default="", help="Append evaluation results to this CSV after each run.")
    args = p.parse_args()

    set_seed(int(args.seed))
    csv_path = resolve_project_path(args.csv_out) if args.csv_out else None
    csv_fields = [
        "dataset",
        "status",
        "message",
        "target_size_frac",
        "budget_mb",
        "backend",
        "metric",
        "acc_cond",
        "acc_full",
        "acc_cond_std",
        "acc_full_std",
        "gap",
        "cond_dir",
        "bonsai_preset",
        "runs",
    ]
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if args.bonsai_preset:
        args.backend = "gcnconv"
        args.model = "gcn"
        args.hidden_dim = 1024 if args.dataset in BONSAI_LARGE_HIDDEN_DATASETS else 128
        args.out_dim = args.hidden_dim
        args.epochs = 100
        args.baseline_train_mode = "neighbor_sampled"
        args.condensed_train_mode = "neighbor_sampled"
        args.eval_mode = "neighbor_exact"
        args.split_mode = "bonsai"
    # Always define metric_name for logging even if some branches return early/raise.
    metric_name: str = _ogb_metric_name(args.dataset) or "acc"

    ds_root = resolve_project_path(args.dataset_root)
    cd_root = resolve_project_path(args.condensed_root)

    # Load full dataset
    dataset = create_dataset(
        name=args.dataset,
        root=str(ds_root),
        task_level="node",
        induced=False,
        feat_reduction=bool(args.feat_reduction),
        feat_reduction_dim=int(args.feat_dim),
    )
    full = dataset[0]
    split_tuple = tuple(float(x) for x in str(args.split).split(","))
    split_tuple = split_tuple if len(split_tuple) == 3 else (0.8, 0.1, 0.1)
    split_tag = format_split_for_name(split_tuple)
    # Default split_root: use shared split root unless user overrides.
    split_root = args.split_root
    if not split_root:
        split_root = str(project_path("data", "splits"))
    train_mask, val_mask, test_mask = _get_splits(
        dataset,
        full,
        dataset_name=str(args.dataset),
        seed=int(args.seed),
        split_root=split_root,
        split=split_tuple,  # type: ignore[arg-type]
        mask_col=int(args.mask_col),
        split_mode=str(args.split_mode),
    )
    num_outputs = _num_outputs_from_y(full.y)

    # Load condensed graph (rigor: allow explicit directory)
    if args.condensed_dir:
        cond_dir = Path(args.condensed_dir)
        if not cond_dir.is_absolute():
            cond_dir = resolve_project_path(cond_dir)
    else:
        cond_dir = _resolve_condensed_dir(cd_root, args.dataset, float(args.target_size_frac), int(args.budget_mb))
    condensed = _load_condensed_from_dir(cond_dir)
    meta_path = cond_dir / "meta.pt"
    meta = None
    if meta_path.is_file():
        try:
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)  # type: ignore[arg-type]
        except Exception:
            meta = None
    features_used = None
    if isinstance(meta, dict):
        raw_features_used = meta.get("features_used")
        if raw_features_used is not None:
            features_used = [int(x) for x in raw_features_used]
    if features_used:
        full = full.clone()
        full.x = full.x[:, features_used]
    in_dim = int(full.num_node_features)

    # Determine condensed train mask
    c_n = int(getattr(condensed, "num_nodes", None) or condensed.x.size(0))
    c_train: Optional[torch.Tensor] = None
    if hasattr(condensed, "target") and condensed.target is not None:
        c_train = condensed.target.view(-1).bool()
    elif hasattr(condensed, "train_mask") and condensed.train_mask is not None:
        c_train = condensed.train_mask.view(-1).bool()
    else:
        c_train = torch.ones(c_n, dtype=torch.bool)

    # Basic sanity checks to avoid misleading results
    if getattr(condensed, "x", None) is None or getattr(condensed, "edge_index", None) is None:
        raise ValueError("Condensed graph missing x/edge_index.")
    if getattr(condensed, "y", None) is None:
        raise ValueError("Condensed graph missing y. (For leakage-safe pipeline, strip y only AFTER evaluation.)")

    # Some node datasets contain unlabeled nodes (e.g., y=-1). CrossEntropy cannot handle them.
    # Filter train/test masks to valid labels for single-label tasks.
    if not _is_multilabel(full.y):
        full_valid = _valid_singlelabel_mask(full.y, num_outputs=num_outputs)
        before_tr = int(train_mask.sum().item())
        before_va = int(val_mask.sum().item())
        before_te = int(test_mask.sum().item())
        train_mask = train_mask & full_valid
        val_mask = val_mask & full_valid
        test_mask = test_mask & full_valid
        after_tr = int(train_mask.sum().item())
        after_va = int(val_mask.sum().item())
        after_te = int(test_mask.sum().item())
        if (after_tr != before_tr) or (after_va != before_va) or (after_te != before_te):
            print(
                f"[Eval] label_validity(full): filtered_invalid "
                f"train={before_tr-after_tr} val={before_va-after_va} test={before_te-after_te} (num_outputs={num_outputs})"
            )
        if after_tr == 0 or after_te == 0 or after_va == 0:
            raise ValueError(f"No valid labeled nodes after filtering (train={after_tr}, val={after_va}, test={after_te}).")

        # Condensed graph validity (aligns with full label domain)
        cond_valid = _valid_singlelabel_mask(condensed.y, num_outputs=num_outputs)
        c_before = int(c_train.sum().item())
        c_train = c_train & cond_valid
        c_after = int(c_train.sum().item())
        if c_after != c_before:
            print(f"[Eval] label_validity(condensed): filtered_invalid train_like={c_before-c_after}")
        if getattr(condensed, "orig_nid", None) is not None:
            try:
                idx = condensed.orig_nid.to(torch.long)
                mapped = train_mask.to(idx.device)[idx]
                mapped = mapped.to(dtype=torch.bool)
                before_map = int(c_train.sum().item())
                c_train = c_train & mapped
                after_map = int(c_train.sum().item())
                if after_map != before_map:
                    print(f"[Eval] align_condensed_train: mapped_to_full_split dropped={before_map-after_map}")
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[Eval] align_condensed_train: failed to map orig_nid -> train_mask ({exc})")
        c_after = int(c_train.sum().item())
        if c_after == 0:
            raise ValueError("No valid labeled nodes in condensed train set after filtering/aligning.")

    # Bonsai-aligned evaluation shortcut (full-batch GCN, Bonsai splits/hparams).
    if args.bonsai_preset:
        if args.backend != "gcnconv":
            raise RuntimeError("bonsai_preset currently expects backend=gcnconv.")
        if _is_multilabel(full.y):
            raise RuntimeError("bonsai_preset assumes single-label classification (matches Bonsai).")

        class _Cfg:
            pass

        def _build_bonsai_encoder() -> nn.Module:
            cfg = _Cfg()
            cfg.model = _Cfg()
            cfg.model.name = "gcn"
            cfg.model.hidden_dim = int(args.hidden_dim)
            cfg.model.out_dim = int(args.out_dim)
            cfg.model.num_layers = int(args.num_layers)
            cfg.model.dropout = float(args.dropout)
            cfg.model.activation = "relu"
            cfg.model.graph_pooling = "mean"
            cfg.model.gat = _Cfg()
            cfg.model.gat.heads = 2
            return build_encoder_from_cfg(cfg=cfg, in_dim=in_dim).to(device)

        def _run_single_bonsai(seed_offset: int) -> Tuple[float, float]:
            set_seed(int(args.seed) + int(seed_offset))
            enc_a = _build_bonsai_encoder()
            head_a = nn.Linear(int(args.out_dim), num_outputs).to(device)
            _train_fullbatch_bonsai(
                model=enc_a,
                head=head_a,
                train_data=condensed,
                val_data=full,
                train_mask=c_train,
                val_mask=val_mask,
                dataset_name=args.dataset,
                num_outputs=num_outputs,
                device=device,
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                epochs=int(args.epochs),
            )
            enc_a.eval()
            head_a.eval()
            with torch.no_grad():
                node_repr_full, _ = enc_a(full.to(device))
                logits_full = head_a(node_repr_full)
                acc_cond = _metric_on_mask(
                    dataset_name=args.dataset,
                    y_true=full.y.reshape(-1),
                    y_pred=logits_full,
                    mask=test_mask,
                    num_outputs=num_outputs,
                )

            enc_b = _build_bonsai_encoder()
            head_b = nn.Linear(int(args.out_dim), num_outputs).to(device)
            _train_fullbatch_bonsai(
                model=enc_b,
                head=head_b,
                train_data=full,
                val_data=full,
                train_mask=train_mask,
                val_mask=val_mask,
                dataset_name=args.dataset,
                num_outputs=num_outputs,
                device=device,
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                epochs=int(args.epochs),
            )
            enc_b.eval()
            head_b.eval()
            with torch.no_grad():
                node_repr_full, _ = enc_b(full.to(device))
                logits_full = head_b(node_repr_full)
                acc_full = _metric_on_mask(
                    dataset_name=args.dataset,
                    y_true=full.y.reshape(-1),
                    y_pred=logits_full,
                    mask=test_mask,
                    num_outputs=num_outputs,
                )
            return float(acc_cond), float(acc_full)

        runs = max(1, int(args.bonsai_runs))
        cond_scores = []
        full_scores = []
        for i in range(runs):
            acc_c, acc_f = _run_single_bonsai(i)
            cond_scores.append(acc_c)
            full_scores.append(acc_f)
            print(f"[Eval-Bonsai] run={i} acc_cond={acc_c:.4f} acc_full={acc_f:.4f}")

        def _mean_std(scores: list[float]) -> Tuple[float, float]:
            arr = np.array(scores, dtype=float)
            return float(arr.mean()), float(arr.std())

        mean_c, std_c = _mean_std(cond_scores)
        mean_f, std_f = _mean_std(full_scores)
        print(f"[Eval-Bonsai] dataset={args.dataset} frac={args.target_size_frac} backend={args.backend}")
        print(f"[Eval-Bonsai] test_acc(train_on_condensed -> test_on_full)={mean_c:.4f}±{std_c:.4f}")
        print(f"[Eval-Bonsai] test_acc(train_on_full -> test_on_full)={mean_f:.4f}±{std_f:.4f}")
        print(f"[Eval-Bonsai] gap(full-cond)={mean_f-mean_c:.4f}")
        if csv_path is not None:
            append_csv_row(
                csv_path,
                fieldnames=csv_fields,
                row={
                    "dataset": args.dataset,
                    "status": "ok",
                    "message": "",
                    "target_size_frac": float(args.target_size_frac),
                    "budget_mb": int(args.budget_mb),
                    "backend": str(args.backend),
                    "metric": metric_name,
                    "acc_cond": mean_c,
                    "acc_full": mean_f,
                    "acc_cond_std": std_c,
                    "acc_full_std": std_f,
                    "gap": mean_f - mean_c,
                    "cond_dir": str(cond_dir),
                    "bonsai_preset": True,
                    "runs": int(runs),
                },
            )
        return 0

    # Prepare adjacency for sparse backend
    if args.backend == "sparse_gcn":
        full_adj = _gcn_norm_sparse_tensor(full.edge_index, int(full.num_nodes), device=device)
        cond_adj = _gcn_norm_sparse_tensor(condensed.edge_index, int(condensed.num_nodes), device=device)

    # Backend A: train on condensed, evaluate on full
    if args.backend == "sparse_gcn":
        model_a = SparseGCN(in_dim=in_dim, hidden_dim=int(args.hidden_dim), out_dim=num_outputs, dropout=float(args.dropout)).to(device)
        opt = optim.Adam(model_a.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        x_c = condensed.x.to(device)
        y_c = condensed.y.to(device)
        m_c = c_train.to(device)
        for _ in range(int(args.epochs)):
            model_a.train()
            opt.zero_grad(set_to_none=True)
            logits = model_a(x_c, cond_adj)
            if _is_multilabel(y_c):
                loss = _bce_with_nan_mask(logits[m_c], y_c[m_c])
            else:
                loss = F.cross_entropy(logits[m_c], y_c.reshape(-1)[m_c])
            loss.backward()
            opt.step()
        model_a.eval()
        logits_full = model_a(full.x.to(device), full_adj)
        m_test = test_mask.to(device)
        if _is_multilabel(full.y):
            y_true = full.y.to(device)[m_test].detach().cpu()
            y_pred = logits_full[m_test].detach().cpu()
        else:
            y_true = full.y.reshape(-1).to(device)[m_test].detach().cpu()
            y_pred = logits_full[m_test].detach().cpu()
        metric_name, acc_cond = _compute_metric(args.dataset, y_true=y_true, y_pred=y_pred)
    else:
        # gcnconv backend currently supports single-label node classification only.
        # (ogbn-proteins is multi-label; use backend=sparse_gcn there.)
        if _is_multilabel(full.y):
            raise RuntimeError("backend=gcnconv does not support multi-label y; use backend=sparse_gcn.")
        # Build encoder from cfg-like namespace (reuse existing builder)
        class _Cfg:
            pass
        cfg = _Cfg()
        cfg.model = _Cfg()
        cfg.model.name = args.model
        cfg.model.hidden_dim = int(args.hidden_dim)
        cfg.model.out_dim = int(args.out_dim)
        cfg.model.num_layers = int(args.num_layers)
        cfg.model.dropout = float(args.dropout)
        cfg.model.activation = "relu"
        cfg.model.graph_pooling = "mean"
        cfg.model.gat = _Cfg()
        cfg.model.gat.heads = 2

        enc_a = build_encoder_from_cfg(cfg=cfg, in_dim=in_dim).to(device)
        head_a = nn.Linear(cfg.model.out_dim, num_outputs).to(device)
        # Train on condensed graph. For large condensed graphs, full-batch GCNConv may OOM;
        # allow NeighborLoader-based training similar to the baseline.
        c_mode = str(args.condensed_train_mode).lower()
        if c_mode == "auto":
            c_nodes = int(condensed.num_nodes)
            c_edges = int(condensed.edge_index.size(1))
            c_mode = "full" if (c_nodes <= 50000 and c_edges <= 2_000_000) else "neighbor_sampled"
        if c_mode == "full":
            _train_on_graph(
                model=enc_a,
                head=head_a,
                data=condensed,
                train_mask=c_train,
                device=device,
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                epochs=int(args.epochs),
            )
        elif c_mode == "neighbor_exact":
            _train_on_graph_neighbor_exact(
                model=enc_a,
                head=head_a,
                full=condensed,
                train_mask=c_train,
                device=device,
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                epochs=int(args.epochs),
                num_layers=int(args.num_layers),
                batch_size=int(args.condensed_train_batch_size),
                num_workers=int(args.condensed_train_num_workers),
            )
        else:  # neighbor_sampled
            c_fanouts = [int(x.strip()) for x in str(args.condensed_num_neighbors).split(",") if x.strip()]
            if not c_fanouts:
                raise ValueError("condensed_num_neighbors must be a comma-separated list, e.g., '15,10'.")
            _train_on_graph_neighbor_sampled(
                model=enc_a,
                head=head_a,
                full=condensed,
                train_mask=c_train,
                device=device,
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                epochs=int(args.epochs),
                num_neighbors=c_fanouts,
                batch_size=int(args.condensed_train_batch_size),
                num_workers=int(args.condensed_train_num_workers),
            )
        if args.eval_mode == "full":
            try:
                acc_cond = _eval_on_full(enc_a, head_a, full, test_mask, device=device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                acc_cond = _eval_on_full(enc_a.to("cpu"), head_a.to("cpu"), full, test_mask, device=torch.device("cpu"))
        else:
            acc_cond = _eval_on_full_neighbor_exact_autobatch(
                enc_a,
                head_a,
                full,
                test_mask,
                device=device,
                num_layers=int(args.num_layers),
                batch_size=int(args.eval_batch_size),
                num_workers=int(args.eval_num_workers),
            )
        metric_name = "acc"

    # Model B: baseline train on full, evaluate on full
    if args.backend == "sparse_gcn":
        model_b = SparseGCN(in_dim=in_dim, hidden_dim=int(args.hidden_dim), out_dim=num_outputs, dropout=float(args.dropout)).to(device)
        opt = optim.Adam(model_b.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        x_f = full.x.to(device)
        y_f = full.y.to(device)
        m_train = train_mask.to(device)
        for _ in range(int(args.epochs)):
            model_b.train()
            opt.zero_grad(set_to_none=True)
            logits = model_b(x_f, full_adj)
            if _is_multilabel(y_f):
                loss = _bce_with_nan_mask(logits[m_train], y_f[m_train])
            else:
                loss = F.cross_entropy(logits[m_train], y_f.reshape(-1)[m_train])
            loss.backward()
            opt.step()
        model_b.eval()
        logits_full = model_b(x_f, full_adj)
        m_test = test_mask.to(device)
        if _is_multilabel(full.y):
            y_true = full.y.to(device)[m_test].detach().cpu()
            y_pred = logits_full[m_test].detach().cpu()
        else:
            y_true = full.y.reshape(-1).to(device)[m_test].detach().cpu()
            y_pred = logits_full[m_test].detach().cpu()
        metric_name, acc_full = _compute_metric(args.dataset, y_true=y_true, y_pred=y_pred)
    else:
        enc_b = build_encoder_from_cfg(cfg=cfg, in_dim=in_dim).to(device)
        head_b = nn.Linear(cfg.model.out_dim, num_outputs).to(device)
        if args.baseline_train_mode == "full":
            _train_on_graph(
                model=enc_b,
                head=head_b,
                data=full,
                train_mask=train_mask,
                device=device,
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                epochs=int(args.epochs),
            )
        elif args.baseline_train_mode == "neighbor_exact":
            _train_on_graph_neighbor_exact(
                model=enc_b,
                head=head_b,
                full=full,
                train_mask=train_mask,
                device=device,
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                epochs=int(args.epochs),
                num_layers=int(args.num_layers),
                batch_size=int(args.baseline_train_batch_size),
                num_workers=int(args.baseline_train_num_workers),
            )
        else:  # neighbor_sampled
            fanouts = [int(x.strip()) for x in str(args.baseline_num_neighbors).split(",") if x.strip()]
            if not fanouts:
                raise ValueError("baseline_num_neighbors must be a comma-separated list, e.g., '15,10'.")
            _train_on_graph_neighbor_sampled(
                model=enc_b,
                head=head_b,
                full=full,
                train_mask=train_mask,
                device=device,
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                epochs=int(args.epochs),
                num_neighbors=fanouts,
                batch_size=int(args.baseline_train_batch_size),
                num_workers=int(args.baseline_train_num_workers),
            )

        if args.eval_mode == "full":
            try:
                acc_full = _eval_on_full(enc_b, head_b, full, test_mask, device=device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                acc_full = _eval_on_full(enc_b.to("cpu"), head_b.to("cpu"), full, test_mask, device=torch.device("cpu"))
        else:
            acc_full = _eval_on_full_neighbor_exact_autobatch(
                enc_b,
                head_b,
                full,
                test_mask,
                device=device,
                num_layers=int(args.num_layers),
                batch_size=int(args.eval_batch_size),
                num_workers=int(args.eval_num_workers),
            )
        metric_name = "acc"

    # Report
    full_nodes = int(full.num_nodes)
    full_edges = int(full.edge_index.size(1))
    cond_nodes = int(condensed.num_nodes)
    cond_edges = int(condensed.edge_index.size(1))
    # Print a strict summary of what was evaluated for reproducibility.
    print(f"[Eval] dataset={args.dataset} tag_dir={cond_dir} metric={metric_name} backend={args.backend}")
    if int(args.budget_mb) > 0:
        print(f"[Eval] requested_budget_mb={int(args.budget_mb)} requested_frac={args.target_size_frac}")
    else:
        print(f"[Eval] requested_frac={args.target_size_frac}")
    if isinstance(meta, dict):
        b = meta.get('budget_bytes')
        u = meta.get('used_bytes_est')
        s = meta.get('num_seed_nodes')
        print(f"[Eval] cond_meta: budget_bytes={b} used_bytes_est={u} num_seed_nodes={s} neighbor_hops={meta.get('neighbor_hops')}")
    # critical counts used by training/eval
    try:
        print(f"[Eval] split_sizes(full): train={int(train_mask.sum().item())} test={int(test_mask.sum().item())}")
    except Exception:
        pass
    try:
        print(f"[Eval] split_sizes(condensed): train_like={int(c_train.sum().item())} total_nodes={int(c_n)}")
    except Exception:
        pass
    # label sanity (helps catch accidental empty/constant labels)
    try:
        y_train = condensed.y[c_train] if _is_multilabel(condensed.y) else condensed.y.reshape(-1)[c_train]
        if _is_multilabel(y_train):
            print(f"[Eval] condensed_train_labels: shape={tuple(y_train.shape)} pos_rate={float(y_train.float().mean().item()):.4f}")
        else:
            uniq = torch.unique(y_train.detach().cpu())
            print(f"[Eval] condensed_train_labels: num_unique={int(uniq.numel())} min={int(uniq.min().item())} max={int(uniq.max().item())}")
    except Exception:
        pass
    print(f"[Eval] full: nodes={full_nodes} edges={full_edges}")
    print(f"[Eval] cond: nodes={cond_nodes} edges={cond_edges} node_ratio={cond_nodes/full_nodes:.4f} edge_ratio={cond_edges/max(1,full_edges):.4f}")
    # NOTE: metric_name can be 'rocauc' for ogbn-proteins; keep logs consistent.
    print(f"[Eval] test_{metric_name}(train_on_condensed -> test_on_full)={acc_cond:.4f}")
    print(f"[Eval] test_{metric_name}(train_on_full -> test_on_full)={acc_full:.4f}")
    print(f"[Eval] gap(full-cond)={acc_full-acc_cond:.4f}")
    if csv_path is not None:
        append_csv_row(
            csv_path,
            fieldnames=csv_fields,
            row={
                "dataset": args.dataset,
                "status": "ok",
                "message": "",
                "target_size_frac": float(args.target_size_frac),
                "budget_mb": int(args.budget_mb),
                "backend": str(args.backend),
                "metric": metric_name,
                "acc_cond": float(acc_cond),
                "acc_full": float(acc_full),
                "acc_cond_std": "",
                "acc_full_std": "",
                "gap": float(acc_full - acc_cond),
                "cond_dir": str(cond_dir),
                "bonsai_preset": bool(args.bonsai_preset),
                "runs": 1,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
