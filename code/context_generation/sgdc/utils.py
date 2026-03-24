"""Utility helpers for the SGDC context-generation package."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional

import torch
from torch_geometric.data import Data


def ensure_x_feat_dim(data: Data, feat_dim: int) -> Optional[Data]:
    """Clone a graph and align `data.x` to the requested feature dimension."""
    if data is None:
        return None
    x = getattr(data, "x", None)
    if x is None:
        return None
    if x.dim() != 2 or x.size(0) <= 0 or x.size(1) <= 0:
        return None
    if not torch.is_floating_point(x):
        x = x.to(torch.float32)
    if x.size(1) != feat_dim:
        if x.size(1) > feat_dim:
            x = x[:, :feat_dim]
        else:
            pad = torch.zeros((x.size(0), feat_dim - x.size(1)), dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
    cloned = data.clone()
    cloned.x = x
    return cloned


def save_graph_list_as_pyg_dataset(
    graphs: Iterable[Data],
    *,
    out_root: str,
    meta: Mapping[str, object] | None = None,
) -> None:
    """
    Save condensed graphs in the simple directory layout used by the rest of the
    project:

    - `<out_root>/raw/data.pt`
    - `<out_root>/processed/data.pt`
    - `<out_root>/meta.pt`
    """

    out_dir = Path(out_root)
    raw_dir = out_dir / "raw"
    proc_dir = out_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    graph_list = [graph.clone() if hasattr(graph, "clone") else graph for graph in graphs]
    torch.save(graph_list, raw_dir / "data.pt")
    torch.save(graph_list, proc_dir / "data.pt")
    torch.save(dict(meta or {}), out_dir / "meta.pt")
