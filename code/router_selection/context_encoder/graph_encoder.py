"""Condensation graph encoder for dataset context embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_add_pool, global_max_pool, global_mean_pool


def _normalize_name(value: str) -> str:
    text = str(value or "").strip().lower()
    for token in (" ", "-", "/"):
        text = text.replace(token, "_")
    return text


def _choose_data_path(base_dir: Path, use_processed: bool) -> Optional[Path]:
    raw_path = base_dir / "raw" / "data.pt"
    if raw_path.exists():
        return raw_path
    if use_processed:
        proc_path = base_dir / "processed" / "data.pt"
        if proc_path.exists():
            return proc_path
    return None


def _select_latest_dir(dirs: Iterable[Path]) -> Optional[Path]:
    candidates = [d for d in dirs if d.is_dir()]
    if not candidates:
        return None
    try:
        return max(candidates, key=lambda d: d.stat().st_mtime)
    except OSError:
        return sorted(candidates)[-1]


def _resolve_bonsai_dataset_path(dataset_dir: Path, tag: str, use_processed: bool) -> Optional[Path]:
    if tag:
        candidate = dataset_dir / tag
        return _choose_data_path(candidate, use_processed)
    candidate = _select_latest_dir(dataset_dir.iterdir())
    if candidate is None:
        return None
    return _choose_data_path(candidate, use_processed)


def _resolve_sgdc_dataset_path(dataset_dir: Path, tag: str, use_processed: bool) -> Optional[Path]:
    tag = str(tag or "")
    tag_key = tag if tag.startswith("split-") else (f"split-{tag}" if tag else "")
    split_dirs: List[Path] = []
    for run_dir in dataset_dir.iterdir():
        if not run_dir.is_dir():
            continue
        for split_dir in run_dir.iterdir():
            if not split_dir.is_dir():
                continue
            if not split_dir.name.startswith("split-"):
                continue
            if tag_key and split_dir.name != tag_key:
                continue
            split_dirs.append(split_dir)
    chosen = _select_latest_dir(split_dirs)
    if chosen is None:
        return None
    return _choose_data_path(chosen, use_processed)


def build_condensation_index(root_dir: str, method: str, tag: str, use_processed: bool) -> Dict[str, Path]:
    """Map dataset name to condensation graph path (raw/processed data.pt)."""
    root = Path(root_dir)
    if not root.exists():
        return {}
    method = str(method or "bonsai").lower()
    index: Dict[str, Path] = {}
    for dataset_dir in root.iterdir():
        if not dataset_dir.is_dir():
            continue
        if method == "sgdc":
            path = _resolve_sgdc_dataset_path(dataset_dir, tag, use_processed)
        else:
            path = _resolve_bonsai_dataset_path(dataset_dir, tag, use_processed)
        if path is None:
            continue
        key = _normalize_name(dataset_dir.name)
        index[key] = path
    return index


def resolve_condensation_path(index: Dict[str, Path], dataset_name: str) -> Optional[Path]:
    key = _normalize_name(dataset_name)
    return index.get(key)


def load_condensed_graphs(path: Path) -> List[Data]:
    """Load list[Data] from a PyG-style data.pt."""
    if not path.exists():
        return []
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, list):
        return [g for g in payload if isinstance(g, Data)]
    if isinstance(payload, Data):
        return [payload]
    if isinstance(payload, tuple) and len(payload) == 2:
        data_obj, slices = payload
        try:
            from torch_geometric.data.separate import separate
        except Exception:
            return []
        keys = list(slices.keys())
        if not keys:
            return []
        count = int(slices[keys[0]].numel() - 1)
        graphs: List[Data] = []
        for idx in range(count):
            graphs.append(separate(Data, data_obj, idx, slices, decrement=True))
        return graphs
    return []


@dataclass
class GraphEncoderConfig:
    root_dir: str
    method: str = "bonsai"
    tag: str = ""
    use_processed: bool = False
    embed_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    pool: str = "mean"
    max_graphs: int = 0
    input_dim: int = 0
    device: Optional[str] = None


class GraphContextEncoder(nn.Module):
    """Encode condensed graphs into dataset embeddings."""

    def __init__(self, cfg: GraphEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_dim = int(cfg.embed_dim)
        self.hidden_dim = int(cfg.hidden_dim)
        self.num_layers = max(1, int(cfg.num_layers))
        self.dropout = float(cfg.dropout)
        self.pool = str(cfg.pool or "mean").lower()
        self.input_dim = int(cfg.input_dim)
        self._expected_dim: Optional[int] = None

        if self.input_dim > 0:
            self.node_proj = nn.Linear(self.input_dim, self.hidden_dim)
        else:
            self.node_proj = nn.LazyLinear(self.hidden_dim)

        self.convs = nn.ModuleList([GCNConv(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)])
        self.out_proj = nn.Linear(self.hidden_dim, self.embed_dim)

    def _resolve_input_dim(self, x_dim: int) -> int:
        if self.input_dim > 0:
            return self.input_dim
        if self._expected_dim is None:
            self._expected_dim = int(x_dim)
        return int(self._expected_dim)

    def _prepare_x(self, x: Optional[torch.Tensor], num_nodes: int, device: torch.device) -> torch.Tensor:
        if x is None:
            dim = self._resolve_input_dim(1)
            return torch.ones((num_nodes, dim), device=device, dtype=torch.float32)
        x = x.to(device=device, dtype=torch.float32)
        if x.dim() == 1:
            x = x.view(-1, 1)
        target_dim = self._resolve_input_dim(int(x.size(1)))
        if int(x.size(1)) < target_dim:
            pad = torch.zeros((x.size(0), target_dim - int(x.size(1))), device=device, dtype=torch.float32)
            x = torch.cat([x, pad], dim=1)
        elif int(x.size(1)) > target_dim:
            x = x[:, :target_dim]
        return x

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pool == "sum":
            return global_add_pool(x, batch)
        if self.pool == "max":
            return global_max_pool(x, batch)
        return global_mean_pool(x, batch)

    def forward(self, batch: Batch) -> torch.Tensor:
        if batch.num_nodes == 0:
            return torch.zeros((batch.num_graphs, self.embed_dim), device=batch.batch.device)
        x = self._prepare_x(getattr(batch, "x", None), int(batch.num_nodes), batch.batch.device)
        x = self.node_proj(x)
        edge_index = getattr(batch, "edge_index", None)
        for conv in self.convs:
            if edge_index is None or edge_index.numel() == 0:
                break
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        pooled = self._pool(x, batch.batch)
        return self.out_proj(pooled)

    def encode_graphs(self, graphs: List[Data], device: Optional[torch.device] = None) -> torch.Tensor:
        if not graphs:
            return torch.empty((0, self.embed_dim), dtype=torch.float32)
        batch = Batch.from_data_list(graphs)
        if device is not None:
            batch = batch.to(device)
        return self.forward(batch)

    def encode_dataset_graphs(
        self,
        dataset_graphs: List[List[Data]],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        num_sets = len(dataset_graphs)
        if num_sets == 0:
            return torch.empty((0, self.embed_dim), dtype=torch.float32)
        flat: List[Data] = []
        owners: List[int] = []
        max_graphs = int(self.cfg.max_graphs or 0)
        for idx, graphs in enumerate(dataset_graphs):
            if not graphs:
                continue
            if max_graphs > 0:
                graphs = graphs[:max_graphs]
            for graph in graphs:
                flat.append(graph)
                owners.append(idx)
        if not flat:
            device = device or torch.device("cpu")
            return torch.zeros((num_sets, self.embed_dim), device=device, dtype=torch.float32)
        emb = self.encode_graphs(flat, device=device)
        device = emb.device
        out = torch.zeros((num_sets, self.embed_dim), device=device, dtype=torch.float32)
        counts = torch.zeros((num_sets, 1), device=device, dtype=torch.float32)
        owner_idx = torch.tensor(owners, device=device, dtype=torch.long)
        out.index_add_(0, owner_idx, emb)
        counts.index_add_(0, owner_idx, torch.ones((emb.size(0), 1), device=device, dtype=torch.float32))
        out = out / counts.clamp(min=1.0)
        return out
