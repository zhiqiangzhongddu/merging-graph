#!/usr/bin/env python3
"""
Convert AGAE datasets into AnyGraph-native matrix format.

Outputs per dataset:
- link task:
  - trn_mat.pkl / val_mat.pkl / tst_mat.pkl   (N x N sparse matrices)
  - feats.pkl                                  (optional, N x F)
- node task:
  - trn_mat.pkl                                (N+C x N+C sparse, graph + train label edges)
  - val_mat.pkl / tst_mat.pkl                  (N+C x C sparse label matrices)
  - feats.pkl                                  (optional, (N+C) x F, with class-node features)

Only datasets that resolve to single-graph node/edge tasks are converted.
"""

import argparse
import ast
import json
import pickle
import sys
from datetime import datetime
from numbers import Integral
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from code.data_loader.datasets import create_dataset, infer_task_level  # noqa: E402
from code.utils import (
    format_split_for_name,
    parse_csv_list,
    project_path,
    read_name_list_file,
    resolve_project_path,
)  # noqa: E402


def _parse_split(raw: str) -> Tuple[float, float, float]:
    vals = tuple(float(x.strip()) for x in raw.split(","))
    if len(vals) != 3:
        raise ValueError(f"Invalid split '{raw}'. Expected 'a,b,c'.")
    return vals  # type: ignore[return-value]


def _parse_split_value(value) -> Tuple[float, float, float]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Split definition cannot be empty.")
        try:
            value = ast.literal_eval(text)
        except Exception:
            return _parse_split(text)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Invalid split definition '{value}'. Expected 3 values.")
    return tuple(float(x) for x in value)  # type: ignore[return-value]


def _parse_split_list(raw: str) -> List[Tuple[float, float, float]]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        chunks = [chunk.strip() for chunk in text.split(";") if chunk.strip()]
        if len(chunks) > 1:
            return [_parse_split(chunk) for chunk in chunks]
        return [_parse_split(text)]
    if isinstance(parsed, (list, tuple)):
        if len(parsed) == 0:
            return []
        if len(parsed) == 3 and not isinstance(parsed[0], (list, tuple)):
            return [_parse_split_value(parsed)]
        return [_parse_split_value(item) for item in parsed]
    return [_parse_split_value(parsed)]


def _parse_int_list(raw: str) -> List[int]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        parsed = parse_csv_list(text)
    if isinstance(parsed, (list, tuple, set)):
        return [int(tok) for tok in parsed]
    return [int(parsed)]


def _add_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str = "") -> None:
    """
    Python 3.6-compatible replacement for argparse.BooleanOptionalAction.
    Creates paired flags: --<name> / --no-<name>.
    """
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--" + name, dest=dest, action="store_true", help=help_text)
    group.add_argument("--no-" + name, dest=dest, action="store_false")
    parser.set_defaults(**{dest: bool(default)})


def _split_suffix(portions: Tuple[float, float, float]) -> str:
    return "-".join(str(int(round(float(p) * 100))) for p in portions)


def _is_few_shot_split(split_def: Tuple[float, float, float]) -> bool:
    if not isinstance(split_def, (list, tuple)) or len(split_def) < 3:
        return False
    first = split_def[0]
    if isinstance(first, bool):
        return False
    try:
        first_val = float(first)
        shots_like = first_val.is_integer() and first_val >= 1.0
    except Exception:
        shots_like = isinstance(first, Integral) and not isinstance(first, bool)
    if not shots_like:
        return False
    try:
        val_ratio = float(split_def[1])
        test_ratio = float(split_def[2])
    except Exception:
        return False
    return val_ratio >= 0.0 and test_ratio >= 0.0 and (val_ratio + test_ratio) > 0.0


def _normalized_split_name(split: Tuple[float, float, float]) -> str:
    parts = tuple(float(item) for item in split)
    if _is_few_shot_split(parts):
        parts = (int(round(parts[0])), float(parts[1]), float(parts[2]))
    return str(format_split_for_name(parts))


def _split_file_tag(split: Tuple[float, float, float]) -> str:
    split_name = _normalized_split_name(split)
    if split_name.startswith("split"):
        return split_name[len("split") :]
    return split_name


def _split_alias_tag(split: Tuple[float, float, float]) -> str:
    split_name = _normalized_split_name(split)
    if split_name.startswith("split"):
        return f"split-{split_name[len('split') :]}"
    return split_name


def _converted_dataset_name(dataset_name: str, split: Tuple[float, float, float], seed: int) -> str:
    return f"{dataset_name}_seed{int(seed)}_{_split_alias_tag(split)}"


def _safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _row_l1_normalize(feats: np.ndarray) -> np.ndarray:
    if feats.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape={feats.shape}")
    feats = feats.astype(np.float32, copy=False)
    row_sum = feats.sum(axis=1, keepdims=True)
    nonzero = row_sum.squeeze(-1) > 0
    if np.any(nonzero):
        feats[nonzero] = feats[nonzero] / row_sum[nonzero]
    return feats


def _reduce_features_svd(feats: np.ndarray, out_dim: int) -> np.ndarray:
    if out_dim <= 0:
        return feats.astype(np.float32, copy=False)
    feats_t = torch.as_tensor(feats)
    if feats_t.dim() != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape={tuple(feats_t.shape)}")
    in_dim = int(feats_t.size(1))
    if in_dim == out_dim:
        return feats_t.detach().cpu().numpy().astype(np.float32, copy=False)
    if in_dim < out_dim:
        x_f = feats_t.to(torch.float32) if not torch.is_floating_point(feats_t) else feats_t
        pad = x_f.new_zeros(x_f.size(0), out_dim - in_dim)
        out = torch.cat([x_f, pad], dim=1)
        return out.detach().cpu().numpy().astype(np.float32, copy=False)
    try:
        u, s, _ = torch.linalg.svd(feats_t.float(), full_matrices=False)
        reduced = u[:, :out_dim] * s[:out_dim]
        return reduced.detach().cpu().numpy().astype(np.float32, copy=False)
    except Exception as exc:
        raise RuntimeError(
            f"SVD feature reduction failed for shape={tuple(feats_t.shape)} out_dim={out_dim}"
        ) from exc


def _to_1d_label(y: torch.Tensor) -> np.ndarray:
    if y is None:
        raise ValueError("Missing labels (y is None).")
    y_t = torch.as_tensor(y)
    if y_t.dim() > 1:
        if y_t.size(-1) != 1:
            raise ValueError(f"Expected single-label y, got shape={tuple(y_t.shape)}")
        y_t = y_t.view(-1)
    return y_t.detach().cpu().to(torch.long).numpy()


def _mask_to_bool(mask: Optional[torch.Tensor], n: int, mask_col: int) -> np.ndarray:
    if mask is None:
        return np.zeros((n,), dtype=bool)
    m = torch.as_tensor(mask).detach().cpu()
    if m.dtype != torch.bool:
        m = m.to(torch.bool)
    if m.dim() == 1:
        if m.numel() != n:
            raise ValueError(f"Mask length mismatch: got {m.numel()} expected {n}")
        return m.numpy()
    if m.dim() == 2:
        if m.size(0) != n:
            raise ValueError(f"2D mask first dim mismatch: got {m.size(0)} expected {n}")
        col = max(0, min(mask_col, m.size(1) - 1))
        return m[:, col].numpy()
    raise ValueError(f"Unsupported mask shape: {tuple(m.shape)}")


def _coo_from_edges(rows: np.ndarray, cols: np.ndarray, shape: Tuple[int, int]) -> sp.coo_matrix:
    if rows.size == 0:
        return sp.coo_matrix(shape, dtype=np.float32)
    data = np.ones(rows.shape[0], dtype=np.float32)
    mat = sp.coo_matrix((data, (rows, cols)), shape=shape, dtype=np.float32)
    mat.sum_duplicates()
    mat.data[:] = 1.0
    return mat


def _edge_index_from_data(data) -> np.ndarray:
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None:
        raise ValueError("Dataset has no edge_index.")
    ei = torch.as_tensor(edge_index).detach().cpu().to(torch.long)
    if ei.dim() != 2 or ei.size(0) != 2:
        raise ValueError(f"Invalid edge_index shape: {tuple(ei.shape)}")
    return ei.numpy()


def _save_sparse(path: Path, mat: sp.coo_matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(mat.tocoo(), f, protocol=pickle.HIGHEST_PROTOCOL)


def _save_feats(path: Path, feats: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(feats.astype(np.float32, copy=False), f, protocol=pickle.HIGHEST_PROTOCOL)


def _validate_single_graph_dataset(ds, name: str, task_level: str) -> None:
    if len(ds) != 1:
        raise ValueError(
            f"Only single-graph {task_level} datasets are supported for AnyGraph conversion; "
            f"dataset={name} len={len(ds)}"
        )


def _unique_sorted_edges(edge_index: np.ndarray, num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    if edge_index.shape[1] == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    rows = edge_index[0].astype(np.int64, copy=False)
    cols = edge_index[1].astype(np.int64, copy=False)
    keep = (rows >= 0) & (cols >= 0) & (rows < num_nodes) & (cols < num_nodes)
    rows = rows[keep]
    cols = cols[keep]
    if rows.size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    hashed = rows * int(num_nodes) + cols
    uniq = np.unique(hashed.astype(np.int64, copy=False))
    out_rows = (uniq // int(num_nodes)).astype(np.int64, copy=False)
    out_cols = (uniq % int(num_nodes)).astype(np.int64, copy=False)
    return out_rows, out_cols


def _as_int_list(values) -> Optional[List[int]]:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        return [int(v) for v in values.view(-1).tolist()]
    if isinstance(values, np.ndarray):
        return [int(v) for v in values.reshape(-1).tolist()]
    if isinstance(values, (list, tuple)):
        return [int(v) for v in values]
    return None


def _as_edge_index(values) -> Optional[np.ndarray]:
    if values is None:
        return np.empty((2, 0), dtype=np.int64)
    edge_index = torch.as_tensor(values, dtype=torch.long)
    if edge_index.numel() == 0:
        return np.empty((2, 0), dtype=np.int64)
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        return None
    return edge_index.detach().cpu().numpy().astype(np.int64, copy=False)


def _lookup_payload(payload: Mapping[str, object], *keys: str):
    for key in keys:
        if key in payload:
            return payload.get(key)
    return None


def _dedup_paths(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for p in paths:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _split_roots(split_root: Path, dataset_name: str) -> List[Path]:
    roots = [split_root, split_root / dataset_name]
    existing = [p for p in roots if p.is_dir()]
    missing = [p for p in roots if not p.is_dir()]
    return _dedup_paths(existing + missing)


def _find_split_file(
    roots: Sequence[Path],
    names: Sequence[str],
    globs: Sequence[str],
    *,
    split_kind: str,
) -> Optional[Path]:
    for name in names:
        matches_for_name: List[Path] = []
        for root in roots:
            path = root / name
            if path.is_file():
                matches_for_name.append(path)
        matches_for_name = _dedup_paths(matches_for_name)
        if not matches_for_name:
            continue
        if len(matches_for_name) == 1:
            return matches_for_name[0]
        raise RuntimeError(
            f"Ambiguous exact {split_kind} split files found for candidate '{name}': "
            f"{[str(p) for p in matches_for_name]}. Please keep only one exact match."
        )

    fallback_matches: List[Path] = []
    for root in roots:
        for pattern in globs:
            fallback_matches.extend(sorted(root.glob(pattern)))
    fallback_matches = _dedup_paths(fallback_matches)
    if fallback_matches:
        raise FileNotFoundError(
            f"No exact {split_kind} split file found; non-exact candidates exist: "
            f"{[str(p) for p in fallback_matches]}. "
            "Please rename to one of the expected exact filenames."
        )
    return None


def _resolve_node_split_file(
    *,
    split_root: Path,
    dataset_name: str,
    split: Tuple[float, float, float],
    seed: int,
) -> Optional[Path]:
    suffix = _split_file_tag(split)
    roots = _split_roots(split_root, dataset_name)
    names = [
        f"{dataset_name}_node_seed{int(seed)}_splits-{suffix}.pt",
        f"{dataset_name}_seed{int(seed)}_splits-{suffix}.pt",
        f"{dataset_name}_splits-{suffix}.pt",
    ]
    globs = [f"{dataset_name}*_splits-{suffix}.pt"]
    return _find_split_file(roots, names, globs, split_kind="node")


def _resolve_edge_split_file(
    *,
    split_root: Path,
    dataset_name: str,
    split: Tuple[float, float, float],
    seed: int,
) -> Optional[Path]:
    suffix = _split_suffix(split)
    neg_pct = int(round(float(split[0]) * 100))
    edge_name = dataset_name if dataset_name.endswith("_edge") else f"{dataset_name}_edge"
    seeded_edge_name = f"{dataset_name}_seed{int(seed)}_edge"
    roots = _split_roots(split_root, dataset_name)
    names = [
        f"{seeded_edge_name}_splits-{suffix}.pt",
        f"{seeded_edge_name}_splits-pos{suffix}-neg{neg_pct}.pt",
        f"{edge_name}_splits-{suffix}.pt",
        f"{edge_name}_splits-pos{suffix}-neg{neg_pct}.pt",
    ]
    globs = [
        f"{edge_name}*_splits-{suffix}.pt",
        f"{edge_name}*_splits-pos{suffix}-neg{neg_pct}.pt",
        f"{dataset_name}*_edge_splits-{suffix}.pt",
        f"{dataset_name}*_edge_splits-pos{suffix}-neg{neg_pct}.pt",
    ]
    return _find_split_file(roots, names, globs, split_kind="edge")


def _load_node_split_indices(path: Path, num_nodes: int) -> Tuple[List[int], List[int], List[int]]:
    payload = _safe_torch_load(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Node split payload is not a dict: {path}")

    train_idx = _as_int_list(_lookup_payload(payload, "train_indices", "train"))
    val_idx = _as_int_list(_lookup_payload(payload, "val_indices", "val"))
    test_idx = _as_int_list(_lookup_payload(payload, "test_indices", "test"))
    if train_idx is None or val_idx is None or test_idx is None:
        raise ValueError(f"Node split payload missing train/val/test indices: {path}")

    def _validate(indices: List[int], tag: str) -> List[int]:
        out = [int(idx) for idx in indices]
        bad = [idx for idx in out if idx < 0 or idx >= int(num_nodes)]
        if bad:
            raise ValueError(f"Node split file contains out-of-range {tag} indices in {path}: sample={bad[:5]}")
        return out

    train_idx = _validate(train_idx, "train")
    val_idx = _validate(val_idx, "val")
    test_idx = _validate(test_idx, "test")
    return train_idx, val_idx, test_idx


def _load_edge_split_payload(path: Path, total_edges: int) -> Dict[str, np.ndarray]:
    payload = _safe_torch_load(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Edge split payload is not a dict: {path}")

    out: Dict[str, np.ndarray] = {}
    for key in ("train_pos_idx", "val_pos_idx", "test_pos_idx", "message_pos_idx"):
        idx = _as_int_list(payload.get(key))
        if idx is None:
            raise ValueError(f"Edge split payload missing key={key}: {path}")
        out[key] = np.asarray(idx, dtype=np.int64)
        if out[key].size > 0:
            if int(out[key].min()) < 0 or int(out[key].max()) >= int(total_edges):
                raise ValueError(f"Edge split payload has out-of-range indices for key={key}: {path}")

    if (
        int(out["train_pos_idx"].size)
        + int(out["val_pos_idx"].size)
        + int(out["test_pos_idx"].size)
        + int(out["message_pos_idx"].size)
        != int(total_edges)
    ):
        raise ValueError(
            f"Edge split positive count mismatch in {path}: "
            f"train={out['train_pos_idx'].size} val={out['val_pos_idx'].size} "
            f"test={out['test_pos_idx'].size} msg={out['message_pos_idx'].size} total={total_edges}"
        )
    merged = np.concatenate(
        [
            out["train_pos_idx"],
            out["val_pos_idx"],
            out["test_pos_idx"],
            out["message_pos_idx"],
        ],
        axis=0,
    ).astype(np.int64, copy=False)
    if int(np.unique(merged).size) != int(total_edges):
        raise ValueError(
            f"Edge split payload indices are not a strict partition of edges (duplicates or missing) in {path}"
        )

    for key in ("train_neg_edge_index", "val_neg_edge_index", "test_neg_edge_index"):
        edge_index = _as_edge_index(payload.get(key))
        if edge_index is None:
            raise ValueError(f"Invalid edge index tensor for key={key}: {path}")
        out[key] = edge_index

    if out["train_neg_edge_index"].shape[1] != out["train_pos_idx"].size:
        raise ValueError(
            f"Edge split payload is not 1:1 for train split in {path}: "
            f"pos={out['train_pos_idx'].size} neg={out['train_neg_edge_index'].shape[1]}"
        )
    if out["val_neg_edge_index"].shape[1] != out["val_pos_idx"].size:
        raise ValueError(
            f"Edge split payload is not 1:1 for val split in {path}: "
            f"pos={out['val_pos_idx'].size} neg={out['val_neg_edge_index'].shape[1]}"
        )
    if out["test_neg_edge_index"].shape[1] != out["test_pos_idx"].size:
        raise ValueError(
            f"Edge split payload is not 1:1 for test split in {path}: "
            f"pos={out['test_pos_idx'].size} neg={out['test_neg_edge_index'].shape[1]}"
        )

    return out


class ConvertResult:
    def __init__(
        self,
        dataset: str,
        task: str,
        status: str,
        source_dataset: str = "",
        split: Optional[Tuple[float, float, float]] = None,
        seed: int = 0,
        out_dir: str = "",
        message: str = "",
        num_nodes: int = 0,
        num_edges_train: int = 0,
        num_edges_val: int = 0,
        num_edges_test: int = 0,
        num_classes: int = 0,
    ):
        self.dataset = dataset
        self.task = task
        self.status = status
        self.source_dataset = source_dataset or dataset
        self.split = tuple(float(item) for item in split) if split is not None else None
        self.seed = int(seed)
        self.out_dir = out_dir
        self.message = message
        self.num_nodes = int(num_nodes)
        self.num_edges_train = int(num_edges_train)
        self.num_edges_val = int(num_edges_val)
        self.num_edges_test = int(num_edges_test)
        self.num_classes = int(num_classes)

    def to_dict(self) -> Dict[str, object]:
        split_tag = _split_alias_tag(self.split) if self.split is not None else ""
        return {
            "dataset": self.dataset,
            "source_dataset": self.source_dataset,
            "split": list(self.split) if self.split is not None else [],
            "split_tag": split_tag,
            "seed": self.seed,
            "task": self.task,
            "status": self.status,
            "out_dir": self.out_dir,
            "message": self.message,
            "num_nodes": self.num_nodes,
            "num_edges_train": self.num_edges_train,
            "num_edges_val": self.num_edges_val,
            "num_edges_test": self.num_edges_test,
            "num_classes": self.num_classes,
        }


class AnyGraphConverter:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dataset_root = resolve_project_path(args.dataset_root)
        self.split_root = resolve_project_path(args.split_root)
        self.out_root = resolve_project_path(args.out_root)
        self.link_root = self.out_root / "link"
        self.node_root = self.out_root / "node"
        if not self.split_root.exists():
            raise FileNotFoundError(f"Split root does not exist: {self.split_root}")
        if not self.split_root.is_dir():
            raise NotADirectoryError(f"Split root is not a directory: {self.split_root}")
        self.link_root.mkdir(parents=True, exist_ok=True)
        self.node_root.mkdir(parents=True, exist_ok=True)

    def _reset_output_dir(self, out_dir: Path) -> None:
        if out_dir.is_dir():
            for item in out_dir.iterdir():
                if item.is_file():
                    item.unlink()
        else:
            out_dir.mkdir(parents=True, exist_ok=True)

    def convert_link(self, dataset_name: str, split: Tuple[float, float, float], seed: int, dataset_alias: str) -> ConvertResult:
        try:
            ds = create_dataset(
                name=dataset_name,
                root=str(self.dataset_root),
                task_level="edge",
                induced=False,
                feat_reduction=bool(self.args.feat_reduction),
                feat_reduction_dim=int(self.args.feat_dim),
            )
            _validate_single_graph_dataset(ds, dataset_name, "edge")
            data = ds[0]
            num_nodes = int(getattr(data, "num_nodes", 0) or 0)
            if num_nodes <= 0:
                raise ValueError("Invalid num_nodes from dataset.")
            edge_index = _edge_index_from_data(data)
            total_edges = int(edge_index.shape[1])
            split_path = _resolve_edge_split_file(
                split_root=self.split_root,
                dataset_name=dataset_name,
                split=split,
                seed=int(seed),
            )
            if split_path is None:
                raise FileNotFoundError(
                    f"Edge split payload not found for dataset={dataset_name}, split={split}, seed={seed}, "
                    f"split_root={self.split_root}"
                )
            split_payload = _load_edge_split_payload(split_path, total_edges=total_edges)

            all_rows = edge_index[0].astype(np.int64, copy=False)
            all_cols = edge_index[1].astype(np.int64, copy=False)

            def _rows_cols_from_indices(indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                if indices.size == 0:
                    return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
                if int(indices.max()) >= total_edges or int(indices.min()) < 0:
                    raise ValueError(f"Found edge index out of range in split payload: {split_path}")
                return all_rows[indices], all_cols[indices]

            train_all_idx = np.concatenate(
                [split_payload["train_pos_idx"], split_payload["message_pos_idx"]],
                axis=0,
            ).astype(np.int64, copy=False)

            trn_rows, trn_cols = _rows_cols_from_indices(train_all_idx)
            val_rows, val_cols = _rows_cols_from_indices(split_payload["val_pos_idx"])
            tst_rows, tst_cols = _rows_cols_from_indices(split_payload["test_pos_idx"])

            trn_mat = _coo_from_edges(trn_rows, trn_cols, (num_nodes, num_nodes))
            val_mat = _coo_from_edges(val_rows, val_cols, (num_nodes, num_nodes))
            tst_mat = _coo_from_edges(tst_rows, tst_cols, (num_nodes, num_nodes))

            out_dir = self.link_root / dataset_alias
            self._reset_output_dir(out_dir)
            _save_sparse(out_dir / "trn_mat.pkl", trn_mat)
            _save_sparse(out_dir / "val_mat.pkl", val_mat)
            _save_sparse(out_dir / "tst_mat.pkl", tst_mat)
            torch.save(
                {
                    "val_pos_edge_index": np.stack([val_rows, val_cols], axis=0).astype(np.int64, copy=False),
                    "val_neg_edge_index": split_payload["val_neg_edge_index"].astype(np.int64, copy=False),
                    "test_pos_edge_index": np.stack([tst_rows, tst_cols], axis=0).astype(np.int64, copy=False),
                    "test_neg_edge_index": split_payload["test_neg_edge_index"].astype(np.int64, copy=False),
                    "meta": {
                        "dataset": dataset_alias,
                        "source_dataset": dataset_name,
                        "split_file": str(split_path),
                        "edge_split": tuple(float(v) for v in split),
                        "split_seed": int(seed),
                    },
                },
                out_dir / str(self.args.edge_eval_payload_name),
            )

            if getattr(data, "x", None) is not None:
                feats = torch.as_tensor(data.x).detach().cpu().numpy().astype(np.float32, copy=False)
                if feats.shape[0] == num_nodes:
                    if bool(self.args.l1_normalize_features):
                        feats = _row_l1_normalize(feats)
                    _save_feats(out_dir / "feats.pkl", feats)

            meta = {
                "dataset": dataset_alias,
                "source_dataset": dataset_name,
                "task": "link",
                "source": "agae_conversion",
                "num_nodes": num_nodes,
                "num_edges_total": total_edges,
                "num_edges_train": int(trn_mat.nnz),
                "num_edges_val": int(val_mat.nnz),
                "num_edges_test": int(tst_mat.nnz),
                "edge_split": list(split),
                "split_seed": int(seed),
                "split_strategy": "edge_payload_train_plus_message",
                "split_file": str(split_path),
                "edge_eval_payload": str(out_dir / str(self.args.edge_eval_payload_name)),
            }
            (out_dir / "conversion_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return ConvertResult(
                dataset=dataset_alias,
                source_dataset=dataset_name,
                split=split,
                seed=seed,
                task="link",
                status="ok",
                out_dir=str(out_dir),
                num_nodes=num_nodes,
                num_edges_train=int(trn_mat.nnz),
                num_edges_val=int(val_mat.nnz),
                num_edges_test=int(tst_mat.nnz),
            )
        except Exception as exc:  # pylint: disable=broad-except
            return ConvertResult(
                dataset=dataset_alias,
                source_dataset=dataset_name,
                split=split,
                seed=seed,
                task="link",
                status="fail",
                message=str(exc),
            )

    def convert_node(self, dataset_name: str, split: Tuple[float, float, float], seed: int, dataset_alias: str) -> ConvertResult:
        try:
            ds = create_dataset(
                name=dataset_name,
                root=str(self.dataset_root),
                task_level="node",
                induced=False,
                feat_reduction=bool(self.args.feat_reduction),
                feat_reduction_dim=int(self.args.feat_dim),
            )
            _validate_single_graph_dataset(ds, dataset_name, "node")

            data = ds[0]
            num_nodes = int(getattr(data, "num_nodes", 0) or 0)
            if num_nodes <= 0:
                raise ValueError("Invalid num_nodes from dataset.")

            y = _to_1d_label(data.y)
            valid = y >= 0
            if not np.any(valid):
                raise ValueError("No non-negative labels found.")
            num_classes = int(y[valid].max() + 1)
            total_nodes = num_nodes + num_classes

            split_path = _resolve_node_split_file(
                split_root=self.split_root,
                dataset_name=dataset_name,
                split=split,
                seed=int(seed),
            )
            if split_path is None:
                raise FileNotFoundError(
                    f"Node split file not found for dataset={dataset_name}, split={split}, seed={seed}, "
                    f"split_root={self.split_root}"
                )
            if not bool(self.args.emit_node_val):
                raise ValueError("Strict node conversion requires --emit_node_val to keep val/test separated.")

            split_train_idx, split_val_idx, split_test_idx = _load_node_split_indices(split_path, num_nodes)
            for tag, indices in (
                ("train", split_train_idx),
                ("val", split_val_idx),
                ("test", split_test_idx),
            ):
                if len(indices) == 0:
                    continue
                idx_np = np.asarray(indices, dtype=np.int64)
                if not bool(np.all(valid[idx_np])):
                    raise ValueError(
                        f"Node split contains unlabeled or invalid {tag} indices for dataset={dataset_name}: {split_path}"
                    )

            train_mask = np.zeros((num_nodes,), dtype=bool)
            val_mask = np.zeros((num_nodes,), dtype=bool)
            test_mask = np.zeros((num_nodes,), dtype=bool)
            train_mask[np.asarray(split_train_idx, dtype=np.int64)] = True
            val_mask[np.asarray(split_val_idx, dtype=np.int64)] = True
            test_mask[np.asarray(split_test_idx, dtype=np.int64)] = True
            mask_source = "split_ckpt"
            coverage = float(np.sum(train_mask | val_mask | test_mask)) / float(max(np.sum(valid), 1))

            edge_index = _edge_index_from_data(data)
            keep = (
                (edge_index[0] >= 0)
                & (edge_index[1] >= 0)
                & (edge_index[0] < num_nodes)
                & (edge_index[1] < num_nodes)
            )
            graph_edges = edge_index[:, keep]

            trn_rows: List[np.ndarray] = [graph_edges[0].astype(np.int64, copy=False)]
            trn_cols: List[np.ndarray] = [graph_edges[1].astype(np.int64, copy=False)]

            train_nodes = np.where(train_mask)[0].astype(np.int64, copy=False)
            if train_nodes.size > 0:
                train_labels = y[train_nodes].astype(np.int64, copy=False)
                trn_rows.append(train_nodes)
                trn_cols.append((num_nodes + train_labels).astype(np.int64, copy=False))
                trn_rows.append((num_nodes + train_labels).astype(np.int64, copy=False))
                trn_cols.append(train_nodes)

            trn_row = np.concatenate(trn_rows) if trn_rows else np.empty((0,), dtype=np.int64)
            trn_col = np.concatenate(trn_cols) if trn_cols else np.empty((0,), dtype=np.int64)
            trn_mat = _coo_from_edges(trn_row, trn_col, (total_nodes, total_nodes))

            val_nodes = np.where(val_mask)[0].astype(np.int64, copy=False)
            val_labels = y[val_nodes].astype(np.int64, copy=False) if val_nodes.size > 0 else np.empty((0,), dtype=np.int64)
            val_mat = _coo_from_edges(val_nodes, val_labels, (total_nodes, num_classes))

            test_nodes = np.where(test_mask)[0].astype(np.int64, copy=False)
            test_labels = y[test_nodes].astype(np.int64, copy=False) if test_nodes.size > 0 else np.empty((0,), dtype=np.int64)
            tst_mat = _coo_from_edges(test_nodes, test_labels, (total_nodes, num_classes))

            out_dir = self.node_root / dataset_alias
            self._reset_output_dir(out_dir)
            _save_sparse(out_dir / "trn_mat.pkl", trn_mat)
            if bool(self.args.emit_node_val):
                _save_sparse(out_dir / "val_mat.pkl", val_mat)
            _save_sparse(out_dir / "tst_mat.pkl", tst_mat)

            x = getattr(data, "x", None)
            if x is not None:
                x_np = torch.as_tensor(x).detach().cpu().numpy().astype(np.float32, copy=False)
                if x_np.shape[0] == num_nodes:
                    node_feat_dim = int(self.args.node_output_feat_dim)
                    if node_feat_dim > 0:
                        x_np = _reduce_features_svd(x_np, node_feat_dim)
                    feat_dim = int(x_np.shape[1])
                    class_feats = np.zeros((num_classes, feat_dim), dtype=np.float32)
                    feats = np.concatenate([x_np, class_feats], axis=0)
                    _save_feats(out_dir / "feats.pkl", feats)

            meta = {
                "dataset": dataset_alias,
                "source_dataset": dataset_name,
                "task": "node",
                "source": "agae_conversion",
                "mask_source": mask_source,
                "split_file": str(split_path) if split_path is not None else "",
                "existing_mask_coverage": coverage,
                "num_nodes_real": num_nodes,
                "num_nodes_total": total_nodes,
                "num_classes": num_classes,
                "num_edges_graph": int(graph_edges.shape[1]),
                "num_train_labels": int(train_nodes.size),
                "num_val_labels": int(val_nodes.size),
                "num_test_labels": int(test_nodes.size),
                "node_split": list(split),
                "split_seed": int(seed),
                "node_output_feat_dim": int(self.args.node_output_feat_dim),
            }
            (out_dir / "conversion_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return ConvertResult(
                dataset=dataset_alias,
                source_dataset=dataset_name,
                split=split,
                seed=seed,
                task="node",
                status="ok",
                out_dir=str(out_dir),
                num_nodes=total_nodes,
                num_edges_train=int(trn_mat.nnz),
                num_edges_val=int(val_mat.nnz),
                num_edges_test=int(tst_mat.nnz),
                num_classes=num_classes,
            )
        except Exception as exc:  # pylint: disable=broad-except
            return ConvertResult(
                dataset=dataset_alias,
                source_dataset=dataset_name,
                split=split,
                seed=seed,
                task="node",
                status="fail",
                message=str(exc),
            )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert AGAE datasets into AnyGraph matrix format.")
    parser.add_argument("--dataset", default="", help="Comma-separated dataset names. Empty means use --dataset_file.")
    parser.add_argument(
        "--dataset_file",
        default=str(project_path("data", "available_node_datasets.tsv")),
        help="TSV file containing dataset names when --dataset is empty.",
    )
    parser.add_argument(
        "--task",
        default="auto",
        choices=["auto", "all", "node", "link"],
        help="Conversion mode: auto picks node/edge by infer_task_level.",
    )
    parser.add_argument("--dataset_root", default=str(project_path("data", "datasets")))
    parser.add_argument("--split_root", default=str(project_path("data", "splits")))
    parser.add_argument("--out_root", default=str(project_path("data", "anygraph_data")))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask_col", type=int, default=0)

    _add_bool_flag(parser, "feat_reduction", default=False)
    _add_bool_flag(
        parser,
        "l1_normalize_features",
        default=True,
        help_text="Apply row-wise L1 normalization for link-task feats.pkl.",
    )
    parser.add_argument("--feat_dim", type=int, default=100)

    parser.add_argument("--edge_split", default="0.8,0.1,0.1")
    parser.add_argument("--edge_splits", default="", help="Optional list of edge splits; empty falls back to --edge_split.")
    parser.add_argument(
        "--edge_eval_payload_name",
        default="agae_edge_eval_payload.pt",
        help="Filename (under each link dataset dir) for AGAE-style edge eval payload.",
    )

    parser.add_argument("--node_split", default="0.8,0.1,0.1")
    parser.add_argument("--node_splits", default="", help="Optional list of node splits; empty falls back to --node_split.")
    parser.add_argument("--seeds", default="", help="Comma-separated split seeds; empty falls back to --seed.")
    parser.add_argument("--node_output_feat_dim", type=int, default=128, help="Output feature dimension for node task; <=0 keeps original.")
    _add_bool_flag(parser, "emit_node_val", default=True, help_text="Emit val_mat.pkl for node task.")

    parser.add_argument("--index_out", default="", help="Optional explicit conversion index output path.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    args.edge_split = _parse_split(args.edge_split)
    args.node_split = _parse_split(args.node_split)
    args.edge_splits = _parse_split_list(args.edge_splits) or [args.edge_split]
    args.node_splits = _parse_split_list(args.node_splits) or [args.node_split]
    args.seeds = _parse_int_list(args.seeds) or [int(args.seed)]

    if args.dataset:
        datasets = parse_csv_list(args.dataset)
    else:
        dataset_file = resolve_project_path(args.dataset_file)
        if not dataset_file.is_file():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        datasets = read_name_list_file(dataset_file)

    converter = AnyGraphConverter(args)

    results: List[ConvertResult] = []
    for idx, name in enumerate(datasets, 1):
        inferred = infer_task_level(name)
        print(f"[{idx}/{len(datasets)}] dataset={name} inferred={inferred}")

        tasks: List[str] = []
        if args.task == "all":
            tasks = ["node", "link"]
        elif args.task == "node":
            tasks = ["node"]
        elif args.task == "link":
            tasks = ["link"]
        else:  # auto
            if inferred == "node":
                tasks = ["node"]
            elif inferred == "edge":
                tasks = ["link"]
            else:
                tasks = []

        if not tasks:
            results.append(
                ConvertResult(
                    dataset=name,
                    task="skip",
                    status="fail",
                    message=f"unsupported task inference: {inferred}",
                )
            )
            continue

        for task in tasks:
            split_defs = args.node_splits if task == "node" else args.edge_splits
            if not split_defs:
                results.append(
                    ConvertResult(
                        dataset=name,
                        source_dataset=name,
                        split=None,
                        seed=0,
                        task=task,
                        status="fail",
                        message=f"missing split definitions for task={task}",
                    )
                )
                continue
            for split_def in split_defs:
                for split_seed in args.seeds:
                    dataset_alias = _converted_dataset_name(name, split_def, int(split_seed))
                    if task == "node":
                        res = converter.convert_node(
                            name,
                            split=split_def,
                            seed=int(split_seed),
                            dataset_alias=dataset_alias,
                        )
                    else:
                        res = converter.convert_link(
                            name,
                            split=split_def,
                            seed=int(split_seed),
                            dataset_alias=dataset_alias,
                        )
                    print(
                        f"  - task={task} split={list(split_def)} seed={int(split_seed)} "
                        f"dataset={dataset_alias} status={res.status} out={res.out_dir or '-'} msg={res.message or '-'}"
                    )
                    results.append(res)

    ok = [r for r in results if r.status == "ok"]
    fail = [r for r in results if r.status == "fail"]
    skip = [r for r in results if r.status == "skip"]

    conversion_index = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "dataset_root": str(resolve_project_path(args.dataset_root)),
        "split_root": str(resolve_project_path(args.split_root)),
        "out_root": str(resolve_project_path(args.out_root)),
        "config": {
            "task": args.task,
            "seed": int(args.seed),
            "seeds": list(args.seeds),
            "feat_reduction": bool(args.feat_reduction),
            "l1_normalize_features": bool(args.l1_normalize_features),
            "feat_dim": int(args.feat_dim),
            "edge_splits": [list(split) for split in args.edge_splits],
            "edge_eval_payload_name": str(args.edge_eval_payload_name),
            "node_splits": [list(split) for split in args.node_splits],
            "node_output_feat_dim": int(args.node_output_feat_dim),
            "emit_node_val": bool(args.emit_node_val),
        },
        "results": [r.to_dict() for r in results],
        "ok_count": len(ok),
        "fail_count": len(fail),
        "skip_count": len(skip),
        "node_datasets": sorted({r.dataset for r in ok if r.task == "node"}),
        "link_datasets": sorted({r.dataset for r in ok if r.task == "link"}),
        "node_source_datasets": sorted({r.source_dataset for r in ok if r.task == "node"}),
        "link_source_datasets": sorted({r.source_dataset for r in ok if r.task == "link"}),
    }

    if args.index_out:
        index_path = resolve_project_path(args.index_out)
    else:
        index_path = resolve_project_path(args.out_root) / "conversion_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(conversion_index, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"Conversion done: ok={len(ok)} fail={len(fail)} skip={len(skip)}")
    print(f"Conversion index: {index_path}")
    print("=" * 72)

    return 0 if len(fail) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
