"""Dataset splitting utilities for node-level, edge-level, and graph-level datasets, including support for fixed random splits, few-shot splits, and edge splits with negative sampling. Provides functions to create or load split indices and payloads, ensuring reproducibility and efficient caching of splits on disk."""

import re
from numbers import Integral
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

from code.utils import ensure_dir

from .dataset_metadata import resolve_count_split_strategy
from .dataset_paths import _split_dataset_dir
from .filter_empty_graph import _graph_filter_cache_meta_for_dataset
from .utils import safe_torch_load


def _split_origin(status: str, path: Path | None = None) -> Dict[str, Optional[str]]:
    return {
        "status": str(status),
        "path": str(path) if path is not None else None,
    }


def _maybe_return_split_indices(
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    split_meta: Dict[str, Optional[str]],
    return_split_meta: bool,
):
    if return_split_meta:
        return train_idx, val_idx, test_idx, dict(split_meta)
    return train_idx, val_idx, test_idx


def _maybe_return_split_payload(
    payload: Dict,
    split_meta: Dict[str, Optional[str]],
    return_split_meta: bool,
):
    if return_split_meta:
        return payload, dict(split_meta)
    return payload


def _canonical_split_dataset_name(dataset_name: str, task_level: str, seed: int) -> str:
    """
    Canonical split file stem:
      - node/graph: <base>_<task>_seed<seed>
      - edge:       <base>_edge_seed<seed>
    """
    name = str(dataset_name).strip()
    if not name:
        return name

    task = str(task_level or "").lower()
    if task not in {"node", "graph", "edge"}:
        return name

    tagged = re.match(r"^(?P<base>.+)_(?P<tag>node|graph|edge)_seed(?P<seed>\d+)$", name)
    if tagged:
        return name

    if task == "edge" and name.endswith("_edge"):
        return f"{name}_seed{seed}"

    for marker in ("_node_seed", "_graph_seed", "_edge_seed", "_seed"):
        if marker in name:
            base = name.split(marker, 1)[0]
            return f"{base}_{task}_seed{seed}" if task != "edge" else f"{base}_edge_seed{seed}"

    return f"{name}_{task}_seed{seed}" if task != "edge" else f"{name}_edge_seed{seed}"


def _split_suffix(portions: Tuple[float, float, float]) -> str:
    train, val, test = portions
    if _is_few_shot_split_def(portions):
        return f"{int(float(train))}-{int(float(val) * 100)}-{int(float(test) * 100)}"
    return "-".join(str(int(float(p) * 100)) for p in portions)


def _validate_split_def(split_def: Tuple[float, float, float]) -> None:
    if split_def is None or len(split_def) != 3:
        raise ValueError("split must be a 3-tuple/list: (train, val, test)")
    train, val, test = split_def
    values = [float(train), float(val), float(test)]
    if any(v < 0 for v in values):
        raise ValueError(f"split values must be non-negative, got {split_def}")

    first = values[0]
    is_count_split = float(first).is_integer()
    if is_count_split:
        if int(first) < 0:
            raise ValueError(f"train count must be >= 0, got {split_def}")
        if values[1] + values[2] > 1.0 + 1e-8:
            raise ValueError(f"val/test ratios must sum to <= 1 for count splits, got {split_def}")
        return

    total = sum(values)
    if not (0.0 < total <= 1.0 + 1e-8):
        raise ValueError(f"ratio split must sum to <= 1, got {split_def}")
    if total <= 0.0:
        raise ValueError(f"ratio split must allocate at least one item, got {split_def}")


def _validate_edge_split_def(split_def: Tuple[float, float, float]) -> None:
    if split_def is None or len(split_def) != 3:
        raise ValueError("edge split must be a 3-tuple/list: (train, val, test)")
    train, val, test = [float(v) for v in split_def]
    if train < 0 or val < 0 or test < 0:
        raise ValueError(f"edge split values must be non-negative, got {split_def}")
    total = train + val + test
    if total > 1.0 + 1e-8:
        raise ValueError(
            "edge split ratios reserve train/val/test positives from the full edge set; "
            f"their sum must be <= 1, got {split_def}"
        )


def _is_few_shot_split_def(split_def) -> bool:
    if not isinstance(split_def, (list, tuple)) or not split_def:
        return False
    first = split_def[0]
    if isinstance(first, bool):
        return False
    if isinstance(first, Integral):
        return True
    return isinstance(first, float) and first.is_integer()


def _mask_to_node_indices(mask, name: str) -> List[int]:
    if mask is None:
        raise ValueError(f"{name} is required but missing.")
    mask_tensor = torch.as_tensor(mask)
    if mask_tensor.dtype != torch.bool:
        raise ValueError(f"{name} must be boolean.")
    if mask_tensor.dim() == 1:
        return mask_tensor.nonzero(as_tuple=False).view(-1).tolist()
    if mask_tensor.dim() == 2:
        if mask_tensor.size(1) != 1:
            raise ValueError(f"{name} must be 1D or Nx1 boolean mask.")
        return mask_tensor[:, 0].nonzero(as_tuple=False).view(-1).tolist()
    raise ValueError(f"{name} must be 1D or Nx1 boolean mask.")


def _load_existing_indices(path: Path, expected_total: int, expected_meta: Dict | None = None):
    if not path.is_file():
        return None
    try:
        payload = safe_torch_load(path)
        train_idx = payload.get("train_indices") or payload.get("train")
        val_idx = payload.get("val_indices") or payload.get("val")
        test_idx = payload.get("test_indices") or payload.get("test")
        if not all(isinstance(idx, (list, tuple)) for idx in (train_idx, val_idx, test_idx)):
            return None
        if len(train_idx) + len(val_idx) + len(test_idx) != expected_total:
            return None
        if expected_meta:
            meta = payload.get("meta", {})
            if not isinstance(meta, dict):
                return None
            for key, value in expected_meta.items():
                if meta.get(key) != value:
                    return None
        return list(train_idx), list(val_idx), list(test_idx)
    except Exception:
        return None


def _get_or_create_split_indices(
    dataset_name: str,
    split: Tuple[float, float, float],
    seed: int,
    split_root_path: Path,
    total: int,
    extra_meta: Dict | None = None,
    return_split_meta: bool = False,
):
    """Load existing split indices or create and persist new ones."""
    if split_root_path is None:
        raise ValueError("split_root_path must be provided (configure cfg.dataset.split_root).")

    dataset_split_dir = _split_dataset_dir(split_root_path, dataset_name)
    split_path = dataset_split_dir / f"{dataset_name}_splits-{_split_suffix(split)}.pt"

    existing = _load_existing_indices(split_path, total, expected_meta=extra_meta)
    if existing:
        train_idx, val_idx, test_idx = existing
        print(f"[Dataset split] Loaded fixed split from {split_path}")
        return _maybe_return_split_indices(
            train_idx,
            val_idx,
            test_idx,
            _split_origin("loaded", split_path),
            return_split_meta,
        )

    train_len = int(split[0] * total)
    val_len = int(split[1] * total)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total, generator=generator).tolist()
    train_idx = perm[:train_len]
    val_idx = perm[train_len : train_len + val_len]
    test_idx = perm[train_len + val_len :]

    ensure_dir(str(dataset_split_dir))
    payload = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
        "meta": {
            "dataset_name": dataset_name,
            "total": total,
            "split": split,
            "seed": seed,
            **(extra_meta or {}),
        },
    }
    torch.save(payload, split_path)
    print(f"[Dataset split] Saved fixed split to {split_path}")
    return _maybe_return_split_indices(
        train_idx,
        val_idx,
        test_idx,
        _split_origin("saved", split_path),
        return_split_meta,
    )


def _get_or_create_split_indices_subset(
    dataset_name: str,
    split: Tuple[float, float, float],
    seed: int,
    split_root_path: Path,
    subset_indices: List[int],
    extra_meta: Dict | None = None,
    return_split_meta: bool = False,
):
    """Create or load split indices for a subset of nodes."""
    dataset_split_dir = _split_dataset_dir(split_root_path, dataset_name)
    split_path = dataset_split_dir / f"{dataset_name}_splits-{_split_suffix(split)}.pt"
    existing = _load_existing_indices(split_path, len(subset_indices), expected_meta=extra_meta)
    if existing:
        train_idx, val_idx, test_idx = existing
        print(f"[Dataset split] Loaded fixed split from {split_path}")
        return _maybe_return_split_indices(
            train_idx,
            val_idx,
            test_idx,
            _split_origin("loaded", split_path),
            return_split_meta,
        )

    train_len = int(split[0] * len(subset_indices))
    val_len = int(split[1] * len(subset_indices))
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(subset_indices), generator=generator).tolist()
    train_idx = [subset_indices[i] for i in perm[:train_len]]
    val_idx = [subset_indices[i] for i in perm[train_len : train_len + val_len]]
    test_idx = [subset_indices[i] for i in perm[train_len + val_len :]]

    ensure_dir(str(dataset_split_dir))
    payload = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
        "meta": {
            "dataset_name": dataset_name,
            "total": len(subset_indices),
            "split": split,
            "seed": seed,
            "type": "subset",
            **(extra_meta or {}),
        },
    }
    torch.save(payload, split_path)
    print(f"[Dataset split] Saved subset split to {split_path}")
    return _maybe_return_split_indices(
        train_idx,
        val_idx,
        test_idx,
        _split_origin("saved", split_path),
        return_split_meta,
    )


def _edge_split_file_path(
    dataset_name: str,
    split: Tuple[float, float, float],
    split_root_path: Path,
) -> Path:
    edge_name = str(dataset_name)
    if "_edge_" not in edge_name and not edge_name.endswith("_edge"):
        edge_name = f"{edge_name}_edge"
    dataset_split_dir = _split_dataset_dir(split_root_path, edge_name)
    return dataset_split_dir / f"{edge_name}_splits-{_split_suffix(split)}.pt"


def _edge_positive_counts(
    total_edges: int,
    split: Tuple[float, float, float],
) -> Tuple[int, int, int, int]:
    train_len = min(total_edges, int(float(split[0]) * total_edges))
    remaining = total_edges - train_len
    val_len = min(remaining, int(float(split[1]) * total_edges))
    remaining -= val_len
    test_len = min(remaining, int(float(split[2]) * total_edges))
    message_len = total_edges - train_len - val_len - test_len
    return train_len, val_len, test_len, message_len


def _edge_negative_targets(train_pos: int, val_pos: int, test_pos: int) -> Tuple[int, int, int]:
    """Match negative counts to positive counts for each split."""
    return max(0, int(train_pos)), max(0, int(val_pos)), max(0, int(test_pos))


def _as_index_list(values) -> Optional[List[int]]:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        return [int(v) for v in values.view(-1).tolist()]
    if isinstance(values, (list, tuple)):
        return [int(v) for v in values]
    return None


def _as_edge_pair_tensor(values) -> Optional[torch.Tensor]:
    if values is None:
        return torch.empty((2, 0), dtype=torch.long)
    tensor = torch.as_tensor(values, dtype=torch.long)
    if tensor.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    if tensor.dim() != 2 or tensor.size(0) != 2:
        return None
    return tensor.cpu()


def _load_existing_edge_split_payload(path: Path, expected_total_edges: int):
    if not path.is_file():
        return None
    try:
        payload = safe_torch_load(path)
    except Exception:
        return None

    required_index_keys = ("train_pos_idx", "val_pos_idx", "test_pos_idx", "message_pos_idx")
    required_neg_keys = ("train_neg_edge_index", "val_neg_edge_index", "test_neg_edge_index")

    parsed = {}
    for key in required_index_keys:
        values = _as_index_list(payload.get(key))
        if values is None:
            return None
        parsed[key] = values

    if (
        len(parsed["train_pos_idx"])
        + len(parsed["val_pos_idx"])
        + len(parsed["test_pos_idx"])
        + len(parsed["message_pos_idx"])
        != expected_total_edges
    ):
        return None

    for key in required_neg_keys:
        values = _as_edge_pair_tensor(payload.get(key))
        if values is None:
            return None
        parsed[key] = values

    meta = payload.get("meta", {})
    parsed["meta"] = meta

    train_pos = len(parsed["train_pos_idx"])
    val_pos = len(parsed["val_pos_idx"])
    test_pos = len(parsed["test_pos_idx"])
    train_neg = int(parsed["train_neg_edge_index"].size(1))
    val_neg = int(parsed["val_neg_edge_index"].size(1))
    test_neg = int(parsed["test_neg_edge_index"].size(1))
    if train_neg != train_pos or val_neg != val_pos or test_neg != test_pos:
        return None

    return parsed


def _sample_negative_edge_pairs(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_neg_samples: int,
    seed: int,
) -> torch.Tensor:
    if num_nodes <= 0 or num_neg_samples <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

    fork_devices: List[int] = []
    if edge_index.is_cuda:
        device_index = edge_index.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        fork_devices = [int(device_index)]

    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(int(seed))
        neg_pairs = negative_sampling(
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_neg_samples=int(num_neg_samples),
            method="sparse",
            force_undirected=False,
        )
    if neg_pairs is None or neg_pairs.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
    return neg_pairs


def _get_or_create_edge_split_payload(
    dataset_name: str,
    split: Tuple[float, float, float],
    seed: int,
    split_root_path: Path,
    data: Data,
    persist: bool = True,
    verbose: bool = True,
    return_split_meta: bool = False,
):
    if split_root_path is None:
        raise ValueError("split_root_path must be provided for edge split generation.")
    _validate_edge_split_def(split)
    split_path = _edge_split_file_path(dataset_name, split, split_root_path)

    total_edges = int(data.edge_index.size(1))
    existing = _load_existing_edge_split_payload(split_path, total_edges)
    if existing is not None:
        if verbose:
            print(f"[Dataset split] Loaded fixed edge split from {split_path}")
        return _maybe_return_split_payload(existing, _split_origin("loaded", split_path), return_split_meta)

    train_len, val_len, test_len, _ = _edge_positive_counts(total_edges, split)
    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(total_edges, generator=generator).tolist()

    train_pos_idx = perm[:train_len]
    val_pos_idx = perm[train_len : train_len + val_len]
    test_pos_idx = perm[train_len + val_len : train_len + val_len + test_len]
    message_pos_idx = perm[train_len + val_len + test_len :]

    num_nodes = getattr(data, "num_nodes", None)
    if num_nodes is None and getattr(data, "x", None) is not None:
        num_nodes = data.x.size(0)
    num_nodes = int(num_nodes or 0)

    train_neg_target, val_neg_target, test_neg_target = _edge_negative_targets(
        len(train_pos_idx),
        len(val_pos_idx),
        len(test_pos_idx),
    )
    max_possible_neg = max(0, (num_nodes * max(0, num_nodes - 1)) - total_edges)
    total_neg_target = min(train_neg_target + val_neg_target + test_neg_target, max_possible_neg)
    neg_pairs_all = _sample_negative_edge_pairs(
        edge_index=data.edge_index,
        num_nodes=num_nodes,
        num_neg_samples=total_neg_target,
        seed=int(seed) + 1,
    )

    offset = 0
    neg_fill_generator = torch.Generator().manual_seed(int(seed) + 17)

    def _take_neg(count: int) -> torch.Tensor:
        nonlocal offset
        if count <= 0 or neg_pairs_all.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=neg_pairs_all.device)
        end = min(offset + count, neg_pairs_all.size(1))
        out = neg_pairs_all[:, offset:end]
        offset = end
        if out.size(1) < count and neg_pairs_all.size(1) > 0:
            need = count - out.size(1)
            fill_idx = torch.randint(
                low=0,
                high=neg_pairs_all.size(1),
                size=(need,),
                generator=neg_fill_generator,
                device=neg_pairs_all.device,
            )
            out = torch.cat([out, neg_pairs_all[:, fill_idx]], dim=1)
        return out

    train_neg_edge_index = _take_neg(train_neg_target)
    val_neg_edge_index = _take_neg(val_neg_target)
    test_neg_edge_index = _take_neg(test_neg_target)

    payload = {
        "train_pos_idx": train_pos_idx,
        "val_pos_idx": val_pos_idx,
        "test_pos_idx": test_pos_idx,
        "message_pos_idx": message_pos_idx,
        "train_neg_edge_index": train_neg_edge_index.cpu(),
        "val_neg_edge_index": val_neg_edge_index.cpu(),
        "test_neg_edge_index": test_neg_edge_index.cpu(),
        "meta": {
            "dataset_name": dataset_name,
            "total_edges": total_edges,
            "split": tuple(float(v) for v in split),
            "seed": int(seed),
            "negative_sampling_mode": "match_split_positive",
            "train_negatives": int(train_neg_edge_index.size(1)),
            "val_negatives": int(val_neg_edge_index.size(1)),
            "test_negatives": int(test_neg_edge_index.size(1)),
        },
    }
    if not persist:
        return _maybe_return_split_payload(payload, _split_origin("builtin", None), return_split_meta)

    ensure_dir(str(split_root_path))
    torch.save(payload, split_path)
    if verbose:
        print(f"[Dataset split] Saved fixed edge split to {split_path}")
    loaded_payload = _load_existing_edge_split_payload(split_path, total_edges) or payload
    return _maybe_return_split_payload(loaded_payload, _split_origin("saved", split_path), return_split_meta)


def split_graph_dataset(
    dataset,
    dataset_name: str,
    split: Tuple[float, float, float],
    seed: int,
    split_root: str,
    return_split_meta: bool = False,
):
    """Split graph-level dataset into train/val/test sets (supports few-shot)."""
    if not split_root:
        raise ValueError("split_root is required to save or load fixed splits.")

    split_root_path = Path(split_root)
    split_dataset_name = _canonical_split_dataset_name(dataset_name, "graph", int(seed))

    _validate_split_def(split)
    extra_meta = None
    graph_filter_meta = _graph_filter_cache_meta_for_dataset(dataset)
    if graph_filter_meta is not None:
        extra_meta = {"graph_filter": graph_filter_meta}

    use_few_shot = _is_few_shot_split_def(split)
    split_meta = None

    if use_few_shot:
        split_strategy = resolve_count_split_strategy(dataset, "graph")
        if split_strategy == "random":
            result = _get_or_create_count_split(
                dataset_name=split_dataset_name,
                train_count=int(split[0]),
                val_ratio=float(split[1]),
                test_ratio=float(split[2]),
                seed=seed,
                split_root_path=split_root_path,
                total=len(dataset),
                extra_meta=extra_meta,
                return_split_meta=return_split_meta,
            )
        elif split_strategy == "balanced":
            labels_list = []
            for item in dataset:
                if not hasattr(item, "y") or item.y is None:
                    raise ValueError("Few-shot split requires labels for each graph instance.")
                target = item.y.view(-1)
                if target.numel() != 1:
                    raise ValueError("Few-shot split currently supports single-label targets.")
                labels_list.append(int(target[0].item()))
            labels = torch.tensor(labels_list, dtype=torch.long)
            result = _get_or_create_few_shot_split(
                dataset_name=split_dataset_name,
                labels=labels,
                shots_per_class=int(split[0]),
                val_ratio=float(split[1]),
                test_ratio=float(split[2]),
                seed=seed,
                split_root_path=split_root_path,
                extra_meta=extra_meta,
                return_split_meta=return_split_meta,
            )
        else:
            raise ValueError("Integer-first splits are not supported for unlabeled graph datasets.")
    else:
        result = _get_or_create_split_indices(
            dataset_name=split_dataset_name,
            split=split,
            seed=seed,
            split_root_path=split_root_path,
            total=len(dataset),
            extra_meta=extra_meta,
            return_split_meta=return_split_meta,
        )

    if return_split_meta:
        train_idx, val_idx, test_idx, split_meta = result
    else:
        train_idx, val_idx, test_idx = result

    from torch.utils.data import Subset

    outputs = (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )
    if return_split_meta:
        return outputs[0], outputs[1], outputs[2], dict(split_meta)
    return outputs


def _get_or_create_few_shot_split(
    dataset_name: str,
    labels: torch.Tensor,
    shots_per_class: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    split_root_path: Path,
    extra_meta: Dict | None = None,
    return_split_meta: bool = False,
):
    """Create or load a class-balanced few-shot split."""
    if split_root_path is None:
        raise ValueError("split_root_path must be provided (configure cfg.dataset.split_root).")

    dataset_split_dir = _split_dataset_dir(split_root_path, dataset_name)
    split_tag = _split_suffix((shots_per_class, val_ratio, test_ratio))
    split_path = dataset_split_dir / f"{dataset_name}_splits-{split_tag}.pt"

    labels = labels.view(-1).cpu()
    total = labels.numel()

    existing = _load_existing_indices(split_path, total, expected_meta=extra_meta)
    if existing:
        train_idx, val_idx, test_idx = existing
        print(f"[Dataset split] Loaded few-shot split from {split_path}")
        return _maybe_return_split_indices(
            train_idx,
            val_idx,
            test_idx,
            _split_origin("loaded", split_path),
            return_split_meta,
        )

    unique_labels = torch.unique(labels)
    generator = torch.Generator().manual_seed(seed)
    train_idx = []
    for label in unique_labels:
        indices = torch.nonzero(labels == label, as_tuple=False).view(-1)
        if indices.numel() == 0:
            continue
        perm = torch.randperm(indices.numel(), generator=generator)
        ordered = indices[perm].tolist()
        take = min(shots_per_class, len(ordered))
        train_idx.extend(ordered[:take])

    mask = torch.ones(total, dtype=torch.bool)
    if train_idx:
        mask[train_idx] = False
    remaining = mask.nonzero(as_tuple=False).view(-1)
    if remaining.numel() > 0:
        perm = torch.randperm(remaining.numel(), generator=generator)
        remaining = remaining[perm]
    remaining_list = remaining.tolist()

    denom = val_ratio + test_ratio
    val_fraction = val_ratio / denom if denom > 0 else 0.0
    val_len = int(val_fraction * len(remaining_list))
    val_idx = remaining_list[:val_len]
    test_idx = remaining_list[val_len:]

    if len(train_idx) + len(val_idx) + len(test_idx) != total:
        raise ValueError(
            f"Few-shot split mismatch: got {len(train_idx)} train, {len(val_idx)} val, "
            f"{len(test_idx)} test for total {total}."
        )

    ensure_dir(str(dataset_split_dir))
    payload = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
        "meta": {
            "dataset_name": dataset_name,
            "total": total,
            "split": (shots_per_class, val_ratio, test_ratio),
            "seed": seed,
            "type": "few_shot",
            **(extra_meta or {}),
        },
    }
    torch.save(payload, split_path)
    print(f"[Dataset split] Saved few-shot split to {split_path}")
    return _maybe_return_split_indices(
        train_idx,
        val_idx,
        test_idx,
        _split_origin("saved", split_path),
        return_split_meta,
    )


def _get_or_create_count_split(
    dataset_name: str,
    train_count: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    split_root_path: Path,
    total: int | None = None,
    subset_indices: List[int] | None = None,
    extra_meta: Dict | None = None,
    return_split_meta: bool = False,
):
    """Create or load a deterministic random train-count split."""
    if split_root_path is None:
        raise ValueError("split_root_path must be provided (configure cfg.dataset.split_root).")

    candidate_indices = list(subset_indices) if subset_indices is not None else list(range(int(total or 0)))
    expected_total = len(candidate_indices)
    dataset_split_dir = _split_dataset_dir(split_root_path, dataset_name)
    split_tag = _split_suffix((train_count, val_ratio, test_ratio))
    split_path = dataset_split_dir / f"{dataset_name}_splits-{split_tag}.pt"

    existing = _load_existing_indices(split_path, expected_total, expected_meta=extra_meta)
    if existing:
        train_idx, val_idx, test_idx = existing
        print(f"[Dataset split] Loaded random count split from {split_path}")
        return _maybe_return_split_indices(
            train_idx,
            val_idx,
            test_idx,
            _split_origin("loaded", split_path),
            return_split_meta,
        )

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(expected_total, generator=generator).tolist() if expected_total > 0 else []
    ordered = [candidate_indices[idx] for idx in perm]

    train_take = min(int(train_count), expected_total)
    train_idx = ordered[:train_take]
    remaining = ordered[train_take:]

    denom = val_ratio + test_ratio
    val_fraction = val_ratio / denom if denom > 0 else 0.0
    val_len = int(val_fraction * len(remaining))
    val_idx = remaining[:val_len]
    test_idx = remaining[val_len:]

    ensure_dir(str(dataset_split_dir))
    payload = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
        "meta": {
            "dataset_name": dataset_name,
            "total": expected_total,
            "split": (train_count, val_ratio, test_ratio),
            "seed": seed,
            "type": "random_count",
            **(extra_meta or {}),
        },
    }
    torch.save(payload, split_path)
    print(f"[Dataset split] Saved random count split to {split_path}")
    return _maybe_return_split_indices(
        train_idx,
        val_idx,
        test_idx,
        _split_origin("saved", split_path),
        return_split_meta,
    )
