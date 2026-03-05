from numbers import Integral
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List
import contextlib
import re
import warnings

import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data
try:
    from torch_geometric.data import InMemoryDataset
except Exception:  # pragma: no cover - older PyG variants
    InMemoryDataset = None
try:
    from torch_geometric.data.data import DataEdgeAttr  # type: ignore
except Exception:  # pragma: no cover - older PyG
    DataEdgeAttr = None
from torch_geometric.datasets import (
    Actor,
    Airports,
    Amazon,
    CitationFull,
    Coauthor,
    CoraFull,
    EllipticBitcoinDataset,
    EmailEUCore,
    GNNBenchmarkDataset,
    # FacebookPagePage,
    Flickr,
    # GemsecDeezer,
    # GitHub,
    HeterophilousGraphDataset,
    # LastFMAsia,
    LINKXDataset,
    LRGBDataset,
    MoleculeNet,
    Planetoid,
    QM7b,
    QM9,
    Reddit,
    Reddit2,
    TUDataset,
    # Twitch,
    WebKB,
    WikiCS,
    WikipediaNetwork,
)
from torch_geometric.loader import DataLoader
try:
    from torch_geometric.loader import LinkNeighborLoader
except Exception:  # pragma: no cover - optional loader
    LinkNeighborLoader = None
from torch_geometric.transforms import Compose
from torch_geometric.utils import degree, k_hop_subgraph, negative_sampling, subgraph

from code.utils import ensure_dir, format_split_for_name
from .dataset_domains import (
    CLASS_TO_DOMAIN,
    KEYWORD_DOMAINS,
    NAME_TO_DOMAIN,
)

# Keep logs clean when third-party internals still touch InMemoryDataset.data.
warnings.filterwarnings(
    "ignore",
    message="It is not recommended to directly access the internal storage format `data` of an 'InMemoryDataset'.*",
    category=UserWarning,
    module="torch_geometric.data.in_memory_dataset",
)

# Ensure torch.load works with PyG Data under torch 2.6+ defaults.
import os
import torch.serialization as ts  # type: ignore

os.environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")
os.environ.setdefault("OGB_ASSUME_YES", "1")
if hasattr(ts, "add_safe_globals"):
    try:
        extras = [Data]
        if DataEdgeAttr is not None:
            extras.append(DataEdgeAttr)
        ts.add_safe_globals(extras)
    except Exception:
        pass
# Force weights_only=False globally when available (torch>=2.6)
try:
    import inspect

    _orig_torch_load = torch.load
    if "weights_only" in inspect.signature(_orig_torch_load).parameters:
        def _torch_load_no_weights_only(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_torch_load(*args, **kwargs)
        torch.load = _torch_load_no_weights_only
except Exception:
    pass

# Compatibility loader to handle Torch 2.6+ weights_only default.
def _safe_torch_load(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def _get_dataset_data_storage(dataset):
    """
    Return underlying dataset-level Data storage without triggering PyG warnings.

    For InMemoryDataset, use `_data` directly instead of `data` to avoid:
    "It is not recommended to directly access the internal storage format `data`..."
    """
    # Prefer direct internal storage access for PyG InMemoryDataset to avoid
    # triggering the `dataset.data` deprecation warning.
    if InMemoryDataset is not None and isinstance(dataset, InMemoryDataset):
        return dataset.__dict__.get("_data", None)

    if "_data" in getattr(dataset, "__dict__", {}):
        return dataset.__dict__.get("_data", None)

    return getattr(dataset, "data", None)


def _set_dataset_data_storage(dataset, data_obj) -> None:
    """Set dataset-level Data storage using `_data` when available."""
    if InMemoryDataset is not None and isinstance(dataset, InMemoryDataset):
        dataset.__dict__["_data"] = data_obj
        return

    if "_data" in getattr(dataset, "__dict__", {}):
        dataset.__dict__["_data"] = data_obj
        return
    setattr(dataset, "data", data_obj)


@contextlib.contextmanager
def _force_ogb_prompts_yes():
    """Force OGB download/version prompts to auto-yes."""
    import builtins
    try:
        import ogb.utils.url as ogb_url
    except Exception:
        ogb_url = None

    orig_input = builtins.input
    orig_decide = getattr(ogb_url, "decide_download", None) if ogb_url else None
    builtins.input = lambda *args, **kwargs: "y"
    if ogb_url and orig_decide:
        ogb_url.decide_download = lambda url: True
    try:
        yield
    finally:
        builtins.input = orig_input
        if ogb_url and orig_decide:
            ogb_url.decide_download = orig_decide


# Note: we don't touch any dynamic, 3D, relational or heterogeneous datasets in this project.

# -------------------------------------------------------------------------- #
# Node-level datasets
# -------------------------------------------------------------------------- #
Actor_NAMES = {"actor"}
Airports_NAMES = {"airports": "USA"}
Amazon_NAMES = {
    "computers": "Computers",
    "photo": "Photo",
}
CitationFull_NAMES = {
    "cora_ml",
    "dblp",
}
Coauthor_NAMES = {
    "cs": "CS",
    "physics": "Physics",
}
CoraFull_NAMES = {"corafull"}
EllipticBitcoinDataset_NAMES = {"elliptic_bitcoin"}
EmailEUCore_NAMES = {
    "email": "email_eu_core",
    "email_eu_core": "email_eu_core",
    "email-eu-core": "email_eu_core",
}
# Disabled datasets (source URLs currently 404).
FacebookPagePage_NAMES = {}  # {"facebook_page-page"}
Flickr_NAMES = {"flickr"}
GemsecDeezer_NAMES = {}  # {"gemsec_deezer_hu": "HU", "gemsec_deezer_hr": "HR", "gemsec_deezer_ro": "RO"}
GitHub_NAMES = {}
HeterophilousGraphDataset_NAMES = {
    "amazon-ratings": "Amazon-ratings",
    "minesweeper": "Minesweeper",
    "questions": "Questions",
    "roman-empire": "Roman-empire",
    "tolokers": "Tolokers",
}
LastFMAsia_NAMES = {}
LINKXDataset_NAMES = {
    "amherst41",
    "cornell5",
    "genius",
    "johnshopkins55",
    "penn94",
    "reed98",
}
LRGB_NODE_NAMES = {
    "coco-sp": "COCO-SP",
    "pascalvoc-sp": "PascalVOC-SP",
}
LRGB_EDGE_NAMES = {
    "pcqm-contact": "PCQM-Contact",
}
LRGB_GRAPH_NAMES = {
    "peptides-func": "Peptides-func",
    "peptides-struct": "Peptides-struct",
}
Planetoid_NAMES = {
    "citeseer": "CiteSeer",
    "cora": "Cora",
    "pubmed": "PubMed",
}
Reddit_NAMES = {"reddit"}
Reddit2_NAMES = {"reddit2"}
Twitch_NAMES = {}  # {"twitch-de": "DE", ...} currently 404
WebKB_NAMES = {"cornell", "texas", "wisconsin"}
WikiCS_NAMES = {"wikics"}
WikipediaNetwork_NAMES = {
    "chameleon",
    # "crocodile",
    "squirrel",
}

# -------------------------------------------------------------------------- #
# Graph-level datasets
# -------------------------------------------------------------------------- #
# Core molecule/graph benchmarks
MoleculeNet_NAMES = {
    "bace",
    "bbbp",
    "clintox",
    "esol",
    "freesolv",
    "hiv",
    "lipo",
    "muv",
    "pcba",
    "sider",
    "tox21",
    "toxcast",
}
QM7b_NAMES = {"qm7b"}
QM9_NAMES = {"qm9"}
TUDataset_NAMES = {
    "collab": "COLLAB",
    "enzymes": "ENZYMES",
    "imdb-binary": "IMDB-BINARY",
    "imdb-multi": "IMDB-MULTI",
    "mutag": "MUTAG",
    "ppi": "PPI",
    "proteins": "PROTEINS",
    "nci1": "NCI1",
    "nci109": "NCI109",
    "dd": "DD",
    "reddit-binary": "REDDIT-BINARY",
    "reddit-multi-5k": "REDDIT-MULTI-5K",
}
GNNBenchmarkDataset_NAMES = {
    "mnist": "MNIST",
    "cifar10": "CIFAR10",
}


def _is_node_dataset_key(key: str) -> bool:
    return (
        key in Actor_NAMES
        or key in Airports_NAMES
        or key in Amazon_NAMES
        or key in CitationFull_NAMES
        or key in Coauthor_NAMES
        or key in CoraFull_NAMES
        or key in EllipticBitcoinDataset_NAMES
        or key in EmailEUCore_NAMES
        or key in Flickr_NAMES
        or key in HeterophilousGraphDataset_NAMES
        or key in LINKXDataset_NAMES
        or key in LRGB_NODE_NAMES
        or key in Planetoid_NAMES
        or key in Reddit_NAMES
        or key in Reddit2_NAMES
        or key.startswith("ogbn-")
        or key in WebKB_NAMES
        or key in WikiCS_NAMES
        or key in WikipediaNetwork_NAMES
    )


def _is_graph_dataset_key(key: str) -> bool:
    return (
        key in MoleculeNet_NAMES
        or key in GNNBenchmarkDataset_NAMES
        or key in LRGB_GRAPH_NAMES
        or key in QM7b_NAMES
        or key in QM9_NAMES
        or key in TUDataset_NAMES
        or key.startswith("ogbg-")
    )


def _is_edge_dataset_key(key: str) -> bool:
    return key in LRGB_EDGE_NAMES or key.startswith("ogbl-")


def infer_task_level(name: str) -> str | None:
    key = name.lower()
    in_node = _is_node_dataset_key(key)
    in_graph = _is_graph_dataset_key(key)
    in_edge = _is_edge_dataset_key(key)
    if in_edge and not in_node and not in_graph:
        return "edge"
    if in_node and not in_graph:
        return "node"
    if in_graph and not in_node and not in_edge:
        return "graph"
    return None


def _sanitize_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    return cleaned or "dataset"


def _dataset_scoped_dir(base_dir: Path | str, dataset_name: str) -> Path:
    base_path = Path(base_dir)
    scoped = _sanitize_name(str(dataset_name))
    if base_path.name == scoped:
        return base_path
    return base_path / scoped


def _base_dataset_for_split_name(dataset_name: str) -> str:
    text = str(dataset_name)
    for marker in ("_node_seed", "_graph_seed", "_edge_seed", "_seed"):
        if marker in text:
            return text.split(marker, 1)[0]
    if text.endswith("_edge"):
        return text[: -len("_edge")]
    return text


def _split_dataset_dir(split_root_path: Path, dataset_name: str) -> Path:
    return _dataset_scoped_dir(split_root_path, _base_dataset_for_split_name(dataset_name))


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

    # Already has a task-qualified seed marker.
    tagged = re.match(r"^(?P<base>.+)_(?P<tag>node|graph|edge)_seed(?P<seed>\d+)$", name)
    if tagged:
        return name

    # Bare edge name without a seed marker.
    if task == "edge" and name.endswith("_edge"):
        return f"{name}_seed{int(seed)}"

    if task == "edge":
        return f"{name}_edge_seed{int(seed)}"
    return f"{name}_{task}_seed{int(seed)}"


def _split_suffix(portions: Tuple[float, float, float]) -> str:
    # Represent split as integer percentages for filenames, e.g., (0.8,0.1,0.1) -> "80-10-10"
    return "-".join(str(int(round(float(p) * 100))) for p in portions)


def _validate_split_def(split_def: Tuple[float, float, float]) -> None:
    if not isinstance(split_def, (list, tuple)) or len(split_def) < 3:
        raise ValueError(
            "Split definition must have 3 values: (train, val, test) or (shots_per_class, val_weight, test_weight)."
        )

    first = split_def[0]
    is_few_shot = False
    try:
        first_val = float(first)
        is_few_shot = float(first_val).is_integer()
    except Exception:
        is_few_shot = isinstance(first, Integral) and not isinstance(first, bool)

    if is_few_shot:
        try:
            shots = int(float(split_def[0]))
            val_ratio = float(split_def[1])
            test_ratio = float(split_def[2])
        except Exception as exc:
            raise ValueError("Few-shot split must contain numeric values.") from exc
        if shots < 1:
            raise ValueError("Few-shot split requires shots_per_class >= 1.")
        if val_ratio < 0.0 or test_ratio < 0.0:
            raise ValueError("Few-shot split requires non-negative val/test weights.")
        if (val_ratio + test_ratio) <= 0.0:
            raise ValueError("Few-shot split requires val_weight + test_weight > 0.")
    else:
        try:
            total = float(split_def[0]) + float(split_def[1]) + float(split_def[2])
        except Exception as exc:
            raise ValueError("Supervised split must contain numeric ratios.") from exc
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Supervised split ratios must sum to 1.0.")


def _validate_edge_split_def(split_def: Tuple[float, float, float]) -> None:
    if not isinstance(split_def, (list, tuple)) or len(split_def) < 3:
        raise ValueError("Edge split definition must have 3 values: (train_pos, val_pos, test_pos).")
    try:
        train_pos = float(split_def[0])
        val_pos = float(split_def[1])
        test_pos = float(split_def[2])
    except Exception as exc:
        raise ValueError("Edge split values must be numeric.") from exc
    if min(train_pos, val_pos, test_pos) < 0.0:
        raise ValueError("Edge split values must be >= 0.")
    total = train_pos + val_pos + test_pos
    if total > 1.0 + 1e-6:
        raise ValueError("Edge split positive ratios must sum to <= 1.0.")


def _is_few_shot_split_def(split_def) -> bool:
    """Return True when split uses few-shot form (shots_per_class, val_weight, test_weight)."""
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


def _mask_to_node_indices(mask, name: str) -> List[int]:
    """Convert boolean node mask(s) to node indices, handling 1D/2D PyG masks."""
    tensor = torch.as_tensor(mask)
    if tensor.dim() == 0:
        return [0] if bool(tensor.item()) else []
    if tensor.dim() == 1:
        return torch.nonzero(tensor, as_tuple=False).view(-1).tolist()
    if tensor.dim() == 2:
        # Some datasets store multiple predefined splits as [num_nodes, num_splits].
        # Use the first split deterministically when explicit split ratios are absent.
        if tensor.size(1) > 1:
            print(f"[Dataset split] {name} has {tensor.size(1)} predefined splits; using column 0.")
        return torch.nonzero(tensor[:, 0], as_tuple=False).view(-1).tolist()
    raise ValueError(f"{name} must be 1D or 2D; got shape={tuple(tensor.shape)}")


def _few_shot_suffix(shots: int, val_ratio: float, test_ratio: float) -> str:
    val_pct = int(round(val_ratio * 100))
    test_pct = int(round(test_ratio * 100))
    return f"fewshot{shots}-{val_pct}-{test_pct}"


def _labels_indicate_regression(labels: torch.Tensor) -> bool:
    """Best-effort regression detection from labels."""
    label_tensor = torch.as_tensor(labels).view(-1)
    if label_tensor.numel() == 0:
        return False

    if label_tensor.dtype.is_floating_point:
        finite = label_tensor[torch.isfinite(label_tensor)]
        if finite.numel() == 0:
            return False
        rounded = finite.round()
        if not torch.allclose(finite, rounded, atol=1e-6):
            return True

        # Float labels with many unique values are likely regression targets.
        if int(torch.unique(finite).numel()) > 32:
            return True

    return False


def is_regression_dataset(dataset, task_level: str) -> bool:
    """
    Infer whether the current task uses regression labels.

    This is used to reject unsupported few-shot splits for regression tasks.
    """
    level = str(task_level).lower()
    if level == "edge":
        return False

    # Explicit class count indicates classification.
    num_classes = getattr(dataset, "num_classes", None)
    try:
        if num_classes is not None and int(num_classes) > 0:
            return False
    except Exception:
        pass

    if level == "node":
        try:
            data = dataset[0]
            labels = getattr(data, "y", None)
            if labels is None:
                return False
            return _labels_indicate_regression(labels)
        except Exception:
            return False

    if level == "graph":
        # Fast path for in-memory graph labels.
        data_obj = _get_dataset_data_storage(dataset)
        labels = getattr(data_obj, "y", None) if data_obj is not None else None
        if labels is not None:
            labels_tensor = torch.as_tensor(labels).view(-1)
            try:
                if labels_tensor.numel() == len(dataset):
                    return _labels_indicate_regression(labels_tensor)
            except Exception:
                pass

        # Fallback: sample per-graph labels.
        sampled = []
        try:
            sample_count = min(len(dataset), 4096)
            for idx in range(sample_count):
                item = dataset[idx]
                y = getattr(item, "y", None)
                if y is None:
                    continue
                y_tensor = torch.as_tensor(y).view(-1)
                if y_tensor.numel() > 0:
                    sampled.append(y_tensor[0].detach().cpu())
        except Exception:
            sampled = []

        if not sampled:
            return False
        stacked = torch.stack(sampled).view(-1)
        return _labels_indicate_regression(stacked)

    return False


def _load_existing_indices(path: Path, expected_total: int):
    if not path.is_file():
        return None
    try:
        payload = _safe_torch_load(path)
        train_idx = payload.get("train_indices") or payload.get("train")
        val_idx = payload.get("val_indices") or payload.get("val")
        test_idx = payload.get("test_indices") or payload.get("test")
        if not all(isinstance(idx, (list, tuple)) for idx in (train_idx, val_idx, test_idx)):
            return None
        if len(train_idx) + len(val_idx) + len(test_idx) != expected_total:
            return None
        return list(train_idx), list(val_idx), list(test_idx)
    except Exception:
        return None


def _scoped_root(root: str, subdir: str) -> str:
    """Place datasets without internal namespacing into their own subfolder."""
    return str(Path(root) / subdir)

def _get_or_create_split_indices(
    dataset_name: str,
    split: Tuple[float, float, float],
    seed: int,
    split_root_path: Path,
    total: int,
):
    """Load existing split indices or create and persist new ones."""
    if split_root_path is None:
        raise ValueError("split_root_path must be provided (configure cfg.dataset.split_root).")

    dataset_split_dir = _split_dataset_dir(split_root_path, dataset_name)
    split_path = dataset_split_dir / f"{dataset_name}_splits-{_split_suffix(split)}.pt"
    
    existing = _load_existing_indices(split_path, total)
    if existing:
        train_idx, val_idx, test_idx = existing
        print(f"[Dataset split] Loaded fixed split from {split_path}")
        return train_idx, val_idx, test_idx

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
        },
    }
    torch.save(payload, split_path)
    print(f"[Dataset split] Saved fixed split to {split_path}")

    return train_idx, val_idx, test_idx


def _get_or_create_split_indices_subset(
    dataset_name: str,
    split: Tuple[float, float, float],
    seed: int,
    split_root_path: Path,
    subset_indices: List[int],
):
    """Create or load split indices for a subset of nodes."""
    dataset_split_dir = _split_dataset_dir(split_root_path, dataset_name)
    split_path = dataset_split_dir / f"{dataset_name}_splits-{_split_suffix(split)}.pt"
    existing = _load_existing_indices(split_path, len(subset_indices))
    if existing:
        train_idx, val_idx, test_idx = existing
        print(f"[Dataset split] Loaded fixed split from {split_path}")
        return train_idx, val_idx, test_idx

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
        },
    }
    torch.save(payload, split_path)
    print(f"[Dataset split] Saved subset split to {split_path}")
    return train_idx, val_idx, test_idx


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
        payload = _safe_torch_load(path)
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

    # Avoid initializing all visible CUDA devices inside fork_rng.
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
        return existing

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
        # Best-effort backfill with replacement when unique negatives are insufficient.
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
        return payload

    ensure_dir(str(split_root_path))
    torch.save(payload, split_path)
    if verbose:
        print(f"[Dataset split] Saved fixed edge split to {split_path}")
    return _load_existing_edge_split_payload(split_path, total_edges) or payload


def _induced_cache_path(
    dataset_name: str,
    task_level: str,
    cache_root_path: Path,
    suffix: str,
):
    sanitized = _sanitize_name(dataset_name)
    dataset_dir = _dataset_scoped_dir(cache_root_path, sanitized)
    return dataset_dir / f"{sanitized}_induced_{task_level}_{suffix}.pt"


def _load_induced_cache(path: Path, expected_meta: Dict) -> Dict | None:
    if not path.is_file():
        return None
    try:
        payload = _safe_torch_load(path)
        meta = payload.get("meta", {})
        for k, v in expected_meta.items():
            if meta.get(k) != v:
                return None
        return payload
    except Exception:
        return None


def _save_induced_cache(path: Path, payload: Dict):
    ensure_dir(path.parent)
    torch.save(payload, path)


class SingleGraphDataLoader:
    """Lightweight wrapper so node-level datasets match the DataLoader interface."""

    def __init__(self, data: Data):
        self.data = data

    def __iter__(self):
        yield self.data

    def __len__(self):
        return 1


def build_induced_graphs(
    data: Data,
    smallest_size: int = 10,
    largest_size: int = 30,
    max_hops: int = 5,
    start_hops: int = 2,
):
    """Generate induced subgraphs centered at each node."""
    induced_graphs = []
    if data is None:
        return induced_graphs

    edge_index = getattr(data, "edge_index", None)
    num_nodes = getattr(data, "num_nodes", None)
    if num_nodes is None and getattr(data, "x", None) is not None:
        num_nodes = data.x.size(0)
    if num_nodes is None or num_nodes <= 0:
        return induced_graphs

    labels = getattr(data, "y", None)
    if labels is None:
        return induced_graphs
    labels = labels.view(-1)
    device = data.x.device if getattr(data, "x", None) is not None else torch.device("cpu")
    has_edges = edge_index is not None and edge_index.numel() > 0

    for idx in range(num_nodes):
        label = int(labels[idx].item())
        if label < 0:
            continue
        hops = start_hops
        subset = torch.tensor([idx], device=device)

        if has_edges:
                subset, _, _, _ = k_hop_subgraph(
                    node_idx=idx,
                    num_hops=hops,
                    edge_index=edge_index,
                    relabel_nodes=True,
                    num_nodes=num_nodes,
                )
                while subset.numel() < smallest_size and hops < max_hops:
                    hops += 1
                    subset, _, _, _ = k_hop_subgraph(
                        node_idx=idx,
                        num_hops=hops,
                        edge_index=edge_index,
                        relabel_nodes=True,
                        num_nodes=num_nodes,
                    )

        subset_cpu = subset.cpu()
        if subset_cpu.numel() > largest_size:
            keep = subset_cpu[torch.randperm(subset_cpu.numel())[: largest_size - 1]]
            subset_cpu = torch.unique(torch.cat([torch.tensor([idx], dtype=torch.long), keep]))

        subset = subset_cpu.to(device)
        if has_edges:
            sub_edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True, num_nodes=num_nodes)
        else:
            sub_edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        x = data.x[subset] if getattr(data, "x", None) is not None else None

        g = Data(x=x, edge_index=sub_edge_index, y=torch.tensor(label))
        g.base_node_id = idx
        g.index = idx
        induced_graphs.append(g)
    return induced_graphs


def _edge_induced_subgraph(
    data: Data,
    endpoints: torch.Tensor,
    max_hops: int,
    max_size: int | None = None,
    label: int | None = None,
):
    """Build a single edge-centered induced subgraph given endpoints."""
    if endpoints.numel() != 2:
        raise ValueError("endpoints must contain exactly two node indices.")

    device = data.edge_index.device
    x = getattr(data, "x", None)
    num_nodes = getattr(data, "num_nodes", None)
    if num_nodes is None and x is not None:
        num_nodes = x.size(0)
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx=endpoints.to(device=device),
        edge_index=data.edge_index,
        num_hops=max_hops,
        relabel_nodes=True,
        num_nodes=num_nodes,
    )
    if max_size is not None and max_size > 0 and subset.numel() > max_size:
        # Keep endpoints and randomly sample the rest in original index space.
        target = max(max_size, 2)
        keep = torch.randperm(subset.numel(), device=subset.device)[: target - 2]
        keep = torch.unique(torch.cat([keep, torch.tensor([mapping[0], mapping[1]], device=subset.device)]))
        subset = subset[keep]
        sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
        # Recompute endpoint mapping after truncation.
        map_u = torch.where(subset == endpoints[0])[0][0]
        map_v = torch.where(subset == endpoints[1])[0][0]
        mapping = torch.stack([map_u, map_v])
    sub_x = x[subset] if x is not None else None
    mapped_u, mapped_v = int(mapping[0]), int(mapping[1])
    target_edge = torch.tensor([[mapped_u], [mapped_v]], dtype=torch.long, device=device)
    g = Data(
        x=sub_x,
        edge_index=sub_edge_index,
        edge_label_index=target_edge,
    )
    if label is not None:
        g.y = torch.tensor(label, dtype=torch.long)
    return g


def build_edge_induced_graphs(
    data: Data,
    edge_indices: torch.Tensor,
    max_hops: int = 2,
    max_size: int | None = None,
):
    """Generate edge-centered induced subgraphs (one per target edge)."""
    if data is None or edge_indices.numel() == 0:
        return []

    edge_pairs = data.edge_index[:, edge_indices.to(dtype=torch.long)]
    graphs = []

    for idx in range(edge_pairs.size(1)):
        endpoints = edge_pairs[:, idx]
        graphs.append(_edge_induced_subgraph(data, endpoints=endpoints, max_hops=max_hops, max_size=max_size))
    return graphs


def build_edge_induced_graphs_supervised(
    data: Data,
    pos_edge_pairs: torch.Tensor,
    neg_edge_pairs: torch.Tensor,
    max_hops: int = 2,
    max_size: int | None = None,
):
    """Generate edge-centered induced subgraphs with binary labels."""
    if data is None:
        return []

    graphs = []
    if pos_edge_pairs is not None and pos_edge_pairs.numel() > 0:
        for idx in range(pos_edge_pairs.size(1)):
            endpoints = pos_edge_pairs[:, idx]
            graphs.append(_edge_induced_subgraph(data, endpoints=endpoints, max_hops=max_hops, max_size=max_size, label=1))

    if neg_edge_pairs is not None and neg_edge_pairs.numel() > 0:
        for idx in range(neg_edge_pairs.size(1)):
            endpoints = neg_edge_pairs[:, idx]
            graphs.append(_edge_induced_subgraph(data, endpoints=endpoints, max_hops=max_hops, max_size=max_size, label=0))
    return graphs


class EnsureFeatureTransform:
    """If a dataset has no node features, attach degree features."""

    def __call__(self, data: Data) -> Data:
        x = getattr(data, "x", None)
        if x is None or (hasattr(x, "numel") and x.numel() == 0):
            num_nodes = data.num_nodes
            if num_nodes is None:
                raise ValueError("Cannot infer num_nodes to build degree features.")
            edge_index = getattr(data, "edge_index", None)
            if edge_index is not None and edge_index.numel() > 0:
                deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
            else:
                deg = torch.zeros(num_nodes, dtype=torch.float)
            # Use scalar degree as node feature (num_nodes x 1), avoid huge diagonal matrices.
            data.x = deg.view(num_nodes, 1)
            #data.x = torch.diag(deg.view(-1))
        return data


class SafeSVDFeatureReduction:
    """Robust feature reduction with padding/truncation fallbacks."""

    def __init__(self, out_channels: int):
        self.out_channels = out_channels

    def __call__(self, data: Data) -> Data:
        x = getattr(data, "x", None)
        
        # Handle missing or empty features by creating zero features.
        if x is None or x.numel() == 0:
            return data

        target = self.out_channels
        in_dim = x.size(-1)
        # # Handle exact match case.
        # if in_dim == target:
        #     return data
        # Handle padding/truncation cases.
        if in_dim < target:
            x_f = x.to(torch.float32) if not torch.is_floating_point(x) else x
            pad = x_f.new_zeros(x_f.size(0), target - in_dim)
            data.x = torch.cat([x_f, pad], dim=1).to(torch.float32)
            return data

        # in_dim > target: attempt SVD, fall back to truncate on failure
        try:
            u, s, _ = torch.linalg.svd(x.float(), full_matrices=False)
            u = u[:, :target]
            s = s[:target]
            reduced = u * s
            data.x = reduced.to(dtype=torch.float32, device=x.device)
        except Exception:
            data.x = x[:, :target].to(torch.float32)
        return data


class InducedGraphDataset(Dataset):
    """Dataset of induced subgraphs (one per original node)."""

    def __init__(
        self, 
        graphs, 
        base_num_nodes: int = None, 
        base_num_edges: int = None,
        base_info: Dict = None,
        split_tags=None,
    ):
        self.graphs = graphs
        self.num_features = graphs[0].num_node_features if graphs else 0
        labels_list = []
        for g in graphs:
            y = getattr(g, "y", None)
            if y is not None:
                labels_list.append(y.view(-1)[0])
        labels = torch.stack(labels_list) if labels_list else torch.tensor([])
        self.num_classes = int(labels.max().item() + 1) if labels.numel() > 0 else None
        self.base_num_nodes = base_num_nodes
        self.base_num_edges = base_num_edges
        self.base_dataset_info = base_info or {}
        self.split_tags = split_tags or ["train"] * len(graphs)
        # Surface the base name to downstream code for nicer logging/LLM prompts.
        if self.base_dataset_info.get("name"):
            self.name = f"Induced({self.base_dataset_info['name']})"

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx]
        # Expose the originating node index for downstream logging/recording.
        if not hasattr(data, "base_node_id"):
            data.base_node_id = idx
        if self.split_tags:
            data.split = self.split_tags[idx]
        return data


def _load_node_dataset(
    name: str, 
    root: str, 
    transform,
    force_reload_raw: bool = False,
):
    """Load node-level dataset by name."""

    key = name.lower()
    if key in Actor_NAMES:
        return Actor(_scoped_root(root, "actor"), transform=transform)
    elif key in Airports_NAMES:
        dataset_key = Airports_NAMES[key]
        return Airports(_scoped_root(root, key), dataset_key, transform=transform)
    elif key in Amazon_NAMES:
        return Amazon(_scoped_root(root, key), Amazon_NAMES[key], transform=transform)
    elif key in CitationFull_NAMES:
        return CitationFull(_scoped_root(root, key), key, transform=transform)
    elif key in Coauthor_NAMES:
        return Coauthor(_scoped_root(root, key), Coauthor_NAMES[key], transform=transform)
    elif key in CoraFull_NAMES:
        return CoraFull(_scoped_root(root, key), transform=transform)
    elif key in EllipticBitcoinDataset_NAMES:
        return EllipticBitcoinDataset(_scoped_root(root, "elliptic_bitcoin"), transform=transform)
    elif key in EmailEUCore_NAMES:
        return EmailEUCore(_scoped_root(root, "email"), transform=transform)
    # elif key in FacebookPagePage_NAMES:
    #     return FacebookPagePage(_scoped_root(root, "facebook_page-page"), transform=transform)
    elif key in Flickr_NAMES:
        return Flickr(_scoped_root(root, "flickr"), transform=transform)
    # elif key in GemsecDeezer_NAMES:
    #     dataset_key = GemsecDeezer_NAMES[key]
    #     return GemsecDeezer(_scoped_root(root, "GemsecDeezer"), dataset_key, transform=transform)
    # elif key in GitHub_NAMES:
    #     return GitHub(_scoped_root(root, "GitHub"), transform=transform)
    elif key in HeterophilousGraphDataset_NAMES:
        dataset_key = HeterophilousGraphDataset_NAMES[key]
        return HeterophilousGraphDataset(_scoped_root(root, key), dataset_key, transform=transform)
    # elif key in LastFMAsia_NAMES:
    #     return LastFMAsia(_scoped_root(root, "LastFMAsia"), transform=transform)
    elif key in LINKXDataset_NAMES:
        return LINKXDataset(_scoped_root(root, key), key, transform=transform)
    elif key in LRGB_NODE_NAMES or key in LRGB_EDGE_NAMES:
        dataset_key = LRGB_NODE_NAMES.get(key) or LRGB_EDGE_NAMES[key]
        return LRGBDataset(_scoped_root(root, key), dataset_key, transform=transform)
    elif key in Planetoid_NAMES:
        # force_reload_raw lets us rebuild processed data from raw without deleting caches.
        return Planetoid(
            _scoped_root(root, key),
            Planetoid_NAMES[key],
            transform=transform,
            force_reload=force_reload_raw,
        )
    elif key in Reddit_NAMES:
        return Reddit(_scoped_root(root, "reddit"), transform=transform)
    elif key in Reddit2_NAMES:
        return Reddit2(_scoped_root(root, "reddit2"), transform=transform)
    # elif key in Twitch_NAMES:
    #     dataset_key = Twitch_NAMES[key]
    #     return Twitch(_scoped_root(root, "Twitch"), dataset_key, transform=transform)
    elif key.startswith("ogbn-"):
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
        except ImportError as exc:
            raise ImportError("Please install ogb to load ogbn-* datasets.") from exc
        with _force_ogb_prompts_yes():
            return PygNodePropPredDataset(name=key, root=_scoped_root(root, key), transform=transform)
    elif key in WebKB_NAMES:
        return WebKB(_scoped_root(root, key), key, transform=transform)
    elif key in WikiCS_NAMES:
        return WikiCS(_scoped_root(root, "wikics"), is_undirected=True, transform=transform)
    elif key in WikipediaNetwork_NAMES:
        return WikipediaNetwork(_scoped_root(root, key), key, transform=transform)
    else:
        raise ValueError(f"Unsupported node-level dataset: {name}")


def _load_graph_dataset(
    name: str, 
    root: str, 
    transform
):
    """Load graph-level dataset by name."""

    key = name.lower()
    if key in MoleculeNet_NAMES:
        return MoleculeNet(_scoped_root(root, key), key, transform=transform)
    elif key in GNNBenchmarkDataset_NAMES:
        dataset_key = GNNBenchmarkDataset_NAMES[key]
        return GNNBenchmarkDataset(_scoped_root(root, key), dataset_key, transform=transform)
    elif key in LRGB_GRAPH_NAMES:
        dataset_key = LRGB_GRAPH_NAMES[key]
        return LRGBDataset(_scoped_root(root, key), dataset_key, transform=transform)
    elif key in QM7b_NAMES:
        return QM7b(_scoped_root(root, key), transform=transform)
    elif key in QM9_NAMES:
        return QM9(_scoped_root(root, key), transform=transform)
    elif key in TUDataset_NAMES:
        dataset_key = TUDataset_NAMES[key]
        # Ensure nested raw dir exists to avoid fs.ls errors when download checks for files
        base = Path(root) / key
        (base / "raw" / dataset_key).mkdir(parents=True, exist_ok=True)
        return TUDataset(_scoped_root(root, key), dataset_key, transform=transform)
    elif key.startswith("ogbg-"):
        try:
            from ogb.graphproppred import PygGraphPropPredDataset
        except ImportError as exc:
            raise ImportError("Please install ogb to load ogbg-* datasets.") from exc
        with _force_ogb_prompts_yes():
            return PygGraphPropPredDataset(name=key, root=_scoped_root(root, key), transform=transform)
    else:
        raise ValueError(f"Unsupported graph-level dataset: {name}")


def create_dataset(
    name: str,
    root: str,
    task_level: str,
    feat_reduction: bool = True,
    feat_reduction_dim: int = 100,
    persist_feature_svd: bool = True,
    feature_svd_dir: str = "",
    induced: bool = False,
    induced_min_size: int = 10,
    induced_max_size: int = 30,
    induced_max_hops: int = 5,
    edge_max_size: int | None = 60,
    edge_sample_ratio: float | None = None,
    edge_sample_max: int | None = 100000,
    edge_sample_seed: int = 42,
    edge_supervised: bool = True,
    edge_neg_ratio: float = 1.0,
    edge_neg_seed: int = 42,
    cache_induced: bool = True,
    split: Tuple[float, float, float] | None = None,
    seed: int = 42,
    split_root: str = "",
    induced_root: str = "",
    force_reload_raw: bool = False,
):
    """Create dataset based on name and task level with optional feature reduction."""
    transforms = [EnsureFeatureTransform()]
    reducer = None
    if feat_reduction:
        if persist_feature_svd:
            reducer = SafeSVDFeatureReduction(out_channels=feat_reduction_dim)
        else:
            transforms.append(SafeSVDFeatureReduction(out_channels=feat_reduction_dim))
    transform = Compose(transforms) if transforms else None
    if task_level in ("node", "edge"):
        if induced:
            # Build induced subgraphs from the base node-level dataset.
            base_dataset = _load_node_dataset(
                name=name, 
                root=root, 
                transform=transform,
                force_reload_raw=force_reload_raw,
            )
            if reducer and hasattr(base_dataset, "data") and persist_feature_svd:
                _apply_feature_svd(
                    base_dataset,
                    name,
                    feat_reduction_dim,
                    reducer,
                    task_level=task_level,
                    output_root=feature_svd_dir,
                )
            base_data = base_dataset[0]
            base_name = name
            cache_root_path = Path(induced_root) if induced_root else (Path(split_root) if split_root else None)
            try:
                if task_level == "edge":
                    if not split_root:
                        raise ValueError("split_root is required for induced edge datasets.")
                    if split is None:
                        raise ValueError("split must be provided for induced edge datasets.")
                    split_root_path = Path(split_root)
                    split_def = split
                    _validate_edge_split_def(split_def)
                    cache_suffix = f"h{induced_max_hops}_s{_split_suffix(split_def)}_seed{seed}"
                    cache_meta = {
                        "task_level": "edge",
                        "max_hops": induced_max_hops,
                        "split": tuple(float(v) for v in split_def),
                        "seed": int(seed),
                    }
                    split_name = _canonical_split_dataset_name(base_name, "edge", int(seed))
                    split_payload = _get_or_create_edge_split_payload(
                        dataset_name=split_name,
                        split=split_def,
                        seed=int(seed),
                        split_root_path=split_root_path,
                        data=base_data,
                        persist=False,
                        verbose=False,
                    )
                    cache_path = _induced_cache_path(base_name, "edge", cache_root_path, cache_suffix) if cache_root_path else None
                    payload = _load_induced_cache(cache_path, cache_meta) if cache_induced and cache_path else None
                    if payload:
                        if cache_path:
                            print(f"[Induced] Loaded cached induced edge graphs from {cache_path}")
                        graphs = payload["graphs"]
                        split_tags = payload.get("split_tags")
                        result = InducedGraphDataset(
                            graphs,
                            base_info=get_basic_dataset_info(base_dataset),
                            base_num_nodes=getattr(base_data, "num_nodes", None),
                            base_num_edges=getattr(base_data, "num_edges", None),
                            split_tags=split_tags,
                        )
                        result.edge_split = tuple(float(v) for v in split_def)
                        result.edge_seed = int(seed)
                        result.edge_context = "train+message"
                        return result
                    if cache_induced and cache_path:
                        print(
                            "[Induced] Cache miss for edge induced graphs "
                            f"(split={_split_suffix(split_def)}, seed={int(seed)}), generating: {cache_path}"
                        )
                    print(f"[Induced] Processing edge induced subgraphs for {base_name}...")

                    edge_device = base_data.edge_index.device

                    def _edge_pairs_from_idx(indices) -> torch.Tensor:
                        idx = torch.as_tensor(indices, dtype=torch.long, device=edge_device)
                        if idx.numel() == 0:
                            return torch.empty((2, 0), dtype=torch.long, device=edge_device)
                        return base_data.edge_index[:, idx]

                    def _neg_pairs_from_payload(key: str) -> torch.Tensor:
                        neg_pairs = split_payload.get(key)
                        if neg_pairs is None:
                            return torch.empty((2, 0), dtype=torch.long, device=edge_device)
                        neg_pairs = torch.as_tensor(neg_pairs, dtype=torch.long, device=edge_device)
                        if neg_pairs.numel() == 0:
                            return torch.empty((2, 0), dtype=torch.long, device=edge_device)
                        if neg_pairs.dim() != 2 or neg_pairs.size(0) != 2:
                            raise ValueError(f"Invalid negative edge tensor for key={key}.")
                        return neg_pairs

                    train_pairs = _edge_pairs_from_idx(split_payload["train_pos_idx"])
                    message_pairs = _edge_pairs_from_idx(split_payload["message_pos_idx"])
                    # Edge-level context excludes val/test positives by construction.
                    context_pairs = torch.cat([train_pairs, message_pairs], dim=1)
                    context_data = Data(
                        x=getattr(base_data, "x", None),
                        edge_index=context_pairs,
                        num_nodes=base_data.num_nodes,
                    )

                    all_graphs = []
                    split_tags = []
                    for split_name, pos_key, neg_key in (
                        ("train", "train_pos_idx", "train_neg_edge_index"),
                        ("val", "val_pos_idx", "val_neg_edge_index"),
                        ("test", "test_pos_idx", "test_neg_edge_index"),
                    ):
                        pos_pairs = _edge_pairs_from_idx(split_payload[pos_key])
                        neg_pairs = _neg_pairs_from_payload(neg_key)
                        gs = build_edge_induced_graphs_supervised(
                            data=context_data,
                            pos_edge_pairs=pos_pairs,
                            neg_edge_pairs=neg_pairs,
                            max_hops=induced_max_hops,
                            max_size=edge_max_size,
                        )
                        all_graphs.extend(gs)
                        split_tags.extend([split_name] * len(gs))

                    if cache_induced and cache_path:
                        _save_induced_cache(
                            cache_path,
                            {
                                "graphs": all_graphs,
                                "split_tags": split_tags,
                                "base_num_nodes": getattr(base_data, "num_nodes", None),
                                "base_num_edges": getattr(base_data, "num_edges", None),
                                "meta": cache_meta,
                            },
                        )
                        print(f"[Induced] Saved induced edge graphs to {cache_path}")
                    result = InducedGraphDataset(
                        all_graphs,
                        base_info=get_basic_dataset_info(base_dataset),
                        base_num_nodes=getattr(base_data, "num_nodes", None),
                        base_num_edges=getattr(base_data, "num_edges", None),
                        split_tags=split_tags,
                    )
                    result.edge_split = tuple(float(v) for v in split_def)
                    result.edge_seed = int(seed)
                    result.edge_context = "train+message"
                    return result
                else: # task_level == "node"
                    graphs = None
                    split_tags = None
                    split_lookup = None
                    if split_root:
                        split_root_path = Path(split_root)
                        labels = getattr(base_data, "y", None)
                        labeled_idx = None
                        if labels is not None:
                            labels = labels.view(-1)
                            labeled_idx = torch.nonzero(labels >= 0, as_tuple=False).view(-1).tolist()

                        # Explicit split config must take precedence over built-in masks.
                        if split is not None:
                            split_def = split
                            _validate_split_def(split_def)
                            split_name = _canonical_split_dataset_name(base_name, "node", int(seed))
                            use_few_shot = _is_few_shot_split_def(split_def)
                            if use_few_shot:
                                train_idx, val_idx, test_idx = _get_or_create_few_shot_split(
                                    dataset_name=split_name,
                                    labels=labels,
                                    shots_per_class=int(split_def[0]),
                                    val_ratio=float(split_def[1]),
                                    test_ratio=float(split_def[2]),
                                    seed=seed,
                                    split_root_path=split_root_path,
                                )
                            elif labeled_idx is not None and len(labeled_idx) < base_data.num_nodes:
                                train_idx, val_idx, test_idx = _get_or_create_split_indices_subset(
                                    dataset_name=split_name,
                                    split=split_def,
                                    seed=seed,
                                    split_root_path=split_root_path,
                                    subset_indices=labeled_idx,
                                )
                            else:
                                train_idx, val_idx, test_idx = _get_or_create_split_indices(
                                    dataset_name=split_name,
                                    split=split_def,
                                    seed=seed,
                                    split_root_path=split_root_path,
                                    total=base_data.num_nodes,
                                )
                        else:
                            train_mask = getattr(base_data, "train_mask", None)
                            val_mask = getattr(base_data, "val_mask", None)
                            test_mask = getattr(base_data, "test_mask", None)
                            if train_mask is None or val_mask is None or test_mask is None:
                                raise ValueError(
                                    "split must be provided for induced node datasets when masks are unavailable."
                                )
                            train_idx = _mask_to_node_indices(train_mask, "train_mask")
                            val_idx = _mask_to_node_indices(val_mask, "val_mask")
                            test_idx = _mask_to_node_indices(test_mask, "test_mask")
                        split_lookup = {idx: "train" for idx in train_idx}
                        split_lookup.update({idx: "val" for idx in val_idx})
                        split_lookup.update({idx: "test" for idx in test_idx})
                    if cache_induced and cache_root_path:
                        cache_suffix = f"h{induced_max_hops}_s{induced_min_size}-{induced_max_size}"
                        cache_meta = {
                            "task_level": "node",
                            "max_hops": induced_max_hops,
                            "min_size": induced_min_size,
                            "max_size": induced_max_size,
                        }
                        cache_path = _induced_cache_path(base_name, "node", cache_root_path, cache_suffix)
                        payload = _load_induced_cache(cache_path, cache_meta)
                        if payload:
                            if cache_path:
                                print(f"[Induced] Loaded cached induced node graphs from {cache_path}")
                            graphs = payload["graphs"]
                            split_tags = payload.get("split_tags")
                            if graphs is not None and split_lookup is not None:
                                split_tags = [
                                    split_lookup.get(getattr(g, "base_node_id", idx), "train")
                                    for idx, g in enumerate(graphs)
                                ]
                    if graphs is None:
                        print(f"[Induced] Processing node induced subgraphs for {base_name}...")
                        graphs = build_induced_graphs(
                            data=base_data,
                            smallest_size=induced_min_size,
                            largest_size=induced_max_size,
                            max_hops=induced_max_hops,
                        )
                        if split_lookup is not None:
                            split_tags = [split_lookup.get(g.base_node_id, "train") for g in graphs]
                        if cache_induced and cache_root_path:
                            _save_induced_cache(
                                cache_path,
                                {
                                    "graphs": graphs,
                                    "split_tags": split_tags,
                                    "base_num_nodes": getattr(base_data, "num_nodes", None),
                                    "base_num_edges": getattr(base_data, "num_edges", None),
                                    "meta": cache_meta,
                                },
                            )
                            print(f"[Induced] Saved induced node graphs to {cache_path}")
            except Exception as exc:
                raise RuntimeError(f"Induced graph generation failed for {name} ({task_level}): {exc}") from exc
            if graphs:
                return InducedGraphDataset(
                    graphs,
                    base_info=get_basic_dataset_info(base_dataset),
                    base_num_nodes=getattr(base_data, "num_nodes", None),
                    base_num_edges=getattr(base_data, "num_edges", None),
                    split_tags=split_tags,
                )
            raise RuntimeError(f"[Induced] No induced graphs generated for {name} ({task_level}).")
        dataset = _load_node_dataset(
            name=name, 
            root=root, 
            transform=transform
        )
        if reducer and _get_dataset_data_storage(dataset) is not None and persist_feature_svd:
            _apply_feature_svd(
                dataset,
                name,
                feat_reduction_dim,
                reducer,
                task_level=task_level,
                output_root=feature_svd_dir,
            )
            try:
                dataset._svd_dim = feat_reduction_dim  # type: ignore[attr-defined]
                dataset._svd_task_level = task_level  # type: ignore[attr-defined]
                dataset._feature_svd_root = feature_svd_dir or dataset.root  # type: ignore[attr-defined]
            except Exception:
                pass
        return dataset

    elif task_level == "graph":
        # For graph-level tasks, induced flag is a no-op; return the graph dataset.
        dataset = _load_graph_dataset(
            name=name, 
            root=root, 
            transform=transform
        )
        # Optional one-time SVD cache for graph datasets
        if reducer and _get_dataset_data_storage(dataset) is not None and persist_feature_svd:
            _apply_feature_svd(
                dataset,
                name,
                feat_reduction_dim,
                reducer,
                task_level=task_level,
                output_root=feature_svd_dir,
            )
            try:
                dataset._svd_dim = feat_reduction_dim  # type: ignore[attr-defined]
                dataset._svd_task_level = task_level  # type: ignore[attr-defined]
                dataset._feature_svd_root = feature_svd_dir or dataset.root  # type: ignore[attr-defined]
            except Exception:
                pass
        return dataset

    else:
        raise ValueError(f"Unsupported task_level: {task_level}")


def _feature_task_suffix(task_level: str | None) -> str:
    if task_level == "edge":
        return "_edge"
    if task_level == "graph":
        return "_graph"
    if task_level == "node" or task_level is None:
        return "_node"
    return f"_{str(task_level)}"


def _feature_svd_path(dataset_root: str, name: str, dim: int, task_level: str | None = None) -> Path:
    suffix = _feature_task_suffix(task_level)
    dataset_dir = _dataset_scoped_dir(dataset_root, name)
    return dataset_dir / f"feature_svd_{name}{suffix}_d{dim}.pt"


def _apply_feature_svd(
    dataset,
    name: str,
    dim: int,
    reducer: SafeSVDFeatureReduction,
    task_level: str | None = None,
    output_root: str | None = None,
):
    """Apply SVD once and persist reduced features to avoid recompute."""
    root = output_root or dataset.root
    save_path = _feature_svd_path(root, name, dim, task_level=task_level)

    dataset_data = _get_dataset_data_storage(dataset)
    target_x = getattr(dataset_data, "x", None) if dataset_data is not None else None
    target_device = target_x.device if target_x is not None else "cpu"
    if save_path.exists():
        try:
            payload = torch.load(save_path, map_location=target_device)
            dataset_data = _get_dataset_data_storage(dataset)
            if dataset_data is None:
                raise ValueError("Dataset has no in-memory data storage.")
            dataset_data.x = payload["x"].to(target_device)
            return
        except Exception:
            save_path.unlink(missing_ok=True)

    print(f"[FeatureSVD] Processing features for {name} (task={task_level or 'node'})...")

    # Apply reducer once on the aggregated data.
    data = _get_dataset_data_storage(dataset)
    if data is None:
        raise ValueError("Dataset has no in-memory data storage for feature SVD.")
    if getattr(data, "x", None) is None and getattr(dataset, "transform", None) is not None:
        try:
            data = dataset.transform(data.clone())
        except Exception:
            pass
    reduced = reducer(data)
    _set_dataset_data_storage(dataset, reduced)
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"x": reduced.x.cpu()}, save_path)
        print(f"[FeatureSVD] Saved features to {save_path}")
    except Exception:
        pass

    # Surface meta so downstream loaders (e.g., LRGB multi-graph node datasets) can reuse persisted features.
    try:
        dataset._svd_dim = dim  # type: ignore[attr-defined]
        dataset._svd_task_level = task_level  # type: ignore[attr-defined]
        dataset._feature_svd_root = root  # type: ignore[attr-defined]
    except Exception:
        pass


def split_graph_dataset(
    dataset,
    dataset_name: str,
    split: Tuple[float, float, float],
    seed: int,
    split_root: str,
):
    """Split graph-level dataset into train/val/test sets (supports few-shot)."""
    if not split_root:
        raise ValueError("split_root is required to save or load fixed splits.")

    split_root_path = Path(split_root)
    split_dataset_name = _canonical_split_dataset_name(dataset_name, "graph", int(seed))

    _validate_split_def(split)

    # Detect few-shot: first element is int-like (e.g., 5 or 5.0)
    use_few_shot = False
    first = split[0] if isinstance(split, (list, tuple)) and split else split
    try:
        first_val = float(first)
        use_few_shot = float(first_val).is_integer()
    except Exception:
        use_few_shot = isinstance(first, Integral) and not isinstance(first, bool)

    if use_few_shot:
        labels_list = []
        for item in dataset:
            if not hasattr(item, "y") or item.y is None:
                raise ValueError("Few-shot split requires labels for each graph instance.")
            target = item.y.view(-1)
            if target.numel() != 1:
                raise ValueError("Few-shot split currently supports single-label targets.")
            labels_list.append(int(target[0].item()))
        labels = torch.tensor(labels_list, dtype=torch.long)
        train_idx, val_idx, test_idx = _get_or_create_few_shot_split(
            dataset_name=split_dataset_name,
            labels=labels,
            shots_per_class=int(split[0]),
            val_ratio=float(split[1]),
            test_ratio=float(split[2]),
            seed=seed,
            split_root_path=split_root_path,
        )
    else:
        train_idx, val_idx, test_idx = _get_or_create_split_indices(
            dataset_name=split_dataset_name,
            split=split,
            seed=seed,
            split_root_path=split_root_path,
            total=len(dataset),
        )

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


def _get_or_create_few_shot_split(
    dataset_name: str,
    labels: torch.Tensor,
    shots_per_class: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    split_root_path: Path,
):
    """Create or load a class-balanced few-shot split."""
    if split_root_path is None:
        raise ValueError("split_root_path must be provided (configure cfg.dataset.split_root).")

    dataset_split_dir = _split_dataset_dir(split_root_path, dataset_name)
    split_tag = _few_shot_suffix(shots_per_class, val_ratio, test_ratio)
    split_path = dataset_split_dir / f"{dataset_name}_splits-{split_tag}.pt"
    
    labels = labels.view(-1).cpu()
    total = labels.numel()

    existing = _load_existing_indices(split_path, total)
    if existing:
        train_idx, val_idx, test_idx = existing
        print(f"[Dataset split] Loaded few-shot split from {split_path}")
        return train_idx, val_idx, test_idx

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

    # Sanity check to avoid silent mistakes.
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
        },
    }
    torch.save(payload, split_path)
    print(f"[Dataset split] Saved few-shot split to {split_path}")
    return train_idx, val_idx, test_idx


def make_loaders(
    dataset,
    dataset_name: str,
    task_level: str,
    batch_size: int,
    num_workers: int,
    split: Tuple[float, float, float],
    seed: int,
    induced: bool = False,
    split_root: str = "",
    edge_pred_cfg=None,
    drop_last_train: bool = False,
):
    """Create data loaders for training, validation, and testing."""
    raw_dataset_name = str(dataset_name)
    split_dataset_name = _canonical_split_dataset_name(raw_dataset_name, task_level, int(seed))
    if split is not None:
        if task_level == "edge":
            _validate_edge_split_def(split)
        else:
            _validate_split_def(split)

    def _is_few_shot_split(split_def) -> bool:
        if not isinstance(split_def, (list, tuple)) or not split_def:
            return False
        first = split_def[0]
        if isinstance(first, bool):
            return False
        if isinstance(first, Integral):
            return True
        if isinstance(first, float) and first.is_integer():
            return True
        return False

    def _few_shot_indices_from_graphs():
        if len(split) < 3:
            raise ValueError("Few-shot split must provide [shots_per_class, val_ratio, test_ratio].")
        shots_per_class = int(split[0])
        val_ratio = float(split[1])
        test_ratio = float(split[2])
        split_root_path = Path(split_root) if split_root else None

        labels = None
        dataset_data = _get_dataset_data_storage(dataset)
        if dataset_data is not None and getattr(dataset_data, "y", None) is not None:
            labels = dataset_data.y
            if labels.numel() == len(dataset):
                labels = labels.view(-1)
        if labels is None or labels.numel() != len(dataset):
            collected = []
            for item in dataset:
                if not hasattr(item, "y") or item.y is None:
                    raise ValueError("Few-shot split requires labels for each graph instance.")
                target = item.y.view(-1)
                if target.numel() != 1:
                    raise ValueError("Few-shot split currently supports single-label targets.")
                collected.append(int(target[0].item()))
            labels = torch.tensor(collected, dtype=torch.long)
        return _get_or_create_few_shot_split(
            dataset_name=split_dataset_name,
            labels=labels,
            shots_per_class=shots_per_class,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            split_root_path=split_root_path,
        )

    use_few_shot = _is_few_shot_split(split)
    if use_few_shot and is_regression_dataset(dataset, task_level):
        raise ValueError("Few-shot split is not supported for regression tasks.")

    # LRGB PascalVOC-SP / COCO-SP are multi-graph node tasks with built-in train/val/test splits.
    if task_level == "node" and not induced and isinstance(dataset, LRGBDataset) and raw_dataset_name.lower() in LRGB_NODE_NAMES:
        loader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=drop_last_train,
        )
        train_ds = dataset  # already loaded (train split)
        val_ds = LRGBDataset(dataset.root, dataset.name, split="val", transform=dataset.transform)
        test_ds = LRGBDataset(dataset.root, dataset.name, split="test", transform=dataset.transform)

        # Reuse persisted SVD reduction if it was applied during dataset creation.
        svd_dim = getattr(dataset, "_svd_dim", None)
        svd_task = getattr(dataset, "_svd_task_level", None)
        feature_svd_root = getattr(dataset, "_feature_svd_root", None)
        if svd_dim and svd_task == "node":
            reducer = SafeSVDFeatureReduction(out_channels=svd_dim)
            for ds in (train_ds, val_ds, test_ds):
                try:
                    _apply_feature_svd(
                        ds,
                        raw_dataset_name,
                        svd_dim,
                        reducer,
                        task_level="node",
                        output_root=feature_svd_root,
                    )
                except Exception:
                    pass

        train_loader = DataLoader(train_ds, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        return train_loader, val_loader, test_loader

    if task_level == "node" and not induced:
        data = dataset[0]
        if not split_root:
            raise ValueError("split_root is required to save or load fixed splits for node datasets.")
        split_root_path = Path(split_root)
        labels = getattr(data, "y", None)
        labeled_idx = None
        if labels is not None:
            labels = labels.view(-1)
            labeled_idx = torch.nonzero(labels >= 0, as_tuple=False).view(-1).tolist()
        if use_few_shot:
            if len(split) < 3:
                raise ValueError("Few-shot split must provide [shots_per_class, val_ratio, test_ratio].")
            train_idx, val_idx, test_idx = _get_or_create_few_shot_split(
                dataset_name=split_dataset_name,
                labels=data.y,
                shots_per_class=int(split[0]),
                val_ratio=float(split[1]),
                test_ratio=float(split[2]),
                seed=seed,
                split_root_path=split_root_path,
            )
        else:
            if labeled_idx is not None and len(labeled_idx) < data.num_nodes:
                train_idx, val_idx, test_idx = _get_or_create_split_indices_subset(
                    dataset_name=split_dataset_name,
                    split=split,
                    seed=seed,
                    split_root_path=split_root_path,
                    subset_indices=labeled_idx,
                )
            else:
                train_idx, val_idx, test_idx = _get_or_create_split_indices(
                    dataset_name=split_dataset_name,
                    split=split,
                    seed=seed,
                    split_root_path=split_root_path,
                    total=data.num_nodes,
                )
        # Populate boolean masks for downstream use (and to reflect loaded splits).
        for mask_name, indices in (
            ("train_mask", train_idx),
            ("val_mask", val_idx),
            ("test_mask", test_idx),
        ):
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[indices] = True
            setattr(data, mask_name, mask)
        return (
            SingleGraphDataLoader(data),
            SingleGraphDataLoader(data),
            SingleGraphDataLoader(data),
        )

    if task_level == "edge" and not induced:
        data = dataset[0]
        if not split_root:
            raise ValueError("split_root is required to save or load fixed splits for edge datasets.")
        split_root_path = Path(split_root)
        if use_few_shot:
            raise ValueError("Few-shot split is not supported for edge-level tasks.")
        split_payload = _get_or_create_edge_split_payload(
            dataset_name=split_dataset_name,
            split=split,
            seed=seed,
            split_root_path=split_root_path,
            data=data,
        )

        edge_device = data.edge_index.device
        message_idx = torch.as_tensor(split_payload["message_pos_idx"], dtype=torch.long, device=edge_device)
        if message_idx.numel() == 0:
            message_edge_index = torch.empty((2, 0), dtype=torch.long, device=edge_device)
        else:
            message_edge_index = data.edge_index[:, message_idx]

        def _edge_subset(pos_key: str, neg_key: str):
            pos_idx = torch.as_tensor(split_payload[pos_key], dtype=torch.long, device=edge_device)
            if pos_idx.numel() == 0:
                pos_pairs = torch.empty((2, 0), dtype=torch.long, device=edge_device)
            else:
                pos_pairs = data.edge_index[:, pos_idx]

            neg_source = split_payload.get(neg_key)
            if neg_source is None:
                neg_pairs = torch.empty((2, 0), dtype=torch.long, device=edge_device)
            else:
                neg_pairs = torch.as_tensor(neg_source, dtype=torch.long, device=edge_device)
                if neg_pairs.numel() == 0:
                    neg_pairs = torch.empty((2, 0), dtype=torch.long, device=edge_device)
                elif neg_pairs.dim() != 2 or neg_pairs.size(0) != 2:
                    raise ValueError(f"Invalid negative edge tensor for key={neg_key}.")

            edge_label_index = torch.cat([pos_pairs, neg_pairs], dim=1)
            edge_label = torch.cat(
                [
                    torch.ones(pos_pairs.size(1), dtype=torch.float, device=edge_label_index.device),
                    torch.zeros(neg_pairs.size(1), dtype=torch.float, device=edge_label_index.device),
                ],
                dim=0,
            )
            subset = Data(
                x=getattr(data, "x", None),
                edge_index=message_edge_index,
                edge_label_index=edge_label_index,
                num_nodes=data.num_nodes,
            )
            subset.edge_label = edge_label
            return subset

        train_data = _edge_subset("train_pos_idx", "train_neg_edge_index")
        val_data = _edge_subset("val_pos_idx", "val_neg_edge_index")
        test_data = _edge_subset("test_pos_idx", "test_neg_edge_index")

        edge_cfg = edge_pred_cfg
        use_neighbor_sampling = bool(getattr(edge_cfg, "use_neighbor_sampling", False)) if edge_cfg else False
        if use_neighbor_sampling:
            if LinkNeighborLoader is None:
                raise ImportError("LinkNeighborLoader is required for neighbor sampling.")
            sizes = list(getattr(edge_cfg, "neighbor_sizes", [15, 10]))
            edge_batch_size = int(getattr(edge_cfg, "edge_batch_size", batch_size))

            def _link_loader(edge_data, shuffle):
                return LinkNeighborLoader(
                    edge_data,
                    edge_label_index=edge_data.edge_label_index,
                    edge_label=edge_data.edge_label,
                    num_neighbors=sizes,
                    batch_size=edge_batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    neg_sampling_ratio=0.0,
                )

            return (
                _link_loader(train_data, shuffle=True),
                _link_loader(val_data, shuffle=False),
                _link_loader(test_data, shuffle=False),
            )

        return (
            SingleGraphDataLoader(train_data),
            SingleGraphDataLoader(val_data),
            SingleGraphDataLoader(test_data),
        )

    if task_level == "edge" and induced and hasattr(dataset, "split_tags"):
        train_idx = [i for i, tag in enumerate(dataset.split_tags) if tag == "train"]
        val_idx = [i for i, tag in enumerate(dataset.split_tags) if tag == "val"]
        test_idx = [i for i, tag in enumerate(dataset.split_tags) if tag == "test"]
        # Edge-induced datasets use explicit split tags; empty val/test is allowed on tiny datasets.
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        test_set = Subset(dataset, test_idx)
    elif task_level == "graph" and hasattr(dataset, "split_tags") and dataset.split_tags:
        train_idx = [i for i, tag in enumerate(dataset.split_tags) if tag == "train"]
        val_idx = [i for i, tag in enumerate(dataset.split_tags) if tag == "val"]
        test_idx = [i for i, tag in enumerate(dataset.split_tags) if tag == "test"]
        if not val_idx or not test_idx:
            train_set, val_set, test_set = split_graph_dataset(
                dataset=dataset,
                dataset_name=split_dataset_name,
                split=split,
                seed=seed,
                split_root=split_root,
            )
        else:
            train_set = Subset(dataset, train_idx)
            val_set = Subset(dataset, val_idx)
            test_set = Subset(dataset, test_idx)
    elif use_few_shot:
        train_idx, val_idx, test_idx = _few_shot_indices_from_graphs()
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        test_set = Subset(dataset, test_idx)
    else:
        train_set, val_set, test_set = split_graph_dataset(
            dataset=dataset, 
            dataset_name=split_dataset_name,
            split=split, 
            seed=seed,
            split_root=split_root,
        )
    loader_kwargs = dict(
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=True,
        drop_last=drop_last_train,
    )
    train_loader = DataLoader(
        dataset=train_set, 
        **loader_kwargs
    )
    val_loader = DataLoader(
        dataset=val_set, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False
    )
    test_loader = DataLoader(
        dataset=test_set, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False
    )
    return train_loader, val_loader, test_loader


def get_basic_dataset_info(dataset) -> dict:
    """
    Extract minimal context about a PyTorch Geometric dataset.
    Returns a dictionary with:
        - domain: estimated domain (str)
        - source: dataset class path (str)
    """
    cls = dataset.__class__
    cls_name = cls.__name__
    # If this is an induced dataset, prefer base dataset info when available.
    base_info = getattr(dataset, "base_dataset_info", None)
    dataset_name = getattr(dataset, "name", None) or cls_name
    if base_info and base_info.get("name"):
        dataset_name = base_info["name"]

    name_key = str(dataset_name).lower()
    dataset_domain = base_info.get("domain") if base_info else None
    if dataset_domain is None:
        dataset_domain = CLASS_TO_DOMAIN.get(cls_name)
    if dataset_domain is None:
        dataset_domain = NAME_TO_DOMAIN.get(name_key)
    if dataset_domain is None:
        combined = f"{cls_name.lower()} {name_key}"
        for domain, keywords in KEYWORD_DOMAINS:
            if any(token in combined for token in keywords):
                dataset_domain = domain
                break
    dataset_domain = dataset_domain or "unknown"

    dataset_source = f"{cls.__module__}.{cls_name}"

    return {
        "domain": dataset_domain,
        "source": dataset_source,
    }


def dataset_info(
    dataset, 
    task_level: str, 
    name: str = "", 
    induced: bool = False
) -> Dict:
    """Get meta information about the dataset."""
    def _log_info(level: str, info: Dict) -> None:
        """Pretty-print dataset statistics."""

        ordered = []
        for key in (
            "num_node_features",
            "num_classes",
            "num_nodes",
            "num_edges",
            "num_graphs",
            "avg_nodes_per_graph",
            "avg_edges_per_graph",
        ):
            if key in info:
                ordered.append(f"{key}={info[key]}")
        summary = ", ".join(ordered)
        prefix = f"{name}: " if name else ""
        print(f"[Dataset info][{level}] {prefix}{summary}")

    if task_level == "node":
        data = dataset[0]
        if induced and not hasattr(dataset, "graphs"):
            raise RuntimeError("Induced node dataset missing graphs; generation failed.")
        num_classes = None
        label_dim = None
        if hasattr(dataset, "num_classes") and dataset.num_classes is not None:
            num_classes = dataset.num_classes
        elif getattr(data, "y", None) is not None and data.y.numel() > 0:
            num_classes = int(data.y.max().item() + 1)
        if getattr(data, "y", None) is not None:
            y = torch.as_tensor(data.y)
            if y.dim() <= 1:
                label_dim = 1
            else:
                label_dim = int(y.size(-1))

        info = {
            "num_node_features": data.num_node_features,
            "num_classes": num_classes,
            "label_dim": label_dim,
            "num_nodes": dataset.base_num_nodes if induced and getattr(dataset, "base_num_nodes", None) is not None else (len(dataset) if induced else data.num_nodes),
            "num_edges": dataset.base_num_edges if induced and getattr(dataset, "base_num_edges", None) is not None else data.num_edges,
        }
        if induced:
            info["num_graphs"] = len(dataset)
            info["avg_nodes_per_graph"] = round(float(sum(g.num_nodes for g in dataset.graphs) / len(dataset)), 2)
            info["avg_edges_per_graph"] = round(float(sum(g.num_edges for g in dataset.graphs) / len(dataset)), 2)
        _log_info("induced-node" if induced else "node", info)
        return info
    
    elif task_level == "edge":
        data = dataset[0]
        if induced and not hasattr(dataset, "graphs"):
            raise RuntimeError("Induced edge dataset missing graphs; generation failed.")
        num_classes = None
        label_dim = None
        if hasattr(dataset, "num_classes") and dataset.num_classes is not None:
            num_classes = dataset.num_classes
        elif getattr(data, "y", None) is not None and data.y.numel() > 0:
            num_classes = int(data.y.max().item() + 1)
        if getattr(data, "y", None) is not None:
            y = torch.as_tensor(data.y)
            if y.dim() <= 1:
                label_dim = 1
            else:
                label_dim = int(y.size(-1))
        info = {
            "num_node_features": data.num_node_features,
            "num_classes": num_classes,
            "label_dim": label_dim,
            "num_nodes": dataset.base_num_nodes if induced and getattr(dataset, "base_num_nodes", None) is not None else (len(dataset) if induced else data.num_nodes),
            "num_edges": dataset.base_num_edges if induced and getattr(dataset, "base_num_edges", None) is not None else data.num_edges,
        }
        if induced and hasattr(dataset, "graphs"):
            info["num_graphs"] = len(dataset)
            info["avg_nodes_per_graph"] = round(float(sum(g.num_nodes for g in dataset.graphs) / len(dataset)), 2)
            info["avg_edges_per_graph"] = round(float(sum(g.num_edges for g in dataset.graphs) / len(dataset)), 2)
        _log_info("induced-edge" if induced and hasattr(dataset, "graphs") else "edge", info)
        return info
    
    elif task_level == "graph":
        sample = dataset[0]
        sample_y = getattr(sample, "y", None)
        if sample_y is None:
            label_dim = None
        else:
            y = torch.as_tensor(sample_y)
            if y.dim() <= 1:
                label_dim = 1
            else:
                label_dim = int(y.size(-1))
        info = {
            "num_node_features": sample.num_features,
            "num_classes": dataset.num_classes if hasattr(dataset, "num_classes") else None,
            "label_dim": label_dim,
            "num_graphs": len(dataset),
            "avg_nodes_per_graph": round(float(sum(g.num_nodes for g in dataset) / len(dataset)), 2),
            "avg_edges_per_graph": round(float(sum(g.num_edges for g in dataset) / len(dataset)), 2),
        }
        _log_info("induced-graph" if induced else "graph", info)
        return info

    raise ValueError(f"Unsupported task_level: {task_level}")


def _subgraph_svd_path(
    output_dir: str,
    dataset_name: str,
    task_level: str,
    feat_dim: int,
    struct_dim: int,
    matrix_type: str,
    edge_split: Optional[Tuple[float, float, float]] = None,
    edge_seed: Optional[int] = None,
) -> Path:
    sanitized = str(dataset_name).replace("/", "_").replace(" ", "_")
    matrix_type = str(matrix_type or "adjacency").lower()
    dataset_dir = _dataset_scoped_dir(output_dir, sanitized)
    parts = [f"subgraph_svd_{sanitized}_{task_level}"]
    if task_level == "edge" and edge_split is not None:
        parts.append(f"split{_split_suffix(edge_split)}")
    if task_level == "edge" and edge_seed is not None:
        parts.append(f"seed{int(edge_seed)}")
    if feat_dim and feat_dim > 0:
        parts.append(f"feat{feat_dim}")
    parts.append(f"struct{struct_dim}")
    parts.append(matrix_type)
    return dataset_dir / f"{'_'.join(parts)}.pt"


def _svd_singular_values(matrix: torch.Tensor, target_dim: int) -> torch.Tensor:
    if target_dim <= 0:
        return torch.empty((0,), dtype=torch.float32)
    if matrix is None or matrix.numel() == 0:
        return torch.zeros(target_dim, dtype=torch.float32)
    try:
        _, s, _ = torch.linalg.svd(matrix.float(), full_matrices=False)
        vec = s[:target_dim]
    except Exception:
        flat = matrix.flatten()
        vec = flat[:target_dim].float() if flat.numel() else torch.zeros(target_dim, dtype=torch.float32)
    if vec.numel() < target_dim:
        pad = torch.zeros(target_dim - vec.numel(), dtype=torch.float32)
        vec = torch.cat([vec, pad], dim=0)
    return vec.to(torch.float32)


def _subgraph_structure_matrix(data: Data, matrix_type: str) -> torch.Tensor:
    num_nodes = int(getattr(data, "num_nodes", 0) or 0)
    if num_nodes <= 0:
        return torch.empty((0, 0), dtype=torch.float32)
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj = (adj + adj.t()).clamp(max=1.0)
    matrix_type = str(matrix_type or "adjacency").lower()
    if matrix_type == "laplacian":
        deg = torch.diag(adj.sum(dim=1))
        return deg - adj
    return adj


def compute_subgraph_svd_features(
    dataset,
    dataset_name: str,
    task_level: str,
    feat_dim: int,
    struct_dim: int,
    matrix_type: str,
    output_dir: str,
    overwrite: bool = False,
) -> Path | None:
    """Compute and cache per-subgraph SVD feature/structure vectors."""
    if feat_dim <= 0 and struct_dim <= 0:
        return None
    edge_split = None
    edge_seed = None
    edge_context = None
    if task_level == "edge":
        raw_split = getattr(dataset, "edge_split", None)
        if isinstance(raw_split, (list, tuple)) and len(raw_split) >= 3:
            edge_split = (float(raw_split[0]), float(raw_split[1]), float(raw_split[2]))
        raw_seed = getattr(dataset, "edge_seed", None)
        if raw_seed is not None:
            edge_seed = int(raw_seed)
        edge_context = getattr(dataset, "edge_context", None)

    output_dir = str(output_dir)
    path = _subgraph_svd_path(
        output_dir,
        dataset_name,
        task_level,
        feat_dim,
        struct_dim,
        matrix_type,
        edge_split=edge_split,
        edge_seed=edge_seed,
    )
    if path.exists() and not overwrite:
        print(f"[SubgraphSVD] Found cached features at {path}, skipping.")
        return path
    print(f"[SubgraphSVD] Processing features for {dataset_name} (task={task_level})...")

    graphs = getattr(dataset, "graphs", None)
    if graphs is None:
        if hasattr(dataset, "__len__"):
            graphs = [dataset[idx] for idx in range(len(dataset))]
        else:
            return None
    if not graphs:
        return None

    feat_out = torch.zeros((len(graphs), feat_dim), dtype=torch.float32) if feat_dim > 0 else torch.empty((len(graphs), 0))
    struct_out = torch.zeros((len(graphs), struct_dim), dtype=torch.float32) if struct_dim > 0 else torch.empty((len(graphs), 0))
    for idx, data in enumerate(graphs):
        if feat_dim > 0:
            x = getattr(data, "x", None)
            feat_out[idx] = _svd_singular_values(x, feat_dim)
        if struct_dim > 0:
            mat = _subgraph_structure_matrix(data, matrix_type)
            struct_out[idx] = _svd_singular_values(mat, struct_dim)

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feat_svd": feat_out,
        "struct_svd": struct_out,
        "meta": {
            "dataset": dataset_name,
            "task_level": task_level,
            "feat_dim": feat_dim,
            "struct_dim": struct_dim,
            "matrix_type": matrix_type,
            "edge_split": edge_split,
            "edge_seed": edge_seed,
            "edge_context": edge_context,
        },
    }
    torch.save(payload, path)
    print(f"[SubgraphSVD] Saved features to {path}")
    return path


def split_instance_counts(train_loader, val_loader, test_loader, task_level: str) -> Dict[str, Optional[int]]:
    """Best-effort counts of items per split based on available loader metadata."""
    def _mask_count(data, split_name: str):
        mask = getattr(data, f"{split_name}_mask", None)
        if mask is None:
            return None
        try:
            return int(torch.as_tensor(mask).sum().item())
        except Exception:
            return None

    def _count(loader, split_name: str):
        if loader is None:
            return None
        data = getattr(loader, "data", None)
        if data is not None:
            if task_level == "node":
                mask_total = _mask_count(data, split_name)
                if mask_total is not None:
                    return mask_total
                num_nodes = getattr(data, "num_nodes", None)
                if num_nodes is None and getattr(data, "x", None) is not None:
                    num_nodes = data.x.size(0)
                return int(num_nodes) if num_nodes is not None else None
            if task_level == "edge":
                mask_total = _mask_count(data, split_name)
                if mask_total is not None:
                    return mask_total
                edge_label_index = getattr(data, "edge_label_index", None)
                if edge_label_index is not None:
                    return int(edge_label_index.size(1))
                edge_index = getattr(data, "edge_index", None)
                if edge_index is not None:
                    return int(edge_index.size(1))
        dataset = getattr(loader, "dataset", None)
        if dataset is not None:
            try:
                return len(dataset)
            except Exception:
                return None
        return None

    counts = {
        split_name: _count(loader, split_name)
        for split_name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader))
    }
    return counts


def log_split_instance_counts(
    train_loader,
    val_loader,
    test_loader,
    task_level: str,
    split: Optional[Tuple[float, float, float]] = None,
    induced: bool = False,
    prefix: str = "[Dataset split]",
) -> Dict[str, Optional[int]]:
    """Print a concise summary of split instance counts."""
    counts = split_instance_counts(train_loader, val_loader, test_loader, task_level=task_level)
    parts = []
    for key in ("train", "val", "test"):
        value = counts.get(key)
        parts.append(f"{key}={value if value is not None else '?'}")
    split_label = ""
    if split is not None:
        formatted = format_split_for_name(split)
        split_label = f", split={formatted or split}"
    induced_label = ", induced" if induced else ""
    print(f"{prefix} ({task_level}{induced_label}{split_label}): " + ", ".join(parts))
    return counts
