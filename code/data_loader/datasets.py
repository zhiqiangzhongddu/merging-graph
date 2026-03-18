import contextlib
from pathlib import Path
from typing import List, Optional, Tuple
import warnings

import torch
from torch.utils.data import Subset
from torch_geometric.data import Data
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
    Flickr,
    HeterophilousGraphDataset,
    LINKXDataset,
    MoleculeNet,
    Planetoid,
    QM7b,
    QM9,
    Reddit,
    Reddit2,
    TUDataset,
    WebKB,
    WikiCS,
    WikipediaNetwork,
    ZINC,
)
from torch_geometric.loader import DataLoader, LinkNeighborLoader
from torch_geometric.transforms import Compose

# OGB pulls in `outdated`, which still imports `pkg_resources.parse_version`.
# Keep that one third-party deprecation warning out of experiment logs.
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module=r"outdated(\..*)?",
)
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset

from .dataset_metadata import (
    dataset_info,
    get_basic_dataset_info,
    is_regression_dataset,
    log_split_instance_counts,
    resolve_count_split_strategy,
    split_instance_counts,
)
from .dataset_paths import _scoped_root
from .dataset_splits import (
    _canonical_split_dataset_name,
    _get_or_create_count_split,
    _get_or_create_edge_split_payload,
    _get_or_create_few_shot_split,
    _get_or_create_split_indices,
    _get_or_create_split_indices_subset,
    _is_few_shot_split_def,
    _mask_to_node_indices,
    _split_suffix,
    _validate_edge_split_def,
    _validate_split_def,
    split_graph_dataset,
)
from .dataset_storage import _get_dataset_data_storage, _unwrap_subset_dataset
from .filter_empty_graph import _graph_filter_cache_meta_for_dataset, _sanitize_graph_dataset
from .induced_graphs import (
    InducedGraphDataset,
    SingleGraphDataLoader,
    _induced_cache_path,
    _load_induced_cache,
    _save_induced_cache,
    build_edge_induced_graphs,
    build_edge_induced_graphs_supervised,
    build_induced_graphs,
)
from .svd_features import (
    EnsureFeatureTransform,
    SafeSVDFeatureReduction,
    _apply_feature_svd,
    compute_subgraph_svd_features,
)
from .zinc15_dataset import ZINC15Dataset


# Keep logs clean when third-party internals still touch InMemoryDataset.data.
warnings.filterwarnings(
    "ignore",
    message="It is not recommended to directly access the internal storage format `data` of an 'InMemoryDataset'.*",
    category=UserWarning,
    module="torch_geometric.data.in_memory_dataset",
)
# Sparse CSR tensor support is still in beta, and some datasets trigger related warnings when loading.
warnings.filterwarnings(
    "ignore",
    message="Sparse CSR tensor support is in beta state.*",
    category=UserWarning,
)


# Note: we don't touch any dynamic, 3D, relational or heterogeneous datasets in this project.

# -------------------------------------------------------------------------- #
# Node-level datasets
# -------------------------------------------------------------------------- #
Actor_NAMES = {
    "actor": "actor",
}
Airports_NAMES = {
    "airports": "USA",
}
Amazon_NAMES = {
    "computers": "Computers",
    "photo": "Photo",
}
CitationFull_NAMES = {
    "cora-ml": "cora_ml",
    "dblp": "dblp",
}
Coauthor_NAMES = {
    "cs": "CS",
    "physics": "Physics",
}
CoraFull_NAMES = {
    "corafull": "corafull",
}
EllipticBitcoinDataset_NAMES = {
    "elliptic-bitcoin": "elliptic_bitcoin",
}
EmailEUCore_NAMES = {
    "email": "email_eu_core",
}
Flickr_NAMES = {
    "flickr": "flickr",
}
HeterophilousGraphDataset_NAMES = {
    "amazon-ratings": "amazon_ratings",
    "minesweeper": "minesweeper",
    "questions": "questions",
    "roman-empire": "roman_empire",
    "tolokers": "tolokers",
}
LINKXDataset_NAMES = {
    "amherst41": "amherst41",
    "cornell5": "cornell5",
    "genius": "genius",
    "johnshopkins55": "johnshopkins55",
    "penn94": "penn94",
    "reed98": "reed98",
}
OGBG_NAMES = {
    "ogbg-molhiv": "ogbg-molhiv",
    "ogbg-molpcba": "ogbg-molpcba",
}
OGBN_NAMES = {
    "ogbn-arxiv": "ogbn-arxiv",
    "ogbn-mag": "ogbn-mag",
    "ogbn-papers100m": "ogbn-papers100M",
    "ogbn-products": "ogbn-products",
    "ogbn-proteins": "ogbn-proteins",
}
Planetoid_NAMES = {
    "citeseer": "CiteSeer",
    "cora": "Cora",
    "pubmed": "PubMed",
}
Reddit_NAMES = {
    "reddit": "reddit",
}
Reddit2_NAMES = {
    "reddit2": "reddit2",
}
WebKB_NAMES = {
    "cornell": "cornell",
    "texas": "texas",
    "wisconsin": "wisconsin",
}
WikiCS_NAMES = {
    "wikics": "wikics",
}
WikipediaNetwork_NAMES = {
    "chameleon": "chameleon",
    "squirrel": "squirrel",
}

# -------------------------------------------------------------------------- #
# Graph-level datasets
# -------------------------------------------------------------------------- #
MoleculeNet_NAMES = {
    "bace": "bace",
    "bbbp": "bbbp",
    "clintox": "clintox",
    "esol": "esol",
    "freesolv": "freesolv",
    "hiv": "hiv",
    "lipo": "lipo",
    "muv": "muv",
    "pcba": "pcba",
    "sider": "sider",
    "tox21": "tox21",
    "toxcast": "toxcast",
}
QM7b_NAMES = {
    "qm7b": "qm7b",
}
QM9_NAMES = {
    "qm9": "qm9",
}
TUDataset_NAMES = {
    "collab": "COLLAB",
    "enzymes": "ENZYMES",
    "imdb-binary": "IMDB-BINARY",
    "imdb-multi": "IMDB-MULTI",
    "mutag": "MUTAG",
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
ZINC_NAMES = {
    "zinc": "zinc",
}
ZINC15_NAMES = {
    "zinc15": "zinc15",
}


def _is_node_dataset_key(key: str) -> bool:
    return (
        key.startswith("ogbn-")
        or key in Actor_NAMES
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
        or key in Planetoid_NAMES
        or key in Reddit_NAMES
        or key in Reddit2_NAMES
        or key in WebKB_NAMES
        or key in WikiCS_NAMES
        or key in WikipediaNetwork_NAMES
    )


def _is_graph_dataset_key(key: str) -> bool:
    return (
        key.startswith("ogbg-")
        or key in MoleculeNet_NAMES
        or key in GNNBenchmarkDataset_NAMES
        or key in ZINC_NAMES
        or key in QM7b_NAMES
        or key in QM9_NAMES
        or key in TUDataset_NAMES
        or key in ZINC15_NAMES
    )


def _is_edge_dataset_key(key: str) -> bool:
    return key.startswith("ogbl-")


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


def _load_node_dataset(
    name: str,
    root: str,
    transform,
):
    """Load node-level dataset by name."""
    key = name.lower()
    if key in Actor_NAMES:
        return Actor(root=_scoped_root(root, key), transform=transform)
    elif key in Airports_NAMES:
        dataset_key = Airports_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return Airports(root=root_key, name=dataset_key, transform=transform)
    elif key in Amazon_NAMES:
        dataset_key = Amazon_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return Amazon(root=root_key, name=dataset_key, transform=transform)
    elif key in CitationFull_NAMES:
        dataset_key = CitationFull_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return CitationFull(root=root_key, name=dataset_key, transform=transform)
    elif key in Coauthor_NAMES:
        dataset_key = Coauthor_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return Coauthor(root=root_key, name=dataset_key, transform=transform)
    elif key in CoraFull_NAMES:
        root_key = _scoped_root(root, key)
        return CoraFull(root=root_key, transform=transform)
    elif key in EllipticBitcoinDataset_NAMES:
        dataset_key = EllipticBitcoinDataset_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return EllipticBitcoinDataset(root=root_key, transform=transform)
    elif key in EmailEUCore_NAMES:
        dataset_key = EmailEUCore_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return EmailEUCore(root=root_key, transform=transform)
    elif key in Flickr_NAMES:
        dataset_key = Flickr_NAMES[key]
        root_key = _scoped_root(root, key)
        return Flickr(root=root_key, transform=transform)
    elif key in HeterophilousGraphDataset_NAMES:
        dataset_key = HeterophilousGraphDataset_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return HeterophilousGraphDataset(root=root_key, name=dataset_key, transform=transform)
    elif key in LINKXDataset_NAMES:
        dataset_key = LINKXDataset_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return LINKXDataset(root=root_key, name=dataset_key, transform=transform)
    elif key in Planetoid_NAMES:
        dataset_key = Planetoid_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return Planetoid(root=root_key, name=dataset_key, transform=transform)
    elif key in Reddit_NAMES:
        root_key = _scoped_root(root, key)
        return Reddit(root=root_key, transform=transform)
    elif key in Reddit2_NAMES:
        root_key = _scoped_root(root, key)
        return Reddit2(root=root_key, transform=transform)
    elif key.startswith("ogbn-"):
        dataset_key = OGBN_NAMES[key]
        root_key = _scoped_root(root, key)
        with _force_ogb_prompts_yes():
            return PygNodePropPredDataset(name=dataset_key, root=root_key)
    elif key in WebKB_NAMES:
        dataset_key = WebKB_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return WebKB(root=root_key, name=dataset_key, transform=transform)
    elif key in WikiCS_NAMES:
        root_key = _scoped_root(root, key)
        return WikiCS(root=root_key, is_undirected=True, transform=transform)
    elif key in WikipediaNetwork_NAMES:
        dataset_key = WikipediaNetwork_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return WikipediaNetwork(root=root_key, name=dataset_key, transform=transform)
    else:
        raise ValueError(f"Unsupported node-level dataset: {name}")


def _load_graph_dataset(
    name: str,
    root: str,
    transform,
):
    """Load graph-level dataset by name."""
    key = name.lower()
    if key in MoleculeNet_NAMES:
        dataset_key = MoleculeNet_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return MoleculeNet(root=root_key, name=dataset_key, transform=transform)
    elif key in GNNBenchmarkDataset_NAMES:
        dataset_key = GNNBenchmarkDataset_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return GNNBenchmarkDataset(root=root_key, name=dataset_key, transform=transform)
    elif key in QM7b_NAMES:
        dataset_key = QM7b_NAMES[key]
        root_key = _scoped_root(root, key)
        return QM7b(root=root_key, transform=transform)
    elif key in QM9_NAMES:
        root_key = _scoped_root(root, key)
        return QM9(root=root_key, transform=transform)
    elif key in ZINC15_NAMES:
        root_key = _scoped_root(root, key)
        return ZINC15Dataset(root=root_key, transform=transform)
    elif key in ZINC_NAMES:
        root_key = _scoped_root(root, key)
        return ZINC(root=root_key, subset=False, split="train", transform=transform)
    elif key in TUDataset_NAMES:
        dataset_key = TUDataset_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return TUDataset(root=root_key, name=dataset_key, transform=transform)
    elif key.startswith("ogbg-"):
        dataset_key = OGBG_NAMES[key]
        root_key = _scoped_root(root, key)
        with _force_ogb_prompts_yes():
            return PygGraphPropPredDataset(name=dataset_key, root=root_key)
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
    cache_induced: bool = True,
    split: Tuple[float, float, float] | None = None,
    seed: int = 42,
    split_root: str = "",
    induced_root: str = "",
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
            base_dataset = _load_node_dataset(name=name, root=root, transform=transform)
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
                        graphs_for_split = build_edge_induced_graphs_supervised(
                            data=context_data,
                            pos_edge_pairs=pos_pairs,
                            neg_edge_pairs=neg_pairs,
                            max_hops=induced_max_hops,
                            max_size=edge_max_size,
                        )
                        all_graphs.extend(graphs_for_split)
                        split_tags.extend([split_name] * len(graphs_for_split))

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
                else:
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
                                    split_lookup.get(getattr(graph, "base_node_id", idx), "train")
                                    for idx, graph in enumerate(graphs)
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
                            split_tags = [split_lookup.get(graph.base_node_id, "train") for graph in graphs]
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

        dataset = _load_node_dataset(name=name, root=root, transform=transform)
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
        dataset = _load_graph_dataset(name=name, root=root, transform=transform)
        dataset = _sanitize_graph_dataset(dataset=dataset, dataset_name=name, split_root=split_root)
        if reducer:
            feature_target = _unwrap_subset_dataset(dataset)
            if _get_dataset_data_storage(feature_target) is not None and persist_feature_svd:
                _apply_feature_svd(
                    feature_target,
                    name,
                    feat_reduction_dim,
                    reducer,
                    task_level=task_level,
                    output_root=feature_svd_dir,
                )
                try:
                    dataset._svd_dim = feat_reduction_dim  # type: ignore[attr-defined]
                    dataset._svd_task_level = task_level  # type: ignore[attr-defined]
                    dataset._feature_svd_root = feature_svd_dir or getattr(feature_target, "root", root)  # type: ignore[attr-defined]
                except Exception:
                    pass
            else:
                pending_datasets = [_unwrap_subset_dataset(dataset)]
                while pending_datasets:
                    current_dataset = pending_datasets.pop()
                    children = getattr(current_dataset, "datasets", None)
                    if children is not None:
                        pending_datasets.extend(children)
                        continue
                    current_transform = getattr(current_dataset, "transform", None)
                    try:
                        current_dataset.transform = reducer if current_transform is None else Compose([current_transform, reducer])
                    except Exception:
                        pass
                try:
                    dataset.num_features = int(feat_reduction_dim)  # type: ignore[attr-defined]
                    dataset.num_node_features = int(feat_reduction_dim)  # type: ignore[attr-defined]
                except Exception:
                    pass
        filter_meta = _graph_filter_cache_meta_for_dataset(dataset)
        if filter_meta is None:
            raise RuntimeError(f"[GraphFilter] Missing filter metadata for graph dataset={name}.")
        return dataset

    else:
        raise ValueError(f"Unsupported task_level: {task_level}")


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
    return_split_meta: bool = False,
):
    """Create data loaders for training, validation, and testing."""
    raw_dataset_name = str(dataset_name)
    split_dataset_name = _canonical_split_dataset_name(raw_dataset_name, task_level, int(seed))
    split_meta = None

    def _builtin_split_meta():
        return {"status": "builtin", "path": None}

    def _finalize_loaders(train_loader, val_loader, test_loader):
        if return_split_meta:
            meta = dict(split_meta) if isinstance(split_meta, dict) else _builtin_split_meta()
            return train_loader, val_loader, test_loader, meta
        return train_loader, val_loader, test_loader

    if split is not None:
        if task_level == "edge":
            _validate_edge_split_def(split)
        else:
            _validate_split_def(split)

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
            return_split_meta=return_split_meta,
        )

    def _count_indices_from_graphs():
        if len(split) < 3:
            raise ValueError("Integer-first split must provide [train_count, val_ratio, test_ratio].")
        split_root_path = Path(split_root) if split_root else None
        return _get_or_create_count_split(
            dataset_name=split_dataset_name,
            train_count=int(split[0]),
            val_ratio=float(split[1]),
            test_ratio=float(split[2]),
            seed=seed,
            split_root_path=split_root_path,
            total=len(dataset),
            return_split_meta=return_split_meta,
        )

    use_few_shot = _is_few_shot_split_def(split)
    split_strategy = resolve_count_split_strategy(dataset, task_level) if use_few_shot else "ratios"
    if use_few_shot and split_strategy == "unsupported":
        raise ValueError("Integer-first splits are not supported for unlabeled or single-target regression datasets.")

    if task_level == "node" and not induced:
        data = dataset[0]
        if not split_root:
            raise ValueError("split_root is required to save or load fixed splits for node datasets.")
        split_root_path = Path(split_root)
        labels = getattr(data, "y", None)
        labeled_idx = None
        if labels is not None:
            label_tensor = torch.as_tensor(labels)
            if label_tensor.dim() <= 1:
                label_matrix = label_tensor.view(-1, 1)
            else:
                label_matrix = label_tensor.view(label_tensor.size(0), -1)
            valid_mask = torch.ones_like(label_matrix, dtype=torch.bool)
            if label_matrix.dtype.is_floating_point:
                valid_mask &= torch.isfinite(label_matrix)
            if label_matrix.dtype != torch.bool:
                valid_mask &= label_matrix >= 0
            labeled_idx = torch.nonzero(valid_mask.any(dim=1), as_tuple=False).view(-1).tolist()
        if use_few_shot and split_strategy == "balanced":
            if len(split) < 3:
                raise ValueError("Few-shot split must provide [shots_per_class, val_ratio, test_ratio].")
            result = _get_or_create_few_shot_split(
                dataset_name=split_dataset_name,
                labels=data.y,
                shots_per_class=int(split[0]),
                val_ratio=float(split[1]),
                test_ratio=float(split[2]),
                seed=seed,
                split_root_path=split_root_path,
                return_split_meta=return_split_meta,
            )
        elif use_few_shot and split_strategy == "random":
            result = _get_or_create_count_split(
                dataset_name=split_dataset_name,
                train_count=int(split[0]),
                val_ratio=float(split[1]),
                test_ratio=float(split[2]),
                seed=seed,
                split_root_path=split_root_path,
                total=data.num_nodes,
                subset_indices=labeled_idx if labeled_idx is not None and len(labeled_idx) < data.num_nodes else None,
                return_split_meta=return_split_meta,
            )
        else:
            if labeled_idx is not None and len(labeled_idx) < data.num_nodes:
                result = _get_or_create_split_indices_subset(
                    dataset_name=split_dataset_name,
                    split=split,
                    seed=seed,
                    split_root_path=split_root_path,
                    subset_indices=labeled_idx,
                    return_split_meta=return_split_meta,
                )
            else:
                result = _get_or_create_split_indices(
                    dataset_name=split_dataset_name,
                    split=split,
                    seed=seed,
                    split_root_path=split_root_path,
                    total=data.num_nodes,
                    return_split_meta=return_split_meta,
                )
        if return_split_meta:
            train_idx, val_idx, test_idx, split_meta = result
        else:
            train_idx, val_idx, test_idx = result
        for mask_name, indices in (
            ("train_mask", train_idx),
            ("val_mask", val_idx),
            ("test_mask", test_idx),
        ):
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[indices] = True
            setattr(data, mask_name, mask)
        return _finalize_loaders(
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
        result = _get_or_create_edge_split_payload(
            dataset_name=split_dataset_name,
            split=split,
            seed=seed,
            split_root_path=split_root_path,
            data=data,
            return_split_meta=return_split_meta,
        )
        if return_split_meta:
            split_payload, split_meta = result
        else:
            split_payload = result

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

            return _finalize_loaders(
                _link_loader(train_data, shuffle=True),
                _link_loader(val_data, shuffle=False),
                _link_loader(test_data, shuffle=False),
            )

        return _finalize_loaders(
            SingleGraphDataLoader(train_data),
            SingleGraphDataLoader(val_data),
            SingleGraphDataLoader(test_data),
        )

    if task_level == "edge" and induced and hasattr(dataset, "split_tags"):
        split_meta = _builtin_split_meta()
        train_idx = [i for i, tag in enumerate(dataset.split_tags) if tag == "train"]
        val_idx = [i for i, tag in enumerate(dataset.split_tags) if tag == "val"]
        test_idx = [i for i, tag in enumerate(dataset.split_tags) if tag == "test"]
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        test_set = Subset(dataset, test_idx)
    elif task_level == "graph" and hasattr(dataset, "split_tags") and dataset.split_tags:
        train_idx = [i for i, tag in enumerate(dataset.split_tags) if tag == "train"]
        val_idx = [i for i, tag in enumerate(dataset.split_tags) if tag == "val"]
        test_idx = [i for i, tag in enumerate(dataset.split_tags) if tag == "test"]
        if not val_idx or not test_idx:
            result = split_graph_dataset(
                dataset=dataset,
                dataset_name=split_dataset_name,
                split=split,
                seed=seed,
                split_root=split_root,
                return_split_meta=return_split_meta,
            )
            if return_split_meta:
                train_set, val_set, test_set, split_meta = result
            else:
                train_set, val_set, test_set = result
        else:
            split_meta = _builtin_split_meta()
            train_set = Subset(dataset, train_idx)
            val_set = Subset(dataset, val_idx)
            test_set = Subset(dataset, test_idx)
    elif use_few_shot and split_strategy == "balanced":
        result = _few_shot_indices_from_graphs()
        if return_split_meta:
            train_idx, val_idx, test_idx, split_meta = result
        else:
            train_idx, val_idx, test_idx = result
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        test_set = Subset(dataset, test_idx)
    elif use_few_shot and split_strategy == "random":
        result = _count_indices_from_graphs()
        if return_split_meta:
            train_idx, val_idx, test_idx, split_meta = result
        else:
            train_idx, val_idx, test_idx = result
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        test_set = Subset(dataset, test_idx)
    else:
        result = split_graph_dataset(
            dataset=dataset,
            dataset_name=split_dataset_name,
            split=split,
            seed=seed,
            split_root=split_root,
            return_split_meta=return_split_meta,
        )
        if return_split_meta:
            train_set, val_set, test_set, split_meta = result
        else:
            train_set, val_set, test_set = result

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=drop_last_train,
    )
    train_loader = DataLoader(dataset=train_set, **loader_kwargs)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return _finalize_loaders(train_loader, val_loader, test_loader)
