"""
Prepare datasets listed in TSV/list inputs.
For each dataset task, this runs dataset creation, split generation, and optional SVD artifacts.
"""
import os
import io
import contextlib
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch

os.environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")
os.environ.setdefault("OGB_ASSUME_YES", "1")

from code.data_loader.datasets import (  # noqa: E402
    create_dataset,
    compute_subgraph_svd_features,
    infer_task_level,
    is_regression_dataset,
    make_loaders,
)

DATASET_SEPARATOR = "=" * 72
TASK_SEPARATOR = "-" * 72


def _dataset_scoped_dir(base_dir: str, dataset_name: str) -> str:
    if not base_dir:
        return ""
    base = Path(base_dir)
    scoped = str(dataset_name)
    if base.name == scoped:
        return str(base)
    return str(base / scoped)


def _feature_task_suffix(task_level: str) -> str:
    if task_level == "edge":
        return "_edge"
    if task_level == "graph":
        return "_graph"
    return "_node"


def _feature_svd_path(dataset_name: str, task_level: str, dim: int, output_dir: str) -> Path:
    suffix = _feature_task_suffix(task_level)
    scoped_dir = Path(_dataset_scoped_dir(output_dir, dataset_name))
    return scoped_dir / f"feature_svd_{dataset_name}{suffix}_d{dim}.pt"


def _legacy_feature_svd_path(dataset_name: str, task_level: str, dim: int, output_dir: str) -> Path:
    suffix = _feature_task_suffix(task_level)
    scoped_dir = Path(_dataset_scoped_dir(output_dir, dataset_name))
    return scoped_dir / f"svd_cache_{dataset_name}{suffix}_d{dim}.pt"


def _feature_svd_old_path(dataset_name: str, dim: int, output_dir: str) -> Path:
    scoped_dir = Path(_dataset_scoped_dir(output_dir, dataset_name))
    return scoped_dir / f"feature_svd_{dataset_name}_d{dim}.pt"


def _legacy_feature_svd_old_path(dataset_name: str, dim: int, output_dir: str) -> Path:
    scoped_dir = Path(_dataset_scoped_dir(output_dir, dataset_name))
    return scoped_dir / f"svd_cache_{dataset_name}_d{dim}.pt"


def _print_feature_svd_info(
    dataset_name: str,
    task_level: str,
    feat_reduction: bool,
    feat_reduction_dim: int,
    feature_svd_dir: str,
) -> None:
    if not feat_reduction or int(feat_reduction_dim) <= 0:
        return
    primary = _feature_svd_path(dataset_name, task_level, int(feat_reduction_dim), feature_svd_dir)
    if primary.exists():
        print(f"[FeatureSVD] Ready features at {primary}")
        return
    legacy = _legacy_feature_svd_path(dataset_name, task_level, int(feat_reduction_dim), feature_svd_dir)
    if legacy.exists():
        print(f"[FeatureSVD] Ready features at {legacy}")
        return
    old_primary = _feature_svd_old_path(dataset_name, int(feat_reduction_dim), feature_svd_dir)
    if old_primary.exists():
        print(f"[FeatureSVD] Ready features at {old_primary}")
        return
    old_legacy = _legacy_feature_svd_old_path(dataset_name, int(feat_reduction_dim), feature_svd_dir)
    if old_legacy.exists():
        print(f"[FeatureSVD] Ready features at {old_legacy}")


def _split_dataset_name(dataset_name: str, task_level: str, seed: int) -> str:
    if task_level == "edge":
        return f"{dataset_name}_edge_seed{seed}"
    return f"{dataset_name}_{task_level}_seed{seed}"


def _fmt_count(value: Optional[int]) -> str:
    return str(value) if value is not None else "?"


def _fmt_avg(value: Optional[float]) -> str:
    return f"{value:.2f}" if value is not None else "?"


def _avg_per_graph(total: Optional[int], graphs: Optional[int]) -> Optional[float]:
    if graphs is not None and int(graphs) == 0:
        return 0.0
    if total is None or graphs is None or int(graphs) < 0:
        return None
    return float(total) / float(graphs)


def _num_nodes(data) -> Optional[int]:
    if data is None:
        return None
    num_nodes = getattr(data, "num_nodes", None)
    if num_nodes is None and getattr(data, "x", None) is not None:
        num_nodes = data.x.size(0)
    return int(num_nodes) if num_nodes is not None else None


def _num_edges(data) -> Optional[int]:
    if data is None:
        return None
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None:
        return None
    return int(edge_index.size(1))


def _split_mask_count(data, mask_key: str) -> Optional[int]:
    if data is None:
        return None
    mask = getattr(data, mask_key, None)
    if mask is None:
        return None
    try:
        return int(torch.as_tensor(mask).sum().item())
    except Exception:
        return None


def _dataset_len(loader) -> Optional[int]:
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return None
    try:
        return int(len(dataset))
    except Exception:
        return None


def _subset_total_from_slices(dataset, key: str) -> Optional[int]:
    """Best-effort fast aggregate from InMemoryDataset slices without loading every graph."""
    if dataset is None:
        return None

    base_dataset = dataset
    indices = None
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base_dataset = dataset.dataset
        try:
            indices = list(dataset.indices)
        except Exception:
            indices = None
    if indices is None:
        try:
            indices = list(range(len(dataset)))
        except Exception:
            return None

    slices = getattr(base_dataset, "slices", None)
    if slices is None or key not in slices:
        return None
    boundaries = slices[key]

    try:
        if torch.is_tensor(boundaries):
            idx = torch.as_tensor(indices, dtype=torch.long)
            if idx.numel() == 0:
                return 0
            span = boundaries[idx + 1] - boundaries[idx]
            return int(span.sum().item())

        total = 0
        for idx in indices:
            total += int(boundaries[idx + 1] - boundaries[idx])
        return total
    except Exception:
        return None


def _graph_nodes_edges(loader) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return None, None, None

    graphs = _dataset_len(loader)
    nodes = _subset_total_from_slices(dataset, "x")
    edges = _subset_total_from_slices(dataset, "edge_index")
    if nodes is not None and edges is not None:
        return nodes, edges, graphs

    # Fallback: materialize only for small datasets.
    if graphs is not None and graphs <= 128:
        total_nodes = 0
        total_edges = 0
        try:
            for item in dataset:
                item_nodes = getattr(item, "num_nodes", None)
                if item_nodes is None and getattr(item, "x", None) is not None:
                    item_nodes = item.x.size(0)
                item_edges = getattr(item, "num_edges", None)
                if item_edges is None and getattr(item, "edge_index", None) is not None:
                    item_edges = item.edge_index.size(1)
                total_nodes += int(item_nodes) if item_nodes is not None else 0
                total_edges += int(item_edges) if item_edges is not None else 0
            return total_nodes, total_edges, graphs
        except Exception:
            pass

    return None, None, graphs


def _split_stats_line(task_level: str, train_loader, val_loader, test_loader) -> str:
    split_meta = (
        ("train", "train", train_loader),
        ("valid", "val", val_loader),
        ("test", "test", test_loader),
    )

    if task_level == "node":
        parts = []
        for display_name, mask_name, loader in split_meta:
            data = getattr(loader, "data", None)
            nodes = _split_mask_count(data, f"{mask_name}_mask")
            if nodes is None:
                nodes = _num_nodes(data)
            edges = _num_edges(data)
            graphs = int(getattr(data, "num_graphs", 1)) if data is not None else _dataset_len(loader)
            parts.append(
                f"{display_name}(nodes={_fmt_count(nodes)},edges={_fmt_count(edges)},graphs={_fmt_count(graphs)})"
            )
        return " ".join(parts)

    if task_level == "edge":
        parts = []
        message_edges = None
        for display_name, _, loader in split_meta:
            data = getattr(loader, "data", None)
            nodes = _num_nodes(data)
            graphs = int(getattr(data, "num_graphs", 1)) if data is not None else _dataset_len(loader)
            if message_edges is None:
                message_edges = _num_edges(data)

            edge_label_index = getattr(data, "edge_label_index", None) if data is not None else None
            labeled_edges = int(edge_label_index.size(1)) if edge_label_index is not None else None

            pos_edges = None
            neg_edges = None
            edge_label = getattr(data, "edge_label", None) if data is not None else None
            if edge_label is not None:
                try:
                    label_tensor = torch.as_tensor(edge_label).view(-1)
                    pos_edges = int((label_tensor > 0.5).sum().item())
                    neg_edges = int((label_tensor <= 0.5).sum().item())
                except Exception:
                    pos_edges = None
                    neg_edges = None

            parts.append(
                f"{display_name}(nodes={_fmt_count(nodes)},edges={_fmt_count(labeled_edges)},"
                f"graphs={_fmt_count(graphs)},pos={_fmt_count(pos_edges)},neg={_fmt_count(neg_edges)})"
            )
        return f"{' '.join(parts)} message_edges={_fmt_count(message_edges)}"

    if task_level == "graph":
        parts = []
        for display_name, _, loader in split_meta:
            nodes, edges, graphs = _graph_nodes_edges(loader)
            avg_nodes = _avg_per_graph(nodes, graphs)
            avg_edges = _avg_per_graph(edges, graphs)
            parts.append(
                f"{display_name}(avg_nodes={_fmt_avg(avg_nodes)},"
                f"avg_edges={_fmt_avg(avg_edges)},graphs={_fmt_count(graphs)})"
            )
        return " ".join(parts)

    return ""


def _is_few_shot_split_def(split_def: Tuple[float, float, float]) -> bool:
    if not isinstance(split_def, (list, tuple)) or not split_def:
        return False
    first = split_def[0]
    try:
        return float(first).is_integer()
    except Exception:
        return isinstance(first, int) and not isinstance(first, bool)


def _filter_unsupported_split_defs(
    dataset_obj,
    dataset_name: str,
    task_level: str,
    split_defs: List[Tuple[float, float, float]],
) -> List[Tuple[float, float, float]]:
    """Drop unsupported split definitions for the given dataset/task pair."""
    if not split_defs:
        return split_defs
    if not is_regression_dataset(dataset_obj, task_level):
        return split_defs

    kept: List[Tuple[float, float, float]] = []
    skipped: List[Tuple[float, float, float]] = []
    for split_def in split_defs:
        if _is_few_shot_split_def(split_def):
            skipped.append(split_def)
        else:
            kept.append(split_def)

    for split_def in skipped:
        print(
            f"[DataPrep][Split] Skip unsupported few-shot split for regression "
            f"dataset={dataset_name} task={task_level} split={split_def}"
        )
    return kept


def _generate_task_splits(
    dataset_obj,
    dataset_name: str,
    task_level: str,
    split_defs: List[Tuple[float, float, float]],
    split_seeds: List[int],
    split_root: str,
    batch_size: int,
    num_workers: int,
) -> int:
    if not split_root or not split_defs or not split_seeds:
        return 0

    dataset_split_root = Path(split_root)
    failures = 0
    total_jobs = len(split_defs) * len(split_seeds)
    for split_def in split_defs:
        for seed in split_seeds:
            split_name = _split_dataset_name(dataset_name, task_level, int(seed))
            try:
                # Suppress per-file split logs; report one concise summary per task.
                with contextlib.redirect_stdout(io.StringIO()):
                    train_loader, val_loader, test_loader = make_loaders(
                        dataset=dataset_obj,
                        dataset_name=split_name,
                        task_level=task_level,
                        batch_size=int(batch_size),
                        num_workers=int(num_workers),
                        split=split_def,
                        seed=int(seed),
                        induced=False,
                        split_root=str(dataset_split_root),
                    )
                split_stats = _split_stats_line(
                    task_level=task_level,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                )
                if split_stats:
                    print(
                        f"[DataPrep][Split][Stats] dataset={dataset_name} task={task_level} "
                        f"split={split_def} seed={seed} {split_stats}"
                    )
            except Exception as exc:  # pylint: disable=broad-except
                failures += 1
                print(
                    f"[DataPrep][Split] Failed dataset={dataset_name} task={task_level} "
                    f"split={split_def} seed={seed} root={dataset_split_root}: {exc}"
                )

    generated = total_jobs - failures
    print(
        f"[DataPrep][Split] dataset={dataset_name} task={task_level} "
        f"splits={len(split_defs)} seeds={len(split_seeds)} generated={generated} "
        f"failed={failures} root={dataset_split_root}"
    )
    return failures


def read_datasets(tsv_path: str) -> List[str]:
    names: List[str] = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if not parts:
                continue
            names.append(parts[0])
    return names


def cleanup_artifacts(dataset: str, root: Path) -> None:
    """Remove stale zip/raw/processed files that can cause zip parsing errors."""
    names = {dataset}
    # Common OGB naming: ogbn-arxiv -> arxiv, ogbg-ppa -> ppa
    if dataset.startswith("ogbn-") or dataset.startswith("ogbg-"):
        names.add(dataset.split("-", 1)[1])
    # LRGB naming: coco-sp -> cocosp, pascalvoc-sp -> pascalvocsp
    names.add(dataset.replace("-", ""))

    candidates = []
    for name in names:
        candidates.extend(
            [
                root / name,
                root / name / "raw",
                root / name / "processed",
                root / f"{name}.zip",
            ]
        )
    for path in candidates:
        if path.is_file():
            try:
                path.unlink()
            except OSError:
                pass
        elif path.is_dir():
            try:
                for sub in path.rglob("*"):
                    if sub.is_file():
                        sub.unlink()
                path.rmdir()
            except OSError:
                pass


def already_processed(dataset: str, root: Path) -> str | None:
    """Return a processed path if dataset appears already prepared, else None."""
    names = {dataset}
    if dataset.startswith("ogbn-") or dataset.startswith("ogbg-"):
        names.add(dataset.split("-", 1)[1])
    names.add(dataset.replace("-", ""))

    candidates = []
    for name in names:
        candidates.extend(
            [
                root / name / "processed",
            ]
        )
    for path in candidates:
        if path.exists() and any(path.iterdir()):
            return str(path)
    return None


def _is_zip_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "zip" in msg or "file is not a zip file" in msg


def _run_with_zip_retry(dataset: str, root_path: Path, process_fn: Callable[[], bool]) -> bool:
    """Retry once after cleaning stale artifacts when dataset archives are corrupted."""
    for attempt in (1, 2):
        try:
            return process_fn()
        except Exception as exc:
            if attempt == 1 and _is_zip_error(exc):
                print(f"[CLEAN] removing stale files for {dataset} due to zip error")
                cleanup_artifacts(dataset, root_path)
                continue
            raise
    return False


def _compute_subgraph_svd_with_fallback(
    dataset_obj,
    dataset_name: str,
    task_level: str,
    feat_dim: int,
    struct_dim: int,
    matrix_type: str,
    output_dir: str,
) -> None:
    """Compute subgraph SVD features and fallback to local default path on permission errors."""
    try:
        compute_subgraph_svd_features(
            dataset_obj,
            dataset_name=dataset_name,
            task_level=task_level,
            feat_dim=feat_dim,
            struct_dim=struct_dim,
            matrix_type=matrix_type,
            output_dir=output_dir,
            overwrite=False,
        )
    except PermissionError:
        fallback_dir = _dataset_scoped_dir("data/subgraph_svd", dataset_name)
        print(
            f"[SubgraphSVD] Permission denied for {output_dir}; retrying with {fallback_dir}."
        )
        compute_subgraph_svd_features(
            dataset_obj,
            dataset_name=dataset_name,
            task_level=task_level,
            feat_dim=feat_dim,
            struct_dim=struct_dim,
            matrix_type=matrix_type,
            output_dir=fallback_dir,
            overwrite=False,
        )


def _num_nodes_edges_from_node_dataset(dataset_obj) -> Tuple[Optional[int], Optional[int]]:
    """Best-effort total node/edge counts for node-level datasets."""
    num_nodes = getattr(dataset_obj, "base_num_nodes", None)
    num_edges = getattr(dataset_obj, "base_num_edges", None)

    if num_nodes is not None and num_edges is not None:
        return int(num_nodes), int(num_edges)

    try:
        data = dataset_obj[0]
    except Exception:
        return None, None

    if num_nodes is None:
        raw_nodes = getattr(data, "num_nodes", None)
        if raw_nodes is None and getattr(data, "x", None) is not None:
            raw_nodes = int(data.x.size(0))
        if raw_nodes is not None:
            num_nodes = int(raw_nodes)

    if num_edges is None:
        edge_index = getattr(data, "edge_index", None)
        if edge_index is not None:
            num_edges = int(edge_index.size(1))

    return (int(num_nodes) if num_nodes is not None else None, int(num_edges) if num_edges is not None else None)


def _num_graphs_from_graph_dataset(dataset_obj) -> Optional[int]:
    """Best-effort graph count for graph-level datasets."""
    try:
        return int(len(dataset_obj))
    except Exception:
        return None


def _label_count_from_tensor(labels: torch.Tensor) -> Optional[int]:
    """Count unique non-negative class labels from a label tensor."""
    labels = torch.as_tensor(labels).view(-1)
    if labels.numel() == 0:
        return None

    if labels.dtype.is_floating_point:
        finite = labels[torch.isfinite(labels)]
        if finite.numel() == 0:
            return None
        rounded = finite.round()
        if not torch.allclose(finite, rounded, atol=1e-6):
            return None
        labels = rounded

    labels = labels.to(torch.long)
    labels = labels[labels >= 0]
    if labels.numel() == 0:
        return None
    return int(torch.unique(labels).numel())


def _classification_label_count(dataset_obj, task_level: str) -> Optional[int]:
    """Count class labels for classification node/graph datasets."""
    num_classes = getattr(dataset_obj, "num_classes", None)
    try:
        if num_classes is not None and int(num_classes) > 0:
            return int(num_classes)
    except Exception:
        pass

    level = str(task_level).lower()

    if level == "graph":
        labels = None
        try:
            data_obj = getattr(dataset_obj, "data", None)
            data_labels = getattr(data_obj, "y", None) if data_obj is not None else None
            if data_labels is not None:
                data_labels = torch.as_tensor(data_labels)
                if data_labels.dim() >= 1 and data_labels.size(0) == len(dataset_obj):
                    labels = data_labels.reshape(-1)
        except Exception:
            labels = None

        if labels is None:
            collected = []
            try:
                sample_count = min(int(len(dataset_obj)), 4096)
                for idx in range(sample_count):
                    item = dataset_obj[idx]
                    item_y = getattr(item, "y", None)
                    if item_y is None:
                        continue
                    item_y = torch.as_tensor(item_y).view(-1)
                    if item_y.numel() > 0:
                        collected.append(item_y.detach().cpu())
            except Exception:
                collected = []
            if collected:
                labels = torch.cat(collected, dim=0).view(-1)

        if labels is None:
            return None
        return _label_count_from_tensor(labels)

    if level != "node":
        return None

    try:
        data = dataset_obj[0]
    except Exception:
        return None

    labels = getattr(data, "y", None)
    if labels is None:
        return None
    labels = torch.as_tensor(labels).view(-1)
    if labels.numel() == 0:
        return None

    if labels.numel() <= 1:
        collected = []
        try:
            sample_count = min(int(len(dataset_obj)), 4096)
            for idx in range(sample_count):
                item = dataset_obj[idx]
                item_y = getattr(item, "y", None)
                if item_y is None:
                    continue
                item_y = torch.as_tensor(item_y).view(-1)
                if item_y.numel() > 0:
                    collected.append(item_y[0].detach().cpu())
        except Exception:
            collected = []
        if collected:
            labels = torch.stack(collected).view(-1)

    return _label_count_from_tensor(labels)


def _log_node_dataset_overview(dataset: str, dataset_obj) -> None:
    """Print one node-dataset summary line at the beginning of node-task prep."""
    num_nodes, num_edges = _num_nodes_edges_from_node_dataset(dataset_obj)
    parts = [
        f"dataset={dataset}",
        f"task=node",
        f"nodes={num_nodes if num_nodes is not None else '?'}",
        f"edges={num_edges if num_edges is not None else '?'}",
    ]
    if not is_regression_dataset(dataset_obj, "node"):
        num_labels = _classification_label_count(dataset_obj=dataset_obj, task_level="node")
        parts.append(f"labels={num_labels if num_labels is not None else '?'}")
    print("[DataPrep][Dataset] " + ", ".join(parts))


def _log_graph_dataset_overview(dataset: str, dataset_obj) -> None:
    """Print one graph-dataset summary line at the beginning of graph-task prep."""
    num_graphs = _num_graphs_from_graph_dataset(dataset_obj)
    parts = [
        f"dataset={dataset}",
        f"task=graph",
        f"graphs={num_graphs if num_graphs is not None else '?'}",
    ]
    if not is_regression_dataset(dataset_obj, "graph"):
        num_labels = _classification_label_count(dataset_obj=dataset_obj, task_level="graph")
        parts.append(f"labels={num_labels if num_labels is not None else '?'}")
    print("[DataPrep][Dataset] " + ", ".join(parts))


def _process_edge_task(
    dataset: str,
    root: str,
    split_defs: List[Tuple[float, float, float]],
    split_seeds: List[int],
    *,
    dataset_split_root: str,
    dataset_feature_svd_dir: str,
    dataset_induced_root: str,
    dataset_subgraph_svd_dir: str,
    split_batch_size: int,
    split_num_workers: int,
    induced: bool,
    induced_min_size: int,
    induced_max_size: int,
    induced_max_hops: int,
    force_reload_raw: bool,
    feat_reduction: bool,
    feat_reduction_dim: int,
    subgraph_svd: bool,
    subgraph_svd_feat_dim: int,
    subgraph_svd_struct_dim: int,
    subgraph_svd_matrix: str,
) -> bool:
    """Prepare edge-level artifacts: fixed splits, induced edge subgraphs, and optional subgraph SVD."""
    if not split_defs:
        raise ValueError("No edge split definitions found in cfg.data_preparation.edge_task_splits.")

    split_failures = 0
    induced_failures = 0
    primary_seed = int(split_seeds[0]) if split_seeds else 42

    # Step 1: materialize all edge splits on non-induced data.
    split_dataset_obj = create_dataset(
        name=dataset,
        root=root,
        task_level="edge",
        induced=False,
        induced_min_size=induced_min_size,
        induced_max_size=induced_max_size,
        induced_max_hops=induced_max_hops,
        induced_root=dataset_induced_root,
        split_root=dataset_split_root,
        split=None,
        seed=primary_seed,
        force_reload_raw=force_reload_raw,
        feat_reduction=feat_reduction,
        feat_reduction_dim=feat_reduction_dim,
        feature_svd_dir=dataset_feature_svd_dir,
    )
    split_failures = _generate_task_splits(
        dataset_obj=split_dataset_obj,
        dataset_name=dataset,
        task_level="edge",
        split_defs=split_defs,
        split_seeds=split_seeds,
        split_root=dataset_split_root,
        batch_size=split_batch_size,
        num_workers=split_num_workers,
    )

    # Step 2: for each split+seed, generate induced edge subgraphs and split-specific SVD.
    if induced:
        total_jobs = len(split_defs) * len(split_seeds)
        for split_def in split_defs:
            for split_seed in split_seeds:
                try:
                    edge_dataset = create_dataset(
                        name=dataset,
                        root=root,
                        task_level="edge",
                        induced=True,
                        induced_min_size=induced_min_size,
                        induced_max_size=induced_max_size,
                        induced_max_hops=induced_max_hops,
                        induced_root=dataset_induced_root,
                        split_root=dataset_split_root,
                        split=split_def,
                        seed=int(split_seed),
                        force_reload_raw=force_reload_raw,
                        feat_reduction=feat_reduction,
                        feat_reduction_dim=feat_reduction_dim,
                        feature_svd_dir=dataset_feature_svd_dir,
                    )
                    if subgraph_svd:
                        _compute_subgraph_svd_with_fallback(
                            edge_dataset,
                            dataset_name=dataset,
                            task_level="edge",
                            feat_dim=subgraph_svd_feat_dim,
                            struct_dim=subgraph_svd_struct_dim,
                            matrix_type=subgraph_svd_matrix,
                            output_dir=dataset_subgraph_svd_dir,
                        )
                except Exception as exc:  # pylint: disable=broad-except
                    induced_failures += 1
                    print(
                        f"[DataPrep][EdgeInduced] Failed dataset={dataset} "
                        f"split={split_def} seed={split_seed}: {exc}"
                    )
        generated = total_jobs - induced_failures
        print(
            f"[DataPrep][EdgeInduced] dataset={dataset} "
            f"splits={len(split_defs)} seeds={len(split_seeds)} "
            f"generated={generated} failed={induced_failures} root={dataset_induced_root}"
        )

    _print_feature_svd_info(
        dataset_name=dataset,
        task_level="edge",
        feat_reduction=feat_reduction,
        feat_reduction_dim=feat_reduction_dim,
        feature_svd_dir=dataset_feature_svd_dir,
    )
    return (split_failures == 0) and (induced_failures == 0)


def _process_non_edge_task(
    dataset: str,
    task_level: str,
    root: str,
    split_defs: List[Tuple[float, float, float]],
    split_seeds: List[int],
    *,
    dataset_split_root: str,
    dataset_feature_svd_dir: str,
    dataset_induced_root: str,
    dataset_subgraph_svd_dir: str,
    split_batch_size: int,
    split_num_workers: int,
    induced: bool,
    induced_min_size: int,
    induced_max_size: int,
    induced_max_hops: int,
    force_reload_raw: bool,
    feat_reduction: bool,
    feat_reduction_dim: int,
    subgraph_svd: bool,
    subgraph_svd_feat_dim: int,
    subgraph_svd_struct_dim: int,
    subgraph_svd_matrix: str,
) -> bool:
    """
    Prepare node/graph artifacts and optionally generate fixed splits.

    Node tasks may use induced subgraphs during preparation; split generation is done
    on the non-induced view to keep split semantics consistent.
    """
    split_failures = 0
    primary_seed = int(split_seeds[0]) if split_seeds else 42
    effective_split_defs = list(split_defs)
    split_for_dataset = effective_split_defs[0] if effective_split_defs else None
    task_uses_induced = induced and task_level == "node"

    dataset_obj = create_dataset(
        name=dataset,
        root=root,
        task_level=task_level,
        induced=task_uses_induced,
        induced_min_size=induced_min_size,
        induced_max_size=induced_max_size,
        induced_max_hops=induced_max_hops,
        induced_root=dataset_induced_root,
        split_root=dataset_split_root,
        split=split_for_dataset,
        seed=primary_seed,
        force_reload_raw=force_reload_raw,
        feat_reduction=feat_reduction,
        feat_reduction_dim=feat_reduction_dim,
        feature_svd_dir=dataset_feature_svd_dir,
    )
    if task_level == "node":
        _log_node_dataset_overview(dataset=dataset, dataset_obj=dataset_obj)
    elif task_level == "graph":
        _log_graph_dataset_overview(dataset=dataset, dataset_obj=dataset_obj)

    # Regression tasks do not support few-shot split definitions.
    effective_split_defs = _filter_unsupported_split_defs(
        dataset_obj=dataset_obj,
        dataset_name=dataset,
        task_level=task_level,
        split_defs=effective_split_defs,
    )

    if effective_split_defs:
        split_dataset_obj = dataset_obj
        if task_uses_induced:
            # Split indices should be produced from the base (non-induced) graph.
            with contextlib.redirect_stdout(io.StringIO()):
                split_dataset_obj = create_dataset(
                    name=dataset,
                    root=root,
                    task_level=task_level,
                    induced=False,
                    induced_min_size=induced_min_size,
                    induced_max_size=induced_max_size,
                    induced_max_hops=induced_max_hops,
                    induced_root=dataset_induced_root,
                    split_root=dataset_split_root,
                    split=None,
                    seed=primary_seed,
                    force_reload_raw=force_reload_raw,
                    feat_reduction=feat_reduction,
                    feat_reduction_dim=feat_reduction_dim,
                    feature_svd_dir=dataset_feature_svd_dir,
                )
        split_failures = _generate_task_splits(
            dataset_obj=split_dataset_obj,
            dataset_name=dataset,
            task_level=task_level,
            split_defs=effective_split_defs,
            split_seeds=split_seeds,
            split_root=dataset_split_root,
            batch_size=split_batch_size,
            num_workers=split_num_workers,
        )

    _print_feature_svd_info(
        dataset_name=dataset,
        task_level=task_level,
        feat_reduction=feat_reduction,
        feat_reduction_dim=feat_reduction_dim,
        feature_svd_dir=dataset_feature_svd_dir,
    )
    should_run_subgraph_svd = subgraph_svd and (task_uses_induced or task_level == "graph")
    if should_run_subgraph_svd:
        _compute_subgraph_svd_with_fallback(
            dataset_obj,
            dataset_name=dataset,
            task_level=task_level,
            feat_dim=subgraph_svd_feat_dim,
            struct_dim=subgraph_svd_struct_dim,
            matrix_type=subgraph_svd_matrix,
            output_dir=dataset_subgraph_svd_dir,
        )

    return split_failures == 0


def try_load(
    dataset: str,
    root: str,
    task_levels: Iterable[str] = ("node", "graph"),
    feat_reduction: bool = True,
    feat_reduction_dim: int = 100,
    feature_svd_dir: str = "data/feature_svd",
    induced: bool = False,
    induced_min_size: int = 10,
    induced_max_size: int = 30,
    induced_max_hops: int = 5,
    induced_root: str = "",
    force_reload_raw: bool = False,
    subgraph_svd: bool = False,
    subgraph_svd_feat_dim: int = 100,
    subgraph_svd_struct_dim: int = 100,
    subgraph_svd_matrix: str = "adjacency",
    subgraph_svd_dir: str = "data/subgraph_svd",
    split_root: str = "",
    split_defs_by_task: Optional[Dict[str, List[Tuple[float, float, float]]]] = None,
    split_seeds: Optional[List[int]] = None,
    split_batch_size: int = 64,
    split_num_workers: int = 0,
) -> bool:
    """Prepare one dataset for all requested task levels."""
    root_path = Path(root)
    dataset_split_root = _dataset_scoped_dir(split_root, dataset)
    dataset_feature_svd_dir = _dataset_scoped_dir(feature_svd_dir, dataset)
    dataset_induced_root = _dataset_scoped_dir(induced_root, dataset) if induced_root else ""
    dataset_subgraph_svd_dir = _dataset_scoped_dir(subgraph_svd_dir, dataset)
    processed_path = already_processed(dataset, root_path)
    if processed_path:
        print(f"[DataPrep] Reusing cached processed data at {processed_path}")

    split_defs_map = split_defs_by_task or {}
    normalized_split_seeds = [int(seed) for seed in (split_seeds or [])]
    if not normalized_split_seeds:
        normalized_split_seeds = [42]

    all_ok = True
    for task_level in task_levels:
        print(TASK_SEPARATOR)
        print(f"[DataPrep] Processing {dataset} (task={task_level})...")
        print(TASK_SEPARATOR)
        task_split_defs = list(split_defs_map.get(task_level, []))
        try:
            if task_level == "edge":
                task_ok = _run_with_zip_retry(
                    dataset,
                    root_path,
                    lambda: _process_edge_task(
                        dataset=dataset,
                        root=root,
                        split_defs=task_split_defs,
                        split_seeds=normalized_split_seeds,
                        dataset_split_root=dataset_split_root,
                        dataset_feature_svd_dir=dataset_feature_svd_dir,
                        dataset_induced_root=dataset_induced_root,
                        dataset_subgraph_svd_dir=dataset_subgraph_svd_dir,
                        split_batch_size=split_batch_size,
                        split_num_workers=split_num_workers,
                        induced=induced,
                        induced_min_size=induced_min_size,
                        induced_max_size=induced_max_size,
                        induced_max_hops=induced_max_hops,
                        force_reload_raw=force_reload_raw,
                        feat_reduction=feat_reduction,
                        feat_reduction_dim=feat_reduction_dim,
                        subgraph_svd=subgraph_svd,
                        subgraph_svd_feat_dim=subgraph_svd_feat_dim,
                        subgraph_svd_struct_dim=subgraph_svd_struct_dim,
                        subgraph_svd_matrix=subgraph_svd_matrix,
                    ),
                )
            else:
                task_ok = _run_with_zip_retry(
                    dataset,
                    root_path,
                    lambda: _process_non_edge_task(
                        dataset=dataset,
                        task_level=task_level,
                        root=root,
                        split_defs=task_split_defs,
                        split_seeds=normalized_split_seeds,
                        dataset_split_root=dataset_split_root,
                        dataset_feature_svd_dir=dataset_feature_svd_dir,
                        dataset_induced_root=dataset_induced_root,
                        dataset_subgraph_svd_dir=dataset_subgraph_svd_dir,
                        split_batch_size=split_batch_size,
                        split_num_workers=split_num_workers,
                        induced=induced,
                        induced_min_size=induced_min_size,
                        induced_max_size=induced_max_size,
                        induced_max_hops=induced_max_hops,
                        force_reload_raw=force_reload_raw,
                        feat_reduction=feat_reduction,
                        feat_reduction_dim=feat_reduction_dim,
                        subgraph_svd=subgraph_svd,
                        subgraph_svd_feat_dim=subgraph_svd_feat_dim,
                        subgraph_svd_struct_dim=subgraph_svd_struct_dim,
                        subgraph_svd_matrix=subgraph_svd_matrix,
                    ),
                )
        except Exception as exc:  # pylint: disable=broad-except
            all_ok = False
            print(f"[FAIL] {dataset} (task={task_level}): {exc}")
            continue

        if not task_ok:
            all_ok = False
            print(f"[FAIL] {dataset} (task={task_level}): preparation completed with failures.")
            continue
        print(f"[OK] {dataset} (task={task_level})")
    return all_ok


def prepare_datasets(
    tsv_path: str | Iterable[str],
    root: Path | str,
    task_levels: Iterable[str] = ("node", "graph"),
    feat_reduction: bool = True,
    feat_reduction_dim: int = 100,
    feature_svd_dir: str = "data/feature_svd",
    induced: bool = False,
    induced_min_size: int = 10,
    induced_max_size: int = 30,
    induced_max_hops: int = 5,
    induced_root: str = "",
    force_reload_raw: bool = False,
    subgraph_svd: bool = False,
    subgraph_svd_feat_dim: int = 100,
    subgraph_svd_struct_dim: int = 100,
    subgraph_svd_matrix: str = "adjacency",
    subgraph_svd_dir: str = "data/subgraph_svd",
    use_infer: bool = True,
    split_root: str = "",
    generate_edge_level: bool = True,
    split_defs_by_task: Optional[Dict[str, List[Tuple[float, float, float]]]] = None,
    split_seeds: Optional[List[int]] = None,
    split_batch_size: int = 64,
    split_num_workers: int = 0,
) -> int:
    """Prepare all requested datasets and return a process exit code."""
    if isinstance(tsv_path, (list, tuple, set)):
        datasets = [str(name) for name in tsv_path]
    else:
        datasets = read_datasets(str(tsv_path))
    if not datasets:
        print(f"[INFO] No datasets found in {tsv_path}")
        return 0

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    success = 0
    for idx, name in enumerate(datasets, 1):
        print(DATASET_SEPARATOR)
        print(f"[{idx}/{len(datasets)}] Preparing {name}")
        print(DATASET_SEPARATOR)
        inferred = infer_task_level(name) if use_infer else None
        if inferred == "node" and generate_edge_level:
            levels = ("node", "edge")
        else:
            levels = (inferred,) if inferred else task_levels
        if try_load(
            name,
            root=str(root),
            task_levels=levels,
            feat_reduction=feat_reduction,
            feat_reduction_dim=feat_reduction_dim,
            feature_svd_dir=feature_svd_dir,
            induced=induced,
            induced_min_size=induced_min_size,
            induced_max_size=induced_max_size,
            induced_max_hops=induced_max_hops,
            induced_root=induced_root,
            force_reload_raw=force_reload_raw,
            subgraph_svd=subgraph_svd,
            subgraph_svd_feat_dim=subgraph_svd_feat_dim,
            subgraph_svd_struct_dim=subgraph_svd_struct_dim,
            subgraph_svd_matrix=subgraph_svd_matrix,
            subgraph_svd_dir=subgraph_svd_dir,
            split_root=split_root,
            split_defs_by_task=split_defs_by_task,
            split_seeds=split_seeds,
            split_batch_size=split_batch_size,
            split_num_workers=split_num_workers,
        ):
            success += 1
    print(f"[SUMMARY] success={success}, fail={len(datasets)-success}, total={len(datasets)}")
    return 0 if success == len(datasets) else 1
