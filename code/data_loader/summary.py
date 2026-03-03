"""Dataset summary generation for all configured available datasets."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch

from code.config import cfg as base_cfg, update_cfg
from code.data_loader.dataset_prepare import read_datasets
from code.data_loader.datasets import create_dataset, is_regression_dataset


DEFAULT_OUTPUT_PATH = Path("data/data_summary.tsv")


@dataclass(frozen=True)
class DatasetSummaryRow:
    """One summary row written to `data_summary.tsv`."""

    name: str
    dataset_type: str
    num_graphs: int
    num_nodes: int
    avg_nodes: float
    num_edges: int
    avg_edges: float
    num_features: int
    task: str
    num_labels: int


def _resolve_list_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path

    typo_fix = Path(str(path).replace("avaulable", "available"))
    if typo_fix.exists():
        return typo_fix

    raise FileNotFoundError(f"Dataset list file not found: {path_str}")


def _sum_from_slices(dataset, key: str) -> Optional[int]:
    """Fast aggregate using InMemoryDataset slices when available."""
    slices = getattr(dataset, "slices", None)
    if slices is None or key not in slices:
        return None

    boundaries = slices[key]
    try:
        if torch.is_tensor(boundaries):
            boundaries = boundaries.view(-1)
            if boundaries.numel() < 2:
                return 0
            span = boundaries[1:] - boundaries[:-1]
            return int(span.sum().item())

        if len(boundaries) < 2:
            return 0
        total = 0
        for idx in range(len(boundaries) - 1):
            total += int(boundaries[idx + 1] - boundaries[idx])
        return total
    except Exception:
        return None


def _compute_graph_totals(dataset) -> tuple[int, int, int]:
    num_graphs = int(len(dataset))
    total_nodes = _sum_from_slices(dataset, "x")
    total_edges = _sum_from_slices(dataset, "edge_index")

    if total_nodes is not None and total_edges is not None:
        return num_graphs, int(total_nodes), int(total_edges)

    # Fallback for datasets without slices.
    nodes = 0
    edges = 0
    for graph in dataset:
        graph_nodes = getattr(graph, "num_nodes", None)
        if graph_nodes is None and getattr(graph, "x", None) is not None:
            graph_nodes = graph.x.size(0)
        graph_edges = getattr(graph, "num_edges", None)
        if graph_edges is None and getattr(graph, "edge_index", None) is not None:
            graph_edges = graph.edge_index.size(1)
        nodes += int(graph_nodes or 0)
        edges += int(graph_edges or 0)
    return num_graphs, nodes, edges


def _extract_labels(dataset, task_level: str) -> Optional[torch.Tensor]:
    try:
        if task_level == "node":
            data = dataset[0]
            labels = getattr(data, "y", None)
            if labels is None:
                return None
            return torch.as_tensor(labels)

        data_obj = getattr(dataset, "data", None)
        labels = getattr(data_obj, "y", None) if data_obj is not None else None
        if labels is not None:
            return torch.as_tensor(labels)

        collected = []
        for graph in dataset:
            y = getattr(graph, "y", None)
            if y is None:
                continue
            collected.append(torch.as_tensor(y).view(-1))
        if not collected:
            return None
        return torch.cat(collected, dim=0)
    except Exception:
        return None


def _num_classes_for_classification(dataset, task_level: str) -> int:
    num_classes = getattr(dataset, "num_classes", None)
    try:
        if num_classes is not None and int(num_classes) > 0:
            return int(num_classes)
    except Exception:
        pass

    labels = _extract_labels(dataset, task_level)
    if labels is None or labels.numel() == 0:
        return -1

    label_tensor = labels.detach().view(-1)
    if label_tensor.dtype.is_floating_point:
        finite = label_tensor[torch.isfinite(label_tensor)]
        if finite.numel() == 0:
            return -1
        label_tensor = finite.round().long()
    else:
        label_tensor = label_tensor.long()

    # Ignore common missing-label markers.
    label_tensor = label_tensor[label_tensor >= 0]
    if label_tensor.numel() == 0:
        return -1

    return int(torch.unique(label_tensor).numel())


def _num_features(dataset, task_level: str) -> int:
    for attr in ("num_node_features", "num_features"):
        value = getattr(dataset, attr, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass

    try:
        if task_level == "node":
            data = dataset[0]
            x = getattr(data, "x", None)
            return int(x.size(1)) if x is not None and x.dim() == 2 else -1

        if len(dataset) == 0:
            return -1
        graph = dataset[0]
        x = getattr(graph, "x", None)
        return int(x.size(1)) if x is not None and x.dim() == 2 else -1
    except Exception:
        return -1


def _summarize_node_dataset(name: str, dataset_root: str) -> DatasetSummaryRow:
    dataset = create_dataset(
        name=name,
        root=dataset_root,
        task_level="node",
        feat_reduction=False,
        induced=False,
    )
    data = dataset[0]

    num_nodes = int(getattr(data, "num_nodes", 0) or (data.x.size(0) if getattr(data, "x", None) is not None else 0))
    edge_index = getattr(data, "edge_index", None)
    num_edges = int(edge_index.size(1)) if edge_index is not None else 0

    regression = is_regression_dataset(dataset, "node")
    task = "regression" if regression else "classification"
    num_labels = -1 if regression else _num_classes_for_classification(dataset, "node")
    num_features = _num_features(dataset, "node")

    return DatasetSummaryRow(
        name=name,
        dataset_type="node",
        num_graphs=1,
        num_nodes=num_nodes,
        avg_nodes=float(num_nodes),
        num_edges=num_edges,
        avg_edges=float(num_edges),
        num_features=num_features,
        task=task,
        num_labels=num_labels,
    )


def _summarize_graph_dataset(name: str, dataset_root: str) -> DatasetSummaryRow:
    dataset = create_dataset(
        name=name,
        root=dataset_root,
        task_level="graph",
        feat_reduction=False,
        induced=False,
    )

    num_graphs, total_nodes, total_edges = _compute_graph_totals(dataset)
    avg_nodes = float(total_nodes) / float(num_graphs) if num_graphs > 0 else 0.0
    avg_edges = float(total_edges) / float(num_graphs) if num_graphs > 0 else 0.0

    regression = is_regression_dataset(dataset, "graph")
    task = "regression" if regression else "classification"
    num_labels = -1 if regression else _num_classes_for_classification(dataset, "graph")
    num_features = _num_features(dataset, "graph")

    return DatasetSummaryRow(
        name=name,
        dataset_type="graph",
        num_graphs=num_graphs,
        num_nodes=total_nodes,
        avg_nodes=avg_nodes,
        num_edges=total_edges,
        avg_edges=avg_edges,
        num_features=num_features,
        task=task,
        num_labels=num_labels,
    )


def _rows_to_tsv(rows: Sequence[DatasetSummaryRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "name",
                "type",
                "#graphs",
                "#nodes",
                "#avg. nodes",
                "#edges",
                "#avg. edges",
                "#features",
                "task",
                "#labels",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.name,
                    row.dataset_type,
                    row.num_graphs,
                    row.num_nodes,
                    f"{row.avg_nodes:.2f}",
                    row.num_edges,
                    f"{row.avg_edges:.2f}",
                    row.num_features,
                    row.task,
                    row.num_labels,
                ]
            )


def _print_summary_status(name: str, dataset_type: str) -> None:
    print(f"[DataSummary] Processing {name} (type={dataset_type})...")


def _sorted_dataset_names(names: Sequence[str]) -> List[str]:
    """Return deterministic A->Z ordering for dataset names."""
    return sorted(names, key=lambda item: item.casefold())


def run_data_summary(cfg) -> int:
    ds_cfg = cfg.data_preparation.dataset
    dataset_root = str(getattr(ds_cfg, "root", "data/datasets"))

    node_list_path = _resolve_list_path(str(getattr(ds_cfg, "available_node_datasets", "")))
    graph_list_path = _resolve_list_path(str(getattr(ds_cfg, "available_graph_datasets", "")))

    node_names = _sorted_dataset_names(read_datasets(str(node_list_path)))
    graph_names = _sorted_dataset_names(read_datasets(str(graph_list_path)))

    rows: List[DatasetSummaryRow] = []
    failures: List[str] = []

    for name in node_names:
        _print_summary_status(name, "node")
        try:
            rows.append(_summarize_node_dataset(name=name, dataset_root=dataset_root))
        except Exception as exc:  # pylint: disable=broad-except
            failures.append(f"{name} (node): {exc}")
            print(f"[DataSummary][FAIL] {name} (node): {exc}")

    for name in graph_names:
        _print_summary_status(name, "graph")
        try:
            rows.append(_summarize_graph_dataset(name=name, dataset_root=dataset_root))
        except Exception as exc:  # pylint: disable=broad-except
            failures.append(f"{name} (graph): {exc}")
            print(f"[DataSummary][FAIL] {name} (graph): {exc}")

    output_path = DEFAULT_OUTPUT_PATH
    _rows_to_tsv(rows, output_path)
    print(f"[DataSummary] Saved {len(rows)} dataset summaries to {output_path}")

    if failures:
        print(f"[DataSummary] Failed datasets: {len(failures)}")
        return 1
    return 0


def run_data_summary_from_cli(argv: Iterable[str]) -> int:
    cfg = update_cfg(base_cfg, " ".join(argv))
    return run_data_summary(cfg)
