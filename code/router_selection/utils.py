"""Utility helpers for router selection."""
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from code.data_loader import create_dataset
from code.router_dataset_generation.utils import serialize_graph
from code.router_graph_construction.builder import _build_subgraph_key

from .data import RouterGraphData


def _shared_split_root(cfg) -> str:
    ds_cfg = getattr(getattr(cfg, "data_preparation", None), "dataset", None)
    return getattr(ds_cfg, "split_root", "data/splits")


def dataset_from_application_meta(meta: Dict) -> Optional[str]:
    if not isinstance(meta, dict):
        return None
    dataset = meta.get("dataset")
    if dataset:
        return dataset
    key = meta.get("key") or {}
    if isinstance(key, dict):
        return key.get("dataset")
    return None


def normalize_dataset_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(value).strip()] if str(value).strip() else []


def resolve_split_datasets(
    graph_data: RouterGraphData,
    test_dataset,
    train_datasets_raw,
) -> Tuple[List[str], List[str], List[str]]:
    test_datasets = normalize_dataset_list(test_dataset)
    if not test_datasets:
        raise ValueError("Please provide router_selection.test_dataset (single or comma-separated).")

    available_datasets = sorted(
        {
            ds
            for ds in (
                dataset_from_application_meta(m)
                for m in graph_data.application_id_map.values()
            )
            if ds
        }
    )
    missing_tests = sorted(set(test_datasets) - set(available_datasets))
    if missing_tests:
        raise ValueError(f"test_dataset not found in graph datasets: {missing_tests}")

    train_datasets = normalize_dataset_list(train_datasets_raw)
    if train_datasets:
        unknown_train = sorted(set(train_datasets) - set(available_datasets))
        if unknown_train:
            raise ValueError(f"train_datasets not found in graph datasets: {unknown_train}")
        overlap = set(train_datasets) & set(test_datasets)
        if overlap:
            raise ValueError(f"train_datasets overlaps with test: {sorted(overlap)}")
    else:
        train_datasets = [ds for ds in available_datasets if ds not in set(test_datasets)]

    return train_datasets, test_datasets, available_datasets


def expand_test_subgraphs(
    cfg,
    graph_data: RouterGraphData,
    test_dataset: str,
) -> Dict[str, int]:
    if not test_dataset:
        return {"added": 0, "total": int(graph_data.num_subgraphs)}

    app_id = None
    task_meta: Dict[str, object] = {}
    for app_id_str, meta in graph_data.application_id_map.items():
        if dataset_from_application_meta(meta) == test_dataset:
            app_id = int(app_id_str)
            task_meta = meta.get("task") or {}
            break
    if app_id is None:
        raise ValueError(f"test_dataset '{test_dataset}' not found in graph datasets.")

    router_ds = getattr(cfg, "router_dataset", None)
    ds_cfg = getattr(router_ds, "target_dataset", None)
    if ds_cfg is None:
        train_cfg = getattr(cfg, "train", None)
        ds_cfg = getattr(train_cfg, "dataset", None)
    if ds_cfg is None:
        ds_cfg = cfg.dataset
    task_level = str(task_meta.get("raw_task_level") or task_meta.get("task_level") or ds_cfg.task_level or "node")
    induced = bool(task_meta.get("induced")) if "induced" in task_meta else bool(ds_cfg.induced)
    if induced and task_level == "graph":
        task_level = "node"

    dataset = create_dataset(
        name=test_dataset,
        root=ds_cfg.root,
        task_level=task_level,
        feat_reduction=ds_cfg.feat_reduction,
        feat_reduction_dim=ds_cfg.feat_reduction_svd_dim,
        feature_svd_dir=getattr(ds_cfg, "feature_svd_dir", "data/feature_svd"),
        induced=induced,
        induced_min_size=ds_cfg.induced_min_size,
        induced_max_size=ds_cfg.induced_max_size,
        induced_max_hops=ds_cfg.induced_max_hops,
        split_root=_shared_split_root(cfg),
        induced_root=ds_cfg.induced_root,
    )

    key_to_id: Dict[str, int] = {}
    for sub_id_str, meta in graph_data.subgraph_id_map.items():
        key = meta.get("key")
        if key:
            key_to_id[key] = int(sub_id_str)

    added = 0
    new_subgraph_ids: List[int] = []
    next_id = int(graph_data.num_subgraphs)
    if induced and hasattr(dataset, "__len__") and len(dataset) > 0:
        for idx in range(len(dataset)):
            data = dataset[idx]
            node_id = int(getattr(data, "base_node_id", idx))
            target = serialize_graph(data)
            target["node_id"] = node_id
            sub_key, _hash_key, summary = _build_subgraph_key(target, dataset=test_dataset)
            if sub_key in key_to_id:
                continue
            summary["dataset"] = test_dataset
            graph_data.subgraph_id_map[str(next_id)] = summary
            graph_data.subgraph_application_ids.append(app_id)
            graph_data.subgraph_expert_ids.append(torch.empty(0, dtype=torch.long))
            graph_data.subgraph_expert_weights.append(torch.empty(0, dtype=torch.float))
            key_to_id[sub_key] = next_id
            new_subgraph_ids.append(next_id)
            next_id += 1
            added += 1
    else:
        if induced and getattr(dataset, "base_num_nodes", None) is not None:
            total_nodes = int(dataset.base_num_nodes)
        elif hasattr(dataset, "data") and getattr(dataset.data, "num_nodes", None) is not None:
            total_nodes = int(dataset.data.num_nodes)
        else:
            total_nodes = int(len(dataset)) if hasattr(dataset, "__len__") else 0
        if total_nodes <= 0:
            return {"added": 0, "total": int(graph_data.num_subgraphs)}
        base_data = None
        if hasattr(dataset, "data"):
            base_data = dataset.data
        for node_idx in range(total_nodes):
            if base_data is not None and getattr(base_data, "x", None) is not None:
                node_x = base_data.x[node_idx].view(1, -1).cpu().tolist()
                node_y = None
                if getattr(base_data, "y", None) is not None:
                    node_y = [int(base_data.y[node_idx].item())]
                target = {
                    "num_nodes": 1,
                    "num_edges": 0,
                    "x": node_x,
                    "y": node_y,
                    "node_id": int(node_idx),
                }
            else:
                target = {"node_id": int(node_idx), "num_nodes": 1, "num_edges": 0}
            sub_key, _hash_key, summary = _build_subgraph_key(target, dataset=test_dataset)
            if sub_key in key_to_id:
                continue
            summary["dataset"] = test_dataset
            graph_data.subgraph_id_map[str(next_id)] = summary
            graph_data.subgraph_application_ids.append(app_id)
            graph_data.subgraph_expert_ids.append(torch.empty(0, dtype=torch.long))
            graph_data.subgraph_expert_weights.append(torch.empty(0, dtype=torch.float))
            key_to_id[sub_key] = next_id
            new_subgraph_ids.append(next_id)
            next_id += 1
            added += 1

    graph_data.num_subgraphs = int(next_id)
    if added > 0:
        new_edges = torch.tensor(
            [[app_id] * added, new_subgraph_ids],
            dtype=torch.long,
        )
        if graph_data.app_to_sub_edge_index.numel() == 0:
            graph_data.app_to_sub_edge_index = new_edges
        else:
            graph_data.app_to_sub_edge_index = torch.cat(
                [graph_data.app_to_sub_edge_index, new_edges],
                dim=1,
            )
        graph = graph_data.graph
        if isinstance(graph, dict):
            graph.get("node_counts", {})["target_subgraph"] = graph_data.num_subgraphs
            app_key = ("application", "to", "target_subgraph")
            rev_key = ("target_subgraph", "rev_to", "application")
            store = graph.get(app_key)
            if store is not None:
                edge_index = store.get("edge_index")
                if edge_index is not None:
                    store["edge_index"] = torch.cat([edge_index, new_edges], dim=1)
                edge_weight = store.get("edge_weight")
                if edge_weight is not None:
                    store["edge_weight"] = torch.cat(
                        [edge_weight, torch.ones(added, dtype=edge_weight.dtype)],
                        dim=0,
                    )
            rev_store = graph.get(rev_key)
            if rev_store is not None:
                edge_index = rev_store.get("edge_index")
                if edge_index is not None:
                    rev_store["edge_index"] = torch.cat([edge_index, new_edges.flip(0)], dim=1)
                edge_weight = rev_store.get("edge_weight")
                if edge_weight is not None:
                    rev_store["edge_weight"] = torch.cat(
                        [edge_weight, torch.ones(added, dtype=edge_weight.dtype)],
                        dim=0,
                    )
        else:
            graph["target_subgraph"].num_nodes = graph_data.num_subgraphs
            if ("application", "to", "target_subgraph") in graph.edge_types:
                store = graph["application", "to", "target_subgraph"]
                store.edge_index = torch.cat([store.edge_index, new_edges], dim=1)
                if getattr(store, "edge_weight", None) is not None:
                    store.edge_weight = torch.cat(
                        [store.edge_weight, torch.ones(added, dtype=store.edge_weight.dtype)],
                        dim=0,
                    )
            if ("target_subgraph", "rev_to", "application") in graph.edge_types:
                rev_store = graph["target_subgraph", "rev_to", "application"]
                rev_store.edge_index = torch.cat([rev_store.edge_index, new_edges.flip(0)], dim=1)
                if getattr(rev_store, "edge_weight", None) is not None:
                    rev_store.edge_weight = torch.cat(
                        [rev_store.edge_weight, torch.ones(added, dtype=rev_store.edge_weight.dtype)],
                        dim=0,
                    )
    return {"added": added, "total": int(graph_data.num_subgraphs)}


def build_subgraph_dataset_index(graph_data: RouterGraphData) -> Dict[int, Optional[str]]:
    dataset_by_subgraph: Dict[int, Optional[str]] = {}
    for sub_id, app_id in enumerate(graph_data.subgraph_application_ids):
        if app_id == -1:
            continue
        app_meta = graph_data.application_id_map.get(str(app_id), {})
        dataset_by_subgraph[sub_id] = app_meta.get("dataset")
    return dataset_by_subgraph


def enrich_predictions(
    predictions: Sequence[Dict[str, object]],
    expert_meta: Dict[str, Dict],
    dataset_by_subgraph: Dict[int, Optional[str]],
) -> List[Dict[str, object]]:
    enriched: List[Dict[str, object]] = []
    for item in predictions:
        sub_id = int(item.get("target_subgraph_id"))
        dataset = dataset_by_subgraph.get(sub_id)
        experts = []
        for expert in item.get("experts", []):
            expert_id = int(expert.get("expert_id"))
            meta = expert_meta.get(str(expert_id), {})
            experts.append(
                {
                    "expert_id": expert_id,
                    "score": expert.get("score"),
                    "expert_model": {
                        "model": meta.get("model"),
                        "checkpoint": meta.get("checkpoint"),
                    },
                }
            )
        enriched.append(
            {
                "target_subgraph_id": sub_id,
                "dataset": dataset,
                "experts": experts,
            }
        )
    return enriched


def count_subgraph_edges(graph_data: RouterGraphData, subgraph_ids: List[int]) -> int:
    total = 0
    for sub_id in subgraph_ids:
        total += int(graph_data.subgraph_expert_ids[sub_id].numel())
    return int(total)


def format_split_stats(
    label: str,
    graph_data: RouterGraphData,
    splits: Dict[str, List[int]],
    edge_splits: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
) -> str:
    train_subgraphs = splits.get("train", [])
    valid_subgraphs = splits.get("valid", [])
    test_subgraphs = splits.get("test", [])
    train_edges = int(
        sum(edges[0].numel() for edges in edge_splits.get("train", {}).values())
    )
    valid_edges = int(
        sum(edges[0].numel() for edges in edge_splits.get("valid", {}).values())
    )
    test_edges = count_subgraph_edges(graph_data, test_subgraphs)
    total_edges = int(sum(t.numel() for t in graph_data.subgraph_expert_ids))
    app_sub_edges = int(graph_data.app_to_sub_edge_index.shape[1])
    sub_exp_edges = int(graph_data.sub_to_expert_edge_index.shape[1])
    return "\n".join(
        [
            f"[RouterSelect]{label} Graph nodes: "
            f"applications={graph_data.num_applications} "
            f"subgraphs={graph_data.num_subgraphs} "
            f"experts={graph_data.num_experts} "
            f"edges={total_edges}",
            f"[RouterSelect]{label} Edge types: "
            f"application->target_subgraph={app_sub_edges} "
            f"target_subgraph->expert={sub_exp_edges}",
            f"[RouterSelect]{label} Split sizes: "
            f"train_subgraphs={len(train_subgraphs)} train_edges={train_edges} "
            f"valid_subgraphs={len(valid_subgraphs)} valid_edges={valid_edges} "
            f"test_subgraphs={len(test_subgraphs)} test_edges={test_edges}",
        ]
    )


def _tokenize_run_name(value: str) -> List[str]:
    if not isinstance(value, str):
        return []
    raw = value.replace("-", "_").lower()
    return [tok for tok in raw.split("_") if tok]


def remove_test_dataset_experts(graph_data: RouterGraphData, test_dataset) -> Dict[str, int]:
    targets = [t.lower() for t in normalize_dataset_list(test_dataset)]
    if not targets:
        return {"removed": 0, "kept": graph_data.num_experts}
    keep_ids: List[int] = []
    removed_ids: List[int] = []
    for expert_id_str, meta in graph_data.expert_id_map.items():
        expert_id = int(expert_id_str)
        model = (meta or {}).get("model", "")
        tokens = _tokenize_run_name(model)
        if any(target in tokens for target in targets):
            removed_ids.append(expert_id)
        else:
            keep_ids.append(expert_id)
    if not removed_ids:
        return {"removed": 0, "kept": graph_data.num_experts}
    keep_ids = sorted(keep_ids)
    id_map = {old: new for new, old in enumerate(keep_ids)}

    new_expert_map: Dict[str, Dict] = {}
    for new_id, old_id in enumerate(keep_ids):
        new_expert_map[str(new_id)] = graph_data.expert_id_map.get(str(old_id), {})
    graph_data.expert_id_map = new_expert_map
    graph_data.num_experts = len(keep_ids)

    new_sub_expert_ids: List[torch.Tensor] = []
    new_sub_expert_weights: List[torch.Tensor] = []
    keep_set = set(keep_ids)
    for expert_ids, weights in zip(graph_data.subgraph_expert_ids, graph_data.subgraph_expert_weights):
        if expert_ids.numel() == 0:
            new_sub_expert_ids.append(expert_ids)
            new_sub_expert_weights.append(weights)
            continue
        mask = torch.tensor([int(eid.item()) in keep_set for eid in expert_ids], dtype=torch.bool)
        if mask.numel() == 0 or not mask.any():
            new_sub_expert_ids.append(torch.empty(0, dtype=torch.long))
            new_sub_expert_weights.append(torch.empty(0, dtype=torch.float))
            continue
        kept_ids = expert_ids[mask]
        remapped = torch.tensor([id_map[int(eid.item())] for eid in kept_ids], dtype=torch.long)
        new_sub_expert_ids.append(remapped)
        new_sub_expert_weights.append(weights[mask])
    graph_data.subgraph_expert_ids = new_sub_expert_ids
    graph_data.subgraph_expert_weights = new_sub_expert_weights

    if graph_data.sub_to_expert_edge_index.numel() > 0:
        expert_ids = graph_data.sub_to_expert_edge_index[1]
        keep_mask = torch.tensor([int(eid.item()) in keep_set for eid in expert_ids], dtype=torch.bool)
        filtered_edges = graph_data.sub_to_expert_edge_index[:, keep_mask]
        filtered_weights = graph_data.sub_to_expert_weight[keep_mask]
        if filtered_edges.numel() > 0:
            remapped_experts = torch.tensor(
                [id_map[int(eid.item())] for eid in filtered_edges[1]],
                dtype=torch.long,
            )
            filtered_edges = torch.stack([filtered_edges[0], remapped_experts], dim=0)
        graph_data.sub_to_expert_edge_index = filtered_edges
        graph_data.sub_to_expert_weight = filtered_weights

    graph = graph_data.graph
    if isinstance(graph, dict):
        graph.get("node_counts", {})["expert"] = graph_data.num_experts
        key = ("target_subgraph", "to", "expert")
        rev_key = ("expert", "rev_to", "target_subgraph")
        store = graph.get(key)
        if store is not None:
            store["edge_index"] = graph_data.sub_to_expert_edge_index
            store["edge_weight"] = graph_data.sub_to_expert_weight
        rev_store = graph.get(rev_key)
        if rev_store is not None:
            rev_store["edge_index"] = graph_data.sub_to_expert_edge_index.flip(0)
            edge_weight = graph_data.sub_to_expert_weight
            rev_store["edge_weight"] = edge_weight
    else:
        graph["expert"].num_nodes = graph_data.num_experts
        if ("target_subgraph", "to", "expert") in graph.edge_types:
            store = graph["target_subgraph", "to", "expert"]
            store.edge_index = graph_data.sub_to_expert_edge_index
            if getattr(store, "edge_weight", None) is not None:
                store.edge_weight = graph_data.sub_to_expert_weight
        if ("expert", "rev_to", "target_subgraph") in graph.edge_types:
            rev_store = graph["expert", "rev_to", "target_subgraph"]
            rev_store.edge_index = graph_data.sub_to_expert_edge_index.flip(0)
            if getattr(rev_store, "edge_weight", None) is not None:
                rev_store.edge_weight = graph_data.sub_to_expert_weight

    return {"removed": len(removed_ids), "kept": graph_data.num_experts}
