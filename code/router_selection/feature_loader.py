"""Node feature loading for router selection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch_geometric.data import Data

from .context_encoder import (
    GraphContextEncoder,
    GraphEncoderConfig,
    TextContextEncoder,
    TextEncoderConfig,
    build_condensation_index,
    build_context_index,
    load_condensed_graphs,
    resolve_dataset_context_path,
    resolve_condensation_path,
    resolve_model_context_path,
)
from .data_loader import build_subgraph_svd_features
from .data import RouterGraphData


def _resolve_context_root(cfg) -> Path:
    source = getattr(cfg.router_selection, "context_source", "") or getattr(cfg.context, "model", "gpt-4o")
    source = str(source).replace(".", "_")
    return Path(cfg.context.output_dir) / source


def _resolve_dataset_context_path(context_index, dataset: str, task_level: str):
    path = resolve_dataset_context_path(context_index, dataset, task_level)
    if path is None:
        for fallback_level in ("node", "graph", "edge"):
            path = resolve_dataset_context_path(context_index, dataset, fallback_level)
            if path is not None:
                break
    return path


@dataclass
class RouterFeatureBundle:
    node_features: Dict[str, torch.Tensor]
    application_graphs: Optional[List[List[Data]]]
    application_graph_encoder: Optional[GraphContextEncoder]
    application_text_features: Optional[torch.Tensor]
    context_mode: str


def _normalize_context_mode(mode: str) -> str:
    value = str(mode or "text").strip().lower()
    if value in ("condensation", "graph", "graph_only", "condensed"):
        return "condensation"
    if value in ("concat", "hybrid", "mix", "combine"):
        return "concat"
    return "text"


def _build_text_encoder(cfg) -> TextContextEncoder:
    rp = cfg.router_selection
    encoder_cfg = TextEncoderConfig(
        model_name=getattr(rp, "text_embed_model", "bert-base-uncased"),
        cache_dir=getattr(rp, "text_embed_cache_dir", "router_selection/cache/text_embeddings"),
        max_length=getattr(rp, "text_max_length", 256),
        batch_size=getattr(rp, "text_batch_size", 16),
    )
    return TextContextEncoder(encoder_cfg)


def _build_application_text_features(
    cfg,
    graph_data: RouterGraphData,
    text_encoder: TextContextEncoder,
) -> torch.Tensor:
    context_root = _resolve_context_root(cfg)
    context_index = build_context_index(str(context_root))
    app_features: List[torch.Tensor] = []
    for app_id in range(graph_data.num_applications):
        meta = graph_data.application_id_map.get(str(app_id), {})
        dataset = meta.get("dataset") or ""
        task_meta = meta.get("task") or {}
        task_level = str(task_meta.get("raw_task_level") or task_meta.get("task_level") or cfg.dataset.task_level)
        context_path = _resolve_dataset_context_path(context_index, dataset, task_level)
        if context_path is not None:
            emb = text_encoder.embed_file(context_path, "dataset")
        else:
            fallback = f"{dataset}\n{task_level}"
            emb = text_encoder.embed_text(fallback, f"dataset:{dataset}:{task_level}")
        app_features.append(emb)
    return torch.stack(app_features, dim=0) if app_features else torch.empty((0, text_encoder.embed_dim))


def _build_expert_text_features(
    cfg,
    graph_data: RouterGraphData,
    text_encoder: TextContextEncoder,
) -> torch.Tensor:
    context_root = _resolve_context_root(cfg)
    expert_features: List[torch.Tensor] = []
    for expert_id in range(graph_data.num_experts):
        meta = graph_data.expert_id_map.get(str(expert_id), {})
        model_name = meta.get("model") or ""
        context_path = resolve_model_context_path(str(context_root), model_name)
        if context_path is not None:
            emb = text_encoder.embed_file(context_path, "model")
        else:
            emb = text_encoder.embed_text(model_name, f"model:{model_name}")
        expert_features.append(emb)
    return torch.stack(expert_features, dim=0) if expert_features else torch.empty((0, text_encoder.embed_dim))


def _build_condensation_encoder(cfg) -> GraphContextEncoder:
    cond_cfg = getattr(cfg.router_selection, "condensation", None)
    if cond_cfg is None:
        raise ValueError("router_selection.condensation config missing.")
    encoder_cfg = GraphEncoderConfig(
        root_dir=str(getattr(cond_cfg, "root_dir", "")),
        method=str(getattr(cond_cfg, "method", "bonsai")),
        tag=str(getattr(cond_cfg, "tag", "")),
        use_processed=bool(getattr(cond_cfg, "use_processed", False)),
        embed_dim=int(getattr(cond_cfg, "embed_dim", 128)),
        hidden_dim=int(getattr(cond_cfg, "hidden_dim", 128)),
        num_layers=int(getattr(cond_cfg, "num_layers", 2)),
        dropout=float(getattr(cond_cfg, "dropout", 0.1)),
        pool=str(getattr(cond_cfg, "pool", "mean")),
        max_graphs=int(getattr(cond_cfg, "max_graphs", 0)),
        input_dim=int(getattr(cond_cfg, "input_dim", 0)),
    )
    return GraphContextEncoder(encoder_cfg)


def _candidate_condensation_roots(root_dir: str, method: str) -> List[Path]:
    candidates: List[Path] = []
    if root_dir:
        candidates.append(Path(root_dir))
    method = str(method or "bonsai").lower()
    if method == "sgdc":
        candidates.extend(
            [
                Path("condensation/graph_level_sgdc/condensation"),
                Path("data/datasets/sgdc"),
            ]
        )
    else:
        candidates.extend(
            [
                Path("condensation/node_level_bonsai"),
                Path("data/datasets/bonsai"),
            ]
        )
    seen = set()
    unique: List[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _build_condensation_index_with_fallback(
    root_dir: str,
    method: str,
    tag: str,
    use_processed: bool,
) -> tuple[Path, Dict[str, Path], List[str]]:
    checked: List[str] = []
    for root in _candidate_condensation_roots(root_dir, method):
        if not root.exists():
            checked.append(str(root))
            continue
        index = build_condensation_index(str(root), method, tag, use_processed)
        if index:
            return root, index, checked
        checked.append(str(root))
    return Path(root_dir), {}, checked


def _build_application_graphs(
    cfg,
    graph_data: RouterGraphData,
) -> List[List[Data]]:
    cond_cfg = getattr(cfg.router_selection, "condensation", None)
    root_dir = str(getattr(cond_cfg, "root_dir", "")) if cond_cfg is not None else ""
    method = str(getattr(cond_cfg, "method", "bonsai")) if cond_cfg is not None else "bonsai"
    tag = str(getattr(cond_cfg, "tag", "")) if cond_cfg is not None else ""
    use_processed = bool(getattr(cond_cfg, "use_processed", False)) if cond_cfg is not None else False

    root_path, index, checked = _build_condensation_index_with_fallback(root_dir, method, tag, use_processed)
    if index and root_dir and Path(root_dir) != root_path:
        print(f"[RouterSelect] Using condensation root {root_path} (configured {root_dir}).")
    if not index:
        checked_list = ", ".join(checked) if checked else root_dir
        print(f"[RouterSelect] Warning: no condensation outputs found under {checked_list}.")
    graphs_by_path: Dict[str, List[Data]] = {}
    app_graphs: List[List[Data]] = []
    missing = 0
    missing_datasets: List[str] = []
    dataset_graph_counts: Dict[str, int] = {}
    for app_id in range(graph_data.num_applications):
        meta = graph_data.application_id_map.get(str(app_id), {})
        dataset = meta.get("dataset") or ""
        path = resolve_condensation_path(index, dataset) if dataset else None
        if path is None:
            missing += 1
            if dataset:
                missing_datasets.append(dataset)
            app_graphs.append([])
            continue
        key = str(path)
        graphs = graphs_by_path.get(key)
        if graphs is None:
            graphs = load_condensed_graphs(path)
            graphs_by_path[key] = graphs
        dataset_graph_counts[dataset] = len(graphs)
        app_graphs.append(graphs)
    if missing > 0:
        print(f"[RouterSelect] Warning: missing condensation graphs for {missing} applications.")
    if dataset_graph_counts:
        counts = list(dataset_graph_counts.values())
        total_graphs = sum(counts)
        avg_graphs = total_graphs / max(1, len(counts))
        print(
            "[RouterSelect] Condensation datasets loaded: "
            f"{len(counts)}/{graph_data.num_applications} "
            f"total_graphs={total_graphs} avg_graphs={avg_graphs:.2f} "
            f"min_graphs={min(counts)} max_graphs={max(counts)}"
        )
    if missing_datasets:
        unique = sorted(set(missing_datasets))
        print(f"[RouterSelect] Missing condensation datasets: {', '.join(unique)}")
    return app_graphs


def build_router_feature_bundle(cfg, graph_data: RouterGraphData) -> RouterFeatureBundle:
    mode = _normalize_context_mode(getattr(cfg.router_selection, "context_mode", "text"))
    text_encoder = _build_text_encoder(cfg)

    application_text_features: Optional[torch.Tensor] = None
    if mode in ("text", "concat"):
        application_text_features = _build_application_text_features(cfg, graph_data, text_encoder)

    expert_tensor = _build_expert_text_features(cfg, graph_data, text_encoder)
    subgraph_features = build_subgraph_svd_features(cfg, graph_data)

    application_graphs: Optional[List[List[Data]]] = None
    application_graph_encoder: Optional[GraphContextEncoder] = None

    if mode in ("condensation", "concat"):
        application_graph_encoder = _build_condensation_encoder(cfg)
        application_graphs = _build_application_graphs(cfg, graph_data)
        has_graphs = any(bool(graphs) for graphs in application_graphs) if application_graphs else False
        if not has_graphs:
            print("[RouterSelect] Warning: condensation mode requested but no graphs were loaded; falling back to text.")
            mode = "text"
            application_graphs = None
            application_graph_encoder = None

    if mode == "text":
        if application_text_features is None:
            application_text_features = _build_application_text_features(cfg, graph_data, text_encoder)
        if application_text_features is None:
            application_features = torch.empty((0, text_encoder.embed_dim))
        else:
            application_features = application_text_features
        application_text_features = None
    else:
        graph_dim = int(application_graph_encoder.embed_dim) if application_graph_encoder is not None else 0
        text_dim = int(application_text_features.size(1)) if application_text_features is not None else 0
        if mode == "concat" and text_dim > 0:
            app_dim = graph_dim + text_dim
        else:
            app_dim = graph_dim
        application_features = torch.zeros((graph_data.num_applications, app_dim), dtype=torch.float32)

    return RouterFeatureBundle(
        node_features={
            "application": application_features,
            "target_subgraph": subgraph_features,
            "expert": expert_tensor,
        },
        application_graphs=application_graphs,
        application_graph_encoder=application_graph_encoder,
        application_text_features=application_text_features,
        context_mode=mode,
    )


def build_router_node_features(cfg, graph_data: RouterGraphData) -> Dict[str, torch.Tensor]:
    bundle = build_router_feature_bundle(cfg, graph_data)
    return bundle.node_features


def router_node_feature_dims(node_features: Dict[str, torch.Tensor]) -> Dict[str, int]:
    dims: Dict[str, int] = {}
    for key, value in node_features.items():
        if value is None or not hasattr(value, "dim"):
            dims[key] = 0
        elif value.dim() == 2:
            dims[key] = int(value.size(1))
        elif value.dim() == 1:
            dims[key] = 1
        else:
            dims[key] = 0
    return dims
