import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from code.data_loader import (
    SingleGraphDataLoader,
    create_dataset,
    dataset_info,
    log_split_instance_counts,
    make_loaders,
)
from code.model import build_encoder_from_cfg
from code.pretrain.checkpoint import cfg_to_dict
from code.pretrain.methods.utils import get_batch_vector, pool_nodes
from code.router_dataset_generation.utils import serialize_graph
from code.router_selection import load_router_graph
from code.router_selection.utils import expand_test_subgraphs, normalize_dataset_list
from code.router_graph_construction import builder as graph_builder
from code.utils import ensure_dir, format_split_for_name, set_seed

from .clustering import apply_cluster_mapping, kmeans, majority_vote_mapping
from .cluster_eval import unsup_eval
from .heads import build_head
from .utils import RoutedQuery, load_router_output, load_weak_labels, save_json
from .train import predict_on_embeddings, train_head_on_embeddings


def _shared_split_root(cfg) -> str:
    ds_cfg = getattr(getattr(cfg, "data_preparation", None), "dataset", None)
    return getattr(ds_cfg, "split_root", "data/splits")


@dataclass
class QueryInstance:
    query_id: int
    label: Optional[float]
    data: Optional[Data]
    node_index: Optional[int]


@dataclass
class ExpertInfo:
    expert_id: int
    run_name: Optional[str]
    checkpoint: Optional[str]


class RouterPredictionRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
        set_seed(cfg.seed)

        rp = cfg.router_prediction
        self.mode = rp.mode
        self.output_dir = Path(rp.output_dir)
        ensure_dir(str(self.output_dir))

        self.run_name = rp.run_name or self._build_run_name()
        self.run_dir = self.output_dir / self.run_name
        ensure_dir(str(self.run_dir))

        self.pretrained_dir = rp.pretrained_dir or getattr(getattr(cfg, "pretrain", None), "checkpoint_dir", "pretrained_models")
        self.task_type = self._resolve_task_type()
        self.topk = self._resolve_topk()
        self.k_expert_feat = self._resolve_k_expert_feat()
        self.keep_query_features = self._resolve_keep_query_features()
        if self.k_expert_feat is not None:
            print(f"[RouterPredict] Using top-{self.k_expert_feat} expert features per query.")
        if self.keep_query_features:
            print("[RouterPredict] Appending raw query features to expert embeddings.")

        if self.mode not in ("cluster", "head"):
            raise ValueError(f"Unsupported router_prediction.mode: {self.mode}")
        if self.task_type not in ("classification", "regression"):
            raise ValueError(f"Unsupported router_prediction.task_type: {self.task_type}")

    def _build_run_name(self) -> str:
        base = Path(self.cfg.router_prediction.router_output_path or "router_output").stem
        return f"router_pred_{self.mode}_{base}_seed{self.cfg.seed}"

    def _ensure_router_output_path(self) -> None:
        path = getattr(self.cfg.router_prediction, "router_output_path", "") or ""
        if path and Path(path).is_file():
            return
        resolved = self._resolve_router_output_path()
        if not resolved:
            raise FileNotFoundError(
                "router_prediction.router_output_path not found; set it explicitly or ensure "
                "router_selection outputs exist for the current config."
            )
        self.cfg.router_prediction.router_output_path = resolved

    def _resolve_router_output_path(self) -> Optional[str]:
        rs_cfg = getattr(self.cfg, "router_selection", None)
        if rs_cfg is None:
            return None
        if not getattr(rs_cfg, "test_dataset", None):
            rp_ds_cfg = getattr(self.cfg.router_prediction, "test_dataset", None)
            if rp_ds_cfg is not None and getattr(rp_ds_cfg, "name", None):
                rs_cfg.test_dataset = rp_ds_cfg.name
        run_name = getattr(rs_cfg, "run_name", "") or self._build_router_selection_run_name(rs_cfg)
        output_dir = getattr(rs_cfg, "output_dir", "router_selection/outputs")
        candidate = Path(output_dir) / f"{run_name}_prediction.json"
        if candidate.is_file():
            return str(candidate)
        return None

    def _build_router_selection_run_name(self, rs_cfg) -> str:
        train_tag = str(getattr(rs_cfg, "train_datasets", "") or "").replace(" ", "").replace(",", "-")
        test_tag = str(getattr(rs_cfg, "test_dataset", "") or "").replace(" ", "")
        valid_edge_ratio = getattr(rs_cfg, "valid_edge_ratio", 0.0)
        split_tag = format_split_for_name(getattr(rs_cfg, "graph_split", None))
        parts = [
            "router_select",
            getattr(rs_cfg, "loss_type", ""),
            split_tag,
            f"train{train_tag}" if train_tag else "",
            f"test{test_tag}" if test_tag else "",
            f"valedges{valid_edge_ratio}",
            f"topk{getattr(rs_cfg, 'topk', '')}",
            f"dim{getattr(rs_cfg, 'embed_dim', '')}",
            f"seed{self.cfg.seed}",
        ]
        if getattr(rs_cfg, "scorer", "") == "mlp":
            parts.append(f"mlp{getattr(rs_cfg, 'mlp_hidden_dim', '')}")
        temperature = getattr(rs_cfg, "temperature", 1.0)
        if temperature and temperature != 1.0:
            parts.append(f"temp{temperature}")
        return "_".join(str(p) for p in parts if p)

    def _resolve_graph_dir(self) -> str:
        graph_dir = self.cfg.router_prediction.graph_dir
        base_path = Path(graph_dir)
        if base_path.is_dir() and (base_path / "router_graph.pt").is_file():
            return str(base_path)
        split = None
        rs_cfg = getattr(self.cfg, "router_selection", None)
        if rs_cfg is not None:
            split = getattr(rs_cfg, "graph_split", None)
        if not split and getattr(self.cfg, "router_dataset", None) is not None:
            split = getattr(self.cfg.router_dataset.target_dataset, "fixed_split", None)
        split_tag = format_split_for_name(split)
        if split_tag:
            return str(Path(graph_dir) / split_tag)
        return graph_dir

    def _resolve_topk(self) -> int:
        rs_cfg = getattr(self.cfg, "router_selection", None)
        if rs_cfg is not None and getattr(rs_cfg, "topk", None) is not None:
            return int(rs_cfg.topk)
        rp_cfg = getattr(self.cfg, "router_prediction", None)
        if rp_cfg is not None and getattr(rp_cfg, "topk", None) is not None:
            return int(rp_cfg.topk)
        return 5

    def _resolve_k_expert_feat(self) -> Optional[int]:
        rp_ds_cfg = getattr(self.cfg.router_prediction, "test_dataset", None)
        if rp_ds_cfg is None:
            return None
        value = getattr(rp_ds_cfg, "k_expert_feat", None)
        if value is None:
            return None
        try:
            value = int(value)
        except (TypeError, ValueError):
            return None
        if value < 0:
            return None
        return value

    def _resolve_keep_query_features(self) -> bool:
        rp_ds_cfg = getattr(self.cfg.router_prediction, "test_dataset", None)
        if rp_ds_cfg is not None:
            value = getattr(rp_ds_cfg, "keep_query_features", None)
            if value is not None:
                return bool(value)
        rp_cfg = getattr(self.cfg, "router_prediction", None)
        value = getattr(rp_cfg, "keep_query_features", None)
        return bool(value) if value is not None else False

    def _resolve_task_type(self) -> str:
        rp_ds_cfg = getattr(self.cfg.router_prediction, "test_dataset", None)
        if rp_ds_cfg is not None:
            value = getattr(rp_ds_cfg, "task_type", None)
            if value:
                return value
        rp_cfg = getattr(self.cfg, "router_prediction", None)
        if rp_cfg is not None:
            value = getattr(rp_cfg, "task_type", None)
            if value:
                return value
        return "classification"

    def run(self) -> None:
        self._ensure_router_output_path()
        routed_queries, router_meta = load_router_output(self.cfg.router_prediction.router_output_path)
        weak_labels = load_weak_labels(self.cfg.router_prediction.weak_labels_path)

        graph_data = load_router_graph(self._resolve_graph_dir())
        test_dataset = None
        rp_ds_cfg = getattr(self.cfg.router_prediction, "test_dataset", None)
        if rp_ds_cfg is not None:
            test_dataset = getattr(rp_ds_cfg, "name", None)
        if not test_dataset:
            rs_cfg = getattr(self.cfg, "router_selection", None)
            if rs_cfg is not None:
                test_dataset = getattr(rs_cfg, "test_dataset", None)
        if not test_dataset:
            test_dataset = getattr(self.cfg.train.dataset, "name", None)
        expand_list = []
        meta_tests = None
        if isinstance(router_meta, dict):
            meta_tests = (router_meta.get("datasets") or {}).get("test")
        if meta_tests:
            expand_list = normalize_dataset_list(meta_tests)
        elif test_dataset:
            expand_list = normalize_dataset_list(test_dataset)
        for ds_name in expand_list:
            expand_test_subgraphs(self.cfg, graph_data, ds_name)
        key_to_subgraph_id = {
            (meta or {}).get("key"): int(sub_id)
            for sub_id, meta in ((sid, graph_data.subgraph_id_map.get(str(sid))) for sid in range(graph_data.num_subgraphs))
            if meta and (meta or {}).get("key")
        }

        query_routes = self._build_route_index(routed_queries)
        print(
            "[RouterPredict] Routed queries: "
            f"total={len(query_routes)} experts={len(self._collect_expert_ids(query_routes))}"
        )

        use_all_nodes = self.mode == "cluster"
        dataset, dataset_meta, loaders, task_level, induced = self._load_dataset_and_loaders(use_all_nodes=use_all_nodes)
        train_loader, val_loader, test_loader = loaders
        if use_all_nodes:
            print("[RouterPredict][Split] cluster mode uses all nodes as test; no split loaded.")
        else:
            log_split_instance_counts(
                train_loader,
                val_loader,
                test_loader,
                task_level=task_level,
                split=self._resolve_split(),
                induced=induced,
                prefix="[RouterPredict][Split]",
            )

        print(
            "[RouterPredict][Data] "
            f"name={dataset_meta.get('name')} "
            f"task_level={task_level} "
            f"induced={induced} "
            f"num_nodes={dataset_meta.get('num_nodes')} "
            f"num_edges={dataset_meta.get('num_edges')} "
            f"num_classes={dataset_meta.get('num_classes')}"
        )

        query_instances = self._collect_query_instances(
            dataset=dataset,
            loaders=loaders,
            task_level=task_level,
            induced=induced,
            dataset_name=dataset_meta.get("name") or getattr(self.cfg.train.dataset, "name", None),
            key_to_subgraph_id=key_to_subgraph_id,
            query_routes=query_routes,
            weak_labels=weak_labels,
            use_all_nodes=use_all_nodes,
        )

        if getattr(self.cfg.router_prediction, "num_classes", None) is None:
            self.cfg.router_prediction.num_classes = dataset_meta.get("num_classes")
        if self.mode == "cluster":
            if not getattr(self.cfg.router_prediction, "num_clusters", None):
                self.cfg.router_prediction.num_clusters = self.cfg.router_prediction.num_classes

        payload = {
            "run_name": self.run_name,
            "mode": self.mode,
            "task_type": self.task_type,
            "router_output": self.cfg.router_prediction.router_output_path,
            "graph_dir": self.cfg.router_prediction.graph_dir,
            "output_dir": str(self.run_dir),
            "seed": self.cfg.seed,
            "config": cfg_to_dict(self.cfg),
            "experts": {},
        }

        features_by_split = self._compute_features(query_instances, query_routes)

        if self.mode == "cluster":
            predictions, metrics = self._run_cluster(features_by_split, weak_labels)
        else:
            predictions, metrics = self._run_head(features_by_split)

        if metrics:
            payload["metrics"] = metrics

        save_json(str(self.run_dir / f"{self.run_name}_metrics.json"), payload)
        save_json(str(self.run_dir / f"{self.run_name}_predictions.json"), predictions)

    def _resolve_split(self) -> Tuple[float, float, float]:
        ds_cfg = getattr(self.cfg.router_prediction, "test_dataset", None)
        split = getattr(ds_cfg, "fixed_split", None) if ds_cfg is not None else None
        if isinstance(split, str):
            try:
                split = tuple(float(x) for x in split.strip("()").split(","))
            except Exception:
                split = None
        if split is None:
            ds_cfg = self.cfg.train.dataset
            split = getattr(ds_cfg, "fixed_split", None)
        if split is None:
            split = getattr(self.cfg.train, "fixed_split", None)
        return tuple(split) if split is not None else (0.8, 0.1, 0.1)

    def _load_dataset_and_loaders(self, use_all_nodes: bool = False):
        ds_cfg = getattr(self.cfg.router_prediction, "test_dataset", None) or self.cfg.train.dataset
        if getattr(ds_cfg, "name", None) in (None, ""):
            rs_cfg = getattr(self.cfg, "router_selection", None)
            if rs_cfg is not None and getattr(rs_cfg, "test_dataset", None):
                ds_cfg = ds_cfg.clone()
                ds_cfg.name = rs_cfg.test_dataset
        raw_task_level = ds_cfg.task_level
        induced = getattr(ds_cfg, "induced", False)
        dataset = create_dataset(
            name=ds_cfg.name,
            root=ds_cfg.root,
            task_level=raw_task_level,
            feat_reduction=ds_cfg.feat_reduction,
            feat_reduction_dim=getattr(ds_cfg, "feat_reduction_dim", 100),
            induced=induced,
            induced_min_size=getattr(ds_cfg, "induced_min_size", 10),
            induced_max_size=getattr(ds_cfg, "induced_max_size", 30),
            induced_max_hops=getattr(ds_cfg, "induced_max_hops", 5),
            split_root=_shared_split_root(self.cfg),
            induced_root=getattr(ds_cfg, "induced_root", ""),
        )
        effective_task_level = "graph" if induced else raw_task_level
        ds_cfg.task_level = effective_task_level
        dataset_meta = dataset_info(dataset=dataset, task_level=raw_task_level, name=ds_cfg.name, induced=induced)
        if getattr(ds_cfg, "num_classes", None) is None:
            ds_cfg.num_classes = dataset_meta.get("num_classes")

        if use_all_nodes:
            if effective_task_level == "graph" or induced:
                full_loader = DataLoader(dataset, batch_size=self.cfg.router_prediction.batch_size)
            else:
                full_loader = SingleGraphDataLoader(dataset)
            loaders = (None, None, full_loader)
        else:
            split = self._resolve_split()
            loaders = make_loaders(
                dataset=dataset,
                dataset_name=ds_cfg.name,
                task_level=ds_cfg.task_level,
                batch_size=self.cfg.router_prediction.batch_size,
                num_workers=getattr(self.cfg.train, "num_workers", 0),
                split=split,
                seed=self.cfg.seed,
                induced=induced,
                split_root=_shared_split_root(self.cfg),
            )
        return dataset, dataset_meta, loaders, effective_task_level, induced

    def _build_route_index(self, routed_queries: Iterable[RoutedQuery]) -> Dict[int, RoutedQuery]:
        routes = {}
        for routed in routed_queries:
            experts = routed.experts[: self.topk]
            routes[routed.query_id] = RoutedQuery(
                query_id=routed.query_id,
                dataset=routed.dataset,
                experts=experts,
            )
        return routes

    def _collect_expert_ids(self, routes: Dict[int, RoutedQuery]) -> List[int]:
        expert_ids = set()
        for routed in routes.values():
            for expert in routed.experts:
                expert_ids.add(expert.expert_id)
        return sorted(expert_ids)

    def _collect_query_instances(
        self,
        dataset,
        loaders,
        task_level: str,
        induced: bool,
        dataset_name: str,
        key_to_subgraph_id: Dict[str, int],
        query_routes: Dict[int, RoutedQuery],
        weak_labels: Dict[int, float],
        use_all_nodes: bool = False,
    ) -> Dict[str, List[QueryInstance]]:
        train_loader, val_loader, test_loader = loaders
        if use_all_nodes:
            split_loaders = {"test": test_loader}
        else:
            split_loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
        instances: Dict[str, List[QueryInstance]] = {"train": [], "val": [], "test": []}
        missing_routes = 0
        missing_ids = 0
        matched = 0

        if task_level == "node" and not induced:
            base_data = dataset[0]
            for split, loader in split_loaders.items():
                if loader is None:
                    continue
                data = next(iter(loader))
                if use_all_nodes:
                    indices = list(range(int(data.num_nodes)))
                else:
                    mask = getattr(data, f"{split}_mask", None)
                    if mask is None:
                        continue
                    indices = torch.nonzero(mask, as_tuple=False).view(-1).tolist()
                for idx in indices:
                    target = {"node_id": int(idx)}
                    key, _, _ = graph_builder._build_subgraph_key(target, dataset=dataset_name)
                    query_id = key_to_subgraph_id.get(key)
                    if query_id is None:
                        missing_ids += 1
                        continue
                    if query_id not in query_routes:
                        missing_routes += 1
                        continue
                    label = float(data.y[idx].item()) if getattr(data, "y", None) is not None else None
                    if query_id in weak_labels:
                        label = weak_labels[query_id]
                    instances[split].append(
                        QueryInstance(query_id=query_id, label=label, data=base_data, node_index=int(idx))
                    )
                    matched += 1
            return instances

        for split, loader in split_loaders.items():
            if loader is None:
                continue
            for batch in loader:
                data_list = batch.to_data_list()
                for data_item in data_list:
                    if induced and hasattr(data_item, "base_node_id"):
                        target = {"node_id": int(getattr(data_item, "base_node_id"))}
                    else:
                        target = serialize_graph(data_item)
                    key, _, _ = graph_builder._build_subgraph_key(target, dataset=dataset_name)
                    query_id = key_to_subgraph_id.get(key)
                    if query_id is None:
                        missing_ids += 1
                        continue
                    if query_id not in query_routes:
                        missing_routes += 1
                        continue
                    label = None
                    if getattr(data_item, "y", None) is not None:
                        label = float(data_item.y.view(-1)[0].item())
                    if query_id in weak_labels:
                        label = weak_labels[query_id]
                    data_item.query_id = query_id
                    instances[split].append(QueryInstance(query_id=query_id, label=label, data=data_item, node_index=None))
                    matched += 1
        if missing_ids:
            print(
                "[RouterPredict] Skipped "
                f"{missing_ids} queries not present in router graph ids."
            )
        if missing_routes:
            print(f"[RouterPredict] Skipped {missing_routes} queries missing router selection routes.")
        print(f"[RouterPredict] Matched {matched} routed queries to dataset instances.")
        return instances

    def _resolve_expert_checkpoint(self, expert: ExpertInfo) -> Optional[str]:
        if expert.checkpoint and Path(expert.checkpoint).is_file():
            return expert.checkpoint
        if expert.run_name:
            root = Path(self.pretrained_dir)
            if not root.is_dir():
                return None

            for dataset_dir in sorted(root.iterdir()):
                if not dataset_dir.is_dir() or dataset_dir.name == expert.run_name:
                    continue
                candidate = dataset_dir / f"{expert.run_name}.pt"
                if candidate.is_file():
                    return str(candidate)
        return None

    def _load_expert_model(self, checkpoint_path: str, in_dim: int) -> torch.nn.Module:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        ckpt_cfg = checkpoint.get("cfg", {}) or {}
        cfg = self.cfg.clone()
        model_cfg = ckpt_cfg.get("model", {}) or {}
        if model_cfg.get("name"):
            cfg.model.name = model_cfg["name"]
        for attr in ("hidden_dim", "out_dim", "num_layers", "activation", "dropout", "graph_pooling"):
            if attr in model_cfg:
                setattr(cfg.model, attr, model_cfg[attr])
        gat_cfg = model_cfg.get("gat", {}) or {}
        if isinstance(gat_cfg, dict) and "heads" in gat_cfg:
            cfg.model.gat.heads = gat_cfg["heads"]
        cfg.model.in_dim = cfg.model.in_dim or in_dim
        model = build_encoder_from_cfg(cfg=cfg, in_dim=cfg.model.in_dim).to(self.device)
        missing, unexpected = model.load_state_dict(checkpoint.get("model_state", {}), strict=False)
        if missing:
            print(f"[RouterPredict] Missing keys when loading expert checkpoint: {missing}")
        if unexpected:
            print(f"[RouterPredict] Unexpected keys when loading expert checkpoint: {unexpected}")
        model.eval()
        return model

    def _compute_features(
        self,
        query_instances: Dict[str, List[QueryInstance]],
        query_routes: Dict[int, RoutedQuery],
    ) -> Dict[str, Dict[str, object]]:
        features_dir = self.run_dir / "features"
        ensure_dir(str(features_dir))
        features_by_split: Dict[str, Dict[str, object]] = {}
        agg = self.cfg.router_prediction.embed_agg
        kfeat_tag = f"kfeat{self.k_expert_feat}" if self.k_expert_feat is not None else "kfeatall"
        qfeat_tag = "qfeat1" if self.keep_query_features else "qfeat0"
        split_tag = format_split_for_name(self._resolve_split()) if self.mode != "cluster" else "all"
        dataset_name = None
        rp_ds_cfg = getattr(self.cfg.router_prediction, "test_dataset", None)
        if rp_ds_cfg is not None:
            dataset_name = getattr(rp_ds_cfg, "name", None)
        if not dataset_name:
            dataset_name = getattr(self.cfg.train.dataset, "name", None)
        dataset_tag = str(dataset_name or "dataset")

        for split, instances in query_instances.items():
            cache_path = features_dir / f"{split}_k{self.topk}_{kfeat_tag}_{qfeat_tag}_{agg}_{dataset_tag}_{split_tag}.pt"
            if self.cfg.router_prediction.cache_features and cache_path.is_file():
                cached = torch.load(cache_path, map_location="cpu")
                if (
                    cached.get("topk") == self.topk
                    and cached.get("k_expert_feat") == self.k_expert_feat
                    and cached.get("keep_query_features") == self.keep_query_features
                    and cached.get("embed_agg") == agg
                    and cached.get("router_output") == self.cfg.router_prediction.router_output_path
                    and cached.get("dataset") == dataset_tag
                    and cached.get("split") == split_tag
                ):
                    features_by_split[split] = cached
                    continue

            feature_map, labels, expert_lists = self._build_features_for_split(instances, query_routes)
            payload = {
                "features": feature_map,
                "labels": labels,
                "experts": expert_lists,
                "topk": self.topk,
                "k_expert_feat": self.k_expert_feat,
                "keep_query_features": self.keep_query_features,
                "embed_agg": agg,
                "router_output": self.cfg.router_prediction.router_output_path,
                "dataset": dataset_tag,
                "split": split_tag,
            }
            if self.cfg.router_prediction.cache_features:
                torch.save(payload, cache_path)
                print(f"[RouterPredict] Saved features: {cache_path}")
            features_by_split[split] = payload

        return features_by_split

    def _build_features_for_split(
        self,
        instances: List[QueryInstance],
        query_routes: Dict[int, RoutedQuery],
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, Optional[float]], Dict[int, List[int]]]:
        if not instances:
            return {}, {}, {}
        feature_map: Dict[int, torch.Tensor] = {}
        labels: Dict[int, Optional[float]] = {}
        expert_lists: Dict[int, List[int]] = {}
        use_expert_features = self.k_expert_feat is None or self.k_expert_feat > 0

        expert_to_queries: Dict[int, List[QueryInstance]] = {}
        for inst in instances:
            route = query_routes.get(inst.query_id)
            if route is None:
                continue
            experts = route.experts
            if self.k_expert_feat is not None:
                if self.k_expert_feat == 0:
                    experts = []
                else:
                    experts = experts[: self.k_expert_feat]
            expert_ids = [exp.expert_id for exp in experts]
            expert_lists[inst.query_id] = expert_ids
            labels[inst.query_id] = inst.label
            for expert_id in expert_ids:
                expert_to_queries.setdefault(expert_id, []).append(inst)

        expert_embeddings: Dict[Tuple[int, int], torch.Tensor] = {}
        if use_expert_features:
            for expert_id, inst_list in expert_to_queries.items():
                run_name = None
                checkpoint = None
                for route in query_routes.values():
                    for exp in route.experts:
                        if exp.expert_id == expert_id:
                            run_name = exp.expert_model.get("model")
                            checkpoint = exp.expert_model.get("checkpoint")
                            break
                    if run_name or checkpoint:
                        break
                expert_info = ExpertInfo(expert_id=expert_id, run_name=run_name, checkpoint=checkpoint)
                ckpt_path = self._resolve_expert_checkpoint(expert_info)
                if not ckpt_path:
                    print(f"[RouterPredict] Missing checkpoint for expert {expert_id} ({run_name}); using zeros.")
                    continue

                sample = inst_list[0]
                in_dim = self._resolve_in_dim(sample)
                if in_dim is None:
                    print(f"[RouterPredict] Missing features for expert {expert_id}; using zeros.")
                    continue
                model = self._load_expert_model(ckpt_path, in_dim)

                if sample.node_index is not None:
                    emb_map = self._embed_node_queries(model, inst_list)
                else:
                    emb_map = self._embed_graph_queries(model, inst_list)

                for query_id, emb in emb_map.items():
                    expert_embeddings[(query_id, expert_id)] = emb
                if torch.cuda.is_available() and self.device.type == "cuda":
                    # Release per-expert allocations before loading the next model.
                    del model
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

        out_dim = None
        for emb in expert_embeddings.values():
            out_dim = emb.numel()
            break
        if out_dim is None:
            out_dim = int(getattr(self.cfg.model, "out_dim", 128))

        agg = self.cfg.router_prediction.embed_agg
        query_feat_dim = self._infer_query_feat_dim(instances) if self.keep_query_features else None
        for inst in instances:
            route = query_routes.get(inst.query_id)
            if route is None:
                continue
            experts = route.experts
            if self.k_expert_feat is not None:
                if self.k_expert_feat == 0:
                    experts = []
                else:
                    experts = experts[: self.k_expert_feat]
            emb_list = []
            valid = 0
            for exp in experts:
                emb = expert_embeddings.get((inst.query_id, exp.expert_id))
                if emb is None:
                    emb = torch.zeros(out_dim)
                else:
                    valid += 1
                emb_list.append(emb)
            if agg == "concat":
                feature = torch.cat(emb_list, dim=0) if emb_list else torch.empty(0)
            elif agg == "mean":
                if emb_list and valid > 0:
                    stacked = torch.stack(emb_list, dim=0)
                    feature = stacked.sum(dim=0) / float(valid)
                elif emb_list:
                    feature = torch.zeros(out_dim)
                else:
                    feature = torch.empty(0)
            else:
                raise ValueError(f"Unknown embed_agg: {agg}")
            if self.keep_query_features:
                query_feat = self._get_query_features(inst, query_feat_dim or 0)
                if query_feat is not None and query_feat.numel() > 0:
                    feature = torch.cat([feature, query_feat], dim=0)
            feature_map[inst.query_id] = feature

        return feature_map, labels, expert_lists

    def _infer_query_feat_dim(self, instances: List[QueryInstance]) -> int:
        for inst in instances:
            data = inst.data
            if data is None:
                continue
            x = getattr(data, "x", None)
            if x is None:
                continue
            if x.dim() == 1:
                return int(x.numel())
            if x.dim() >= 2:
                return int(x.size(-1))
        return int(getattr(self.cfg.model, "in_dim", 0) or 0)

    def _get_query_features(self, inst: QueryInstance, fallback_dim: int) -> Optional[torch.Tensor]:
        data = inst.data
        if data is None:
            return torch.zeros(fallback_dim)
        x = getattr(data, "x", None)
        if x is None:
            return torch.zeros(fallback_dim)
        if inst.node_index is not None:
            if x.dim() == 2 and x.size(0) > inst.node_index:
                return x[inst.node_index].detach().cpu()
            if x.dim() == 1 and inst.node_index == 0:
                return x.detach().cpu()
            feat_dim = int(x.size(-1)) if x.dim() >= 1 else fallback_dim
            return torch.zeros(feat_dim)
        if x.dim() == 1:
            return x.detach().cpu()
        if x.numel() == 0:
            feat_dim = int(x.size(-1)) if x.dim() == 2 else fallback_dim
            return torch.zeros(feat_dim)
        return x.mean(dim=0).detach().cpu()

    def _resolve_in_dim(self, instance: QueryInstance) -> Optional[int]:
        data = instance.data
        if data is None:
            return None
        if getattr(data, "x", None) is None:
            return None
        return int(data.x.size(-1))

    def _embed_node_queries(self, model: torch.nn.Module, instances: List[QueryInstance]) -> Dict[int, torch.Tensor]:
        base_data = instances[0].data
        if base_data is None:
            return {}
        data = base_data.to(self.device)
        with torch.inference_mode():
            node_repr, _ = model(data)
        emb_map: Dict[int, torch.Tensor] = {}
        for inst in instances:
            if inst.node_index is None:
                continue
            emb_map[inst.query_id] = node_repr[inst.node_index].detach().cpu()
        return emb_map

    def _embed_graph_queries(self, model: torch.nn.Module, instances: List[QueryInstance]) -> Dict[int, torch.Tensor]:
        data_list = []
        for inst in instances:
            if inst.data is None:
                continue
            inst.data.subgraph_id = inst.query_id
            data_list.append(inst.data)
        loader = DataLoader(data_list, batch_size=self.cfg.router_prediction.batch_size)
        emb_map: Dict[int, torch.Tensor] = {}
        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(self.device)
                node_repr, graph_repr = model(batch)
                if graph_repr is None:
                    graph_repr = pool_nodes(node_repr, get_batch_vector(batch), mode=self.cfg.model.graph_pooling)
                data_items = batch.to_data_list()
                for idx, item in enumerate(data_items):
                    emb_map[int(getattr(item, "subgraph_id"))] = graph_repr[idx].detach().cpu()
        return emb_map

    def _run_cluster(
        self,
        features_by_split: Dict[str, Dict[str, object]],
        weak_labels: Dict[int, float],
    ) -> Tuple[Dict[str, List[Dict[str, object]]], Dict[str, float]]:
        train = features_by_split.get("train", {})
        train_features = train.get("features", {})
        train_labels = train.get("labels", {})

        all_features = {}
        for split, payload in features_by_split.items():
            all_features.update(payload.get("features", {}))

        if not all_features:
            return {"train": [], "val": [], "test": []}, {}

        feature_tensor, query_ids = self._stack_features(all_features)
        num_clusters = int(self.cfg.router_prediction.num_clusters or 0)
        if num_clusters <= 0:
            num_clusters = int(self.cfg.router_prediction.num_classes or 0)
        if num_clusters <= 0:
            raise ValueError("num_clusters or num_classes must be provided for cluster mode.")
        if self.cfg.router_prediction.cluster_method != "kmeans":
            raise ValueError("Only kmeans is supported for cluster mode.")

        assignments, centers = kmeans(feature_tensor, num_clusters=num_clusters, seed=self.cfg.seed)

        calibration_labels = [None for _ in query_ids]
        if self.cfg.router_prediction.calibration_ratio > 0 and train_features:
            candidates = [idx for idx, qid in enumerate(query_ids) if qid in train_features and train_labels.get(qid) is not None]
            if candidates:
                sample_size = max(1, int(len(candidates) * self.cfg.router_prediction.calibration_ratio))
                rng = torch.Generator().manual_seed(self.cfg.seed)
                perm = torch.randperm(len(candidates), generator=rng).tolist()
                for pick_idx in perm[:sample_size]:
                    qid = query_ids[candidates[pick_idx]]
                    calibration_labels[candidates[pick_idx]] = train_labels.get(qid)
        elif weak_labels:
            for idx, qid in enumerate(query_ids):
                if qid in weak_labels:
                    calibration_labels[idx] = weak_labels[qid]

        mapping = majority_vote_mapping(assignments, calibration_labels, num_clusters)
        mapped_preds = apply_cluster_mapping(assignments, mapping)

        predictions = {"train": [], "val": [], "test": []}
        metrics = {}
        split_lookup = {}
        for split, payload in features_by_split.items():
            for qid in payload.get("features", {}).keys():
                split_lookup[qid] = split

        for idx, qid in enumerate(query_ids):
            pred = mapped_preds[idx]
            entry = {
                "query_id": int(qid),
                "cluster": int(assignments[idx].item()),
            }
            if pred is not None:
                entry["pred"] = int(pred)
            label = None
            for payload in features_by_split.values():
                if qid in payload.get("labels", {}):
                    label = payload.get("labels", {}).get(qid)
                    break
            if label is not None:
                entry["label"] = float(label)
            split = split_lookup.get(qid)
            if split:
                predictions[split].append(entry)

        labeled_idx = []
        true_labels = []
        for idx, qid in enumerate(query_ids):
            label = None
            for payload in features_by_split.values():
                if qid in payload.get("labels", {}):
                    label = payload.get("labels", {}).get(qid)
                    break
            if label is None:
                continue
            labeled_idx.append(idx)
            true_labels.append(int(label))
        if labeled_idx:
            pred_labels = assignments[labeled_idx].cpu()
            acc, nmi, ari, f1 = unsup_eval(true_labels, pred_labels)
            metrics.update(
                {
                    "cluster_acc": acc,
                    "cluster_nmi": nmi,
                    "cluster_ari": ari,
                    "cluster_f1": f1,
                }
            )

        if centers is not None:
            dists = torch.cdist(feature_tensor, centers, p=2)
            min_dists = dists.gather(1, assignments.view(-1, 1)).squeeze(1)
            inertia = float((min_dists ** 2).sum().item())
        else:
            inertia = 0.0
        print(
            "[RouterPredict][Cluster] "
            f"inertia={inertia:.4f} "
            f"acc={metrics.get('cluster_acc', 0.0):.4f} "
            f"nmi={metrics.get('cluster_nmi', 0.0):.4f} "
            f"ari={metrics.get('cluster_ari', 0.0):.4f} "
            f"f1={metrics.get('cluster_f1', 0.0):.4f}"
        )

        return predictions, metrics

    def _run_head(self, features_by_split: Dict[str, Dict[str, object]]) -> Tuple[Dict[str, List[Dict[str, object]]], Dict[str, float]]:
        train = features_by_split.get("train", {})
        val = features_by_split.get("val", {})
        test = features_by_split.get("test", {})

        train_feats, train_ids = self._stack_features(train.get("features", {}))
        val_feats, val_ids = self._stack_features(val.get("features", {}))
        test_feats, test_ids = self._stack_features(test.get("features", {}))

        print(
            "[RouterPredict][Features] "
            f"train={train_feats.shape} val={val_feats.shape} test={test_feats.shape}"
        )

        train_labels = self._collect_labels(train.get("labels", {}), train_ids)
        val_labels = self._collect_labels(val.get("labels", {}), val_ids)
        test_labels = self._collect_labels(test.get("labels", {}), test_ids)

        head_cfg = self.cfg.router_prediction.head
        head_type = getattr(head_cfg, "type", "mlp")
        hidden_dim = getattr(head_cfg, "hidden_dim", 128)
        num_layers = getattr(head_cfg, "num_layers", 2)
        dropout = getattr(head_cfg, "dropout", 0.0)

        head = build_head(
            in_dim=train_feats.shape[1],
            task_type=self.task_type,
            num_classes=self.cfg.router_prediction.num_classes,
            head_type=head_type,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)
        print(f"[RouterPredict] Head architecture ({head_type}):\n{head}")

        metrics, state, history = train_head_on_embeddings(
            head=head,
            train_embeddings=train_feats,
            train_labels=train_labels,
            val_embeddings=val_feats,
            val_labels=val_labels,
            test_embeddings=test_feats,
            test_labels=test_labels,
            device=self.device,
            task_type=self.task_type,
            lr_head=self.cfg.router_prediction.lr,
            weight_decay=self.cfg.router_prediction.weight_decay,
            epochs=self.cfg.router_prediction.epochs,
            early_stop_patience=self.cfg.router_prediction.early_stop_patience,
            batch_size=self.cfg.router_prediction.batch_size,
        )

        predictions = {
            "train": predict_on_embeddings(head, train_feats, train_labels, train_ids, self.device, self.task_type, self.cfg.router_prediction.batch_size),
            "val": predict_on_embeddings(head, val_feats, val_labels, val_ids, self.device, self.task_type, self.cfg.router_prediction.batch_size),
            "test": predict_on_embeddings(head, test_feats, test_labels, test_ids, self.device, self.task_type, self.cfg.router_prediction.batch_size),
        }

        head_path = self.run_dir / "head.pt"
        torch.save(
            {
                "head_state": head.state_dict(),
                "metrics": metrics,
                "history": history,
            "cfg": cfg_to_dict(self.cfg),
            },
            head_path,
        )
        print(f"[RouterPredict] Saved head checkpoint: {head_path}")

        return predictions, metrics

    def _stack_features(self, feature_map: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, List[int]]:
        if not feature_map:
            return torch.empty((0, 0)), []
        ids = sorted(feature_map.keys())
        feats = torch.stack([feature_map[qid] for qid in ids], dim=0)
        return feats, ids

    def _collect_labels(self, label_map: Dict[int, Optional[float]], ids: List[int]) -> torch.Tensor:
        labels = []
        for qid in ids:
            value = label_map.get(qid)
            if value is None:
                labels.append(0.0)
            else:
                labels.append(float(value))
        return torch.tensor(labels)
