"""Training and evaluation loops for router selection with GNN routing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import hashlib
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from code.utils import ensure_dir, format_split_for_name
from .data import RouterGraphData
from .feature_loader import build_router_feature_bundle
from .router_gnn import RouterGNN, RouterScorer


class SubgraphIndexDataset(Dataset):
    def __init__(self, indices: Iterable[int]):
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> int:
        return int(self.indices[idx])


@dataclass
class TrainState:
    best_epoch: int = 0
    best_metric: float = float("-inf")
    best_checkpoint: Optional[str] = None


def _build_run_name(cfg) -> str:
    rp = cfg.router_selection
    train_tag = str(getattr(rp, "train_datasets", "") or "").replace(" ", "").replace(",", "-")
    test_tag = str(getattr(rp, "test_dataset", "") or "").replace(" ", "")
    valid_edge_ratio = getattr(rp, "valid_edge_ratio", 0.0)
    split_tag = format_split_for_name(getattr(rp, "graph_split", None))
    graph_topk = getattr(rp, "graph_topk_expert", None)
    graph_topk_neg = bool(getattr(rp, "graph_topk_neg_expert", False))
    context_mode = str(getattr(rp, "context_mode", "text") or "text").replace(" ", "")
    parts = [
        "router_select",
        "regression",
        split_tag,
        f"train{train_tag}" if train_tag else "",
        f"test{test_tag}" if test_tag else "",
        f"valedge{valid_edge_ratio}",
        f"topk{graph_topk}" if graph_topk is not None else "",
        f"neg{graph_topk}" if graph_topk_neg and graph_topk is not None else "",
        f"context{context_mode}",
        f"scorer{rp.scorer}",
        f"seed{cfg.seed}",
    ]
    name = "_".join(str(p) for p in parts if p)
    return _truncate_run_name(name)


def _truncate_run_name(name: str, max_len: int = 180) -> str:
    if len(name) <= max_len:
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
    keep = max(1, max_len - (len(digest) + 2))
    return f"{name[:keep]}_{digest}"


class RouterSelectionTrainer:
    """Train and evaluate expert ranking with GNN routing supervision."""

    def __init__(
        self,
        cfg,
        graph_data: RouterGraphData,
        splits: Dict[str, List[int]],
        edge_splits: Optional[Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]] = None,
        node_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        self.cfg = cfg
        self.graph_data = graph_data
        self.splits = splits
        self.edge_splits = edge_splits or {}
        self.device = self._resolve_device()

        self.application_graph_encoder = None
        self.application_graphs = None
        self.application_text_features = None
        self.application_context_mode = "text"
        if node_features is None:
            bundle = build_router_feature_bundle(cfg, graph_data)
            self.node_features = bundle.node_features
            self.application_graph_encoder = bundle.application_graph_encoder
            self.application_graphs = bundle.application_graphs
            self.application_text_features = bundle.application_text_features
            self.application_context_mode = bundle.context_mode
        else:
            self.node_features = node_features
        if self.application_graph_encoder is not None:
            self.application_graph_encoder.to(self.device)
        self.edge_index_dict, self.edge_attr_dict = self._build_edge_dicts()
        self._logged_graph_encoder_grad = False
        self._logged_context_stats = False

        node_input_dims = {k: int(v.shape[-1]) for k, v in self.node_features.items()}
        rp = cfg.router_selection
        self.gnn = RouterGNN(
            node_input_dims=node_input_dims,
            embed_dim=rp.embed_dim,
            num_layers=getattr(rp, "gnn_layers", 2),
            dropout=getattr(rp, "gnn_dropout", 0.1),
            edge_types=list(self.edge_index_dict.keys()),
            use_edge_attr=True,
        ).to(self.device)
        self.scorer = RouterScorer(
            embed_dim=rp.embed_dim,
            scorer=rp.scorer,
            mlp_hidden_dim=rp.mlp_hidden_dim,
            temperature=getattr(rp, "temperature", 1.0),
        ).to(self.device)

        params = list(self.gnn.parameters()) + list(self.scorer.parameters())
        if self.application_graph_encoder is not None:
            params += list(self.application_graph_encoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=rp.lr, weight_decay=rp.weight_decay)

        self.run_name = rp.run_name or _build_run_name(cfg)
        self.checkpoint_dir = Path(rp.checkpoint_dir)
        ensure_dir(str(self.checkpoint_dir))
        self.filtered_splits = {name: list(indices) for name, indices in splits.items()}
        self._log_model_info()

    def _log_model_info(self) -> None:
        rp = self.cfg.router_selection
        app_dim = int(self.node_features.get("application", torch.empty(0, 0)).shape[-1]) if self.node_features.get("application") is not None else 0
        sub_dim = int(self.node_features.get("target_subgraph", torch.empty(0, 0)).shape[-1]) if self.node_features.get("target_subgraph") is not None else 0
        exp_dim = int(self.node_features.get("expert", torch.empty(0, 0)).shape[-1]) if self.node_features.get("expert") is not None else 0
        print(
            "[RouterSelect] Node feature dims (init): "
            f"application={app_dim} target_subgraph={sub_dim} expert={exp_dim}"
        )
        print(f"[RouterSelect] GNN:\n{self.gnn}")
        print(f"[RouterSelect] Scorer:\n{self.scorer}")
        if self.application_graph_encoder is not None:
            print(f"[RouterSelect] Context mode: {self.application_context_mode}")
            if self.application_graphs is not None:
                counts = [len(graphs) for graphs in self.application_graphs if graphs is not None]
                if counts:
                    total = sum(counts)
                    avg = total / max(1, len(counts))
                    print(
                        "[RouterSelect] Condensation graphs: "
                        f"apps={len(counts)} total_graphs={total} avg_graphs={avg:.2f}"
                    )

    def _resolve_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.cfg.device}")
        return torch.device("cpu")

    def _build_edge_dicts(self) -> Tuple[Dict[tuple, torch.Tensor], Dict[tuple, torch.Tensor]]:
        graph = self.graph_data.graph
        edge_index_dict: Dict[tuple, torch.Tensor] = {}
        edge_attr_dict: Dict[tuple, torch.Tensor] = {}
        edge_types = [
            ("application", "to", "target_subgraph"),
            ("target_subgraph", "rev_to", "application"),
            ("target_subgraph", "to", "expert"),
            ("expert", "rev_to", "target_subgraph"),
        ]
        for edge_type in edge_types:
            store = graph.get(edge_type) if isinstance(graph, dict) else graph[edge_type] if edge_type in graph.edge_types else None
            if store is None:
                continue
            edge_index = store.get("edge_index") if isinstance(store, dict) else store.edge_index
            edge_weight = store.get("edge_weight") if isinstance(store, dict) else getattr(store, "edge_weight", None)
            if edge_index is None:
                continue
            edge_index_dict[edge_type] = edge_index
            if edge_weight is not None:
                edge_attr_dict[edge_type] = edge_weight.view(-1, 1).to(torch.float32)
            else:
                edge_attr_dict[edge_type] = torch.ones((edge_index.shape[1], 1), dtype=torch.float32)
        return edge_index_dict, edge_attr_dict

    def _get_edges(
        self,
        sub_id: int,
        split_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        split_map = self.edge_splits.get(split_name)
        if split_map and sub_id in split_map:
            return split_map[sub_id]
        return (
            self.graph_data.subgraph_expert_ids[sub_id],
            self.graph_data.subgraph_expert_weights[sub_id],
        )

    def _filter_empty_subgraphs(self, indices: List[int], split_name: str) -> List[int]:
        filtered: List[int] = []
        empty = 0
        for sub_id in indices:
            expert_ids, _ = self._get_edges(sub_id, split_name)
            if expert_ids.numel() == 0:
                empty += 1
                continue
            filtered.append(sub_id)
        if empty > 0:
            print(f"[RouterSelect] Filtered {empty} empty subgraphs from {split_name} split.")
        return filtered

    def _build_loader(self, indices: List[int], shuffle: bool) -> DataLoader:
        dataset = SubgraphIndexDataset(indices)
        batch_size = self.cfg.router_selection.batch_size
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.router_selection.num_workers,
            drop_last=False,
        )

    def _assemble_node_features(self) -> Dict[str, torch.Tensor]:
        x_dict = {k: v.to(self.device) for k, v in self.node_features.items()}
        if self.application_graph_encoder is not None and self.application_graphs is not None:
            graph_emb = self.application_graph_encoder.encode_dataset_graphs(
                self.application_graphs,
                device=self.device,
            )
            if self.application_text_features is not None and self.application_context_mode == "concat":
                text_feat = self.application_text_features.to(self.device)
                app_feat = torch.cat([text_feat, graph_emb], dim=-1)
            else:
                app_feat = graph_emb
            x_dict["application"] = app_feat
        return x_dict

    def _compute_embeddings(self) -> Dict[str, torch.Tensor]:
        x_dict = self._assemble_node_features()
        edge_index_dict = {k: v.to(self.device) for k, v in self.edge_index_dict.items()}
        edge_attr_dict = {k: v.to(self.device) for k, v in self.edge_attr_dict.items()} if self.edge_attr_dict else None
        return self.gnn(x_dict, edge_index_dict, edge_attr_dict)

    def _log_tensor_stats(self, label: str, tensor: Optional[torch.Tensor]) -> None:
        if tensor is None:
            print(f"[RouterSelect] {label}: <none>")
            return
        if tensor.numel() == 0:
            print(f"[RouterSelect] {label}: empty shape={tuple(tensor.shape)}")
            return
        t = tensor.detach().float()
        stats = {
            "shape": tuple(t.shape),
            "mean": float(t.mean().item()),
            "std": float(t.std().item()),
            "min": float(t.min().item()),
            "max": float(t.max().item()),
            "l2_mean": float(t.norm(dim=-1).mean().item()) if t.dim() == 2 else float(t.norm().item()),
        }
        print(
            "[RouterSelect] "
            f"{label}: shape={stats['shape']} mean={stats['mean']:.4f} std={stats['std']:.4f} "
            f"min={stats['min']:.4f} max={stats['max']:.4f} l2_mean={stats['l2_mean']:.4f}"
        )

    def _log_context_stats(self) -> None:
        if self._logged_context_stats:
            return
        if self.application_graph_encoder is None or self.application_graphs is None:
            return
        with torch.no_grad():
            graph_emb = self.application_graph_encoder.encode_dataset_graphs(
                self.application_graphs,
                device=self.device,
            )
        self._log_tensor_stats("Condensation embeddings", graph_emb)
        if self.application_text_features is not None and self.application_context_mode == "concat":
            self._log_tensor_stats("Text embeddings (application)", self.application_text_features)
        self._logged_context_stats = True

    def _score_all(self, subgraph_ids: torch.Tensor, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        sub_emb = embeddings["target_subgraph"][subgraph_ids]
        exp_emb = embeddings["expert"]
        return self.scorer.score_all(sub_emb, exp_emb)

    def _score_all_regression(self, subgraph_ids: torch.Tensor, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        scores = self._score_all(subgraph_ids, embeddings)
        return torch.sigmoid(scores)

    def _score_pairs_regression(
        self,
        subgraph_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        embeddings: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        sub_emb = embeddings["target_subgraph"][subgraph_ids]
        exp_emb = embeddings["expert"][expert_ids]
        scores = self.scorer.score_pairs(sub_emb, exp_emb)
        return torch.sigmoid(scores)

    def _gather_regression_targets(
        self,
        subgraph_ids: torch.Tensor,
        split_name: str,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        all_sub: List[int] = []
        all_exp: List[int] = []
        all_targets: List[float] = []
        for sub_id in subgraph_ids.tolist():
            expert_ids, weights = self._get_edges(sub_id, split_name)
            if weights.numel() == 0:
                continue
            all_sub.extend([int(sub_id)] * int(expert_ids.numel()))
            all_exp.extend([int(eid) for eid in expert_ids.tolist()])
            all_targets.extend([float(w) for w in weights.tolist()])
        if not all_sub:
            return None
        sub_tensor = torch.tensor(all_sub, device=self.device, dtype=torch.long)
        exp_tensor = torch.tensor(all_exp, device=self.device, dtype=torch.long)
        target_tensor = torch.tensor(all_targets, device=self.device, dtype=torch.float)
        return sub_tensor, exp_tensor, target_tensor

    def _train_epoch(self, loader: DataLoader, split_name: str) -> float:
        self.gnn.train()
        self.scorer.train()
        if self.application_graph_encoder is not None:
            self.application_graph_encoder.train()
        total_loss = 0.0
        steps = 0
        for batch in loader:
            embeddings = self._compute_embeddings()
            subgraph_ids = batch.to(self.device)
            gathered = self._gather_regression_targets(subgraph_ids, split_name)
            if gathered is None:
                continue
            sub_tensor, exp_tensor, targets = gathered
            preds = self._score_pairs_regression(sub_tensor, exp_tensor, embeddings)
            loss = nn.functional.smooth_l1_loss(preds, targets)
            self.optimizer.zero_grad()
            loss.backward()
            if self.application_graph_encoder is not None and not self._logged_graph_encoder_grad:
                grad_norm = 0.0
                for param in self.application_graph_encoder.parameters():
                    if param.grad is None:
                        continue
                    grad_norm += float(param.grad.detach().norm().item())
                print(f"[RouterSelect] Graph encoder grad_norm={grad_norm:.6f}")
                self._logged_graph_encoder_grad = True
            self.optimizer.step()
            total_loss += loss
            steps += 1
        if steps == 0:
            return 0.0
        loss = total_loss / float(steps)
        return float(loss.item())

    def _evaluate_regression(self, indices: List[int], split_name: str) -> Dict[str, float]:
        if not indices:
            return {"loss_reg": 0.0, "mae": 0.0}
        loader = self._build_loader(indices, shuffle=False)
        self.gnn.eval()
        self.scorer.eval()
        if self.application_graph_encoder is not None:
            self.application_graph_encoder.eval()
        total_loss = 0.0
        total_abs = 0.0
        total_count = 0
        with torch.no_grad():
            embeddings = self._compute_embeddings()
            for batch in loader:
                subgraph_ids = batch.to(self.device)
                gathered = self._gather_regression_targets(subgraph_ids, split_name)
                if gathered is None:
                    continue
                sub_tensor, exp_tensor, targets = gathered
                preds = self._score_pairs_regression(sub_tensor, exp_tensor, embeddings)
                total_loss += float(nn.functional.smooth_l1_loss(preds, targets, reduction="sum").item())
                total_abs += float(torch.abs(preds - targets).sum().item())
                total_count += int(targets.numel())
        if total_count == 0:
            return {"loss_reg": 0.0, "mae": 0.0}
        return {
            "loss_reg": total_loss / float(total_count),
            "mae": total_abs / float(total_count),
        }

    def evaluate(self, indices: List[int], split_name: str) -> Dict[str, float]:
        return self._evaluate_regression(indices, split_name)

    def predict(self, indices: List[int]) -> List[Dict[str, object]]:
        if not indices:
            return []
        loader = self._build_loader(indices, shuffle=False)
        self.gnn.eval()
        self.scorer.eval()
        if self.application_graph_encoder is not None:
            self.application_graph_encoder.eval()
        predictions: List[Dict[str, object]] = []
        top_l = min(int(self.cfg.router_selection.topk), self.graph_data.num_experts)
        if top_l <= 0:
            return predictions
        with torch.no_grad():
            embeddings = self._compute_embeddings()
            for batch in loader:
                subgraph_ids = batch.to(self.device)
                scores = self._score_all_regression(subgraph_ids, embeddings)
                top_scores, top_ids = torch.topk(scores, k=top_l, dim=-1)
                for idx, sub_id in enumerate(subgraph_ids.tolist()):
                    experts = [
                        {"expert_id": int(expert_id), "score": float(score)}
                        for expert_id, score in zip(top_ids[idx].tolist(), top_scores[idx].tolist())
                    ]
                    predictions.append({"target_subgraph_id": int(sub_id), "experts": experts})
        return predictions

    def _monitor_value(self, metrics: Dict[str, float], loss: float) -> float:
        monitor = self.cfg.router_selection.monitor_metric
        if monitor == "loss_reg":
            return -float(metrics.get("loss_reg", 0.0))
        return -float(metrics.get("mae", 0.0))

    def fit(self) -> Dict[str, object]:
        rp = self.cfg.router_selection
        train_loader = self._build_loader(self.filtered_splits.get("train", []), shuffle=True)
        valid_ids = self.filtered_splits.get("valid", [])
        test_ids = self.filtered_splits.get("test", [])

        self._log_context_stats()
        state = TrainState()
        for epoch in range(int(rp.epochs)):
            loss = self._train_epoch(train_loader, split_name="train")
            metrics = self.evaluate(valid_ids, split_name="valid")
            test_metrics = self.evaluate(test_ids, split_name="test")
            print(
                f"[RouterSelect][Epoch {epoch + 1}/{rp.epochs}] "
                f"train_loss_reg={loss:.4f} "
                f"val_loss_reg={metrics.get('loss_reg', 0):.4f} "
                f"val_MAE={metrics.get('mae', 0):.4f} "
                f"test_MAE={test_metrics.get('mae', 0):.4f}"
            )
            value = self._monitor_value(metrics, loss)
            if value > state.best_metric:
                state.best_metric = value
                state.best_epoch = epoch
                checkpoint = self.save_checkpoint(epoch)
                state.best_checkpoint = checkpoint
            elif (epoch - state.best_epoch) >= int(rp.early_stopping):
                print(f"[RouterSelect] Early stopping at epoch {epoch + 1}.")
                break
        return {
            "best_checkpoint": state.best_checkpoint,
            "best_epoch": state.best_epoch,
            "best_metric": state.best_metric,
        }

    def save_checkpoint(self, epoch: int) -> str:
        ensure_dir(str(self.checkpoint_dir))
        path = self.checkpoint_dir / f"{self.run_name}_best.pt"
        payload = {
            "epoch": epoch,
            "gnn": self.gnn.state_dict(),
            "scorer": self.scorer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "run_name": self.run_name,
        }
        if self.application_graph_encoder is not None:
            payload["graph_encoder"] = self.application_graph_encoder.state_dict()
        torch.save(payload, path)
        return str(path)

    def load_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.gnn.load_state_dict(payload.get("gnn", {}))
        self.scorer.load_state_dict(payload.get("scorer", {}))
        graph_state = payload.get("graph_encoder")
        if graph_state and self.application_graph_encoder is not None:
            self.application_graph_encoder.load_state_dict(graph_state)
        opt_state = payload.get("optimizer")
        if opt_state:
            self.optimizer.load_state_dict(opt_state)

    def save_metrics(self, path: str, payload: Dict[str, object]) -> None:
        ensure_dir(str(Path(path).parent))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[RouterSelect] Saved metrics: {path}")

    def impute_empty_subgraph_embeddings(self, target_ids: List[int]) -> int:
        if not target_ids:
            return 0
        train_ids = self.filtered_splits.get("train", [])
        if not train_ids:
            return 0
        sub_feats = self.node_features.get("target_subgraph")
        if sub_feats is None or sub_feats.numel() == 0:
            return 0
        with torch.no_grad():
            mean_feat = sub_feats[train_ids].mean(dim=0)
            sub_feats[target_ids] = mean_feat
        return len(target_ids)
