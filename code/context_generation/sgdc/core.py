"""
SGDC condensation core for graph-level context generation.

This module is kept under `code.context_generation.sgdc` so the
graph-condensation pipeline stays grouped with the rest of context generation
code.
"""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch

from .src.antk import TNTK
from .src.model.molecule_encoder import MoleculeEncoder
from .src.model.sgdc_gin import G_GIN
from .src.model.wrapper import get_model
from .src.modelpool import ModelPool
from .src.pretrain.wrapper import get_algorithm
from .src.utils import SynDataset, dense2batch

from .utils import ensure_x_feat_dim, save_graph_list_as_pyg_dataset


@dataclass
class SGDCResult:
    graphs: List[Data]
    meta: Dict[str, object]
    scores: Dict[str, float]


def _graph_feat_mean(graph: Data) -> Optional[torch.Tensor]:
    x = getattr(graph, "x", None)
    if x is None:
        return None
    if not torch.is_floating_point(x):
        x = x.to(torch.float32)
    if x.numel() == 0 or x.dim() < 2:
        return None
    return x.mean(dim=0)


def _infer_label_dim(dataset) -> int:
    labels = []
    for graph in dataset:
        y = getattr(graph, "y", None)
        if y is not None and y.numel() > 0:
            y_view = y.view(1, -1) if y.ndim <= 1 else y.view(y.shape[0], -1)
            labels.append(y_view)
    if not labels:
        return 1
    y_all = torch.cat(labels, dim=0)
    return int(y_all.shape[1]) if y_all.ndim > 1 else 1


def _infer_class_count(dataset) -> int:
    if hasattr(dataset, "num_classes") and getattr(dataset, "num_classes") is not None:
        return int(getattr(dataset, "num_classes"))
    inner = getattr(dataset, "dataset", None)
    if inner is not None and hasattr(inner, "num_classes") and getattr(inner, "num_classes") is not None:
        return int(getattr(inner, "num_classes"))

    labels = []
    for graph in dataset:
        y = getattr(graph, "y", None)
        if y is not None and y.numel() > 0:
            labels.append(y.view(-1))
    if not labels:
        return 1

    y_all = torch.cat(labels)
    if y_all.dtype.is_floating_point:
        valid = ~torch.isnan(y_all)
        y_valid = y_all[valid]
        if y_valid.numel() == 0:
            return 1
        rounded = y_valid.round()
        if not torch.allclose(y_valid, rounded, atol=1e-6):
            return 1
        y_valid = rounded.to(torch.long)
    else:
        y_valid = y_all

    if y_valid.ndim > 1:
        return int(y_valid.shape[1])
    return int(torch.max(y_valid).item() + 1) if y_valid.numel() > 0 else 1

def _load_teacher_checkpoint(target_model, ckpt_path: str, args, dataset_name: str, device: torch.device) -> None:
    payload = torch.load(ckpt_path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
        meta = payload.get("meta", {}) or {}
    else:
        state = payload
        meta = {}

    if not isinstance(state, dict):
        raise TypeError(
            f"Unsupported checkpoint format at {ckpt_path}: expected state dict or payload with 'state_dict'."
        )

    expected_model = str(getattr(args, "model", getattr(args, "train_model", "")) or "").strip()
    if meta:
        meta_model = str(meta.get("model", meta.get("train_model", "")) or "").strip()
        if meta_model and expected_model and meta_model != expected_model:
            raise ValueError(
                f"Checkpoint model mismatch for {dataset_name}: "
                f"expected model='{expected_model}', found '{meta_model}' in {ckpt_path}."
            )

    model_state = target_model.state_dict()
    filtered_state = {}
    missing_keys = []
    mismatched_keys = []
    for key, tensor in model_state.items():
        if key.startswith("project."):
            continue
        if key not in state:
            missing_keys.append(key)
            continue
        if state[key].shape != tensor.shape:
            mismatched_keys.append((key, tuple(state[key].shape), tuple(tensor.shape)))
            continue
        filtered_state[key] = state[key]

    if missing_keys or mismatched_keys:
        details = []
        if missing_keys:
            details.append(f"missing={missing_keys[:5]}")
        if mismatched_keys:
            sample = [f"{key}:{src}->{dst}" for key, src, dst in mismatched_keys[:5]]
            details.append(f"shape_mismatch={sample}")
        raise ValueError(
            f"Incompatible SGDC teacher checkpoint for {dataset_name}: {ckpt_path}. "
            f"{'; '.join(details)}"
        )

    target_model.load_state_dict(filtered_state, strict=False)


def _select_init_indices(feats: torch.Tensor, budget: int, mode: str, rng: torch.Generator) -> List[int]:
    num_graphs = feats.size(0)
    if budget >= num_graphs:
        return list(range(num_graphs))
    mode = str(mode).lower()
    if mode == "random":
        return torch.randperm(num_graphs, generator=rng).tolist()[:budget]

    mean = feats.mean(dim=0, keepdim=True)
    dist = torch.sum((feats - mean) ** 2, dim=1)

    if mode == "herding":
        order = torch.argsort(dist)
        return order.tolist()[:budget]

    if mode == "kcenter":
        selected: List[int] = []
        remaining = torch.ones(num_graphs, dtype=torch.bool)
        first = int(torch.argmax(dist).item())
        selected.append(first)
        remaining[first] = False
        distance_matrix = torch.cdist(feats, feats, p=2)
        while len(selected) < budget and bool(remaining.any()):
            sel = torch.tensor(selected, device=distance_matrix.device)
            min_dist = torch.min(distance_matrix[:, sel], dim=1).values
            min_dist[~remaining] = -1
            nxt = int(torch.argmax(min_dist).item())
            selected.append(nxt)
            remaining[nxt] = False
        return selected[:budget]

    return torch.randperm(num_graphs, generator=rng).tolist()[:budget]


def condense_sgdc(
    *,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    test_loader: DataLoader,
    dataset_name: str,
    args,
    device: torch.device,
) -> SGDCResult:
    rng = torch.Generator().manual_seed(int(args.seed))

    if int(args.layers) < 3:
        raise ValueError(
            "SGDC condensation requires context.sgdc.condense.layers >= 3 "
            "because the TNTK kernel expects at least one intermediate graph-kernel block."
        )

    args.loss_func = lambda x, y: F.mse_loss(x, y)

    feat_dim = int(getattr(args, "feat_dim", getattr(args, "nfeat", 0)) or 0)
    if feat_dim <= 0:
        sample_graph = train_loader.dataset[0] if len(train_loader.dataset) > 0 else None  # type: ignore[index]
        if sample_graph is None or getattr(sample_graph, "x", None) is None:
            raise ValueError("Training dataset is empty or has no node features.")
        feat_dim = int(sample_graph.x.size(-1))

    train_graphs = [
        g for g in (ensure_x_feat_dim(train_loader.dataset[i], feat_dim) for i in range(len(train_loader.dataset)))  # type: ignore[index]
        if g is not None
    ]
    if not train_graphs:
        raise ValueError("No valid graphs after feature sanitize for training.")
    train_loader = DataLoader(train_graphs, batch_size=int(args.outer_bs), shuffle=True)

    if val_loader is not None:
        val_graphs = [
            g for g in (ensure_x_feat_dim(val_loader.dataset[i], feat_dim) for i in range(len(val_loader.dataset)))  # type: ignore[index]
            if g is not None
        ]
        val_loader = DataLoader(val_graphs, batch_size=64) if val_graphs else None
    test_graphs = [
        g for g in (ensure_x_feat_dim(test_loader.dataset[i], feat_dim) for i in range(len(test_loader.dataset)))  # type: ignore[index]
        if g is not None
    ]
    test_loader = DataLoader(test_graphs, batch_size=64)

    sample_graph = train_loader.dataset[0] if len(train_loader.dataset) > 0 else None
    if sample_graph is None or getattr(sample_graph, "x", None) is None:
        raise ValueError("Training dataset is empty or has no node features.")
    args.nfeat = int(sample_graph.x.size(-1))

    metric_name = str(getattr(args, "metric", "acc")).strip().lower()
    args.target_dim = _infer_label_dim(train_loader.dataset)
    if metric_name in {"rmse", "mae"}:
        args.nclass = int(args.target_dim)
    else:
        args.nclass = _infer_class_count(train_loader.dataset)

    print("========== SGDC CONFIG ==========")
    print(f"dataset: {dataset_name}")
    print(f"budget_base: {args.budget}")
    print(f"budget_total_override: {int(getattr(args, 'budget_total', 0) or 0)}")
    print(f"hidden_dim: {args.nhid}")
    print(f"outer_epoch: {args.outer_epoch}")
    print(f"outer_lr: {args.outer_lr}")
    print(f"initialize: {args.initialize}")
    print(f"model: {args.model}")
    print(f"outer_grad_norm: {args.outer_grad_norm}")
    print(f"pretrained_ckpt: {args.pretrained_ckpt}")
    print("=================================")

    if args.model == "gin":
        target_model = G_GIN(
            input_dim=args.nfeat,
            hidden_dim=args.nhid,
            output_dim=args.nclass,
            nconvs=args.layers,
            training=False,
        ).to(device)
    elif args.model == "gin_var":
        target_model = MoleculeEncoder(
            emb_dim=args.nhid,
            num_gc_layers=args.layers,
            drop_ratio=args.drop_ratio,
            pooling_type="standard",
        ).to(device)
    else:
        raise NotImplementedError(f"Unsupported model {args.model}")

    if not args.pretrained_ckpt or not os.path.exists(args.pretrained_ckpt):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained_ckpt}")
    _load_teacher_checkpoint(target_model, args.pretrained_ckpt, args, dataset_name, device)

    real_graphs: List[Data] = []
    real_targets: List[torch.Tensor] = []
    target_model.eval()
    with torch.no_grad():
        for data_train in train_loader:
            data_train = data_train.to(device)
            x, edge_index, batch = data_train.x, data_train.edge_index, data_train.batch
            if args.model == "gin":
                y_real = target_model(edge_index, x, batch)
            elif args.model == "gin_var":
                edge_attr = getattr(data_train, "edge_attr", None)
                edge_weight = getattr(data_train, "edge_weight", None)
                y_real, _ = target_model(batch, x, edge_index, edge_attr, edge_weight)
            else:
                raise NotImplementedError
            data_with_target = data_train.cpu()
            data_with_target.y = y_real.detach().cpu()
            real_graphs.append(data_with_target)
            real_targets.append(y_real.detach().cpu())
    real_loader = DataLoader(SynDataset(real_graphs, real_targets), batch_size=1, shuffle=True)
    real_full_loader = DataLoader(real_graphs, batch_size=1, shuffle=True)

    training_graphs: List[Data] = list(train_loader.dataset)
    feat_list: List[torch.Tensor] = []
    keep_idx: List[int] = []
    for idx, graph in enumerate(training_graphs):
        feat = _graph_feat_mean(graph)
        if feat is None:
            continue
        feat_list.append(feat)
        keep_idx.append(idx)

    budget_total = int(getattr(args, "budget_total", 0) or 0)
    if budget_total > 0:
        budget = budget_total
    elif metric_name in {"rmse", "mae"}:
        budget = int(args.budget)
    else:
        budget = int(args.budget) * int(args.nclass)
    print(f"effective_budget: {budget}")

    if not feat_list:
        init_idx = torch.randperm(len(training_graphs), generator=rng).tolist()[:budget]
    else:
        feats = torch.stack(feat_list, dim=0)
        rel_idx = _select_init_indices(feats, budget, mode=args.initialize, rng=rng)
        init_idx = [keep_idx[i] for i in rel_idx if i < len(keep_idx)]
        if len(init_idx) < budget:
            extra = torch.randperm(len(training_graphs), generator=rng).tolist()
            for candidate in extra:
                if len(init_idx) >= budget:
                    break
                if candidate not in init_idx:
                    init_idx.append(candidate)

    selected_list: List[Batch] = []
    for idx in init_idx:
        sub_graph = Batch.from_data_list([training_graphs[idx]])
        num_nodes = sub_graph.x.shape[0]
        identity = torch.eye(num_nodes)
        sub_graph.edge_index = identity.nonzero().T
        selected_list.append(sub_graph)
    batch_syn = Batch.from_data_list(selected_list)

    with torch.no_grad():
        batch_ = batch_syn.to(device)
        x, edge_index, batch = batch_.x.to(device), batch_.edge_index.to(device), batch_.batch.to(device)
        if args.model == "gin":
            y_syn = target_model(edge_index, x, batch)
        elif args.model == "gin_var":
            edge_attr = getattr(batch_, "edge_attr", None)
            edge_weight = getattr(batch_, "edge_weight", None)
            y_syn, _ = target_model(batch, x, edge_index, edge_attr, edge_weight)
        else:
            raise NotImplementedError

    x_syn, _mask = to_dense_batch(batch_syn.x, batch=batch_syn.batch, max_num_nodes=None)
    adj_syn = to_dense_adj(batch_syn.edge_index, batch=batch_syn.batch, max_num_nodes=None)
    x_syn = x_syn.to(device=device, dtype=torch.float32)
    adj_syn = adj_syn.to(device=device, dtype=torch.float32)
    x_syn.requires_grad_(True)
    y_syn = y_syn.to(device)
    y_syn.requires_grad_(True)
    args.num_target_features = x_syn.shape[1]

    syn_loader = dense2batch(x_syn, adj_syn, y_syn)

    outer_opt = torch.optim.Adam([x_syn, y_syn], lr=args.outer_lr, weight_decay=args.outer_wd)

    model_pool = ModelPool(args, device)
    model_pool.init(syn_loader)

    pretrain_algo = get_algorithm("pretrain_ours")
    test_algo = get_algorithm("linear_evaluation")
    train_algo = get_algorithm("pretrain_ssl")
    tntk = TNTK(args.layers, num_mlp_layers=1, scale=args.scale, reg_lambda=args.reg_lambda).to(device)

    best_ckpt_path = None
    full_ckpt_path = None
    save_dir_path = None
    if getattr(args, "save_dir", None):
        save_dir_path = Path(str(args.save_dir))
        save_dir_path.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = save_dir_path / "checkpoint_best.pt"
        full_ckpt_path = save_dir_path / "checkpoint_full.pt"

    score_avg_list: List[float] = []
    score_std_list: List[float] = []
    best_score = float("-inf")
    best_score_std = float("nan")
    best_epoch = None
    best_state = None
    best_syn_tensors = None

    for outer_step in range(1, args.outer_epoch + 1):
        outer_loss, new_loader, _batch_save = train_algo.run(
            args, device, real_loader, model_pool, (x_syn, adj_syn, y_syn), outer_opt, tntk
        )
        print(f"[SGDC] Epoch: {outer_step}, Outer loss: {outer_loss:.4f}")
        if outer_step % args.eval_every == 0:
            init_model = pretrain_algo.run(args, device, args.model, new_loader)
            eval_loader = val_loader if val_loader is not None else test_loader
            score_mean, score_std = test_algo.run(
                args, device, args.model, init_model, train_loader, eval_loader, None
            )
            score_avg_list.append(score_mean)
            score_std_list.append(score_std)
            split_name = "Val" if val_loader is not None else "Test"
            print(
                f"[SGDC] Epoch: {outer_step}, {split_name} Score Mean({args.metric}): {score_mean:.4f}, "
                f"{split_name} Score Std({args.metric}): {score_std:.4f}"
            )
            if score_mean > best_score:
                best_score = score_mean
                best_score_std = score_std
                best_epoch = outer_step
                best_state = deepcopy(init_model.state_dict())
                best_syn_tensors = (
                    x_syn.detach().cpu().clone(),
                    adj_syn.detach().cpu().clone(),
                    y_syn.detach().cpu().clone(),
                )
                if best_ckpt_path is not None:
                    torch.save(best_state, best_ckpt_path)
            del init_model

    if best_state is not None:
        best_model = get_model(args.model, args, args.nclass).to(device)
        best_model.load_state_dict(best_state, strict=False)
        test_mean, test_std = test_algo.run(args, device, args.model, best_model, train_loader, test_loader, None)
    else:
        test_mean, test_std = float("nan"), float("nan")

    if val_loader is not None:
        full_model = pretrain_algo.run(args, device, args.model, real_full_loader)
        full_val_mean, full_val_std = test_algo.run(args, device, args.model, full_model, train_loader, val_loader, None)
        full_test_mean, full_test_std = test_algo.run(args, device, args.model, full_model, train_loader, test_loader, None)
        if full_ckpt_path is not None:
            torch.save(full_model.state_dict(), full_ckpt_path)
    else:
        full_val_mean = full_val_std = full_test_mean = full_test_std = float("nan")

    x_out = x_syn.detach().cpu()
    adj_out = adj_syn.detach().cpu()
    y_out = y_syn.detach().cpu()
    syn_graphs = []
    for idx in range(x_out.shape[0]):
        graph = Data(x=x_out[idx], edge_index=adj_out[idx].nonzero().T, y=y_out[idx].unsqueeze(0))
        syn_graphs.append(graph)

    meta = {
        "method": "sgdc",
        "dataset": dataset_name,
        "budget_base": int(args.budget),
        "budget_total": int(getattr(args, "budget_total", 0) or 0),
        "effective_budget": int(budget),
        "hidden_dim": int(args.nhid),
        "outer_epoch": int(args.outer_epoch),
        "outer_lr": float(args.outer_lr),
        "outer_bs": int(args.outer_bs),
        "initialize": str(args.initialize),
        "model": str(args.model),
        "outer_grad_norm": float(args.outer_grad_norm),
        "scale": str(args.scale),
        "reg_lambda": float(args.reg_lambda),
        "eval_every": int(args.eval_every),
        "metric": str(args.metric),
        "init_indices": init_idx,
        "score_mean_tail": score_avg_list[-5:],
        "score_std_tail": score_std_list[-5:],
        "score_mean_best": best_score,
        "score_std_best": best_score_std,
        "score_best_epoch": best_epoch,
        "score_test_from_best_mean": test_mean,
        "score_test_from_best_std": test_std,
        "score_full_val_mean": full_val_mean,
        "score_full_val_std": full_val_std,
        "score_full_test_mean": full_test_mean,
        "score_full_test_std": full_test_std,
        "checkpoint_best": str(best_ckpt_path) if best_ckpt_path else None,
        "checkpoint_full": str(full_ckpt_path) if full_ckpt_path else None,
    }

    if best_syn_tensors is not None and save_dir_path is not None:
        best_x, best_adj, best_y = best_syn_tensors
        best_graphs = []
        for idx in range(best_x.shape[0]):
            graph = Data(x=best_x[idx], edge_index=best_adj[idx].nonzero().T, y=best_y[idx].unsqueeze(0))
            best_graphs.append(graph)
        best_dir = save_dir_path / "condensed_best"
        best_meta = {
            "method": "sgdc",
            "dataset": dataset_name,
            "tag": "best_val",
            "model": str(args.model),
            "budget_base": int(args.budget),
            "budget_total": int(getattr(args, "budget_total", 0) or 0),
            "effective_budget": int(budget),
            "best_epoch": best_epoch,
            "metric": str(args.metric),
            "score_mean_best": best_score,
            "score_std_best": best_score_std,
            "score_test_from_best_mean": test_mean,
            "score_test_from_best_std": test_std,
            "score_full_val_mean": full_val_mean,
            "score_full_val_std": full_val_std,
            "score_full_test_mean": full_test_mean,
            "score_full_test_std": full_test_std,
        }
        save_graph_list_as_pyg_dataset(best_graphs, out_root=str(best_dir), meta=best_meta)
        meta["condensed_best_dir"] = str(best_dir)

    scores = {
        "score_mean_last": score_avg_list[-1] if score_avg_list else float("nan"),
        "score_std_last": score_std_list[-1] if score_std_list else float("nan"),
        "score_mean_best": best_score,
        "score_std_best": best_score_std,
        "score_best_epoch": best_epoch,
        "score_test_from_best_mean": test_mean,
        "score_test_from_best_std": test_std,
        "score_full_val_mean": full_val_mean,
        "score_full_val_std": full_val_std,
        "score_full_test_mean": full_test_mean,
        "score_full_test_std": full_test_std,
        "checkpoint_best": str(best_ckpt_path) if best_ckpt_path else None,
        "checkpoint_full": str(full_ckpt_path) if full_ckpt_path else None,
    }
    return SGDCResult(graphs=syn_graphs, meta=meta, scores=scores)
