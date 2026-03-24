import copy
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as GraphDataLoader

from code.pretrain.methods.utils import get_batch_vector, pool_nodes


def set_encoder_trainable(model: nn.Module, mode: str) -> None:
    mode = (mode or "none").lower()
    if mode == "all":
        for param in model.parameters():
            param.requires_grad = True
        return
    for param in model.parameters():
        param.requires_grad = False
    if mode == "none":
        return
    if mode != "last":
        raise ValueError(f"Unknown unfreeze mode: {mode}")

    candidates = [
        "convs",
        "layers",
        "gnn_layers",
        "blocks",
        "encoder_layers",
    ]
    for name in candidates:
        module_list = getattr(model, name, None)
        if isinstance(module_list, (list, torch.nn.ModuleList)) and len(module_list) > 0:
            for param in module_list[-1].parameters():
                param.requires_grad = True
            return
    fallback = getattr(model, "out_lin", None)
    if fallback is not None:
        for param in fallback.parameters():
            param.requires_grad = True
        return
    for param in model.parameters():
        param.requires_grad = True
    print("[RouterPredict] Warning: could not identify last block; unfreezing full encoder.")


def _forward_batch(model, head, batch, task_type: str, pool_mode: str):
    node_repr, graph_repr = model(batch)
    if graph_repr is None:
        graph_repr = pool_nodes(node_repr, get_batch_vector(batch), mode=pool_mode)
    logits = head(graph_repr)
    if task_type == "classification":
        labels = batch.y.view(-1).long()
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean().item()
        return loss, {"acc": acc}, preds, labels, logits
    labels = batch.y.view(-1).float()
    preds = logits.view(-1)
    loss = F.mse_loss(preds, labels)
    mae = F.l1_loss(preds, labels).item()
    rmse = torch.sqrt(loss).item()
    return loss, {"mae": mae, "rmse": rmse}, preds, labels, logits


def train_supervised(
    model: nn.Module,
    head: nn.Module,
    train_loader: GraphDataLoader,
    val_loader: GraphDataLoader,
    test_loader: GraphDataLoader,
    device: torch.device,
    task_type: str,
    pool_mode: str,
    lr_head: float,
    lr_expert: float,
    weight_decay: float,
    epochs: int,
    early_stop_patience: int,
    unfreeze: str,
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor], List[Dict[str, float]]]:
    set_encoder_trainable(model, unfreeze)
    head.train()
    model.train()

    params = [
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": lr_expert},
        {"params": head.parameters(), "lr": lr_head},
    ]
    params = [group for group in params if group["params"]]
    optimizer = optim.Adam(params=params, weight_decay=weight_decay)

    best_metric = float("-inf") if task_type == "classification" else float("inf")
    best_state = None
    best_metrics: Dict[str, float] = {}
    history: List[Dict[str, float]] = []
    epochs_since_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        total_loss = 0.0
        total_metric = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss, metrics, _, _, _ = _forward_batch(model, head, batch, task_type, pool_mode)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            key = "acc" if task_type == "classification" else "mae"
            total_metric += float(metrics[key])
            num_batches += 1

        if num_batches == 0:
            break
        avg_loss = total_loss / num_batches
        avg_metric = total_metric / num_batches

        val_metrics = evaluate(model, head, val_loader, device, task_type, pool_mode, prefix="val")
        test_metrics = evaluate(model, head, test_loader, device, task_type, pool_mode, prefix="test")

        log = {
            "epoch": float(epoch),
            "train_loss": float(avg_loss),
            "train_metric": float(avg_metric),
        }
        log.update(val_metrics)
        log.update(test_metrics)
        history.append(log)

        monitor = val_metrics.get("val_acc" if task_type == "classification" else "val_rmse", None)
        if monitor is None:
            monitor = avg_loss
        improved = (monitor > best_metric) if task_type == "classification" else (monitor < best_metric)
        if improved:
            best_metric = monitor
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "head": copy.deepcopy(head.state_dict()),
            }
            best_metrics = {k: float(v) for k, v in log.items() if k not in ("epoch",)}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if early_stop_patience > 0 and epochs_since_improvement >= early_stop_patience:
            print(
                "[RouterPredict] Early stopping at epoch "
                f"{epoch} (no improvement in {early_stop_patience} epochs)."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        head.load_state_dict(best_state["head"])

    return best_metrics, best_state or {}, history


def evaluate(
    model: nn.Module,
    head: nn.Module,
    loader: GraphDataLoader,
    device: torch.device,
    task_type: str,
    pool_mode: str,
    prefix: str,
) -> Dict[str, float]:
    if loader is None:
        return {}
    model.eval()
    head.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            loss, metrics, _, _, _ = _forward_batch(model, head, batch, task_type, pool_mode)
            total_loss += loss.item()
            if task_type == "classification":
                total_acc += float(metrics["acc"])
            else:
                total_mae += float(metrics["mae"])
                total_rmse += float(metrics["rmse"])
            num_batches += 1

    if num_batches == 0:
        return {}

    metrics_out = {f"{prefix}_loss": total_loss / num_batches}
    if task_type == "classification":
        metrics_out[f"{prefix}_acc"] = total_acc / num_batches
    else:
        metrics_out[f"{prefix}_mae"] = total_mae / num_batches
        metrics_out[f"{prefix}_rmse"] = total_rmse / num_batches
    return metrics_out


def predict(
    model: nn.Module,
    head: nn.Module,
    loader: GraphDataLoader,
    device: torch.device,
    task_type: str,
    pool_mode: str,
) -> List[Dict[str, object]]:
    outputs: List[Dict[str, object]] = []
    if loader is None:
        return outputs
    model.eval()
    head.eval()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            loss, metrics, preds, labels, logits = _forward_batch(model, head, batch, task_type, pool_mode)
            data_list = batch.to_data_list()
            if task_type == "classification":
                probs = torch.softmax(logits, dim=-1).cpu()
            else:
                probs = None
            for idx, data_item in enumerate(data_list):
                subgraph_id = int(getattr(data_item, "subgraph_id"))
                entry: Dict[str, object] = {
                    "target_subgraph_id": subgraph_id,
                    "pred": float(preds[idx].item()),
                }
                if labels is not None:
                    entry["label"] = float(labels[idx].item())
                if task_type == "classification" and probs is not None:
                    entry["probs"] = probs[idx].tolist()
                outputs.append(entry)
    return outputs


def train_head_on_embeddings(
    head: nn.Module,
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    test_embeddings: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
    task_type: str,
    lr_head: float,
    weight_decay: float,
    epochs: int,
    early_stop_patience: int,
    batch_size: int,
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor], List[Dict[str, float]]]:
    head = head.to(device)
    optimizer = optim.Adam(head.parameters(), lr=lr_head, weight_decay=weight_decay)
    best_metric = float("-inf") if task_type == "classification" else float("inf")
    best_state = None
    best_metrics: Dict[str, float] = {}
    history: List[Dict[str, float]] = []
    epochs_since_improvement = 0
    best_epoch = None

    train_loader = _make_embedding_loader(train_embeddings, train_labels, batch_size)
    val_loader = _make_embedding_loader(val_embeddings, val_labels, batch_size, shuffle=False)
    test_loader = _make_embedding_loader(test_embeddings, test_labels, batch_size, shuffle=False)

    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()
        head.train()
        total_loss = 0.0
        total_metric = 0.0
        num_batches = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = head(batch_x)
            if task_type == "classification":
                labels = batch_y.long()
                loss = F.cross_entropy(logits, labels)
                preds = logits.argmax(dim=-1)
                metric = (preds == labels).float().mean().item()
            else:
                labels = batch_y.float()
                preds = logits.view(-1)
                loss = F.mse_loss(preds, labels)
                metric = F.l1_loss(preds, labels).item()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_metric += metric
            num_batches += 1
        if num_batches == 0:
            break
        train_loss = total_loss / num_batches
        train_metric = total_metric / num_batches

        val_metrics = _evaluate_embeddings_loader(head, val_loader, device, task_type, prefix="val")
        test_metrics = _evaluate_embeddings_loader(head, test_loader, device, task_type, prefix="test")

        log = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "train_metric": float(train_metric),
        }
        log.update(val_metrics)
        log.update(test_metrics)
        history.append(log)

        monitor = val_metrics.get("val_acc" if task_type == "classification" else "val_rmse", None)
        if monitor is None:
            monitor = train_loss
        improved = (monitor > best_metric) if task_type == "classification" else (monitor < best_metric)
        epoch_time = time.perf_counter() - start_time
        if task_type == "classification":
            train_name = "acc"
            val_metric = float(val_metrics.get("val_acc", 0.0))
            test_metric = float(test_metrics.get("test_acc", 0.0))
        else:
            train_name = "mae"
            val_metric = float(val_metrics.get("val_rmse", 0.0))
            test_metric = float(test_metrics.get("test_rmse", 0.0))
        print(
            "[RouterPredict][Epoch "
            f"{epoch}/{epochs}] "
            f"train_loss={train_loss:.4f} "
            f"train_{train_name}={train_metric:.4f} "
            f"val_loss={float(val_metrics.get('val_loss', 0.0)):.4f} "
            f"val_{'acc' if task_type == 'classification' else 'rmse'}={val_metric:.4f} "
            f"test_loss={float(test_metrics.get('test_loss', 0.0)):.4f} "
            f"test_{'acc' if task_type == 'classification' else 'rmse'}={test_metric:.4f} "
            f"time={epoch_time:.1f}s"
        )
        if improved:
            best_metric = monitor
            best_state = copy.deepcopy(head.state_dict())
            best_metrics = {k: float(v) for k, v in log.items() if k not in ("epoch",)}
            best_epoch = epoch
            epochs_since_improvement = 0
            if task_type == "classification":
                print(f"[RouterPredict] Best epoch updated: epoch={epoch} val_acc={val_metric:.4f}")
            else:
                print(f"[RouterPredict] Best epoch updated: epoch={epoch} val_rmse={val_metric:.4f}")
        else:
            epochs_since_improvement += 1

        if early_stop_patience > 0 and epochs_since_improvement >= early_stop_patience:
            print(
                "[RouterPredict] Early stopping at epoch "
                f"{epoch} (no improvement in {early_stop_patience} epochs)."
            )
            break

    if best_state is not None:
        head.load_state_dict(best_state)
        if task_type == "classification":
            best_val = float(best_metrics.get("val_acc", 0.0))
            best_test = float(best_metrics.get("test_acc", 0.0))
            print(f"[RouterPredict] Complete. Best val_acc: {best_val:.4f} at epoch {best_epoch}.")
            print(f"[RouterPredict] Best-epoch test_acc={best_test:.4f}")
        else:
            best_val = float(best_metrics.get("val_rmse", 0.0))
            best_test = float(best_metrics.get("test_rmse", 0.0))
            print(f"[RouterPredict] Complete. Best val_rmse: {best_val:.4f} at epoch {best_epoch}.")
            print(f"[RouterPredict] Best-epoch test_rmse={best_test:.4f}")
    return best_metrics, {"head": best_state} if best_state is not None else {}, history


def _evaluate_embeddings_loader(
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_type: str,
    prefix: str,
) -> Dict[str, float]:
    if loader is None:
        return {}
    head.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = head(batch_x)
            if task_type == "classification":
                labels = batch_y.long()
                loss = F.cross_entropy(logits, labels)
                preds = logits.argmax(dim=-1)
                total_acc += (preds == labels).float().mean().item()
            else:
                labels = batch_y.float()
                preds = logits.view(-1)
                loss = F.mse_loss(preds, labels)
                total_mae += F.l1_loss(preds, labels).item()
                total_rmse += torch.sqrt(loss).item()
            total_loss += loss.item()
            num_batches += 1
    if num_batches == 0:
        return {}
    metrics_out = {f"{prefix}_loss": total_loss / num_batches}
    if task_type == "classification":
        metrics_out[f"{prefix}_acc"] = total_acc / num_batches
    else:
        metrics_out[f"{prefix}_mae"] = total_mae / num_batches
        metrics_out[f"{prefix}_rmse"] = total_rmse / num_batches
    return metrics_out


def predict_on_embeddings(
    head: nn.Module,
    embeddings: torch.Tensor,
    labels: Optional[torch.Tensor],
    subgraph_ids: List[int],
    device: torch.device,
    task_type: str,
    batch_size: int,
) -> List[Dict[str, object]]:
    outputs: List[Dict[str, object]] = []
    if embeddings.numel() == 0:
        return outputs
    head.eval()
    loader = _make_embedding_loader(embeddings, labels, batch_size, shuffle=False)
    start = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = head(batch_x)
            if task_type == "classification":
                preds = logits.argmax(dim=-1).cpu()
                probs = torch.softmax(logits, dim=-1).cpu()
            else:
                preds = logits.view(-1).cpu()
                probs = None
            batch_size_local = preds.size(0)
            for idx in range(batch_size_local):
                subgraph_id = subgraph_ids[start + idx]
                entry: Dict[str, object] = {
                    "target_subgraph_id": int(subgraph_id),
                    "pred": float(preds[idx].item()),
                }
                if batch_y is not None:
                    entry["label"] = float(batch_y[idx].item())
                if task_type == "classification" and probs is not None:
                    entry["probs"] = probs[idx].tolist()
                outputs.append(entry)
            start += batch_size_local
    return outputs


def _make_embedding_loader(
    embeddings: torch.Tensor,
    labels: Optional[torch.Tensor],
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    if labels is None or labels.numel() == 0:
        labels = torch.zeros((embeddings.shape[0],), dtype=torch.float)
    dataset = TensorDataset(embeddings, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
