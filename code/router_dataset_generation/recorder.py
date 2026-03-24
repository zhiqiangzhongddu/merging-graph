import os
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.data import Batch

from code.data_loader import create_dataset, dataset_info, log_split_instance_counts, make_loaders
from code.model import build_encoder_from_cfg
from code.pretrain.methods.utils import get_batch_vector, pool_nodes
from code.utils import format_split_for_name, set_seed
from .utils import serialize_graph


def _shared_split_root(cfg) -> str:
    ds_cfg = getattr(getattr(cfg, "data_preparation", None), "dataset", None)
    return getattr(ds_cfg, "split_root", "data/splits")


class MismatchedCheckpointError(RuntimeError):
    """Raised when pretrained checkpoint parameters do not align with model."""

    def __init__(self, pretrained_run: str, checkpoint: str, mismatches: List[Tuple[str, torch.Size, torch.Size]]):
        self.pretrained_run = pretrained_run
        self.checkpoint = checkpoint
        self.mismatches = mismatches
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = [f"{k} expected={exp} found={found}" for k, exp, found in self.mismatches]
        return f"Mismatched checkpoint for {self.pretrained_run}: " + "; ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pretrained_run": self.pretrained_run,
            "checkpoint": self.checkpoint,
            "mismatches": [
                {"key": k, "expected": list(exp), "found": list(found)}
                for k, exp, found in self.mismatches
            ],
        }


@dataclass
class RouterRunResult:
    """Container for a single (pretrained model, dataset) run."""

    pretrained_run: str
    dataset: str
    task_level: str
    induced: bool
    split: Tuple[float, float, float]
    records: List[Dict[str, Any]]


class PredictionHead(nn.Module):
    """Simple classifier head."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, features):
        return self.classifier(features)


class RouterRecorder:
    """
    Fine-tune a pretrained encoder on a dataset and emit per-example records
    from the test split.
    """

    def __init__(self, cfg, pretrained_checkpoint: str, pretrained_run_name: str, dataset_name: str):
        if not os.path.isfile(pretrained_checkpoint):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_checkpoint}")

        self.cfg = cfg
        # Ensure provided dataset overrides default/None values on the config copy.
        self.cfg.router_dataset.target_dataset.name = dataset_name
        self.device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
        set_seed(cfg.seed)

        self.pretrained_checkpoint = pretrained_checkpoint
        self.pretrained_run_name = pretrained_run_name or Path(pretrained_checkpoint).stem
        self.dataset_name = dataset_name
        self.router_cfg = cfg.router_dataset

        self.checkpoint = torch.load(pretrained_checkpoint, map_location="cpu")
        self.pretrain_cfg = self.checkpoint.get("cfg", {}) or {}

        self._apply_pretrained_model_cfg()
        self._build_dataset()
        self._build_model_and_head()

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    def _apply_pretrained_model_cfg(self) -> None:
        """Align model hyperparameters with the pretrained run when available."""
        if not isinstance(self.pretrain_cfg, dict):
            return
        model_cfg = self.pretrain_cfg.get("model", {}) or {}
        model_name = model_cfg.get("name")
        if model_name:
            self.cfg.model.name = model_name
        for attr in ("hidden_dim", "out_dim", "num_layers", "activation", "dropout", "graph_pooling"):
            if attr in model_cfg:
                setattr(self.cfg.model, attr, model_cfg[attr])
        gat_cfg = model_cfg.get("gat", {}) or {}
        if isinstance(gat_cfg, dict) and "heads" in gat_cfg:
            self.cfg.model.gat.heads = gat_cfg["heads"]
        dataset_cfg = self.pretrain_cfg.get("dataset", {}) or {}
        target_ds_cfg = self.cfg.router_dataset.target_dataset
        for attr in ("feat_reduction", "feat_reduction_dim", "induced_min_size", "induced_max_size", "induced_max_hops"):
            if attr in dataset_cfg:
                setattr(target_ds_cfg, attr, dataset_cfg[attr])

    def _build_dataset(self) -> None:
        """Create dataset and loaders for the target task."""
        ds_cfg = self.cfg.router_dataset.target_dataset
        raw_task_level = ds_cfg.task_level
        induced = ds_cfg.induced

        ds_cfg.name = self.dataset_name
        ds_cfg.task_level = raw_task_level
        ds_cfg.induced = induced
        self.raw_task_level = raw_task_level

        self.dataset = create_dataset(
            name=ds_cfg.name,
            root=ds_cfg.root,
            task_level=raw_task_level,
            feat_reduction=ds_cfg.feat_reduction,
            feat_reduction_dim=ds_cfg.feat_reduction_dim,
            induced=induced,
            induced_min_size=ds_cfg.induced_min_size,
            induced_max_size=ds_cfg.induced_max_size,
            induced_max_hops=ds_cfg.induced_max_hops,
            split_root=_shared_split_root(self.cfg),
            induced_root=getattr(ds_cfg, "induced_root", ""),
        )
        self.dataset_meta = dataset_info(
            dataset=self.dataset,
            task_level=raw_task_level,
            name=ds_cfg.name,
            induced=induced,
        )

        effective_task_level = "graph" if induced else raw_task_level
        ds_cfg.task_level = effective_task_level
        ds_cfg.num_classes = ds_cfg.num_classes or self.dataset_meta.get("num_classes")
        if ds_cfg.num_classes is None:
            ds_cfg.num_classes = getattr(self.dataset, "num_classes", None)

        # Robustly infer class count from labels to avoid out-of-range errors.
        label_max = None
        if hasattr(self.dataset, "data") and getattr(self.dataset.data, "y", None) is not None:
            try:
                label_max = int(self.dataset.data.y.max().item())
            except Exception:
                label_max = None

        if ds_cfg.num_classes is None and label_max is not None:
            ds_cfg.num_classes = label_max + 1
        elif ds_cfg.num_classes is not None and label_max is not None:
            ds_cfg.num_classes = max(int(ds_cfg.num_classes), label_max + 1)
        self.split = tuple(ds_cfg.fixed_split) if getattr(ds_cfg, "fixed_split", None) else (0.8, 0.1, 0.1)

        self.train_loader, self.val_loader, self.test_loader = make_loaders(
            dataset=self.dataset,
            dataset_name=ds_cfg.name,
            task_level=ds_cfg.task_level,
            batch_size=self.router_cfg.batch_size,
            num_workers=self.router_cfg.num_workers,
            split=self.split,
            seed=self.cfg.seed,
            induced=induced,
            split_root=_shared_split_root(self.cfg),
        )
        log_split_instance_counts(
            self.train_loader,
            self.val_loader,
            self.test_loader,
            task_level=ds_cfg.task_level,
            split=self.split,
            induced=induced,
            prefix=f"[RouterRecorder][Split][{self.pretrained_run_name} -> {self.dataset_name}]",
        )

    def _build_model_and_head(self) -> None:
        """Instantiate encoder and prediction head."""
        ds_cfg = self.cfg.router_dataset.target_dataset
        self.cfg.model.in_dim = self.cfg.model.in_dim or self.dataset_meta["num_node_features"]

        self.model = build_encoder_from_cfg(
            cfg=self.cfg,
            in_dim=self.cfg.model.in_dim,
        ).to(self.device)
        model_name = getattr(self.cfg.model, "name", "model")
        print(f"[RouterRecorder] Encoder architecture ({model_name}):\n{self.model}")
        mismatched = self._load_pretrained_weights()
        if mismatched:
            # Skip this run entirely when checkpoint shapes do not match.
            raise MismatchedCheckpointError(
                pretrained_run=self.pretrained_run_name,
                checkpoint=self.pretrained_checkpoint,
                mismatches=mismatched,
            )
        missing, unexpected = self.model.load_state_dict(
            self._filtered_state_dict,
            strict=False,
        )
        if missing:
            print(f"[RouterRecorder] Missing keys when loading pretrained weights: {missing}")
        if unexpected:
            print(f"[RouterRecorder] Unexpected keys when loading pretrained weights: {unexpected}")

        self.head = PredictionHead(
            in_dim=self.cfg.model.out_dim,
            num_classes=ds_cfg.num_classes,
        ).to(self.device)
        print(f"[RouterRecorder] Task head:\n{self.head}")

        freeze_encoder = getattr(self.router_cfg, "freeze_pretrained", False)
        if mismatched and freeze_encoder:
            freeze_encoder = False
            print("[RouterRecorder] Detected mismatched encoder weights; leaving encoder trainable.")

        if freeze_encoder:
            frozen = 0
            for param in self.model.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen += param.numel()
            print(f"[RouterRecorder] Frozen encoder parameters (count={frozen})")

        params = [p for p in self.model.parameters() if p.requires_grad] + list(self.head.parameters())
        self.optimizer = optim.Adam(
            params=params,
            lr=self.router_cfg.lr,
            weight_decay=self.router_cfg.weight_decay,
        )

    # ------------------------------------------------------------------ #
    # Training and evaluation
    # ------------------------------------------------------------------ #
    def _forward_graph_batch(self, batch: Batch):
        """Forward pass for graph-level tasks."""
        node_repr, graph_repr = self.model(batch.to(self.device))
        if graph_repr is None:
            batch_vector = get_batch_vector(batch)
            graph_repr = pool_nodes(
                x=node_repr,
                batch=batch_vector,
                mode=self.cfg.model.graph_pooling,
            )
        logits = self.head(graph_repr)
        labels = batch.y.view(-1).to(self.device)
        return logits, labels

    def _forward_node_batch(self, data):
        """Forward pass for node-level tasks (non-induced)."""
        data = data.to(self.device)
        node_repr, _ = self.model(data)
        logits = self.head(node_repr)
        return logits, data

    def train(self) -> None:
        """Simple supervised training loop."""
        if getattr(self.router_cfg, "epochs", 0) <= 0:
            return
        ds_cfg = self.cfg.router_dataset.target_dataset
        best_val_acc = float("-inf")
        best_state = None
        best_epoch = None
        patience = int(getattr(self.router_cfg, "early_stopping", 0) or 0)
        epochs_since_improvement = 0
        for epoch in range(1, self.router_cfg.epochs + 1):
            self.model.train()
            self.head.train()
            total_loss = 0.0
            num_batches = 0

            for batch in self.train_loader:
                self.optimizer.zero_grad()
                if ds_cfg.task_level == "graph":
                    logits, labels = self._forward_graph_batch(batch)
                    valid = (labels >= 0) & (labels < ds_cfg.num_classes)
                    if valid.sum() == 0:
                        continue
                    loss = F.cross_entropy(logits[valid], labels[valid])
                else:
                    logits, data = self._forward_node_batch(batch)
                    mask = getattr(data, "train_mask", None)
                    if mask is None:
                        mask = torch.ones(data.num_nodes, dtype=torch.bool, device=logits.device)
                    loss = F.cross_entropy(logits[mask], data.y[mask])
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            val_loss, val_acc = self._evaluate_split(self.val_loader, split="val")
            test_loss, test_acc = self._evaluate_split(self.test_loader, split="test")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = {
                    "model": copy.deepcopy(self.model.state_dict()),
                    "head": copy.deepcopy(self.head.state_dict()),
                }
                epochs_since_improvement = 0
            elif patience > 0:
                epochs_since_improvement += 1
            print(
                f"[RouterRecorder][{self.pretrained_run_name} -> {self.dataset_name}] "
                f"Epoch {epoch}/{self.router_cfg.epochs} "
                f"train_loss={avg_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
            )
            if patience > 0 and epochs_since_improvement >= patience:
                print(
                    f"[RouterRecorder] Early stopping at epoch {epoch} "
                    f"(no validation improvement in {patience} epochs)."
                )
                break
        if best_state is not None:
            self.model.load_state_dict(best_state["model"])
            self.head.load_state_dict(best_state["head"])
            print(
                f"[RouterRecorder] Restored best epoch {best_epoch} with val_acc={best_val_acc:.4f} "
                f"for {self.pretrained_run_name} -> {self.dataset_name}"
            )

    def _record_graph_batch(self, batch: Batch) -> List[Dict[str, Any]]:
        """Collect per-graph records for a batch."""
        self.model.eval()
        self.head.eval()
        with torch.no_grad():
            logits, labels = self._forward_graph_batch(batch)
            probs = torch.softmax(logits, dim=-1)
            pred_labels = torch.argmax(logits, dim=-1)

        data_list = batch.to_data_list()
        records: List[Dict[str, Any]] = []
        for idx, data_item in enumerate(data_list):
            true_label = int(labels[idx].item())
            prob_correct = float(probs[idx, true_label].item())
            if self.cfg.router_dataset.target_dataset.induced and hasattr(data_item, "base_node_id"):
                target = {"node_id": int(getattr(data_item, "base_node_id", idx))}
            else:
                target = data_item
            record = self._build_record(
                subgraph=target,
                true_label=true_label,
                pred_label=int(pred_labels[idx].item()),
                prob_correct=prob_correct,
            )
            records.append(record)
        return records

    def _record_node_graph(self, data) -> List[Dict[str, Any]]:
        """Collect per-node records from a single-graph loader."""
        self.model.eval()
        self.head.eval()
        data = data.to(self.device)
        with torch.no_grad():
            node_repr, _ = self.model(data)
            logits = self.head(node_repr)
            probs = torch.softmax(logits, dim=-1)
        test_mask = getattr(data, "test_mask", None)
        if test_mask is None:
            return []
        indices = torch.nonzero(test_mask, as_tuple=False).view(-1)
        records: List[Dict[str, Any]] = []
        for idx in indices:
            idx_int = int(idx.item())
            true_label = int(data.y[idx_int].item())
            prob_correct = float(probs[idx_int, true_label].item())
            pred_label = int(torch.argmax(logits[idx_int]).item())
            # Minimal subgraph: single node context to satisfy serialization requirement.
            node_data = {
                "num_nodes": 1,
                "num_edges": 0,
                "x": data.x[idx_int].view(1, -1).cpu().tolist() if getattr(data, "x", None) is not None else None,
                "y": [true_label],
                "node_index": idx_int,
            }
            records.append(
                self._build_record(
                    subgraph=node_data,
                    true_label=true_label,
                    pred_label=pred_label,
                    prob_correct=prob_correct,
                )
            )
        return records

    def collect_test_records(self) -> List[Dict[str, Any]]:
        """Iterate over test split and collect per-example records."""
        ds_cfg = self.cfg.router_dataset.target_dataset
        all_records: List[Dict[str, Any]] = []
        for batch in self.test_loader:
            if ds_cfg.task_level == "graph":
                all_records.extend(self._record_graph_batch(batch))
            else:
                all_records.extend(self._record_node_graph(batch))
        return all_records

    def _evaluate_split(self, loader, split: str) -> Tuple[float, float]:
        """Evaluate loss/accuracy on a given split."""
        ds_cfg = self.cfg.router_dataset.target_dataset
        self.model.eval()
        self.head.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in loader:
                if ds_cfg.task_level == "graph":
                    logits, labels = self._forward_graph_batch(batch)
                    valid = (labels >= 0) & (labels < ds_cfg.num_classes)
                    if valid.sum() == 0:
                        continue
                    logits = logits[valid]
                    labels = labels[valid]
                    loss = F.cross_entropy(logits, labels)
                    preds = logits.argmax(dim=-1)
                    acc = (preds == labels).float().mean().item()
                else:
                    logits, data = self._forward_node_batch(batch)
                    mask_attr = f"{split}_mask"
                    mask = getattr(data, mask_attr, None)
                    if mask is None:
                        mask = torch.ones(data.num_nodes, dtype=torch.bool, device=logits.device)
                    loss = F.cross_entropy(logits[mask], data.y[mask])
                    preds = logits.argmax(dim=-1)
                    acc = (preds[mask] == data.y[mask]).float().mean().item()
                total_loss += loss.item()
                total_acc += acc
                num_batches += 1

        if num_batches == 0:
            return 0.0, 0.0
        return total_loss / num_batches, total_acc / num_batches

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self) -> RouterRunResult:
        """Run fine-tuning + test recording for one dataset."""
        self.train()
        records = self.collect_test_records()
        return RouterRunResult(
            pretrained_run=self.pretrained_run_name,
            dataset=self.dataset_name,
            task_level=self.cfg.router_dataset.target_dataset.task_level,
            induced=self.cfg.router_dataset.target_dataset.induced,
            split=self.split,
            records=records,
        )

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def _build_record(self, subgraph: Any, true_label: int, pred_label: int, prob_correct: float) -> Dict[str, Any]:
        """Assemble a single router record."""
        ds_cfg = self.cfg.router_dataset.target_dataset
        target = serialize_graph(subgraph) if not isinstance(subgraph, dict) else subgraph
        return {
            "dataset": self.dataset_name,
            "task": {
                "raw_level": self.raw_task_level,
                "effective_level": ds_cfg.task_level,
                "induced": ds_cfg.induced,
                "split": self.split,
                "split_tag": format_split_for_name(self.split),
            },
            "pretrained_model": {
                "run_name": self.pretrained_run_name,
                "checkpoint": self.pretrained_checkpoint,
            },
            "target": target,
            "prediction": {
                "true_label": true_label,
                "pred_label": pred_label,
                "prob_correct": prob_correct,
            },
        }

    def _load_pretrained_weights(self) -> List[Tuple[str, torch.Size, torch.Size]]:
        """
        Filter out any pretrained parameters whose shapes do not match the
        current model to avoid load_state_dict RuntimeErrors.
        Returns a list of skipped (key, expected_shape, found_shape).
        """
        model_state = self.checkpoint.get("model_state", {}) or {}
        current_state = self.model.state_dict()
        filtered_state = {}
        mismatched: List[Tuple[str, torch.Size, torch.Size]] = []
        for key, tensor in model_state.items():
            target_tensor = current_state.get(key)
            if target_tensor is None:
                filtered_state[key] = tensor
                continue
            if target_tensor.shape != tensor.shape:
                mismatched.append((key, target_tensor.shape, tensor.shape))
                continue
            filtered_state[key] = tensor

        # Store for reuse when actually loading.
        self._filtered_state_dict = filtered_state
        return mismatched
