import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

try:
    from torch_geometric.data import HeteroData

    HAS_PYG = True
except Exception:
    HAS_PYG = False


@dataclass
class GraphArtifacts:
    graph: Any
    application_id_map: Dict[str, Dict[str, Any]]
    subgraph_id_map: Dict[str, Dict[str, Any]]
    expert_id_map: Dict[str, Dict[str, Any]]
    key_to_id: Dict[str, Dict[str, int]]
    stats: Dict[str, Any]


def _canonical_dumps(obj: Any) -> str:
    """Serialize using sorted keys so hashes are stable."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_hash(obj: Any) -> str:
    """Deterministic hash for JSON-serializable objects."""
    payload = _canonical_dumps(obj)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_records_dir(records_dir: Path) -> List[Dict[str, Any]]:
    """
    Load router record runs from a directory produced by router dataset generation.
    Expected layout: `records_dir/<pretrained_run_name>.json` (each file contains a list of runs).
    """
    if not records_dir.is_dir():
        raise ValueError(f"Router records dir not found: {records_dir}")

    runs: List[Dict[str, Any]] = []
    
    skip_names = {"router_records_failed_runs.json"}
    json_paths = sorted(p for p in records_dir.glob("*.json") if p.name not in skip_names)
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        tqdm = None
    iterator = tqdm(json_paths, desc="[RouterGraph] Loading records") if tqdm else json_paths
    for json_path in iterator:
        try:
            obj = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        candidates = obj if isinstance(obj, list) else [obj]
        runs.extend([o for o in candidates if isinstance(o, dict)])

    if not runs:
        raise ValueError(f"No router record runs found under directory: {records_dir}")
    print(f"[RouterGraph] Loaded records (files={len(json_paths)}, runs={len(runs)}) from {records_dir}")
    return runs


def _clean_split(split: Any) -> Optional[List[float]]:
    """Normalize split info to a list of floats when possible."""
    if split is None:
        return None
    try:
        return [float(x) for x in split]
    except Exception:
        return None


def _extract_task_descriptor(run: Dict[str, Any], prefer_raw_task_level: bool = True) -> Dict[str, Any]:
    target_ds = run.get("target_dataset") or {}
    task = run.get("task") or {}
    raw_task_level = task.get("raw_level")
    effective_task_level = target_ds.get("task_level") or task.get("effective_level")
    task_level = raw_task_level if (prefer_raw_task_level and raw_task_level) else effective_task_level or raw_task_level
    descriptor: Dict[str, Any] = {
        "task_level": task_level,
        "induced": target_ds.get("induced"),
        "split": _clean_split(target_ds.get("split") or task.get("split")),
        "raw_task_level": raw_task_level,
        "split_tag": task.get("split_tag"),
    }
    return {k: v for k, v in descriptor.items() if v is not None}


def _build_application_key(run: Dict[str, Any], prefer_raw_task_level: bool = True) -> Tuple[str, Dict[str, Any]]:
    target_ds = run.get("target_dataset") or {}
    dataset = target_ds.get("name") or run.get("dataset") or "unknown_dataset"
    task_desc = _extract_task_descriptor(run, prefer_raw_task_level=prefer_raw_task_level)
    key_payload = {"dataset": dataset, "task": task_desc}
    key = _canonical_dumps(key_payload)
    display_name = f"{dataset} | level={task_desc.get('task_level', 'n/a')} induced={task_desc.get('induced', 'n/a')}"
    meta = {
        "dataset": dataset,
        "task": task_desc,
        "display_name": display_name,
        "key": key_payload,
    }
    return key, meta


def _build_subgraph_key(target: Any, dataset: Optional[str] = None) -> Tuple[str, str, Dict[str, Any]]:
    """
    Returns (node_key, hash_key, summary).
    node_key prefers explicit identifiers; hash_key always derives from a canonical dump.
    """
    identifier = None
    target_payload = target if target is not None else {}
    if isinstance(target_payload, dict):
        for field in ("subgraph_id", "graph_id", "node_id", "node_index", "id", "instance_id", "center_node"):
            if field in target_payload:
                identifier = f"{field}:{target_payload[field]}"
                break
    hash_payload = {"target": target_payload, "identifier": identifier, "dataset": dataset}
    hash_key = _stable_hash(hash_payload)
    node_key = identifier or f"hash:{hash_key}"
    if dataset and identifier:
        node_key = f"{dataset}:{identifier}"
    summary = _summarize_subgraph(target_payload)
    summary["hash_key"] = hash_key
    summary["key"] = node_key
    return node_key, hash_key, summary


def _summarize_subgraph(target: Any) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    if isinstance(target, dict):
        for field in ("node_id", "node_index", "graph_id", "subgraph_id", "instance_id", "center_node"):
            if field in target:
                summary[field] = target[field]
        for field in ("num_nodes", "num_edges"):
            if field in target:
                summary[field] = target[field]
    return summary


def _extract_expert_key(run: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    pretrained = run.get("pretrained_model") or {}
    identity = pretrained.get("run_name") or pretrained.get("checkpoint") or "unknown_expert"
    meta = {
        "model": identity,
        "checkpoint": pretrained.get("checkpoint"),
    }
    return identity, meta


def _extract_prob_correct(record: Dict[str, Any]) -> Optional[float]:
    pred = record.get("prediction") or {}
    true_label = pred.get("true_label", record.get("true_label", None))
    if true_label is not None:
        try:
            true_label = int(true_label)
        except Exception:
            true_label = None

    if "prob_correct" in pred:
        try:
            return float(pred["prob_correct"])
        except Exception:
            return None

    if "probs" in pred and true_label is not None:
        probs = pred["probs"]
        try:
            if isinstance(probs, dict):
                if str(true_label) in probs:
                    return float(probs[str(true_label)])
            elif isinstance(probs, (list, tuple)) and 0 <= true_label < len(probs):
                return float(probs[true_label])
        except Exception:
            return None

    if "logits" in pred and true_label is not None:
        logits = pred["logits"]
        try:
            tensor = torch.tensor(logits, dtype=torch.float)
            probs = torch.softmax(tensor, dim=-1)
            return float(probs[true_label].item())
        except Exception:
            return None
    return None


def _basic_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "p50": 0.0}
    ordered = sorted(values)
    count = len(ordered)
    mean = float(sum(ordered)) / float(count)
    median = ordered[count // 2]
    return {
        "count": count,
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "mean": float(mean),
        "p50": float(median),
    }


def _tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    if tensor.numel() == 0:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "p50": 0.0}
    return _basic_stats([float(v) for v in tensor.detach().cpu().view(-1).tolist()])


def _build_application_features(
    application_meta: Dict[str, Dict[str, Any]]
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    task_levels = []
    for meta in application_meta.values():
        task_levels.append((meta.get("task") or {}).get("task_level"))
    vocab = sorted({lvl for lvl in task_levels if lvl is not None})
    if "unknown" not in vocab:
        vocab.append("unknown")
    task_level_to_idx = {lvl: idx for idx, lvl in enumerate(vocab)}

    rows: List[List[float]] = []
    for key in sorted(application_meta):
        meta = application_meta[key]
        task = meta.get("task") or {}
        raw_level = task.get("task_level") or "unknown"
        task_idx = float(task_level_to_idx.get(raw_level, task_level_to_idx["unknown"]))
        induced = 1.0 if bool(task.get("induced")) else 0.0
        split = task.get("split") or []
        split_vals = [0.0, 0.0, 0.0]
        for i in range(min(3, len(split))):
            try:
                split_vals[i] = float(split[i])
            except Exception:
                split_vals[i] = 0.0
        rows.append([task_idx, induced, split_vals[0], split_vals[1], split_vals[2]])

    features = torch.tensor(rows, dtype=torch.float) if rows else torch.empty((0, 5), dtype=torch.float)
    schema = {
        "feature_schema": ["task_level_index", "induced", "split_0", "split_1", "split_2"],
        "task_level_vocab": vocab,
    }
    return features, schema


def _build_subgraph_features(
    subgraph_meta: Dict[str, Dict[str, Any]]
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    rows: List[List[float]] = []
    for key in sorted(subgraph_meta):
        meta = subgraph_meta[key]
        num_nodes = float(meta.get("num_nodes") or 0.0)
        num_edges = float(meta.get("num_edges") or 0.0)
        rows.append([num_nodes, num_edges])
    features = torch.tensor(rows, dtype=torch.float) if rows else torch.empty((0, 2), dtype=torch.float)
    schema = {"feature_schema": ["num_nodes", "num_edges"]}
    return features, schema


def _build_expert_features(expert_meta: Dict[str, Dict[str, Any]]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    count = len(expert_meta)
    features = torch.ones((count, 1), dtype=torch.float) if count else torch.empty((0, 1), dtype=torch.float)
    schema = {"feature_schema": ["bias"]}
    return features, schema


def _build_edge_indices(
    application_ids: Dict[str, int],
    subgraph_ids: Dict[str, int],
    expert_ids: Dict[str, int],
    app_sub_pairs: Iterable[Tuple[str, str]],
    sub_expert_pairs: Dict[Tuple[str, str], float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    app_sub_pairs_sorted = sorted({(application_ids[a], subgraph_ids[s]) for a, s in app_sub_pairs})
    if app_sub_pairs_sorted:
        app_sub_tensor = torch.tensor(app_sub_pairs_sorted, dtype=torch.long).t().contiguous()
    else:
        app_sub_tensor = torch.empty((2, 0), dtype=torch.long)
    app_sub_weights = torch.ones(app_sub_tensor.shape[1], dtype=torch.float)

    sub_expert_entries = sorted(
        ((subgraph_ids[sub], expert_ids[exp], weight) for (sub, exp), weight in sub_expert_pairs.items()),
        key=lambda tup: (tup[0], tup[1]),
    )
    if sub_expert_entries:
        coords = [(s, e) for s, e, _ in sub_expert_entries]
        weights = [w for _, _, w in sub_expert_entries]
        sub_expert_tensor = torch.tensor(coords, dtype=torch.long).t().contiguous()
        sub_expert_weights = torch.tensor(weights, dtype=torch.float)
    else:
        sub_expert_tensor = torch.empty((2, 0), dtype=torch.long)
        sub_expert_weights = torch.empty((0,), dtype=torch.float)
    return app_sub_tensor, app_sub_weights, sub_expert_tensor, sub_expert_weights


def _filter_topk_experts(
    sub_expert_pairs: Dict[Tuple[str, str], float],
    topk_expert: Optional[int],
) -> Tuple[Dict[Tuple[str, str], float], Dict[str, Any]]:
    if topk_expert is None:
        return sub_expert_pairs, {"topk_expert": None}

    try:
        k = int(topk_expert)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"topk_expert must be an int or None, got {topk_expert!r}") from exc

    if k <= 0:
        return {}, {
            "topk_expert": k,
            "topk_expert_edges_before": len(sub_expert_pairs),
            "topk_expert_edges_after": 0,
        }

    by_subgraph: Dict[str, List[Tuple[str, float]]] = {}
    for (sub_key, expert_key), prob in sub_expert_pairs.items():
        by_subgraph.setdefault(sub_key, []).append((expert_key, prob))

    filtered: Dict[Tuple[str, str], float] = {}
    for sub_key, entries in by_subgraph.items():
        entries.sort(key=lambda item: item[1], reverse=True)
        for expert_key, prob in entries[:k]:
            filtered[(sub_key, expert_key)] = prob

    return filtered, {
        "topk_expert": k,
        "topk_expert_edges_before": len(sub_expert_pairs),
        "topk_expert_edges_after": len(filtered),
    }


def _filter_topk_neg_experts(
    sub_expert_pairs: Dict[Tuple[str, str], float],
    enabled: bool,
    k: int,
) -> Tuple[Dict[Tuple[str, str], float], Dict[str, Any]]:
    if not enabled:
        return sub_expert_pairs, {"topk_neg_expert": False}
    if k is None:
        raise ValueError("topk_neg_expert=True requires router_graph.topk_expert to set k.")
    if k <= 0:
        return {}, {
            "topk_neg_expert": True,
            "topk_neg_expert_k": k,
            "topk_neg_expert_edges_before": len(sub_expert_pairs),
            "topk_neg_expert_edges_after": 0,
        }

    by_subgraph: Dict[str, List[Tuple[str, float]]] = {}
    for (sub_key, expert_key), prob in sub_expert_pairs.items():
        by_subgraph.setdefault(sub_key, []).append((expert_key, prob))

    filtered: Dict[Tuple[str, str], float] = {}
    for sub_key, entries in by_subgraph.items():
        entries.sort(key=lambda item: item[1])
        for expert_key, prob in entries[:k]:
            filtered[(sub_key, expert_key)] = prob

    return filtered, {
        "topk_neg_expert": True,
        "topk_neg_expert_k": k,
        "topk_neg_expert_edges_before": len(sub_expert_pairs),
        "topk_neg_expert_edges_after": len(filtered),
    }


def _split_matches(split: Optional[List[float]], expected: Optional[List[float]]) -> bool:
    if expected is None:
        return True
    if not split:
        return False
    if len(split) != len(expected):
        return False
    for left, right in zip(split, expected):
        if abs(float(left) - float(right)) > 1e-6:
            return False
    return True


def build_router_graph(
    dataset_dir: str,
    prefer_raw_task_level: bool = True,
    strict_duplicates: bool = True,
    expected_split: Optional[List[float]] = None,
    topk_expert: Optional[int] = None,
    topk_neg_expert: bool = False,
) -> GraphArtifacts:
    """
    Construct a heterogeneous router graph from router records.
    `dataset_dir` must be a directory of per-expert JSON files produced by router dataset generation.
    """
    path = Path(dataset_dir)
    runs = _load_records_dir(path)
    expected_split = _clean_split(expected_split)
    if expected_split:
        filtered_runs: List[Dict[str, Any]] = []
        skipped_split = 0
        for run in runs:
            target_ds = run.get("target_dataset") or {}
            task_meta = run.get("task") or {}
            split = _clean_split(target_ds.get("split") or task_meta.get("split"))
            if not _split_matches(split, expected_split):
                skipped_split += 1
                continue
            filtered_runs.append(run)
        runs = filtered_runs
        print(f"[RouterGraph] Filtered runs by split={expected_split}: kept={len(runs)} skipped={skipped_split}")
    else:
        skipped_split = 0

    application_meta: Dict[str, Dict[str, Any]] = {}
    subgraph_meta: Dict[str, Dict[str, Any]] = {}
    expert_meta: Dict[str, Dict[str, Any]] = {}
    app_sub_pairs: List[Tuple[str, str]] = []
    sub_expert_pairs: Dict[Tuple[str, str], float] = {}
    sub_expert_counts: Dict[Tuple[str, str], int] = {}

    seen_app_sub: set = set()
    seen_sub_expert: set = set()
    seen_run_keys: set = set()
    num_records = 0
    skipped_probs = 0
    duplicate_runs = 0
    duplicate_sub_expert = 0
    global_record_idx = 0
    for run_idx, run in enumerate(runs):
        app_key, app_info = _build_application_key(run, prefer_raw_task_level=prefer_raw_task_level)
        application_meta.setdefault(app_key, app_info)

        expert_key, expert_info = _extract_expert_key(run)
        expert_meta.setdefault(expert_key, expert_info)

        run_key = (app_key, expert_key)
        if run_key in seen_run_keys:
            if strict_duplicates:
                raise ValueError(
                    "Duplicate router run detected while building graph: "
                    f"application_key={app_key!r}, expert_key={expert_key!r}."
                )
            duplicate_runs += 1
        else:
            seen_run_keys.add(run_key)

        for record_idx, record in enumerate(run.get("records", [])):
            num_records += 1
            target = record.get("target")
            sub_key, hash_key, summary = _build_subgraph_key(target, dataset=app_info["dataset"])
            if sub_key not in subgraph_meta:
                summary["dataset"] = app_info["dataset"]
                subgraph_meta[sub_key] = summary

            app_sub_pair = (app_key, sub_key)
            if app_sub_pair not in seen_app_sub:
                seen_app_sub.add(app_sub_pair)
                app_sub_pairs.append(app_sub_pair)

            prob = _extract_prob_correct(record)
            if prob is None:
                skipped_probs += 1
                global_record_idx += 1
                continue

            sub_exp_pair = (sub_key, expert_key)
            if sub_exp_pair in seen_sub_expert:
                if strict_duplicates:
                    raise ValueError(
                        "Duplicate subgraph->expert edge detected while building graph: "
                        f"subgraph_key={sub_key!r}, expert_key={expert_key!r}."
                    )
                duplicate_sub_expert += 1
                count = sub_expert_counts.get(sub_exp_pair, 1)
                sub_expert_pairs[sub_exp_pair] = (sub_expert_pairs[sub_exp_pair] * count + prob) / (count + 1)
                sub_expert_counts[sub_exp_pair] = count + 1
            else:
                seen_sub_expert.add(sub_exp_pair)
                sub_expert_pairs[sub_exp_pair] = prob
                sub_expert_counts[sub_exp_pair] = 1
            global_record_idx += 1

    if topk_neg_expert:
        pos_pairs, topk_stats = _filter_topk_experts(sub_expert_pairs, topk_expert)
        neg_pairs, neg_stats = _filter_topk_neg_experts(sub_expert_pairs, True, k=topk_expert)
        sub_expert_pairs = {**pos_pairs, **neg_pairs}
        neg_stats["topk_neg_expert_edges_total"] = len(sub_expert_pairs)
    else:
        sub_expert_pairs, topk_stats = _filter_topk_experts(sub_expert_pairs, topk_expert)
        neg_stats = {"topk_neg_expert": False}

    application_ids = {key: idx for idx, key in enumerate(sorted(application_meta))}
    subgraph_ids = {key: idx for idx, key in enumerate(sorted(subgraph_meta))}
    expert_ids = {key: idx for idx, key in enumerate(sorted(expert_meta))}

    app_sub_ei, app_sub_w, sub_expert_ei, sub_expert_w = _build_edge_indices(
        application_ids,
        subgraph_ids,
        expert_ids,
        app_sub_pairs,
        sub_expert_pairs,
    )

    if HAS_PYG:
        graph = HeteroData()
        graph["application"].num_nodes = len(application_ids)
        graph["target_subgraph"].num_nodes = len(subgraph_ids)
        graph["expert"].num_nodes = len(expert_ids)
        graph["application", "to", "target_subgraph"].edge_index = app_sub_ei
        graph["application", "to", "target_subgraph"].edge_weight = app_sub_w
        graph["target_subgraph", "to", "expert"].edge_index = sub_expert_ei
        graph["target_subgraph", "to", "expert"].edge_weight = sub_expert_w
        graph["target_subgraph", "to", "expert"].edge_label_index = sub_expert_ei
        graph["target_subgraph", "to", "expert"].edge_label = sub_expert_w
        graph["target_subgraph", "rev_to", "application"].edge_index = app_sub_ei.flip(0)
        graph["target_subgraph", "rev_to", "application"].edge_weight = app_sub_w
        graph["expert", "rev_to", "target_subgraph"].edge_index = sub_expert_ei.flip(0)
        graph["expert", "rev_to", "target_subgraph"].edge_weight = sub_expert_w
    else:
        graph = {
            "node_counts": {
                "application": len(application_ids),
                "target_subgraph": len(subgraph_ids),
                "expert": len(expert_ids),
            },
            ("application", "to", "target_subgraph"): {"edge_index": app_sub_ei, "edge_weight": app_sub_w},
            ("target_subgraph", "to", "expert"): {"edge_index": sub_expert_ei, "edge_weight": sub_expert_w},
            ("target_subgraph", "rev_to", "application"): {
                "edge_index": app_sub_ei.flip(0),
                "edge_weight": app_sub_w,
            },
            ("expert", "rev_to", "target_subgraph"): {
                "edge_index": sub_expert_ei.flip(0),
                "edge_weight": sub_expert_w,
            },
        }

    application_id_map = {}
    for key in sorted(application_meta):
        idx = application_ids[key]
        meta = application_meta[key]
        application_id_map[str(idx)] = {
            "name": meta.get("display_name"),
            "dataset": meta.get("dataset"),
            "task": meta.get("task"),
            "key": meta.get("key"),
        }

    subgraph_id_map = {}
    for key in sorted(subgraph_meta):
        idx = subgraph_ids[key]
        meta = subgraph_meta[key]
        subgraph_id_map[str(idx)] = meta

    expert_id_map = {}
    for key in sorted(expert_meta):
        idx = expert_ids[key]
        meta = expert_meta[key]
        expert_id_map[str(idx)] = {
            "model": meta.get("model"),
            "checkpoint": meta.get("checkpoint"),
        }

    key_to_id = {
        "application": application_ids,
        "target_subgraph": subgraph_ids,
        "expert": expert_ids,
    }

    app_out_deg = (
        torch.bincount(app_sub_ei[0], minlength=len(application_ids))
        if app_sub_ei.numel()
        else torch.zeros(len(application_ids), dtype=torch.long)
    )
    sub_in_deg_from_app = (
        torch.bincount(app_sub_ei[1], minlength=len(subgraph_ids))
        if app_sub_ei.numel()
        else torch.zeros(len(subgraph_ids), dtype=torch.long)
    )
    sub_out_deg_to_expert = (
        torch.bincount(sub_expert_ei[0], minlength=len(subgraph_ids))
        if sub_expert_ei.numel()
        else torch.zeros(len(subgraph_ids), dtype=torch.long)
    )
    expert_in_deg = (
        torch.bincount(sub_expert_ei[1], minlength=len(expert_ids))
        if sub_expert_ei.numel()
        else torch.zeros(len(expert_ids), dtype=torch.long)
    )
    sub_total_deg = sub_in_deg_from_app + sub_out_deg_to_expert

    dataset_counts: Dict[str, int] = {}
    task_level_counts: Dict[str, int] = {}
    induced_counts: Dict[str, int] = {}
    for meta in application_meta.values():
        dataset_counts[meta.get("dataset", "unknown")] = dataset_counts.get(meta.get("dataset", "unknown"), 0) + 1
        task_level = (meta.get("task") or {}).get("task_level") or "unknown"
        task_level_counts[str(task_level)] = task_level_counts.get(str(task_level), 0) + 1
        induced_flag = (meta.get("task") or {}).get("induced")
        induced_key = "unknown" if induced_flag is None else str(bool(induced_flag))
        induced_counts[induced_key] = induced_counts.get(induced_key, 0) + 1

    stats = {
        "num_runs": len(runs),
        "num_runs_filtered_by_split": skipped_split,
        "num_records": num_records,
        "num_records_with_probs": num_records - skipped_probs,
        "num_application_nodes": len(application_ids),
        "num_subgraph_nodes": len(subgraph_ids),
        "num_expert_nodes": len(expert_ids),
        "num_app_to_sub_edges": app_sub_ei.shape[1],
        "num_sub_to_expert_edges": sub_expert_ei.shape[1],
        "records_missing_probs": skipped_probs,
        "duplicate_runs": duplicate_runs,
        "duplicate_sub_expert_edges": duplicate_sub_expert,
        "topk_expert": topk_stats.get("topk_expert"),
        "topk_expert_edges_before": topk_stats.get("topk_expert_edges_before"),
        "topk_expert_edges_after": topk_stats.get("topk_expert_edges_after"),
        "topk_neg_expert": neg_stats.get("topk_neg_expert"),
        "topk_neg_expert_k": neg_stats.get("topk_neg_expert_k"),
        "topk_neg_expert_edges_before": neg_stats.get("topk_neg_expert_edges_before"),
        "topk_neg_expert_edges_after": neg_stats.get("topk_neg_expert_edges_after"),
        "dataset_counts": dataset_counts,
        "task_level_counts": task_level_counts,
        "induced_counts": induced_counts,
        "edge_weight_stats": _tensor_stats(sub_expert_w),
        "degree_stats": {
            "application_out": _tensor_stats(app_out_deg),
            "target_subgraph_in": _tensor_stats(sub_in_deg_from_app),
            "target_subgraph_out": _tensor_stats(sub_out_deg_to_expert),
            "target_subgraph_total": _tensor_stats(sub_total_deg),
            "expert_in": _tensor_stats(expert_in_deg),
        },
        "isolated_nodes": {
            "application": int((app_out_deg == 0).sum().item()) if app_out_deg.numel() else 0,
            "target_subgraph": int((sub_total_deg == 0).sum().item()) if sub_total_deg.numel() else 0,
            "expert": int((expert_in_deg == 0).sum().item()) if expert_in_deg.numel() else 0,
        },
        "edge_types": [
            "application->target_subgraph",
            "target_subgraph->expert",
            "target_subgraph->application",
            "expert->target_subgraph",
        ],
    }

    return GraphArtifacts(
        graph=graph,
        application_id_map=application_id_map,
        subgraph_id_map=subgraph_id_map,
        expert_id_map=expert_id_map,
        key_to_id=key_to_id,
        stats=stats,
    )


def save_graph_artifacts(
    artifacts: GraphArtifacts,
    output_dir: str,
    metadata: Optional[Dict[str, Any]] = None,
    graph_filename: str = "router_graph.pt",
) -> None:
    """Persist graph tensor object and mapping files."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(artifacts.graph, out_dir / graph_filename)


    with open(out_dir / "application_id_map.json", "w", encoding="utf-8") as f:
        json.dump(artifacts.application_id_map, f, indent=2)
    with open(out_dir / "subgraph_id_map.json", "w", encoding="utf-8") as f:
        json.dump(artifacts.subgraph_id_map, f, indent=2)
    with open(out_dir / "expert_id_map.json", "w", encoding="utf-8") as f:
        json.dump(artifacts.expert_id_map, f, indent=2)

    merged = []
    for type_name, id_map in (
        ("application", artifacts.application_id_map),
        ("target_subgraph", artifacts.subgraph_id_map),
        ("expert", artifacts.expert_id_map),
    ):
        for id_str, meta in id_map.items():
            merged.append({"type": type_name, "id": int(id_str), "meta": meta})
    with open(out_dir / "node_id_map.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    meta_payload = {"stats": artifacts.stats, "has_pyg": HAS_PYG}
    if metadata:
        meta_payload.update(metadata)
    with open(out_dir / "router_graph_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)

    print(
        "[RouterGraph] Saved graph and mappings to "
        f"{out_dir} (app={len(artifacts.application_id_map)}, "
        f"subgraph={len(artifacts.subgraph_id_map)}, expert={len(artifacts.expert_id_map)})"
    )
