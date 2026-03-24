from typing import Dict, List, Optional, Tuple

import torch


def kmeans(
    embeddings: torch.Tensor,
    num_clusters: int,
    seed: int,
    num_iters: int = 100,
    tol: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if embeddings.numel() == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0)
    num_points = embeddings.shape[0]
    k = min(int(num_clusters), num_points)
    if k <= 0:
        raise ValueError("num_clusters must be positive")

    generator = torch.Generator(device=embeddings.device)
    generator.manual_seed(int(seed))
    init_idx = torch.randperm(num_points, generator=generator)[:k]
    centers = embeddings[init_idx].clone()

    prev_centers = centers.clone()
    for _ in range(num_iters):
        distances = torch.cdist(embeddings, centers, p=2)
        assignments = distances.argmin(dim=1)
        for idx in range(k):
            members = embeddings[assignments == idx]
            if members.numel() == 0:
                continue
            centers[idx] = members.mean(dim=0)
        delta = torch.norm(centers - prev_centers) / (torch.norm(prev_centers) + 1e-8)
        if delta.item() < tol:
            break
        prev_centers = centers.clone()
    return assignments, centers


def majority_vote_mapping(
    assignments: torch.Tensor,
    labels: List[Optional[float]],
    num_clusters: int,
) -> Dict[int, Optional[int]]:
    mapping: Dict[int, Optional[int]] = {}
    label_lists: Dict[int, List[int]] = {idx: [] for idx in range(num_clusters)}
    for idx, cluster_id in enumerate(assignments.tolist()):
        label = labels[idx]
        if label is None:
            continue
        label_lists[cluster_id].append(int(label))

    for cluster_id in range(num_clusters):
        values = label_lists.get(cluster_id, [])
        if not values:
            mapping[cluster_id] = None
            continue
        counts: Dict[int, int] = {}
        for label in values:
            counts[label] = counts.get(label, 0) + 1
        mapping[cluster_id] = max(counts.items(), key=lambda item: item[1])[0]
    return mapping


def apply_cluster_mapping(
    assignments: torch.Tensor,
    mapping: Dict[int, Optional[int]],
) -> List[Optional[int]]:
    outputs: List[Optional[int]] = []
    for cluster_id in assignments.tolist():
        outputs.append(mapping.get(cluster_id))
    return outputs
