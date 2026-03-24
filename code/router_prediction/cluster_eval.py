from typing import Tuple

import numpy as np
import torch
from munkres import Munkres
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
)


def _to_numpy(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def cluster_eval(y_true, y_pred) -> Tuple[float, float]:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    true_labels = list(set(y_true.tolist()))
    pred_labels = list(set(y_pred.tolist()))

    if len(true_labels) != len(pred_labels):
        pad = 0
        for label in true_labels:
            if label in pred_labels:
                continue
            y_pred[pad] = label
            pad += 1
        pred_labels = list(set(y_pred.tolist()))

    cost = np.zeros((len(true_labels), len(pred_labels)), dtype=int)
    for i, c1 in enumerate(true_labels):
        true_idx = np.where(y_true == c1)[0]
        for j, c2 in enumerate(pred_labels):
            pred_idx = np.where(y_pred == c2)[0]
            cost[i][j] = len(np.intersect1d(true_idx, pred_idx))

    matcher = Munkres()
    indexes = matcher.compute(cost.__neg__().tolist())

    new_pred = np.zeros_like(y_pred)
    for i, c1 in enumerate(true_labels):
        c2 = pred_labels[indexes[i][1]]
        new_pred[y_pred == c2] = c1

    acc = float((new_pred == y_true).mean())
    f1_macro = f1_score(y_true, new_pred, average="macro")
    return acc, float(f1_macro)


def unsup_eval(y_true, y_pred) -> Tuple[float, float, float, float]:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    acc, f1 = cluster_eval(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")
    ari = adjusted_rand_score(y_true, y_pred)
    return float(acc), float(nmi), float(ari), float(f1)
