from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(y_pred)
    total = 0
    for label in labels:
        mask = y_pred == label
        if not np.any(mask):
            continue
        counts = np.bincount(y_true[mask].astype(int))
        total += counts.max()
    return float(total / len(y_true))


def cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pred_labels = np.unique(y_pred)
    true_labels = np.unique(y_true)
    matrix = np.zeros((len(pred_labels), len(true_labels)), dtype=int)
    for i, pred_label in enumerate(pred_labels):
        for j, true_label in enumerate(true_labels):
            matrix[i, j] = int(np.sum((y_pred == pred_label) & (y_true == true_label)))
    row_ind, col_ind = linear_sum_assignment(matrix.max() - matrix)
    matched = matrix[row_ind, col_ind].sum()
    return float(matched / len(y_true))


def matched_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pred_labels = np.unique(y_pred)
    true_labels = np.unique(y_true)
    matrix = np.zeros((len(pred_labels), len(true_labels)), dtype=int)
    for i, pred_label in enumerate(pred_labels):
        for j, true_label in enumerate(true_labels):
            matrix[i, j] = int(np.sum((y_pred == pred_label) & (y_true == true_label)))
    row_ind, col_ind = linear_sum_assignment(matrix.max() - matrix)
    mapping = {pred_labels[row]: true_labels[col] for row, col in zip(row_ind, col_ind)}
    mapped = np.array([mapping.get(label, -1) for label in y_pred])
    return float(f1_score(y_true, mapped, average="macro", zero_division=0))


def noise_ratio(y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred == -1))


def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray, features: np.ndarray | None = None) -> dict[str, float]:
    scores = {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "purity": purity_score(y_true, y_pred),
        "cluster_accuracy": cluster_accuracy(y_true, y_pred),
        "macro_f1": matched_macro_f1(y_true, y_pred),
        "noise_ratio": noise_ratio(y_pred),
    }
    valid_labels = np.unique(y_pred[y_pred >= 0])
    if features is not None and len(valid_labels) > 1:
        scores["silhouette"] = float(silhouette_score(features, y_pred))
    else:
        scores["silhouette"] = float("nan")
    return scores
