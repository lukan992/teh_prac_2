from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def _cluster_score(points: np.ndarray, previous_center: np.ndarray | None) -> float:
    density = len(points)
    center = points.mean(axis=0)
    continuity = 0.0 if previous_center is None else np.linalg.norm(center - previous_center)
    compactness = np.mean(np.linalg.norm(points - center, axis=1))
    return density - 0.2 * continuity - compactness


def run_sparse_pointcloud_estimation(frame_df: pd.DataFrame, eps: float, min_samples: int) -> np.ndarray:
    predictions: list[np.ndarray] = []
    previous_center: np.ndarray | None = None
    for _, frame in frame_df.groupby("frame_id"):
        filtered = frame[frame["z"] > frame["z"].quantile(0.2)]
        points = filtered[["x", "y", "z"]].to_numpy()
        if len(points) < min_samples:
            predictions.append(np.array([np.nan, np.nan, np.nan]))
            continue
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
        best_points = None
        best_score = -np.inf
        for label in np.unique(labels):
            if label == -1:
                continue
            cluster_points = points[labels == label]
            score = _cluster_score(cluster_points, previous_center)
            if score > best_score:
                best_score = score
                best_points = cluster_points
        if best_points is None:
            predictions.append(np.array([np.nan, np.nan, np.nan]))
            continue
        center = best_points.mean(axis=0)
        if previous_center is not None:
            center = 0.65 * center + 0.35 * previous_center
        previous_center = center
        predictions.append(center)
    return np.vstack(predictions)
