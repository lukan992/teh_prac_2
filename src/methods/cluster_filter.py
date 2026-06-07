from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def _voxel_score(points: np.ndarray, voxel_size: float) -> float:
    voxels = np.floor(points / max(voxel_size, 1e-8)).astype(int)
    unique_voxels = np.unique(voxels, axis=0)
    return float(len(points) / max(len(unique_voxels), 1))


def run_cluster_filter(frame_df: pd.DataFrame, eps: float, min_samples: int, voxel_size: float) -> np.ndarray:
    predictions: list[np.ndarray] = []
    previous_center: np.ndarray | None = None
    for _, frame in frame_df.groupby("frame_id"):
        filtered = frame[(frame["z"] > 1.0) & (frame["intensity"] > 0.05)]
        points = filtered[["x", "y", "z"]].to_numpy()
        if len(points) < min_samples:
            predictions.append(np.array([np.nan, np.nan, np.nan]))
            continue
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
        best_cluster = None
        best_score = -np.inf
        for label in np.unique(labels):
            if label == -1:
                continue
            cluster_points = points[labels == label]
            center = cluster_points.mean(axis=0)
            temporal_penalty = 0.0 if previous_center is None else np.linalg.norm(center - previous_center)
            score = _voxel_score(cluster_points, voxel_size) - 0.15 * temporal_penalty
            if score > best_score:
                best_score = score
                best_cluster = cluster_points
        if best_cluster is None:
            predictions.append(np.array([np.nan, np.nan, np.nan]))
            continue
        center = best_cluster.mean(axis=0)
        if previous_center is not None:
            center = 0.7 * center + 0.3 * previous_center
        previous_center = center
        predictions.append(center)
    return np.vstack(predictions)
