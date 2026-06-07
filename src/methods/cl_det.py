from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def run_cl_det_tracking(frame_df: pd.DataFrame, eps: float, min_samples: int, alpha: float) -> np.ndarray:
    predictions: list[np.ndarray] = []
    previous_center: np.ndarray | None = None
    velocity = np.zeros(3, dtype=float)
    for _, frame in frame_df.groupby("frame_id"):
        points = frame[["x", "y", "z"]].to_numpy()
        if len(points) < min_samples:
            if previous_center is None:
                predictions.append(np.array([np.nan, np.nan, np.nan]))
            else:
                previous_center = previous_center + velocity
                predictions.append(previous_center.copy())
            continue
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
        best_center = None
        best_distance = np.inf
        for label in np.unique(labels):
            if label == -1:
                continue
            cluster_points = points[labels == label]
            center = cluster_points.mean(axis=0)
            distance = -len(cluster_points) if previous_center is None else np.linalg.norm(center - previous_center)
            if distance < best_distance:
                best_distance = distance
                best_center = center
        if best_center is None:
            predictions.append(np.array([np.nan, np.nan, np.nan]))
            continue
        if previous_center is not None:
            velocity = alpha * (best_center - previous_center) + (1 - alpha) * velocity
            best_center = previous_center + velocity
        previous_center = best_center
        predictions.append(best_center.copy())
    return np.vstack(predictions)
