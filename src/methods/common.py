from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan  # type: ignore
except ImportError:  # pragma: no cover
    hdbscan = None


@dataclass
class ClusteringResult:
    labels: np.ndarray
    features: np.ndarray
    feature_names: list[str]
    summary: pd.DataFrame
    notes: list[str]


def run_density_clustering(
    features: np.ndarray,
    min_cluster_size: int = 8,
    eps: float = 0.8,
    prefer_hdbscan: bool = True,
) -> tuple[np.ndarray, list[str]]:
    notes: list[str] = []
    scaled = StandardScaler().fit_transform(features)
    if prefer_hdbscan and hdbscan is not None:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(scaled)
        notes.append("Used HDBSCAN for density clustering.")
    else:
        clusterer = DBSCAN(eps=eps, min_samples=min_cluster_size)
        labels = clusterer.fit_predict(scaled)
        notes.append("Used DBSCAN fallback for density clustering.")
    return labels.astype(int), notes


def run_kmeans(features: np.ndarray, n_clusters: int, seed: int = 42) -> np.ndarray:
    scaled = StandardScaler().fit_transform(features)
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(scaled).astype(int)


def summarize_trajectory_frame(frame: pd.DataFrame) -> dict[str, float]:
    coords = frame[["x", "y", "z"]].to_numpy()
    velocity = frame[["vx", "vy", "vz"]].to_numpy()
    diffs = np.diff(coords, axis=0)
    turn = np.linalg.norm(np.diff(velocity, axis=0), axis=1).mean() if len(velocity) > 2 else 0.0
    return {
        "mean_x": float(coords[:, 0].mean()),
        "mean_y": float(coords[:, 1].mean()),
        "mean_z": float(coords[:, 2].mean()),
        "std_x": float(coords[:, 0].std()),
        "std_y": float(coords[:, 1].std()),
        "std_z": float(coords[:, 2].std()),
        "path_length": float(np.linalg.norm(diffs, axis=1).sum()) if len(diffs) else 0.0,
        "mean_speed": float(frame["speed"].mean()),
        "std_speed": float(frame["speed"].std()),
        "altitude_delta": float(coords[-1, 2] - coords[0, 2]),
        "yaw_range": float(frame["yaw"].max() - frame["yaw"].min()),
        "turn_energy": float(turn),
        "duration": float(frame["t"].iloc[-1] - frame["t"].iloc[0]),
    }
