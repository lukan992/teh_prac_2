from __future__ import annotations

import numpy as np
import pandas as pd

from src.methods.common import ClusteringResult, run_kmeans


def _vector_field_descriptor(frame: pd.DataFrame, grid_size: int) -> np.ndarray:
    coords = frame[["x", "y", "z"]].to_numpy()
    velocity = frame[["vx", "vy", "vz"]].to_numpy()
    xy = coords[:, :2]
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    normalized = (xy - mins) / span
    bins = np.clip((normalized * grid_size).astype(int), 0, grid_size - 1)
    descriptor = np.zeros((grid_size, grid_size, 3), dtype=float)
    counts = np.zeros((grid_size, grid_size, 1), dtype=float)
    for (ix, iy), speed in zip(bins, velocity):
        descriptor[ix, iy] += speed
        counts[ix, iy] += 1
    descriptor = descriptor / np.maximum(counts, 1.0)
    return descriptor.reshape(-1)


def run_vector_field_kmeans(trajectories: pd.DataFrame, config: dict[str, float]) -> ClusteringResult:
    rows = []
    features = []
    for trajectory_id, frame in trajectories.groupby("trajectory_id"):
        descriptor = _vector_field_descriptor(frame.sort_values("t"), int(config["grid_size"]))
        rows.append({"trajectory_id": int(trajectory_id)})
        features.append(descriptor)
    summary = pd.DataFrame(rows).sort_values("trajectory_id").reset_index(drop=True)
    feature_matrix = np.vstack(features)
    labels = run_kmeans(feature_matrix, n_clusters=int(config["n_clusters"]))
    notes = ["Adapted Vector Field k-Means using grid-based velocity field descriptors."]
    return ClusteringResult(
        labels=labels,
        features=feature_matrix,
        feature_names=[f"vf_{index}" for index in range(feature_matrix.shape[1])],
        summary=summary,
        notes=notes,
    )
