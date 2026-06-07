from __future__ import annotations

import numpy as np

from src.methods.common import ClusteringResult, run_kmeans


def _segment_feature_vector(segment: np.ndarray) -> np.ndarray:
    coords = segment[:, :3]
    velocity = segment[:, 3:6]
    speed = segment[:, 6]
    yaw = segment[:, 7]
    return np.array(
        [
            coords[:, 0].mean(),
            coords[:, 1].mean(),
            coords[:, 2].mean(),
            np.ptp(coords[:, 2]),
            np.linalg.norm(np.diff(coords, axis=0), axis=1).sum(),
            speed.mean(),
            speed.std(),
            np.abs(np.diff(yaw)).mean(),
            np.linalg.norm(velocity, axis=1).mean(),
        ],
        dtype=float,
    )


def run_spatiotemporal_clustering(segment_split: dict[str, np.ndarray], config: dict[str, float]) -> ClusteringResult:
    x = segment_split["x"]
    features = np.vstack([_segment_feature_vector(segment) for segment in x])
    labels = run_kmeans(features, n_clusters=int(config["n_clusters"]))
    summary = None
    import pandas as pd

    summary = pd.DataFrame({"segment_id": np.arange(len(x)), "trajectory_id": segment_split["trajectory_id"]})
    notes = ["Adapted spatio-temporal segment clustering on handcrafted short-window descriptors."]
    return ClusteringResult(
        labels=labels,
        features=features,
        feature_names=[
            "mean_x",
            "mean_y",
            "mean_z",
            "z_range",
            "path_length",
            "mean_speed",
            "speed_std",
            "mean_yaw_change",
            "mean_velocity_norm",
        ],
        summary=summary,
        notes=notes,
    )
