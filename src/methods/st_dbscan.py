from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from src.methods.common import ClusteringResult, summarize_trajectory_frame


def run_st_dbscan(trajectories: pd.DataFrame, config: dict[str, float]) -> ClusteringResult:
    feature_rows = []
    for trajectory_id, frame in trajectories.groupby("trajectory_id"):
        stats = summarize_trajectory_frame(frame.sort_values("t"))
        stats["trajectory_id"] = int(trajectory_id)
        feature_rows.append(stats)
    summary = pd.DataFrame(feature_rows).sort_values("trajectory_id").reset_index(drop=True)
    feature_names = [column for column in summary.columns if column != "trajectory_id"]
    spatial_features = summary[["mean_x", "mean_y", "mean_z", "path_length"]].to_numpy(dtype=float)
    temporal_features = summary[["duration", "altitude_delta", "turn_energy"]].to_numpy(dtype=float)
    scaled_spatial = StandardScaler().fit_transform(spatial_features)
    scaled_temporal = StandardScaler().fit_transform(temporal_features)
    combined = np.concatenate(
        [
            scaled_spatial / max(float(config["eps_spatial"]), 1e-8),
            scaled_temporal / max(float(config["eps_temporal"]), 1e-8),
        ],
        axis=1,
    )
    labels = DBSCAN(eps=1.0, min_samples=int(config["min_samples"])).fit_predict(combined).astype(int)
    notes = ["Adapted ST-DBSCAN using trajectory-level spatial and temporal descriptors."]
    return ClusteringResult(labels=labels, features=combined, feature_names=feature_names, summary=summary, notes=notes)
