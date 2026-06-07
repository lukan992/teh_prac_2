from __future__ import annotations

import numpy as np
import pandas as pd

from src.methods.common import ClusteringResult, run_density_clustering


def _segment_trajectory(coords: np.ndarray, speeds: np.ndarray, curvature_threshold: float) -> list[np.ndarray]:
    if len(coords) < 3:
        return [np.arange(len(coords))]
    velocity = np.diff(coords, axis=0)
    headings = np.arctan2(velocity[:, 1], velocity[:, 0])
    heading_diff = np.abs(np.diff(headings, prepend=headings[:1]))
    boundaries = np.where(heading_diff > curvature_threshold)[0]
    indices = [0] + boundaries.tolist() + [len(coords) - 1]
    segments = []
    for start, stop in zip(indices[:-1], indices[1:]):
        if stop - start >= 2:
            segments.append(np.arange(start, stop + 1))
    return segments or [np.arange(len(coords))]


def _trajectory_segment_features(frame: pd.DataFrame, curvature_threshold: float) -> dict[str, float]:
    coords = frame[["x", "y", "z"]].to_numpy()
    speeds = frame["speed"].to_numpy()
    segments = _segment_trajectory(coords, speeds, curvature_threshold)
    lengths = []
    directions = []
    mean_speeds = []
    for segment in segments:
        seg_coords = coords[segment]
        delta = seg_coords[-1] - seg_coords[0]
        lengths.append(float(np.linalg.norm(np.diff(seg_coords, axis=0), axis=1).sum()))
        directions.append(float(np.arctan2(delta[1], delta[0])))
        mean_speeds.append(float(speeds[segment].mean()))
    return {
        "segment_count": float(len(segments)),
        "mean_segment_length": float(np.mean(lengths)),
        "std_segment_length": float(np.std(lengths)),
        "direction_entropy": float(np.std(directions)),
        "mean_segment_speed": float(np.mean(mean_speeds)),
        "max_segment_speed": float(np.max(mean_speeds)),
    }


def run_traclus_like(trajectories: pd.DataFrame, config: dict[str, float]) -> ClusteringResult:
    feature_rows = []
    for trajectory_id, frame in trajectories.groupby("trajectory_id"):
        features = _trajectory_segment_features(frame.sort_values("t"), config["curvature_threshold"])
        features["trajectory_id"] = int(trajectory_id)
        feature_rows.append(features)
    summary = pd.DataFrame(feature_rows).sort_values("trajectory_id").reset_index(drop=True)
    feature_names = [column for column in summary.columns if column != "trajectory_id"]
    feature_matrix = summary[feature_names].to_numpy(dtype=float)
    labels, notes = run_density_clustering(
        feature_matrix,
        min_cluster_size=int(config["min_samples"]),
        eps=float(config["eps"]),
    )
    notes.append("Adapted TRACLUS-like implementation based on turn-driven segmentation.")
    return ClusteringResult(labels=labels, features=feature_matrix, feature_names=feature_names, summary=summary, notes=notes)
