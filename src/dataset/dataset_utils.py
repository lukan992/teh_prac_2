from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir, project_root, write_dataframe, write_json

PATTERN_NAMES = [
    "straight_flight",
    "circular_orbit",
    "rectangular_patrol",
    "hover",
    "spiral_climb",
    "zigzag",
    "sharp_turn",
    "descent_approach",
    "random_anomalous",
]

SEGMENT_FEATURES = ["x", "y", "z", "vx", "vy", "vz", "speed", "yaw"]


@dataclass
class DatasetBundle:
    trajectories: pd.DataFrame
    labels: pd.DataFrame
    config: dict[str, Any]


def _resample_polyline(vertices: np.ndarray, length: int) -> np.ndarray:
    segments = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segments)])
    total = cumulative[-1] if cumulative[-1] > 0 else 1.0
    positions = np.linspace(0.0, total, length)
    xs = np.interp(positions, cumulative, vertices[:, 0])
    ys = np.interp(positions, cumulative, vertices[:, 1])
    zs = np.interp(positions, cumulative, vertices[:, 2])
    return np.column_stack([xs, ys, zs])


def _generate_pattern_points(pattern_name: str, length: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    times = np.arange(length) * dt
    x0, y0 = rng.uniform(-50, 50, size=2)
    z0 = rng.uniform(20, 120)

    if pattern_name == "straight_flight":
        direction = rng.normal(size=3)
        direction[2] *= 0.2
        direction /= np.linalg.norm(direction) + 1e-8
        speed = rng.uniform(6.0, 16.0)
        coords = np.column_stack(
            [
                x0 + direction[0] * speed * times,
                y0 + direction[1] * speed * times,
                z0 + direction[2] * speed * times,
            ]
        )
    elif pattern_name == "circular_orbit":
        radius = rng.uniform(18.0, 35.0)
        omega = rng.uniform(0.12, 0.26)
        coords = np.column_stack(
            [
                x0 + radius * np.cos(omega * times),
                y0 + radius * np.sin(omega * times),
                z0 + rng.uniform(-0.5, 0.5) * times,
            ]
        )
    elif pattern_name == "rectangular_patrol":
        width = rng.uniform(25.0, 55.0)
        height = rng.uniform(20.0, 45.0)
        vertices = np.array(
            [
                [x0, y0, z0],
                [x0 + width, y0, z0 + 1],
                [x0 + width, y0 + height, z0 + 1.5],
                [x0, y0 + height, z0 + 0.5],
                [x0, y0, z0],
            ]
        )
        coords = _resample_polyline(vertices, length)
    elif pattern_name == "hover":
        drift = rng.normal(scale=0.4, size=(length, 3)).cumsum(axis=0)
        drift[:, 2] *= 0.2
        coords = np.column_stack(
            [
                np.full(length, x0),
                np.full(length, y0),
                np.full(length, z0),
            ]
        ) + drift
    elif pattern_name == "spiral_climb":
        radius = np.linspace(rng.uniform(8.0, 15.0), rng.uniform(25.0, 35.0), length)
        omega = rng.uniform(0.18, 0.32)
        climb_rate = rng.uniform(0.6, 1.8)
        coords = np.column_stack(
            [
                x0 + radius * np.cos(omega * times),
                y0 + radius * np.sin(omega * times),
                z0 + climb_rate * times,
            ]
        )
    elif pattern_name == "zigzag":
        speed = rng.uniform(6.0, 12.0)
        base_x = x0 + speed * times
        amplitude = rng.uniform(8.0, 18.0)
        frequency = rng.uniform(0.16, 0.28)
        coords = np.column_stack(
            [
                base_x,
                y0 + amplitude * np.sign(np.sin(frequency * times)),
                z0 + rng.uniform(-0.3, 0.3) * times,
            ]
        )
    elif pattern_name == "sharp_turn":
        split_index = length // 2
        first_leg = np.column_stack(
            [
                x0 + np.linspace(0, rng.uniform(25.0, 40.0), split_index),
                np.full(split_index, y0),
                np.full(split_index, z0),
            ]
        )
        second_leg = np.column_stack(
            [
                np.full(length - split_index, first_leg[-1, 0]),
                y0 + np.linspace(0, rng.uniform(25.0, 40.0), length - split_index),
                np.full(length - split_index, z0 + rng.uniform(-6.0, 6.0)),
            ]
        )
        coords = np.vstack([first_leg, second_leg])
    elif pattern_name == "descent_approach":
        distance = rng.uniform(60.0, 110.0)
        approach = np.linspace(0.0, distance, length)
        coords = np.column_stack(
            [
                x0 + approach,
                y0 + rng.uniform(-10.0, 10.0) * np.sin(np.linspace(0, np.pi, length)),
                z0 + np.linspace(rng.uniform(25.0, 40.0), 0.0, length),
            ]
        )
    elif pattern_name == "random_anomalous":
        steps = rng.normal(scale=[4.0, 4.0, 1.4], size=(length, 3))
        steps += rng.normal(scale=[1.4, 1.4, 0.5], size=(length, 3)) * np.sin(times[:, None] * 2.0)
        coords = np.column_stack(
            [
                np.full(length, x0),
                np.full(length, y0),
                np.full(length, z0),
            ]
        ) + np.cumsum(steps, axis=0)
    else:
        raise ValueError(f"Unsupported pattern: {pattern_name}")
    return coords


def _apply_noise(coords: np.ndarray, noise_level: str, config: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    profile = config["noise_levels"][noise_level]
    noisy = coords + rng.normal(scale=profile["position_std"], size=coords.shape)
    drop_probability = profile["drop_probability"]
    if drop_probability > 0:
        mask = rng.random(len(noisy)) >= drop_probability
        mask[0] = True
        mask[-1] = True
        noisy = noisy[mask]
    return noisy


def compute_kinematic_features(coords: np.ndarray, dt: float) -> pd.DataFrame:
    velocities = np.gradient(coords, dt, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    yaw = np.arctan2(velocities[:, 1], velocities[:, 0])
    horizontal_speed = np.linalg.norm(velocities[:, :2], axis=1) + 1e-8
    pitch = np.arctan2(velocities[:, 2], horizontal_speed)
    roll = np.gradient(yaw, dt)
    data = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "vx": velocities[:, 0],
            "vy": velocities[:, 1],
            "vz": velocities[:, 2],
            "speed": speeds,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
        }
    )
    return data


def _assign_splits(count: int, train_fraction: float, val_fraction: float) -> np.ndarray:
    train_boundary = int(count * train_fraction)
    val_boundary = int(count * (train_fraction + val_fraction))
    split = np.array(["test"] * count, dtype=object)
    split[:train_boundary] = "train"
    split[train_boundary:val_boundary] = "val"
    return split


def generate_trajectory_dataset(config: dict[str, Any]) -> DatasetBundle:
    rng = np.random.default_rng(config["seed"])
    sample_rate_hz = config["sample_rate_hz"]
    dt = 1.0 / sample_rate_hz
    trajectory_count = config["trajectory_count"]
    noise_names = list(config["noise_levels"].keys())
    pattern_names = config["patterns"]

    rows: list[pd.DataFrame] = []
    label_rows: list[dict[str, Any]] = []
    split_assignment = _assign_splits(
        trajectory_count,
        config["train_fraction"],
        config["val_fraction"],
    )

    for trajectory_id in range(trajectory_count):
        pattern_id = trajectory_id % len(pattern_names)
        pattern_name = pattern_names[pattern_id]
        noise_level = noise_names[trajectory_id % len(noise_names)]
        length = int(rng.integers(config["trajectory_length_min"], config["trajectory_length_max"] + 1))
        coords = _generate_pattern_points(pattern_name, length, dt, rng)
        coords = _apply_noise(coords, noise_level, config, rng)
        features = compute_kinematic_features(coords, dt)
        features.insert(0, "trajectory_id", trajectory_id)
        features.insert(1, "t", np.arange(len(features)) * dt)
        features["pattern_id"] = pattern_id
        features["noise_level"] = noise_level
        rows.append(features)
        label_rows.append(
            {
                "trajectory_id": trajectory_id,
                "pattern_id": pattern_id,
                "pattern_name": pattern_name,
                "noise_level": noise_level,
                "split": split_assignment[trajectory_id],
                "num_points": len(features),
            }
        )

    trajectories = pd.concat(rows, ignore_index=True)
    labels = pd.DataFrame(label_rows)
    return DatasetBundle(trajectories=trajectories, labels=labels, config=config)


def save_trajectory_bundle(bundle: DatasetBundle, output_dir: str | Path | None = None) -> None:
    target_dir = ensure_dir(output_dir or project_root() / "data" / "trajectories")
    write_dataframe(bundle.trajectories, target_dir / "trajectories.csv")
    write_dataframe(bundle.labels, target_dir / "trajectory_labels.csv")
    write_json(bundle.config, target_dir / "trajectory_config.json")


def load_trajectory_bundle(data_dir: str | Path | None = None) -> DatasetBundle:
    base = Path(data_dir or project_root() / "data" / "trajectories")
    trajectories = pd.read_csv(base / "trajectories.csv")
    labels = pd.read_csv(base / "trajectory_labels.csv")
    from src.utils.io import read_json

    return DatasetBundle(
        trajectories=trajectories,
        labels=labels,
        config=read_json(base / "trajectory_config.json"),
    )


def _future_targets(window: np.ndarray, full_sequence: np.ndarray, start_index: int, horizons: list[int]) -> np.ndarray:
    targets: list[np.ndarray] = []
    last_index = start_index + len(window) - 1
    for horizon in horizons:
        target_index = min(last_index + horizon, len(full_sequence) - 1)
        targets.append(full_sequence[target_index])
    return np.concatenate(targets, axis=0)


def build_segment_arrays(bundle: DatasetBundle) -> dict[str, dict[str, np.ndarray]]:
    config = bundle.config
    segment_length = config["segment_length"]
    stride = config["segment_stride"]
    horizons = config["forecast_horizons_steps"]
    grouped = bundle.trajectories.groupby("trajectory_id")
    split_lookup = bundle.labels.set_index("trajectory_id")["split"].to_dict()

    data_by_split: dict[str, dict[str, list[np.ndarray]]] = {
        "train": {"x": [], "y": [], "targets": [], "trajectory_id": []},
        "val": {"x": [], "y": [], "targets": [], "trajectory_id": []},
        "test": {"x": [], "y": [], "targets": [], "trajectory_id": []},
    }

    for trajectory_id, frame in grouped:
        frame = frame.sort_values("t").reset_index(drop=True)
        features = frame[SEGMENT_FEATURES].to_numpy(dtype=np.float32)
        full_coords = frame[["x", "y", "z"]].to_numpy(dtype=np.float32)
        label = int(frame["pattern_id"].iloc[0])
        split_name = split_lookup[int(trajectory_id)]
        if len(frame) < segment_length:
            continue
        for start in range(0, len(frame) - segment_length + 1, stride):
            stop = start + segment_length
            window = features[start:stop]
            future = _future_targets(window, full_coords, start, horizons)
            data_by_split[split_name]["x"].append(window)
            data_by_split[split_name]["y"].append(np.array(label, dtype=np.int64))
            data_by_split[split_name]["targets"].append(future.astype(np.float32))
            data_by_split[split_name]["trajectory_id"].append(np.array(trajectory_id, dtype=np.int64))

    arrays: dict[str, dict[str, np.ndarray]] = {}
    for split_name, values in data_by_split.items():
        arrays[split_name] = {
            "x": np.stack(values["x"]).astype(np.float32),
            "y": np.stack(values["y"]).astype(np.int64),
            "targets": np.stack(values["targets"]).astype(np.float32),
            "trajectory_id": np.stack(values["trajectory_id"]).astype(np.int64),
        }
    return arrays


def save_segment_arrays(segment_arrays: dict[str, dict[str, np.ndarray]], config: dict[str, Any]) -> None:
    target_dir = ensure_dir(project_root() / "data" / "segments")
    for split_name, arrays in segment_arrays.items():
        np.savez_compressed(target_dir / f"segments_{split_name}.npz", **arrays)
    metadata = {
        "segment_length": config["segment_length"],
        "segment_stride": config["segment_stride"],
        "segment_features": SEGMENT_FEATURES,
        "forecast_horizons_steps": config["forecast_horizons_steps"],
    }
    write_json(metadata, target_dir / "segment_metadata.json")


def load_segment_split(split_name: str) -> dict[str, np.ndarray]:
    base = project_root() / "data" / "segments" / f"segments_{split_name}.npz"
    data = np.load(base)
    return {key: data[key] for key in data.files}


def _make_background_points(center: np.ndarray, difficulty: str, config: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    count = config["pointcloud_background_points"]
    scale = {"clean": 30.0, "medium": 40.0, "hard": 55.0}[difficulty]
    background = rng.uniform(low=center - scale, high=center + scale, size=(count, 3))
    background[:, 2] = np.clip(background[:, 2], 0.0, None)
    return background


def _make_false_clusters(center: np.ndarray, difficulty: str, config: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    profile = config["noise_levels"][difficulty]
    clusters: list[np.ndarray] = []
    for _ in range(profile["false_clusters"]):
        false_center = center + rng.uniform(-25, 25, size=3)
        false_center[2] = max(2.0, false_center[2])
        cluster_size = int(rng.integers(6, 18))
        clusters.append(false_center + rng.normal(scale=1.4 + profile["position_std"], size=(cluster_size, 3)))
    return np.vstack(clusters) if clusters else np.empty((0, 3))


def build_pointcloud_dataset(bundle: DatasetBundle) -> dict[str, pd.DataFrame]:
    config = bundle.config
    rng = np.random.default_rng(config["seed"] + 11)
    frames_per_trajectory = config["pointcloud_frames_per_trajectory"]
    datasets: dict[str, list[dict[str, Any]]] = {level: [] for level in config["noise_levels"].keys()}

    for trajectory_id, frame in bundle.trajectories.groupby("trajectory_id"):
        frame = frame.sort_values("t").reset_index(drop=True)
        if len(frame) < 4:
            continue
        sampled_indices = np.linspace(0, len(frame) - 1, frames_per_trajectory).astype(int)
        sampled = frame.iloc[sampled_indices]
        object_id = int(frame["pattern_id"].iloc[0])
        for difficulty in datasets.keys():
            missing_probability = config["missing_point_probability"][difficulty]
            for local_frame_id, row in enumerate(sampled.itertuples(index=False)):
                center = np.array([row.x, row.y, row.z], dtype=float)
                uav_count = {"clean": rng.integers(20, 31), "medium": rng.integers(10, 22), "hard": rng.integers(5, 13)}[
                    difficulty
                ]
                if rng.random() < missing_probability:
                    uav_count = max(1, uav_count // 3)
                uav_points = center + rng.normal(
                    scale=config["noise_levels"][difficulty]["position_std"] + 0.25,
                    size=(uav_count, 3),
                )
                background = _make_background_points(center, difficulty, config, rng)
                false_clusters = _make_false_clusters(center, difficulty, config, rng)
                point_sets = [
                    (uav_points, 1),
                    (background, 0),
                    (false_clusters, 0),
                ]
                point_index = 0
                for points, is_uav in point_sets:
                    for point in points:
                        datasets[difficulty].append(
                            {
                                "frame_id": local_frame_id,
                                "trajectory_id": int(trajectory_id),
                                "t": float(row.t),
                                "x": float(point[0]),
                                "y": float(point[1]),
                                "z": float(point[2]),
                                "intensity": float(rng.uniform(0.2, 1.0) if is_uav else rng.uniform(0.0, 0.7)),
                                "is_uav_point": is_uav,
                                "true_uav_x": float(center[0]),
                                "true_uav_y": float(center[1]),
                                "true_uav_z": float(center[2]),
                                "object_id": object_id,
                                "point_id": point_index,
                            }
                        )
                        point_index += 1

    return {difficulty: pd.DataFrame(rows) for difficulty, rows in datasets.items()}


def save_pointcloud_dataset(pointcloud_frames: dict[str, pd.DataFrame], config: dict[str, Any]) -> None:
    base = ensure_dir(project_root() / "data" / "pointcloud")
    metadata = {"difficulties": list(pointcloud_frames.keys()), "config": config}
    for difficulty, df in pointcloud_frames.items():
        difficulty_dir = ensure_dir(base / difficulty)
        write_dataframe(df, difficulty_dir / "pointcloud_frames.csv")
    write_json(metadata, base / "pointcloud_metadata.json")
