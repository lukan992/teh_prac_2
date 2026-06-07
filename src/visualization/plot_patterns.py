from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import ensure_dir


def plot_pattern_examples(trajectories: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    pattern_names = trajectories.groupby("pattern_id")["trajectory_id"].first().to_dict()
    fig = plt.figure(figsize=(14, 12))
    for index, (pattern_id, trajectory_id) in enumerate(pattern_names.items(), start=1):
        ax = fig.add_subplot(3, 3, index, projection="3d")
        frame = trajectories[trajectories["trajectory_id"] == trajectory_id].sort_values("t")
        ax.plot(frame["x"], frame["y"], frame["z"], linewidth=1.8)
        ax.set_title(f"Pattern {pattern_id}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target


def plot_experiment_scheme(path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    nodes = [
        ("Synthetic\ndata", 0.08),
        ("Trajectory /\nsegment /\npointcloud views", 0.3),
        ("Methods", 0.54),
        ("Metrics", 0.74),
        ("Comparison", 0.9),
    ]
    for label, xpos in nodes:
        ax.text(
            xpos,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#dfefff", "edgecolor": "#1f4b99"},
        )
    for start, end in zip(nodes[:-1], nodes[1:]):
        ax.annotate("", xy=(end[1] - 0.06, 0.5), xytext=(start[1] + 0.06, 0.5), arrowprops={"arrowstyle": "->", "lw": 2})
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target


def plot_trajectory_segments(trajectory: pd.DataFrame, segment_length: int, stride: int, path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    fig, ax = plt.subplots(figsize=(10, 6))
    frame = trajectory.sort_values("t").reset_index(drop=True)
    ax.plot(frame["x"], frame["y"], color="black", linewidth=1.5, alpha=0.8, label="trajectory")
    colors = plt.cm.tab10.colors
    segment_index = 0
    for start in range(0, len(frame) - segment_length + 1, stride):
        stop = start + segment_length
        window = frame.iloc[start:stop]
        ax.plot(window["x"], window["y"], color=colors[segment_index % len(colors)], linewidth=2.2)
        segment_index += 1
        if segment_index >= 6:
            break
    ax.set_title("Trajectory windows / segments")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target
