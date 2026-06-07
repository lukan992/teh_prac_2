from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import ensure_dir


def plot_pointcloud_levels(pointcloud_by_level: dict[str, pd.DataFrame], path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    fig = plt.figure(figsize=(14, 4))
    for index, (difficulty, frame_df) in enumerate(pointcloud_by_level.items(), start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        first_frame = frame_df[frame_df["frame_id"] == frame_df["frame_id"].min()]
        ax.scatter(first_frame["x"], first_frame["y"], first_frame["z"], c=first_frame["is_uav_point"], s=6, cmap="coolwarm")
        ax.set_title(difficulty)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target
