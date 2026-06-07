from __future__ import annotations

from src.dataset.dataset_utils import build_pointcloud_dataset, load_trajectory_bundle, save_pointcloud_dataset
from src.utils.seed import set_global_seed


def main() -> None:
    bundle = load_trajectory_bundle()
    set_global_seed(bundle.config["seed"])
    pointcloud_frames = build_pointcloud_dataset(bundle)
    save_pointcloud_dataset(pointcloud_frames, bundle.config)
    counts = {difficulty: len(df) for difficulty, df in pointcloud_frames.items()}
    print(f"Saved pointcloud frames: {counts}")


if __name__ == "__main__":
    main()
