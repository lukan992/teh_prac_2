from __future__ import annotations

from src.dataset.dataset_utils import build_segment_arrays, load_trajectory_bundle, save_segment_arrays
from src.utils.seed import set_global_seed


def main() -> None:
    bundle = load_trajectory_bundle()
    set_global_seed(bundle.config["seed"])
    segment_arrays = build_segment_arrays(bundle)
    save_segment_arrays(segment_arrays, bundle.config)
    counts = {split: values["x"].shape[0] for split, values in segment_arrays.items()}
    print(f"Saved segment splits: {counts}")


if __name__ == "__main__":
    main()
