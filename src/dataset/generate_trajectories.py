from __future__ import annotations

from src.dataset.dataset_utils import generate_trajectory_dataset, save_trajectory_bundle
from src.utils.config import load_config
from src.utils.seed import set_global_seed


def main() -> None:
    config = load_config("dataset_config.yaml")
    set_global_seed(config["seed"])
    bundle = generate_trajectory_dataset(config)
    save_trajectory_bundle(bundle)
    print(f"Saved {len(bundle.labels)} trajectories to data/trajectories/")


if __name__ == "__main__":
    main()
