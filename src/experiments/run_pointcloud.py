from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.experiments.helpers import mirror_figure, mirror_table, results_dirs
from src.metrics.resource_metrics import build_resource_metrics
from src.metrics.tracking_metrics import evaluate_tracking
from src.methods.cl_det import run_cl_det_tracking
from src.methods.cluster_filter import run_cluster_filter
from src.methods.sparse_pointcloud_estimation import run_sparse_pointcloud_estimation
from src.utils.config import load_config
from src.utils.hardware import ResourceMonitor
from src.utils.io import write_dataframe
from src.utils.logging_utils import build_run_log, save_run_log
from src.utils.seed import set_global_seed
from src.visualization.plot_pointcloud import plot_pointcloud_levels


def _truth_centers(frame_df: pd.DataFrame) -> np.ndarray:
    truth = (
        frame_df.groupby("frame_id")[["true_uav_x", "true_uav_y", "true_uav_z"]]
        .first()
        .to_numpy(dtype=float)
    )
    return truth


def run(output_root: str | None = None) -> pd.DataFrame:
    config = load_config("pointcloud_config.yaml")
    set_global_seed(int(config["seed"]))
    dirs = results_dirs(output_root)
    pointcloud_base = Path("data") / "pointcloud"
    levels = {}
    for difficulty in ["clean", "medium", "hard"]:
        levels[difficulty] = pd.read_csv(pointcloud_base / difficulty / "pointcloud_frames.csv")

    figure_path = plot_pointcloud_levels(levels, dirs["figures"] / "pointcloud_clean_medium_hard.png")
    mirror_figure(figure_path, dirs["report_figures"])

    rows = []
    resource_rows = []
    method_functions = {
        "sparse_pointcloud": lambda frame: run_sparse_pointcloud_estimation(frame, config["dbscan_eps"], config["dbscan_min_samples"]),
        "cluster_filter": lambda frame: run_cluster_filter(
            frame,
            config["dbscan_eps"],
            config["dbscan_min_samples"],
            config["cluster_filter_voxel_size"],
        ),
        "cl_det": lambda frame: run_cl_det_tracking(frame, config["dbscan_eps"], config["dbscan_min_samples"], config["smoothing_alpha"]),
    }

    for method_name, method_fn in method_functions.items():
        metric_rows = []
        method_logs = []
        for difficulty, frame_df in levels.items():
            start_time = time.perf_counter()
            monitor = ResourceMonitor()
            monitor.start()
            predictions = method_fn(frame_df)
            resource_usage = monitor.stop()
            runtime = time.perf_counter() - start_time
            metrics = evaluate_tracking(_truth_centers(frame_df), predictions, runtime)
            metric_rows.append({"method": method_name, "difficulty": difficulty, **metrics})
            resource_rows.append(build_resource_metrics(f"{method_name}_{difficulty}", resource_usage))
            method_logs.append({"difficulty": difficulty, "metrics": metrics, "resource_usage": resource_usage})
        save_run_log(dirs["logs"] / f"{method_name}_run.json", build_run_log(method_name, config, {"runs": method_logs}, {}))
        method_df = pd.DataFrame(metric_rows)
        rows.append(method_df)

    results = pd.concat(rows, ignore_index=True)
    resources = pd.DataFrame(resource_rows)
    write_dataframe(results, dirs["metrics"] / "pointcloud_results.csv")
    write_dataframe(resources, dirs["metrics"] / "resource_usage_pointcloud.csv")
    medium = results[results["difficulty"] == "medium"]
    rmse_table = medium[["method", "position_rmse"]]
    ax = rmse_table.plot(x="method", y="position_rmse", kind="bar", figsize=(8, 5), legend=False, title="Point cloud RMSE")
    ax.set_ylabel("RMSE")
    ax.figure.tight_layout()
    rmse_path = dirs["figures"] / "pointcloud_rmse.png"
    ax.figure.savefig(rmse_path, dpi=180)
    import matplotlib.pyplot as plt

    plt.close(ax.figure)
    mirror_figure(rmse_path, dirs["report_figures"])
    mirror_table(results, "pointcloud_results_table.csv", dirs["report_tables"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
