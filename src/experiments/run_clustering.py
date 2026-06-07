from __future__ import annotations

import argparse

import pandas as pd

from src.dataset.dataset_utils import load_segment_split, load_trajectory_bundle
from src.experiments.helpers import mirror_figure, mirror_table, results_dirs, serialize_secondary_metrics
from src.metrics.clustering_metrics import evaluate_clustering
from src.metrics.resource_metrics import build_resource_metrics
from src.methods.spatiotemporal_clustering import run_spatiotemporal_clustering
from src.methods.st_dbscan import run_st_dbscan
from src.methods.traclus import run_traclus_like
from src.methods.vector_field_kmeans import run_vector_field_kmeans
from src.utils.config import load_config
from src.utils.hardware import ResourceMonitor
from src.utils.io import write_dataframe
from src.utils.logging_utils import build_run_log, save_run_log
from src.utils.seed import set_global_seed
from src.visualization.plot_clustering import plot_clustering_comparison


def run(output_root: str | None = None) -> pd.DataFrame:
    config = load_config("clustering_config.yaml")
    set_global_seed(int(config["seed"]))
    bundle = load_trajectory_bundle()
    trajectories = bundle.trajectories[bundle.trajectories["noise_level"] == config["noise_level"]]
    truth = (
        trajectories.groupby("trajectory_id")["pattern_id"].first().sort_index().to_numpy(dtype=int)
    )
    runners = {
        "traclus": lambda: run_traclus_like(trajectories, config["methods"]["traclus"]),
        "st_dbscan": lambda: run_st_dbscan(trajectories, config["methods"]["st_dbscan"]),
        "vector_field_kmeans": lambda: run_vector_field_kmeans(trajectories, config["methods"]["vector_field_kmeans"]),
    }
    segment_split = load_segment_split("test")
    runners["spatiotemporal_clustering"] = lambda: run_spatiotemporal_clustering(
        segment_split,
        config["methods"]["spatiotemporal_clustering"],
    )

    dirs = results_dirs(output_root)
    rows = []
    resource_rows = []
    for name, runner in runners.items():
        monitor = ResourceMonitor()
        monitor.start()
        result = runner()
        resource_usage = monitor.stop()
        local_truth = truth if name != "spatiotemporal_clustering" else segment_split["y"]
        metrics = evaluate_clustering(local_truth, result.labels, result.features)
        metric_row = {"method": name, **metrics}
        rows.append(metric_row)
        resource_rows.append(build_resource_metrics(name, resource_usage))
        log_path = dirs["logs"] / f"{name}_run.json"
        save_run_log(log_path, build_run_log(name, config, metrics, resource_usage, result.notes))

    results = pd.DataFrame(rows)
    resources = pd.DataFrame(resource_rows)
    write_dataframe(results, dirs["metrics"] / "clustering_results.csv")
    write_dataframe(resources, dirs["metrics"] / "resource_usage_clustering.csv")
    figure_path = plot_clustering_comparison(results, dirs["figures"] / "clustering_comparison.png")
    mirror_figure(figure_path, dirs["report_figures"])
    mirror_table(results, "clustering_results_table.csv", dirs["report_tables"])

    comparison_rows = []
    for row in rows:
        comparison_rows.append(
            {
                "method": row["method"],
                "source": "adapted literature method",
                "input_type": "trajectory" if row["method"] != "spatiotemporal_clustering" else "segment",
                "task_type": "clustering",
                "main_metric": "ari",
                "main_metric_value": row["ari"],
                "secondary_metrics": serialize_secondary_metrics(row, {"method", "ari"}),
            }
        )
    write_dataframe(pd.DataFrame(comparison_rows), dirs["tables"] / "clustering_comparison_partial.csv")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
