from __future__ import annotations

import argparse

import pandas as pd

from src.dataset.dataset_utils import load_trajectory_bundle
from src.experiments.helpers import mirror_figure, mirror_table, results_dirs
from src.metrics.clustering_metrics import evaluate_clustering
from src.metrics.resource_metrics import build_resource_metrics
from src.methods.resnet_hdbscan import train_resnet_embedding_clusterer
from src.utils.config import load_config
from src.utils.hardware import ResourceMonitor
from src.utils.io import write_dataframe
from src.utils.logging_utils import build_run_log, save_run_log
from src.utils.seed import set_global_seed
from src.visualization.plot_embedding import plot_embedding_space


def run(output_root: str | None = None) -> pd.DataFrame:
    config = load_config("resnet_hdbscan_config.yaml")
    set_global_seed(int(config["seed"]))
    bundle = load_trajectory_bundle()
    dirs = results_dirs(output_root)
    monitor = ResourceMonitor(device="cuda")
    monitor.start()
    output = train_resnet_embedding_clusterer(bundle.trajectories, bundle.labels, config)
    resource_usage = monitor.stop()
    rows = []
    for name, labels in output.results.items():
        metrics = evaluate_clustering(output.y_true, labels, output.embeddings if "resnet" in name else None)
        rows.append({"method": name, **metrics})
    results = pd.DataFrame(rows)
    resources = pd.DataFrame(
        [
            build_resource_metrics(
                "resnet_hdbscan_family",
                resource_usage,
                parameter_count=output.parameter_count,
                mean_inference_time=output.mean_inference_time,
            )
        ]
    )
    write_dataframe(results, dirs["metrics"] / "resnet_hdbscan_results.csv")
    write_dataframe(resources, dirs["metrics"] / "resource_usage_resnet.csv")
    figure_path = plot_embedding_space(output.embeddings, output.y_true, dirs["figures"] / "resnet_embedding_umap.png")
    mirror_figure(figure_path, dirs["report_figures"])
    mirror_table(results, "resnet_hdbscan_results_table.csv", dirs["report_tables"])
    save_run_log(
        dirs["logs"] / "resnet_hdbscan_run.json",
        build_run_log("resnet_hdbscan_family", config, results.to_dict(orient="records")[0], resource_usage, output.notes),
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
