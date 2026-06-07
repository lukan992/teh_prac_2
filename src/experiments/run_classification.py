from __future__ import annotations

import argparse

import pandas as pd

from src.dataset.dataset_utils import load_segment_split
from src.experiments.helpers import mirror_figure, mirror_table, results_dirs
from src.metrics.classification_metrics import evaluate_classification
from src.metrics.resource_metrics import build_resource_metrics
from src.methods.cnn_segment_classifier import train_cnn_classifier
from src.utils.config import load_config
from src.utils.hardware import ResourceMonitor
from src.utils.io import write_dataframe
from src.utils.logging_utils import build_run_log, save_run_log
from src.utils.seed import set_global_seed
from src.visualization.plot_confusion_matrix import plot_confusion_matrix


def run(output_root: str | None = None) -> pd.DataFrame:
    config = load_config("classification_config.yaml")
    set_global_seed(int(config["seed"]))
    train_split = load_segment_split("train")
    val_split = load_segment_split("val")
    test_split = load_segment_split("test")
    dirs = results_dirs(output_root)
    monitor = ResourceMonitor(device="cuda")
    monitor.start()
    output = train_cnn_classifier(train_split, val_split, test_split, config)
    resource_usage = monitor.stop()
    metrics = evaluate_classification(output.y_true, output.y_pred)
    row = {"method": "cnn_segment_classifier", **{key: value for key, value in metrics.items() if key != "confusion_matrix"}}
    results = pd.DataFrame([row])
    resources = pd.DataFrame(
        [
            build_resource_metrics(
                "cnn_segment_classifier",
                resource_usage,
                parameter_count=output.parameter_count,
                mean_inference_time=output.mean_inference_time,
            )
        ]
    )
    write_dataframe(results, dirs["metrics"] / "classification_results.csv")
    write_dataframe(resources, dirs["metrics"] / "resource_usage_classification.csv")
    figure_path = plot_confusion_matrix(metrics["confusion_matrix"], dirs["figures"] / "cnn_confusion_matrix.png")
    mirror_figure(figure_path, dirs["report_figures"])
    mirror_table(results, "classification_results_table.csv", dirs["report_tables"])
    save_run_log(
        dirs["logs"] / "cnn_run.json",
        build_run_log("cnn_segment_classifier", config, metrics, resource_usage, [f"model_path={output.model_path}"]),
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
