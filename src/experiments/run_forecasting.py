from __future__ import annotations

import argparse

import pandas as pd

from src.dataset.dataset_utils import load_segment_split
from src.experiments.helpers import mirror_figure, mirror_table, results_dirs
from src.metrics.forecasting_metrics import evaluate_forecasting
from src.metrics.resource_metrics import build_resource_metrics
from src.methods.lstm_predictor import train_forecasting_models
from src.utils.config import load_config
from src.utils.hardware import ResourceMonitor
from src.utils.io import write_dataframe
from src.utils.logging_utils import build_run_log, save_run_log
from src.utils.seed import set_global_seed
from src.visualization.plot_forecasting import plot_forecast_errors


def run(output_root: str | None = None) -> pd.DataFrame:
    config = load_config("forecasting_config.yaml")
    dataset_config = load_config("dataset_config.yaml")
    set_global_seed(int(config["seed"]))
    train_split = load_segment_split("train")
    val_split = load_segment_split("val")
    test_split = load_segment_split("test")
    dirs = results_dirs(output_root)
    monitor = ResourceMonitor(device="cuda")
    monitor.start()
    outputs = train_forecasting_models(train_split, val_split, test_split, config)
    resource_usage = monitor.stop()
    rows = []
    resource_rows = []
    last_metrics = None
    log_payload = []
    for output in outputs:
        metrics = evaluate_forecasting(output.y_true, output.y_pred, dataset_config["forecast_horizons_steps"])
        last_metrics = metrics
        rows.append(
            {
                "method": output.name,
                "mae": metrics["mae"],
                "mse": metrics["mse"],
                "rmse": metrics["rmse"],
                "ade": metrics["ade"],
                "fde": metrics["fde"],
            }
        )
        resource_rows.append(
            build_resource_metrics(
                output.name,
                resource_usage,
                parameter_count=output.parameter_count,
                mean_inference_time=output.mean_inference_time,
            )
        )
        log_payload.append({"name": output.name, "metrics": metrics, "model_path": str(output.model_path)})
    results = pd.DataFrame(rows)
    resources = pd.DataFrame(resource_rows)
    write_dataframe(results, dirs["metrics"] / "forecasting_results.csv")
    write_dataframe(resources, dirs["metrics"] / "resource_usage_forecasting.csv")
    if last_metrics is not None:
        figure_path = plot_forecast_errors(last_metrics["horizon_error"], dirs["figures"] / "lstm_forecast_error.png")
        mirror_figure(figure_path, dirs["report_figures"])
    mirror_table(results, "forecasting_results_table.csv", dirs["report_tables"])
    save_run_log(dirs["logs"] / "lstm_run.json", build_run_log("lstm_family", config, {"models": log_payload}, resource_usage))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
