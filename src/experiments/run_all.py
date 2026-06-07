from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset.generate_pointcloud import main as generate_pointcloud
from src.dataset.generate_segments import main as generate_segments
from src.dataset.generate_trajectories import main as generate_trajectories
from src.experiments.helpers import mirror_figure, mirror_table, results_dirs, serialize_secondary_metrics
from src.experiments.run_classification import run as run_classification
from src.experiments.run_clustering import run as run_clustering
from src.experiments.run_forecasting import run as run_forecasting
from src.experiments.run_pointcloud import run as run_pointcloud
from src.experiments.run_resnet_hdbscan import run as run_resnet_hdbscan
from src.utils.io import write_dataframe
from src.visualization.plot_patterns import plot_experiment_scheme, plot_pattern_examples, plot_trajectory_segments
from src.visualization.plot_resource_usage import plot_quality_vs_time, plot_resource_usage


def _markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _build_final_comparison(
    clustering: pd.DataFrame,
    classification: pd.DataFrame,
    forecasting: pd.DataFrame,
    pointcloud: pd.DataFrame,
    resnet: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for row in clustering.itertuples(index=False):
        rows.append(
            {
                "method": row.method,
                "source": "adapted literature method",
                "input_type": "trajectory/segment",
                "task_type": "clustering",
                "main_metric": "ari",
                "main_metric_value": row.ari,
                "secondary_metrics": serialize_secondary_metrics(row._asdict(), {"method", "ari"}),
                "runtime_seconds": None,
                "peak_ram_mb": None,
                "peak_vram_mb": None,
                "strengths": "Interpretable clustering of motion patterns",
                "limitations": "Adapted implementation",
                "applicability_to_uav_pattern_detection": "Useful for exploratory grouping of patterns",
            }
        )
    for row in classification.itertuples(index=False):
        rows.append(
            {
                "method": row.method,
                "source": "synthetic deep baseline",
                "input_type": "segment",
                "task_type": "classification",
                "main_metric": "macro_f1",
                "main_metric_value": row.macro_f1,
                "secondary_metrics": serialize_secondary_metrics(row._asdict(), {"method", "macro_f1"}),
                "runtime_seconds": None,
                "peak_ram_mb": None,
                "peak_vram_mb": None,
                "strengths": "Direct behavior classification",
                "limitations": "Needs labeled synthetic windows",
                "applicability_to_uav_pattern_detection": "Useful when segment labels are available",
            }
        )
    for row in forecasting.itertuples(index=False):
        rows.append(
            {
                "method": row.method,
                "source": "synthetic deep baseline",
                "input_type": "segment",
                "task_type": "forecasting",
                "main_metric": "rmse",
                "main_metric_value": row.rmse,
                "secondary_metrics": serialize_secondary_metrics(row._asdict(), {"method", "rmse"}),
                "runtime_seconds": None,
                "peak_ram_mb": None,
                "peak_vram_mb": None,
                "strengths": "Predictive motion modeling",
                "limitations": "Error grows with horizon",
                "applicability_to_uav_pattern_detection": "Useful for anticipation and tracking support",
            }
        )
    medium_pointcloud = pointcloud[pointcloud["difficulty"] == "medium"]
    for row in medium_pointcloud.itertuples(index=False):
        rows.append(
            {
                "method": row.method,
                "source": "adapted pointcloud method",
                "input_type": "pointcloud",
                "task_type": "tracking",
                "main_metric": "position_rmse",
                "main_metric_value": row.position_rmse,
                "secondary_metrics": serialize_secondary_metrics(row._asdict(), {"method", "difficulty", "position_rmse"}),
                "runtime_seconds": None,
                "peak_ram_mb": None,
                "peak_vram_mb": None,
                "strengths": "Recovery from sparse observations",
                "limitations": "Simplified synthetic scene",
                "applicability_to_uav_pattern_detection": "Useful for sparse sensor tracking",
            }
        )
    for row in resnet.itertuples(index=False):
        rows.append(
            {
                "method": row.method,
                "source": "user baseline",
                "input_type": "trajectory image",
                "task_type": "clustering",
                "main_metric": "ari",
                "main_metric_value": row.ari,
                "secondary_metrics": serialize_secondary_metrics(row._asdict(), {"method", "ari"}),
                "runtime_seconds": None,
                "peak_ram_mb": None,
                "peak_vram_mb": None,
                "strengths": "Learned embeddings for motion shape",
                "limitations": "Uses adapted 2D representation",
                "applicability_to_uav_pattern_detection": "Useful for non-linear pattern grouping",
            }
        )
    return pd.DataFrame(rows)


def _attach_resource_usage(final_comparison: pd.DataFrame, resource_df: pd.DataFrame) -> pd.DataFrame:
    if resource_df.empty:
        return final_comparison
    resource_lookup = resource_df.set_index("method").to_dict(orient="index")
    enriched_rows = []
    for row in final_comparison.to_dict(orient="records"):
        lookup_key = row["method"]
        if lookup_key not in resource_lookup and row["task_type"] == "tracking":
            lookup_key = f"{row['method']}_medium"
        if lookup_key not in resource_lookup and row["method"] in {"handcrafted_hdbscan", "resnet_kmeans", "resnet_hdbscan"}:
            lookup_key = "resnet_hdbscan_family"
        if lookup_key in resource_lookup:
            usage = resource_lookup[lookup_key]
            row["runtime_seconds"] = usage.get("runtime_seconds")
            row["peak_ram_mb"] = usage.get("peak_ram_mb")
            row["peak_vram_mb"] = usage.get("peak_vram_mb")
        enriched_rows.append(row)
    return pd.DataFrame(enriched_rows)


def _write_report_fragments(final_comparison: pd.DataFrame) -> None:
    dirs = results_dirs()
    text_dir = dirs["text_fragments"]
    appendix_dir = dirs["appendices"]
    dataset_config = Path("experiments/configs/dataset_config.yaml").read_text(encoding="utf-8")
    text_fragments = {
        "dataset_description.md": "Синтетический датасет содержит траектории, сегменты и point cloud представления для 9 паттернов движения БПЛА на трех уровнях шума.",
        "methods_short_descriptions.md": "В пакет включены адаптированные методы кластеризации траекторий, классификации сегментов, прогнозирования и восстановления траектории по point cloud данным.",
        "metrics_description.md": "Для сравнения используются метрики кластеризации, классификации, прогнозирования, трекинга и ресурсоемкости.",
        "experiment_protocol_summary.md": "Все эксперименты запускаются из CLI, используют фиксированный seed и сохраняют CSV, JSON-логи и фигуры.",
        "hardware_description.md": "Пакет ориентирован на GPU-совместимое выполнение нейросетевых методов и CPU/GPU мониторинг ресурсов.",
        "results_summary.md": _markdown_table(final_comparison[["method", "task_type", "main_metric", "main_metric_value"]]),
        "limitations_summary.md": Path("limitations.md").read_text(encoding="utf-8"),
        "conclusion_points.md": "Итоговое сравнение позволяет оценить сильные стороны методов разных классов без претензии на полную эквивалентность исходным публикациям.",
    }
    for filename, content in text_fragments.items():
        (text_dir / filename).write_text(content, encoding="utf-8")

    appendices = {
        "appendix_a_dataset_params.md": Path("dataset_spec.md").read_text(encoding="utf-8"),
        "appendix_b_method_params.md": Path("methods_list.md").read_text(encoding="utf-8"),
        "appendix_c_extra_results.md": _markdown_table(final_comparison),
        "appendix_d_code_fragments.md": "CLI entry points: `src.dataset.*` and `src.experiments.*`.",
        "appendix_e_additional_figures.md": "Дополнительные фигуры сохранены в `results/figures/` и `report_assets/figures/`.",
    }
    for filename, content in appendices.items():
        (appendix_dir / filename).write_text(content, encoding="utf-8")

    dataset_params = pd.DataFrame(
        [
            {"parameter": "dataset_name", "value": "synthetic_v1"},
            {"parameter": "trajectory_count", "value": 1000},
            {"parameter": "segment_length", "value": 50},
            {"parameter": "noise_levels", "value": "clean, medium, hard"},
        ]
    )
    methods_table = pd.DataFrame(
        [
            {"method": "traclus", "group": "clustering"},
            {"method": "st_dbscan", "group": "clustering"},
            {"method": "vector_field_kmeans", "group": "clustering"},
            {"method": "spatiotemporal_clustering", "group": "clustering"},
            {"method": "cnn_segment_classifier", "group": "classification"},
            {"method": "lstm_baseline", "group": "forecasting"},
            {"method": "lstm_class_aware", "group": "forecasting"},
            {"method": "sparse_pointcloud", "group": "tracking"},
            {"method": "cluster_filter", "group": "tracking"},
            {"method": "cl_det", "group": "tracking"},
            {"method": "resnet_hdbscan", "group": "embedding clustering"},
        ]
    )
    method_params = pd.DataFrame(
        [
            {"config_file": "clustering_config.yaml", "contents": Path("experiments/configs/clustering_config.yaml").read_text(encoding="utf-8")},
            {"config_file": "classification_config.yaml", "contents": Path("experiments/configs/classification_config.yaml").read_text(encoding="utf-8")},
            {"config_file": "forecasting_config.yaml", "contents": Path("experiments/configs/forecasting_config.yaml").read_text(encoding="utf-8")},
            {"config_file": "pointcloud_config.yaml", "contents": Path("experiments/configs/pointcloud_config.yaml").read_text(encoding="utf-8")},
            {"config_file": "resnet_hdbscan_config.yaml", "contents": Path("experiments/configs/resnet_hdbscan_config.yaml").read_text(encoding="utf-8")},
            {"config_file": "dataset_config.yaml", "contents": dataset_config},
        ]
    )
    write_dataframe(methods_table, dirs["report_tables"] / "methods_table.csv")
    write_dataframe(dataset_params, dirs["report_tables"] / "dataset_parameters_table.csv")
    write_dataframe(method_params, dirs["report_tables"] / "method_parameters_table.csv")


def run(dataset: str, output_root: str | None = None) -> pd.DataFrame:
    del dataset
    generate_trajectories()
    generate_segments()
    generate_pointcloud()
    clustering = run_clustering(output_root)
    classification = run_classification(output_root)
    forecasting = run_forecasting(output_root)
    pointcloud = run_pointcloud(output_root)
    resnet = run_resnet_hdbscan(output_root)
    dirs = results_dirs(output_root)
    import pandas as pd

    trajectories = pd.read_csv("data/trajectories/trajectories.csv")
    pattern_figure = plot_pattern_examples(trajectories, dirs["figures"] / "pattern_examples.png")
    mirror_figure(pattern_figure, dirs["report_figures"])
    first_traj = trajectories[trajectories["trajectory_id"] == trajectories["trajectory_id"].min()]
    segments_figure = plot_trajectory_segments(first_traj, 50, 25, dirs["figures"] / "trajectory_segments.png")
    mirror_figure(segments_figure, dirs["report_figures"])

    final_comparison = _build_final_comparison(clustering, classification, forecasting, pointcloud, resnet)
    resources = []
    for filename in [
        "resource_usage_clustering.csv",
        "resource_usage_classification.csv",
        "resource_usage_forecasting.csv",
        "resource_usage_pointcloud.csv",
        "resource_usage_resnet.csv",
    ]:
        csv_path = dirs["metrics"] / filename
        if csv_path.exists():
            resources.append(pd.read_csv(csv_path))
    resource_df = pd.concat(resources, ignore_index=True) if resources else pd.DataFrame()
    final_comparison = _attach_resource_usage(final_comparison, resource_df)
    write_dataframe(final_comparison, dirs["metrics"] / "final_comparison.csv")
    write_dataframe(final_comparison, dirs["tables"] / "final_comparison.csv")
    mirror_table(final_comparison, "final_applicability_table.csv", dirs["report_tables"])
    if not resource_df.empty:
        write_dataframe(resource_df, dirs["metrics"] / "resource_usage.csv")
        mirror_table(resource_df, "resource_usage_table.csv", dirs["report_tables"])
        resource_plot = plot_resource_usage(resource_df, dirs["figures"] / "resource_usage.png")
        mirror_figure(resource_plot, dirs["report_figures"])

    runtime_placeholder = pd.Series(np.arange(1, len(final_comparison) + 1), index=final_comparison.index, dtype=float)
    quality_plot = plot_quality_vs_time(
        final_comparison.assign(
            runtime_seconds=final_comparison["runtime_seconds"].fillna(runtime_placeholder),
            main_metric_value=final_comparison["main_metric_value"].astype(float),
        ),
        dirs["figures"] / "quality_vs_time.png",
    )
    mirror_figure(quality_plot, dirs["report_figures"])

    scheme_path = plot_experiment_scheme(dirs["figures"] / "experiment_scheme.png")
    mirror_figure(scheme_path, dirs["report_figures"])
    _write_report_fragments(final_comparison)
    return final_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="synthetic_v1")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(args.dataset, args.output)


if __name__ == "__main__":
    main()
