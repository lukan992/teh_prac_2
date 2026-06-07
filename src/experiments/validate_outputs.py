from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import project_root


@dataclass
class CheckResult:
    section: str
    label: str
    passed: bool
    detail: str


def check_exists(path: Path, section: str, label: str) -> CheckResult:
    exists = path.exists()
    detail = "exists" if exists else f"missing: {path}"
    return CheckResult(section=section, label=label, passed=exists, detail=detail)


def check_nonempty_png(path: Path, section: str, label: str) -> CheckResult:
    exists = path.exists()
    nonempty = exists and path.stat().st_size > 0
    detail = f"size={path.stat().st_size} bytes" if exists else f"missing: {path}"
    return CheckResult(section=section, label=label, passed=bool(nonempty), detail=detail)


def check_csv_readable(path: Path, section: str, label: str) -> CheckResult:
    try:
        pd.read_csv(path)
    except Exception as exc:
        return CheckResult(section=section, label=label, passed=False, detail=f"read failed: {exc}")
    return CheckResult(section=section, label=label, passed=True, detail="readable via pandas")


def status_line(result: CheckResult) -> str:
    status = "PASS" if result.passed else "FAIL"
    return f"- [{status}] {result.label}: {result.detail}"


def validation_report(results: list[CheckResult]) -> str:
    total = len(results)
    passed = sum(1 for item in results if item.passed)
    failed = total - passed
    ready = failed == 0
    sections = ["Documentation", "Dataset", "Metrics", "Figures", "Report assets"]
    lines = [
        "# Validation report",
        "",
        "## Summary",
        "",
        f"- total checks: {total}",
        f"- passed checks: {passed}",
        f"- failed checks: {failed}",
        f"- status: {'READY' if ready else 'NOT READY'}",
        "",
        f"**{'READY FOR REPORT GENERATION' if ready else 'NOT READY FOR REPORT GENERATION'}**",
        "",
    ]
    for section in sections:
        lines.append(f"## {section}")
        lines.append("")
        section_results = [result for result in results if result.section == section]
        for result in section_results:
            lines.append(status_line(result))
        lines.append("")
    lines.append("## Problems to fix")
    lines.append("")
    problems = [result for result in results if not result.passed]
    if not problems:
        lines.append("- No blocking problems detected.")
    else:
        for result in problems:
            lines.append(f"- {result.section}: {result.label} -> {result.detail}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    root = project_root()
    results: list[CheckResult] = []

    documentation_files = [
        "README.md",
        "project_statement.md",
        "methods_list.md",
        "dataset_spec.md",
        "experiment_protocol.md",
        "limitations.md",
        "references.md",
        "environment.md",
        "requirements.txt",
    ]
    for filename in documentation_files:
        results.append(check_exists(root / filename, "Documentation", filename))

    dataset_paths = [
        root / "data" / "trajectories" / "trajectories.csv",
        root / "data" / "trajectories" / "trajectory_labels.csv",
        root / "data" / "trajectories" / "trajectory_config.json",
        root / "data" / "segments" / "segments_train.npz",
        root / "data" / "segments" / "segments_val.npz",
        root / "data" / "segments" / "segments_test.npz",
        root / "data" / "pointcloud" / "clean",
        root / "data" / "pointcloud" / "medium",
        root / "data" / "pointcloud" / "hard",
        root / "data" / "pointcloud" / "pointcloud_metadata.json",
    ]
    for path in dataset_paths:
        results.append(check_exists(path, "Dataset", str(path.relative_to(root))))

    metric_paths = [
        "clustering_results.csv",
        "classification_results.csv",
        "forecasting_results.csv",
        "pointcloud_results.csv",
        "resource_usage.csv",
        "final_comparison.csv",
    ]
    for filename in metric_paths:
        metric_path = root / "results" / "metrics" / filename
        results.append(check_exists(metric_path, "Metrics", f"results/metrics/{filename}"))
        results.append(check_csv_readable(metric_path, "Metrics", f"read results/metrics/{filename}"))

    figure_files = [
        "experiment_scheme.png",
        "pattern_examples.png",
        "trajectory_segments.png",
        "pointcloud_clean_medium_hard.png",
        "clustering_comparison.png",
        "cnn_confusion_matrix.png",
        "lstm_forecast_error.png",
        "pointcloud_rmse.png",
        "quality_vs_time.png",
        "resnet_embedding_umap.png",
    ]
    for filename in figure_files:
        results.append(check_nonempty_png(root / "results" / "figures" / filename, "Figures", f"results/figures/{filename}"))

    report_asset_paths = [
        root / "report_assets" / "tables",
        root / "report_assets" / "figures",
        root / "report_assets" / "text_fragments",
        root / "report_assets" / "appendices",
        root / "results" / "tables",
    ]
    for path in report_asset_paths:
        results.append(check_exists(path, "Report assets", str(path.relative_to(root))))

    required_report_tables = [
        "clustering_results_table.csv",
        "classification_results_table.csv",
        "forecasting_results_table.csv",
        "pointcloud_results_table.csv",
        "resnet_hdbscan_results_table.csv",
        "resource_usage_table.csv",
        "final_comparison_table.csv",
        "final_applicability_table.csv",
    ]
    for filename in required_report_tables:
        table_path = root / "report_assets" / "tables" / filename
        results.append(check_exists(table_path, "Report assets", f"report_assets/tables/{filename}"))
        results.append(check_csv_readable(table_path, "Report assets", f"read report_assets/tables/{filename}"))
        mirror_table_path = root / "results" / "tables" / filename
        results.append(check_exists(mirror_table_path, "Report assets", f"results/tables/{filename}"))
        results.append(check_csv_readable(mirror_table_path, "Report assets", f"read results/tables/{filename}"))

    final_comparison_table = root / "report_assets" / "tables" / "final_comparison_table.csv"
    if final_comparison_table.exists():
        try:
            table_df = pd.read_csv(final_comparison_table)
            enough_methods = len(table_df) >= 10
            detail = f"rows={len(table_df)}"
            results.append(CheckResult("Report assets", "final_comparison_table has >= 10 methods", enough_methods, detail))
        except Exception as exc:
            results.append(CheckResult("Report assets", "final_comparison_table has >= 10 methods", False, str(exc)))
    else:
        results.append(CheckResult("Report assets", "final_comparison_table has >= 10 methods", False, "file missing"))

    resource_usage_table = root / "report_assets" / "tables" / "resource_usage_table.csv"
    required_vram_columns = {"peak_vram_gpu0_mb", "peak_vram_gpu1_mb", "peak_vram_total_mb"}
    if resource_usage_table.exists():
        try:
            resource_df = pd.read_csv(resource_usage_table)
            has_columns = required_vram_columns.issubset(resource_df.columns)
            detail = f"columns present={sorted(set(resource_df.columns) & required_vram_columns)}"
            results.append(CheckResult("Report assets", "resource_usage_table VRAM columns", has_columns, detail))
        except Exception as exc:
            results.append(CheckResult("Report assets", "resource_usage_table VRAM columns", False, str(exc)))
    else:
        results.append(CheckResult("Report assets", "resource_usage_table VRAM columns", False, "file missing"))

    report_text_files = [
        "dataset_description.md",
        "methods_short_descriptions.md",
        "metrics_description.md",
        "experiment_protocol_summary.md",
        "hardware_description.md",
        "results_summary.md",
        "limitations_summary.md",
        "conclusion_points.md",
        "clustering_results_table.md",
        "classification_results_table.md",
        "forecasting_results_table.md",
        "pointcloud_results_table.md",
        "resource_usage_table.md",
        "final_comparison_table.md",
        "final_applicability_table.md",
    ]
    for filename in report_text_files:
        results.append(check_exists(root / "report_assets" / "text_fragments" / filename, "Report assets", f"report_assets/text_fragments/{filename}"))

    report = validation_report(results)
    (root / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    print("Wrote VALIDATION_REPORT.md")
    print("READY FOR REPORT GENERATION" if all(result.passed for result in results) else "NOT READY FOR REPORT GENERATION")


if __name__ == "__main__":
    main()
