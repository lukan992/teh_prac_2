from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import ensure_dir, project_root, write_dataframe


def results_dirs(output_root: str | Path | None = None) -> dict[str, Path]:
    root = ensure_dir(output_root or project_root() / "results")
    paths = {
        "root": root,
        "metrics": ensure_dir(root / "metrics"),
        "figures": ensure_dir(root / "figures"),
        "tables": ensure_dir(root / "tables"),
        "logs": ensure_dir(root / "logs"),
        "report_figures": ensure_dir(project_root() / "report_assets" / "figures"),
        "report_tables": ensure_dir(project_root() / "report_assets" / "tables"),
        "text_fragments": ensure_dir(project_root() / "report_assets" / "text_fragments"),
        "appendices": ensure_dir(project_root() / "report_assets" / "appendices"),
    }
    return paths


def mirror_figure(path: str | Path, report_dir: str | Path) -> None:
    source = Path(path)
    shutil.copy2(source, Path(report_dir) / source.name)


def mirror_table(df: pd.DataFrame, filename: str, report_dir: str | Path) -> None:
    write_dataframe(df, Path(report_dir) / filename)


def serialize_secondary_metrics(metrics: dict[str, Any], excluded: set[str]) -> str:
    parts = []
    for key, value in metrics.items():
        if key in excluded:
            continue
        parts.append(f"{key}={value}")
    return "; ".join(parts)
