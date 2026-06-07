from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import ensure_dir


def plot_quality_vs_time(results: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(results["runtime_seconds"], results["main_metric_value"], s=80)
    for row in results.itertuples(index=False):
        ax.annotate(row.method, (row.runtime_seconds, row.main_metric_value))
    ax.set_title("Quality vs time")
    ax.set_xlabel("runtime seconds")
    ax.set_ylabel("main metric")
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target


def plot_resource_usage(resources: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    resources.plot(x="method", y="runtime_seconds", kind="bar", ax=axes[0], legend=False, title="Runtime")
    resources.plot(x="method", y="peak_ram_mb", kind="bar", ax=axes[1], legend=False, title="Peak RAM")
    axes[0].set_ylabel("seconds")
    axes[1].set_ylabel("MB")
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target
