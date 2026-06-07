from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import ensure_dir


def plot_clustering_comparison(results: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    subset = results.set_index("method")[["ari", "nmi", "macro_f1"]]
    ax = subset.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Clustering quality comparison")
    ax.set_ylabel("score")
    ax.set_xlabel("method")
    ax.legend(loc="best")
    ax.figure.tight_layout()
    ax.figure.savefig(target, dpi=180)
    plt.close(ax.figure)
    return target
