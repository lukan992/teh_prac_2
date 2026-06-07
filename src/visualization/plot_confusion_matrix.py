from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.utils.io import ensure_dir


def plot_confusion_matrix(matrix: list[list[int]], path: str | Path, title: str = "CNN confusion matrix") -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    array = np.array(matrix)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(array, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target
