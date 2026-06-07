from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from src.utils.io import ensure_dir

try:
    import umap  # type: ignore
except ImportError:  # pragma: no cover
    umap = None


def plot_embedding_space(embeddings: np.ndarray, labels: np.ndarray, path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    if umap is not None:
        reducer = umap.UMAP(random_state=42)
        projected = reducer.fit_transform(embeddings)
    else:
        projected = TSNE(n_components=2, random_state=42, init="pca").fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(projected[:, 0], projected[:, 1], c=labels, cmap="tab10", s=18)
    ax.set_title("Embedding space")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.colorbar(scatter, ax=ax)
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target
