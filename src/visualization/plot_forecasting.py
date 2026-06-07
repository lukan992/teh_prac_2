from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.io import ensure_dir


def plot_forecast_errors(horizon_errors: dict[str, float], path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    horizons = list(horizon_errors.keys())
    values = list(horizon_errors.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(horizons, values, marker="o")
    ax.set_title("Forecast error by horizon")
    ax.set_xlabel("horizon (steps)")
    ax.set_ylabel("mean displacement error")
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target
