from __future__ import annotations

import numpy as np


def evaluate_tracking(y_true: np.ndarray, y_pred: np.ndarray, runtime_seconds: float) -> dict[str, float]:
    valid_mask = ~np.isnan(y_pred).any(axis=1)
    detection_rate = float(valid_mask.mean())
    false_positive_rate = float(np.mean(~valid_mask))
    if np.any(valid_mask):
        rmse = float(np.sqrt(np.mean((y_pred[valid_mask] - y_true[valid_mask]) ** 2)))
    else:
        rmse = float("nan")
    fragmentation = float(np.sum(np.diff(valid_mask.astype(int)) == -1))
    fps = float(len(y_true) / max(runtime_seconds, 1e-8))
    return {
        "position_rmse": rmse,
        "detection_rate": detection_rate,
        "false_positive_rate": false_positive_rate,
        "track_fragmentation": fragmentation,
        "fps": fps,
    }
