from __future__ import annotations

import numpy as np


def evaluate_forecasting(y_true: np.ndarray, y_pred: np.ndarray, horizons: list[int]) -> dict[str, object]:
    diff = y_pred - y_true
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    points_true = y_true.reshape(y_true.shape[0], len(horizons), 3)
    points_pred = y_pred.reshape(y_pred.shape[0], len(horizons), 3)
    displacement = np.linalg.norm(points_pred - points_true, axis=2)
    ade = float(displacement.mean())
    fde = float(displacement[:, -1].mean())
    horizon_errors = {
        str(horizon): float(displacement[:, index].mean())
        for index, horizon in enumerate(horizons)
    }
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "ade": ade,
        "fde": fde,
        "horizon_error": horizon_errors,
    }
