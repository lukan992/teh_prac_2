from __future__ import annotations

from typing import Any

import pandas as pd


def build_resource_metrics(
    method: str,
    resource_usage: dict[str, Any],
    parameter_count: int | None = None,
    mean_inference_time: float | None = None,
) -> dict[str, Any]:
    peak_vram = resource_usage.get("peak_vram_mb", [])
    payload = {
        "method": method,
        "runtime_seconds": float(resource_usage.get("runtime_seconds", 0.0)),
        "peak_ram_mb": float(resource_usage.get("peak_ram_mb", 0.0)),
        "peak_vram_mb": ";".join(str(value) for value in peak_vram) if peak_vram else "",
        "parameter_count": parameter_count if parameter_count is not None else "",
        "mean_inference_time": mean_inference_time if mean_inference_time is not None else "",
    }
    return payload


def metrics_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)
