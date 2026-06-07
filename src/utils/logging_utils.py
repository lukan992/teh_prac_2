from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.io import write_json


def build_run_log(
    method: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    resource_usage: dict[str, Any],
    notes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "method": method,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "metrics": metrics,
        "resource_usage": resource_usage,
        "notes": notes or [],
    }


def save_run_log(path: str | Path, payload: dict[str, Any]) -> Path:
    return write_json(payload, path)
