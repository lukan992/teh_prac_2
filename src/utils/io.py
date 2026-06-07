from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    df.to_csv(target, index=index)
    return target


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
