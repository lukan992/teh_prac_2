from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.io import project_root


def config_path(name: str) -> Path:
    return project_root() / "experiments" / "configs" / name


def load_config(name: str) -> dict[str, Any]:
    path = config_path(name)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
