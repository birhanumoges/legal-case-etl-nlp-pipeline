"""utils/config_loader.py — Load YAML/env overrides on top of config.py."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_env(key: str, default: Any = None) -> Any:
    return os.getenv(key, default)
