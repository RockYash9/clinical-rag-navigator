"""Loads configs/config.yaml as a plain dict for modules that need pipeline settings."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path = "configs/config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())
