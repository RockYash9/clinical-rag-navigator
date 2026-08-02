"""Loads configs/logging.yaml and configures the root logger."""
from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml


def setup_logging(config_path: str | Path = "configs/logging.yaml") -> None:
    path = Path(config_path)
    Path("logs").mkdir(exist_ok=True)

    if not path.exists():
        logging.basicConfig(level=logging.INFO)
        logging.warning("No logging config at %s, using basicConfig fallback", path)
        return

    config = yaml.safe_load(path.read_text())
    logging.config.dictConfig(config)
