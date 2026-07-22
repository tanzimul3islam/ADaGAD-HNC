import json
import logging
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger("AdaGAD-HNC")


def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    LOGGER.info(f"Saved JSON to {path}")


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def save_yaml(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, default_flow_style=False)
    LOGGER.info(f"Saved YAML to {path}")


def load_yaml(path: Path) -> Any:
    with open(path) as f:
        return yaml.safe_load(f)
