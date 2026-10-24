"""Configuration and atomic output helpers.

No helper in this module downloads content or executes commands from a config.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML mapping from ``path``."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        loaded = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("PyYAML is required to read YAML configurations") from exc
        loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError("The top-level configuration value must be a mapping")
    return loaded


def atomic_write_json(
    path: str | Path,
    value: Mapping[str, Any] | list[Any],
    *,
    indent: int = 2,
) -> Path:
    """Write JSON atomically, creating only the requested parent directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=indent, sort_keys=True, default=str)
            stream.write("\n")
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination
