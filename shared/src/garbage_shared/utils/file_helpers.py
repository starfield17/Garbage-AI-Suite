"""File utility functions."""

import json
from pathlib import Path


def ensure_dir(path: Path | str) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path | str) -> dict:
    """Read JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path | str, data: dict | list) -> None:
    """Write data to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
