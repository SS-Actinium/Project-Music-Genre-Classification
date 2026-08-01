"""Genre label mapping load/save."""

from __future__ import annotations

import json
from pathlib import Path

from .config import GENRES, default_mapping_path


class MappingError(ValueError):
    pass


def load_mapping(path: str | Path | None = None) -> list[str]:
    """Load genre list from JSON; fall back to canonical GENRES."""
    p = Path(path) if path else default_mapping_path()
    if not p.is_file():
        return list(GENRES)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise MappingError(f"Invalid mapping JSON: {p}") from e
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise MappingError("Mapping must be a JSON list of strings")
    if len(data) < 2:
        raise MappingError("Mapping must contain at least 2 genres")
    return data


def save_mapping(genres: list[str], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(genres, indent=2) + "\n", encoding="utf-8")
