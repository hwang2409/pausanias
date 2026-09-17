from __future__ import annotations

import sqlite3
from pathlib import Path

from .config import Config, Root

SCHEMA_VERSION = 2


def connect_readonly(database: Path) -> sqlite3.Connection:
    uri = f"{database.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def require_schema_version(connection: sqlite3.Connection) -> None:
    stored_version = connection.execute("PRAGMA user_version").fetchone()[0]
    if stored_version != SCHEMA_VERSION:
        raise ValueError(f"index schema is v{stored_version}; run `pausanias index` to rebuild")


def markdown(path: Path) -> bool:
    return path.suffix.lower() in {".md", ".markdown"}


def safe_source(config: Config, raw_path: str | Path) -> tuple[Root, Path] | None:
    requested = Path(raw_path).expanduser()
    options = [requested] if requested.is_absolute() else [Path.cwd() / requested, *(root.path / requested for root in config.roots)]
    seen: set[Path] = set()
    for option in options:
        try:
            canonical = option.resolve(strict=True)
        except OSError:
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        root = config.root_for(canonical)
        if root and canonical.is_file() and markdown(canonical) and not config.is_excluded(canonical, root):
            return root, canonical
    return None
