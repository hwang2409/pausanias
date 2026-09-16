from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import re
import sqlite3
import time

from .config import Config, Root, _contained
from .splitter import explicit_links, split_markdown


SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    canonical_path TEXT PRIMARY KEY,
    root_id TEXT NOT NULL,
    mtime_ns INTEGER NOT NULL,
    content_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sections (
    section_id TEXT PRIMARY KEY,
    canonical_path TEXT NOT NULL,
    heading TEXT,
    heading_path TEXT NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    root_id TEXT NOT NULL,
    project_scope TEXT NOT NULL,
    text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    index_generation INTEGER NOT NULL,
    indexed_at REAL NOT NULL,
    note_type TEXT,
    updated_date TEXT,
    created_date TEXT,
    links TEXT NOT NULL,
    is_global INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS sections_path_idx ON sections(canonical_path);
CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts5(section_id UNINDEXED, heading, text);
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


@dataclass(frozen=True)
class Candidate:
    section_id: str
    canonical_path: str
    heading: str | None
    heading_path: tuple[str, ...]
    line_start: int
    line_end: int
    root_id: str
    project_scope: str
    text: str
    content_hash: str
    note_type: str | None
    updated_date: str | None
    created_date: str | None
    score: float


def connect(database: Path) -> sqlite3.Connection:
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)
    return connection


def _markdown(path: Path) -> bool:
    return path.suffix.lower() in {".md", ".markdown"}


def _files(config: Config) -> dict[str, tuple[Root, Path]]:
    found: dict[str, tuple[Root, Path]] = {}
    for root in config.roots:
        for directory, dirnames, filenames in os.walk(root.path, followlinks=False):
            dirnames[:] = [name for name in dirnames if not config.is_excluded(Path(directory) / name, root)]
            for filename in filenames:
                path = Path(directory) / filename
                if not _markdown(path) or config.is_excluded(path, root):
                    continue
                try:
                    canonical = path.resolve(strict=True)
                except OSError:
                    continue
                if _contained(canonical, root.path):
                    found[str(canonical)] = (root, path)
    return found


def _remove_file(connection: sqlite3.Connection, canonical: str) -> None:
    connection.execute("DELETE FROM sections_fts WHERE section_id IN (SELECT section_id FROM sections WHERE canonical_path = ?)", (canonical,))
    connection.execute("DELETE FROM sections WHERE canonical_path = ?", (canonical,))
    connection.execute("DELETE FROM files WHERE canonical_path = ?", (canonical,))


def index(config: Config, rebuild: bool = False) -> int:
    found = _files(config)
    connection = connect(config.database)
    try:
        connection.execute("BEGIN IMMEDIATE")
        if rebuild:
            connection.execute("DELETE FROM sections_fts")
            connection.execute("DELETE FROM sections")
            connection.execute("DELETE FROM files")
        generation_row = connection.execute("SELECT value FROM metadata WHERE key = 'generation'").fetchone()
        old_generation = int(generation_row["value"]) if generation_row else 0
        generation = old_generation + 1
        existing = {row["canonical_path"]: row for row in connection.execute("SELECT * FROM files")}
        for canonical, (root, path) in found.items():
            stat = path.stat()
            old = existing.get(canonical)
            content = path.read_text(encoding="utf-8")
            document = split_markdown(canonical, content, config.section_bytes)
            if old and old["root_id"] == root.id and old["mtime_ns"] == stat.st_mtime_ns and old["content_hash"] == document.content_hash:
                connection.execute("UPDATE sections SET index_generation = ?, project_scope = ?, is_global = ? WHERE canonical_path = ?",
                                   (generation, root.project, int(config.is_global(Path(canonical))), canonical))
                continue
            _remove_file(connection, canonical)
            for section in document.sections:
                connection.execute(
                    """INSERT INTO sections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (section.section_id, canonical, section.heading, "\n".join(section.heading_path), section.line_start,
                     section.line_end, root.id, root.project, section.text, document.content_hash, generation, time.time(),
                     document.frontmatter.note_type, document.frontmatter.updated, document.frontmatter.created,
                     json.dumps(explicit_links(content)), int(config.is_global(Path(canonical)))),
                )
                connection.execute("INSERT INTO sections_fts(section_id, heading, text) VALUES (?, ?, ?)",
                                   (section.section_id, section.heading or "", section.text))
            connection.execute("INSERT INTO files VALUES (?, ?, ?, ?)", (canonical, root.id, stat.st_mtime_ns, document.content_hash))
        current = set(found)
        for canonical in set(existing) - current:
            _remove_file(connection, canonical)
        connection.execute("INSERT INTO metadata(key, value) VALUES ('generation', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(generation),))
        connection.commit()
        return generation
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _safe_source(config: Config, raw_path: str | Path) -> tuple[Root, Path] | None:
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
        if root and canonical.is_file() and _markdown(canonical) and not config.is_excluded(canonical, root):
            return root, canonical
    return None


def read_source(config: Config, raw_path: str, heading: str | None = None, max_bytes: int = 20000) -> str:
    if max_bytes < 1:
        raise ValueError("max-bytes must be positive")
    source = _safe_source(config, raw_path)
    if source is None:
        raise ValueError(f"path is outside allowed roots or does not exist: {raw_path}")
    _, path = source
    content = path.read_text(encoding="utf-8")
    if heading is not None:
        document = split_markdown(str(path), content, max(max_bytes, 256))
        matches = [section for section in document.sections if section.heading == heading or " > ".join(section.heading_path) == heading]
        if not matches:
            raise ValueError(f"heading not found: {heading}")
        result = "\n\n".join(section.text for section in matches)
    else:
        result = content
    encoded = result.encode("utf-8")
    if len(encoded) <= max_bytes:
        return result
    marker = b"\n[truncated]"
    if max_bytes <= len(marker):
        return marker[:max_bytes].decode("utf-8", errors="ignore")
    bounded = encoded[:max_bytes - len(marker)].decode("utf-8", errors="ignore").rstrip().encode("utf-8")
    return (bounded + marker)[:max_bytes].decode("utf-8", errors="ignore")


def _fts_query(query: str) -> tuple[str, list[str]]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9]*-\d+|[\w]+", query, flags=re.UNICODE)
    tokens = [token for token in tokens if token]
    return " AND ".join('"' + token.replace('"', '""') + '"' for token in tokens), tokens


def search(config: Config, query: str, project: str | None = None, root_id: str | None = None,
           all_projects: bool = False, limit: int = 20, refresh: set[str] | None = None) -> list[Candidate]:
    if limit < 1:
        raise ValueError("limit must be positive")
    fts, tokens = _fts_query(query)
    if not fts:
        return []
    connection = connect(config.database)
    try:
        conditions = ["sections_fts MATCH ?"]
        params: list[object] = [fts]
        if root_id:
            conditions.append("s.root_id = ?")
            params.append(root_id)
        if project and not all_projects:
            conditions.append("(s.project_scope = ? OR s.is_global = 1)")
            params.append(project)
        elif project is None and not all_projects:
            conditions.append("s.is_global = 1")
        rows = connection.execute(
            "SELECT s.*, bm25(sections_fts) AS fts_score FROM sections s JOIN sections_fts ON sections_fts.section_id = s.section_id WHERE "
            + " AND ".join(conditions) + " ORDER BY fts_score LIMIT ?", (*params, max(limit * 20, 100))).fetchall()
    finally:
        connection.close()

    results: list[Candidate] = []
    for row in rows:
        source = _safe_source(config, row["canonical_path"])
        if source is None:
            if refresh is not None:
                refresh.add(row["canonical_path"])
            continue
        try:
            current_hash = hashlib.sha256(source[1].read_bytes()).hexdigest()
        except OSError:
            if refresh is not None:
                refresh.add(row["canonical_path"])
            continue
        if current_hash != row["content_hash"]:
            if refresh is not None:
                refresh.add(row["canonical_path"])
            continue
        heading_path = tuple(filter(None, row["heading_path"].split("\n")))
        lower_tokens = [token.lower() for token in tokens]
        haystack = (row["heading"] or "").lower() + " " + row["text"].lower()
        identifier_boost = sum(2.0 for token in lower_tokens if re.fullmatch(r"[a-z]{2,}-\d+", token) and token in haystack)
        heading_boost = sum(0.75 for token in lower_tokens if token in (row["heading"] or "").lower())
        score = -float(row["fts_score"]) + identifier_boost + heading_boost
        results.append(Candidate(row["section_id"], row["canonical_path"], row["heading"], heading_path,
                                 row["line_start"], row["line_end"], row["root_id"], row["project_scope"],
                                 row["text"], row["content_hash"], row["note_type"], row["updated_date"],
                                 row["created_date"], score))
    results.sort(key=lambda item: item.score, reverse=True)
    return results[:limit]


def excerpt(text: str, max_chars: int = 500) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= max_chars else clean[:max_chars].rstrip() + "..."
