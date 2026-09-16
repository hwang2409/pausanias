from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import re
import sqlite3
import time

from .config import Config, Root, _contained
from .splitter import content_hash, explicit_links, split_markdown


MAX_QUERY_TERMS = 64
MAX_CANDIDATES = 200
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9]*-\d+|[\w]+", flags=re.UNICODE)


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
    reason: str


def connect(database: Path, initialize: bool = True) -> sqlite3.Connection:
    if initialize:
        database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    if initialize:
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
            raw_bytes = path.read_bytes()
            content = raw_bytes.decode("utf-8")
            document = split_markdown(canonical, content, config.section_bytes, raw_bytes)
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
    raw_bytes = path.read_bytes()
    content = raw_bytes.decode("utf-8")
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
    clauses: list[str] = []
    tokens: list[str] = []

    def add_tokens(value: str, phrase: bool) -> bool:
        found = TOKEN_PATTERN.findall(value)
        remaining = MAX_QUERY_TERMS - len(tokens)
        if remaining <= 0:
            return False
        found = found[:remaining]
        if found:
            if phrase:
                clauses.append('"' + " ".join(found).replace('"', '""') + '"')
            else:
                clauses.extend('"' + token.replace('"', '""') + '"' for token in found)
            tokens.extend(found)
        return len(tokens) < MAX_QUERY_TERMS

    position = 0
    while position < len(query) and len(tokens) < MAX_QUERY_TERMS:
        quote = query.find('"', position)
        if quote < 0:
            add_tokens(query[position:], False)
            break
        if not add_tokens(query[position:quote], False):
            break
        closing = query.find('"', quote + 1)
        if closing < 0:
            add_tokens(query[quote + 1:], False)
            break
        if not add_tokens(query[quote + 1:closing], True):
            break
        position = closing + 1
    return " AND ".join(clauses), tokens


def _column_query(column: str, fts: str) -> str:
    return f"{column} : ({fts})"


def _token_set(value: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(value)}


def search(config: Config, query: str, project: str | None = None, root_id: str | None = None,
           all_projects: bool = False, limit: int = 20, refresh: set[str] | None = None) -> list[Candidate]:
    if limit < 1:
        raise ValueError("limit must be positive")
    fts, tokens = _fts_query(query)
    if not fts:
        return []
    if not config.database.exists():
        return []

    if root_id and not any(root.id == root_id for root in config.roots):
        return []
    global_paths = sorted(str(path) for path in config.global_notes)
    selected_root = next((root for root in config.roots if root.id == root_id), None)
    effective_project = project or (selected_root.project if root_id and not all_projects else None)
    if effective_project and not all_projects:
        scope_root_ids = [root.id for root in config.roots if root.project == effective_project]
        scope_paths = global_paths
    elif effective_project is None and not all_projects:
        scope_root_ids = []
        scope_paths = global_paths
    else:
        scope_root_ids = [root.id for root in config.roots]
        scope_paths = global_paths

    def placeholders(values: list[str]) -> str:
        return ", ".join("?" for _ in values) or "NULL"

    connection = connect(config.database, initialize=False)
    try:
        conditions = ["sections_fts MATCH ?"]
        params: list[object] = [fts]
        global_condition = f"s.canonical_path IN ({placeholders(scope_paths)})"
        if root_id:
            if effective_project and not all_projects:
                conditions.append(f"((s.root_id = ? AND s.project_scope = ?) OR {global_condition})")
                params.extend((root_id, effective_project))
            else:
                conditions.append(f"(s.root_id = ? OR {global_condition})")
                params.append(root_id)
            params.extend(scope_paths)
        elif effective_project and not all_projects:
            if scope_root_ids:
                conditions.append(
                    f"((s.root_id IN ({placeholders(scope_root_ids)}) AND s.project_scope = ?)"
                    f" OR {global_condition})"
                )
                params.extend(scope_root_ids)
                params.append(effective_project)
                params.extend(scope_paths)
            elif scope_paths:
                conditions.append(global_condition)
                params.extend(scope_paths)
            else:
                conditions.append("0")
        elif scope_paths and scope_root_ids:
            conditions.append(
                f"(s.root_id IN ({placeholders(scope_root_ids)}) OR {global_condition})"
            )
            params.extend(scope_root_ids)
            params.extend(scope_paths)
        elif scope_root_ids:
            conditions.append(f"s.root_id IN ({placeholders(scope_root_ids)})")
            params.extend(scope_root_ids)
        elif scope_paths:
            conditions.append(global_condition)
            params.extend(scope_paths)
        else:
            conditions.append("0")
        candidate_limit = min(MAX_CANDIDATES, max(limit * 5, 50))
        rows_by_id: dict[str, sqlite3.Row] = {}
        for lane in (None, "heading", "text"):
            lane_params = list(params)
            lane_params[0] = fts if lane is None else _column_query(lane, fts)
            rows_by_id.update({
                row["section_id"]: row
                for row in connection.execute(
                    "SELECT s.*, bm25(sections_fts) AS fts_score FROM sections s JOIN sections_fts ON sections_fts.section_id = s.section_id WHERE "
                    + " AND ".join(conditions) + " ORDER BY fts_score LIMIT ?",
                    (*lane_params, candidate_limit),
                ).fetchall()
            })
        rows = list(rows_by_id.values())
    finally:
        connection.close()

    results: list[Candidate] = []
    source_cache: dict[str, tuple[Root, Path] | None] = {}
    hash_cache: dict[str, str | None] = {}
    for row in rows:
        canonical = row["canonical_path"]
        if canonical not in source_cache:
            source_cache[canonical] = _safe_source(config, canonical)
        source = source_cache[canonical]
        if source is None:
            if refresh is not None:
                refresh.add(canonical)
            continue
        current_root = source[0]
        is_global = config.is_global(source[1])
        if root_id and current_root.id != root_id and not is_global:
            continue
        if not all_projects:
            if effective_project is None:
                if not is_global:
                    continue
            elif current_root.project != effective_project and not is_global:
                continue
        if canonical not in hash_cache:
            try:
                hash_cache[canonical] = content_hash(source[1].read_bytes())
            except OSError:
                hash_cache[canonical] = None
        current_hash = hash_cache[canonical]
        if current_hash is None:
            if refresh is not None:
                refresh.add(canonical)
            continue
        if current_hash != row["content_hash"]:
            if refresh is not None:
                refresh.add(canonical)
            continue
        heading_path = tuple(filter(None, row["heading_path"].split("\n")))
        lower_tokens = {token.lower() for token in tokens}
        heading_tokens = _token_set(row["heading"] or "")
        body_tokens = _token_set(row["text"])
        identifier_boost = sum(2.0 for token in lower_tokens
                               if re.fullmatch(r"[a-z]{2,}-\d+", token)
                               and token in heading_tokens | body_tokens)
        heading_matches = bool(lower_tokens & heading_tokens)
        body_matches = bool(lower_tokens & body_tokens)
        heading_boost = sum(0.75 for token in lower_tokens if token in heading_tokens)
        score = -float(row["fts_score"]) + identifier_boost + heading_boost
        reasons: list[str] = []
        if is_global:
            reasons.append("global-note inclusion")
        if identifier_boost:
            reasons.append("identifier match")
        if heading_matches:
            reasons.append("heading match")
        if body_matches:
            reasons.append("body match")
        results.append(Candidate(row["section_id"], canonical, row["heading"], heading_path,
                                 row["line_start"], row["line_end"], current_root.id, current_root.project,
                                 row["text"], row["content_hash"], row["note_type"], row["updated_date"],
                                 row["created_date"], score, ", ".join(reasons) or "text match"))
    results.sort(key=lambda item: item.score, reverse=True)
    return results[:limit]


def excerpt(text: str, max_chars: int = 500) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= max_chars else clean[:max_chars].rstrip() + "..."
