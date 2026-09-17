from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path

from .config import Config, Root, _contained
from .model_bundle import (
    MODEL_BUNDLE_MANIFEST,
    BundleError,
    _resolve_active_bundle,
    manifest_fingerprint,
    package_version,
)
from .splitter import content_hash, explicit_links, split_markdown
from .store import SCHEMA_VERSION
from .store import markdown as _markdown
from .store import require_schema_version as _require_schema_version
from .store import safe_source as _safe_source
from .vectors import (
    EMBEDDING_FORMAT_VERSION,
    EMBEDDING_VERSION,
    SEMANTIC_DISABLED_REASON,
    SEMANTIC_REASON_REFRESH_FAILED,
    SEMANTIC_STATE_DISABLED,
    SEMANTIC_STATE_READY,
    SEMANTIC_STATE_STALE,
    Candidate,
    SemanticError,
    semantic_backend_reason,
    vector_candidates,
)
from .vectors import OnnxEncoder as _OnnxEncoder
from .vectors import encoder_vectors as _encoder_vectors

MAX_QUERY_TERMS = 64
MAX_CANDIDATES = 200
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9]*-\d+|[\w]+", flags=re.UNICODE)


def _semantic_backend_reason(config: Config) -> str | None:
    return semantic_backend_reason(config, version_checker=package_version)


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
    text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    index_generation INTEGER NOT NULL,
    indexed_at REAL NOT NULL,
    note_type TEXT,
    updated_date TEXT,
    created_date TEXT,
    links TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sections_path_idx ON sections(canonical_path);
CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts5(section_id UNINDEXED, heading, text);
CREATE TABLE IF NOT EXISTS embeddings (
    section_id TEXT PRIMARY KEY,
    generation INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    embedding_version INTEGER NOT NULL,
    embedding_format_version INTEGER NOT NULL,
    manifest_fingerprint TEXT NOT NULL,
    vector BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS embedding_metadata (
    generation INTEGER PRIMARY KEY,
    created_at REAL NOT NULL,
    semantic_state TEXT NOT NULL CHECK (semantic_state IN ('ready', 'stale', 'disabled')),
    semantic_reason TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_hash TEXT NOT NULL,
    model_fingerprint TEXT NOT NULL,
    model_version TEXT NOT NULL,
    tokenizer_version TEXT NOT NULL,
    tokenizer_fingerprint TEXT NOT NULL,
    runtime_name TEXT NOT NULL,
    runtime_version TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    dtype TEXT NOT NULL,
    pooling TEXT NOT NULL,
    normalization TEXT NOT NULL,
    metric TEXT NOT NULL,
    embedding_version INTEGER NOT NULL,
    embedding_format_version INTEGER NOT NULL,
    manifest_fingerprint TEXT NOT NULL
);
"""

SCHEMA_STATEMENTS = tuple(statement.strip() for statement in SCHEMA.split(";") if statement.strip())
DROP_SCHEMA_STATEMENTS = (
    "DROP TABLE IF EXISTS embedding_metadata",
    "DROP TABLE IF EXISTS embeddings",
    "DROP TABLE IF EXISTS sections_fts",
    "DROP TABLE IF EXISTS sections",
    "DROP TABLE IF EXISTS files",
    "DROP TABLE IF EXISTS metadata",
)


def _create_schema(connection: sqlite3.Connection) -> None:
    for statement in SCHEMA_STATEMENTS:
        connection.execute(statement)
    fingerprint = manifest_fingerprint()
    connection.execute(
        """INSERT OR IGNORE INTO metadata(key, value) VALUES
           ('generation', '0'),
           ('semantic_state', ?),
           ('semantic_reason', ?),
           ('semantic_generation', '0'),
           ('semantic_manifest_fingerprint', ?)""",
        (SEMANTIC_STATE_DISABLED, SEMANTIC_DISABLED_REASON, fingerprint),
    )


def _tokenizer_fingerprint(manifest: dict[str, object]) -> str:
    tokenizer = manifest["tokenizer"]
    tokenizer_files = set(tokenizer["files"])
    artifact_digests = {
        file["path"]: file["sha256"]
        for file in manifest["files"]
        if file["path"] in tokenizer_files
    }
    payload = json.dumps(
        {"artifacts": artifact_digests, "settings": tokenizer},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _embedding_metadata() -> dict[str, object]:
    runtime = MODEL_BUNDLE_MANIFEST["runtime"]
    model_file = next(file for file in MODEL_BUNDLE_MANIFEST["files"] if file["path"] == "onnx/model.onnx")
    return {
        "model_id": MODEL_BUNDLE_MANIFEST["model_id"],
        "model_hash": model_file["sha256"],
        "model_fingerprint": model_file["sha256"],
        "model_version": MODEL_BUNDLE_MANIFEST["model_version"],
        "tokenizer_version": MODEL_BUNDLE_MANIFEST["tokenizer_version"],
        "tokenizer_fingerprint": _tokenizer_fingerprint(MODEL_BUNDLE_MANIFEST),
        "runtime_name": runtime["name"],
        "runtime_version": runtime["version"],
        "dimension": MODEL_BUNDLE_MANIFEST["dimension"],
        "dtype": MODEL_BUNDLE_MANIFEST["dtype"],
        "pooling": MODEL_BUNDLE_MANIFEST["pooling"],
        "normalization": MODEL_BUNDLE_MANIFEST["normalization"],
        "metric": MODEL_BUNDLE_MANIFEST["metric"],
        "embedding_version": EMBEDDING_VERSION,
        "embedding_format_version": EMBEDDING_FORMAT_VERSION,
        "manifest_fingerprint": manifest_fingerprint(),
    }


def _publish_generation(
    connection: sqlite3.Connection,
    generation: int,
    semantic_state: str = SEMANTIC_STATE_DISABLED,
    semantic_reason: str = SEMANTIC_DISABLED_REASON,
) -> None:
    metadata = _embedding_metadata()
    connection.execute(
        """INSERT INTO embedding_metadata
           (generation, created_at, semantic_state, semantic_reason, model_id,
            model_hash, model_fingerprint, model_version,
            tokenizer_version, tokenizer_fingerprint, runtime_name, runtime_version,
            dimension, dtype, pooling, normalization, metric, embedding_version,
            embedding_format_version, manifest_fingerprint)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (generation, time.time(), semantic_state, semantic_reason,
         *(metadata[key] for key in (
            "model_id", "model_hash", "model_fingerprint", "model_version",
            "tokenizer_version", "tokenizer_fingerprint", "runtime_name", "runtime_version",
            "dimension", "dtype", "pooling", "normalization", "metric", "embedding_version",
            "embedding_format_version", "manifest_fingerprint",
        ))),
    )
    fingerprint = str(metadata["manifest_fingerprint"])
    connection.execute(
        """INSERT INTO metadata(key, value) VALUES
           ('semantic_state', ?),
           ('semantic_reason', ?),
           ('semantic_generation', ?),
           ('semantic_manifest_fingerprint', ?)
           ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
        (semantic_state, semantic_reason, str(generation), fingerprint),
    )


def _reset_schema(connection: sqlite3.Connection, generation: str | None = None) -> None:
    for statement in DROP_SCHEMA_STATEMENTS:
        connection.execute(statement)
    _create_schema(connection)
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    if generation is not None:
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES ('generation', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (generation,),
        )


def connect(database: Path, initialize: bool = True, readonly: bool = False) -> sqlite3.Connection:
    if readonly and initialize:
        raise ValueError("read-only connections cannot initialize the schema")
    if initialize and not readonly:
        database.parent.mkdir(parents=True, exist_ok=True)
    if readonly:
        uri = f"{database.resolve().as_uri()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
    else:
        connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    if not readonly:
        connection.execute("PRAGMA foreign_keys = ON")
    if initialize:
        connection.execute("BEGIN IMMEDIATE")
        try:
            stored_version = connection.execute("PRAGMA user_version").fetchone()[0]
            if stored_version != SCHEMA_VERSION:
                _reset_schema(connection)
            else:
                _create_schema(connection)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return connection


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
    connection.execute("DELETE FROM embeddings WHERE section_id IN (SELECT section_id FROM sections WHERE canonical_path = ?)", (canonical,))
    connection.execute("DELETE FROM sections_fts WHERE section_id IN (SELECT section_id FROM sections WHERE canonical_path = ?)", (canonical,))
    connection.execute("DELETE FROM sections WHERE canonical_path = ?", (canonical,))
    connection.execute("DELETE FROM files WHERE canonical_path = ?", (canonical,))


def _sources_unchanged(config: Config, observed: dict[str, str]) -> bool:
    current = _files(config)
    if set(current) != set(observed):
        return False
    try:
        return {
            canonical: content_hash(path.read_bytes())
            for canonical, (_, path) in current.items()
        } == observed
    except OSError:
        return False


def index(config: Config, rebuild: bool = False, encoder: object | None = None) -> int:
    found = _files(config)
    config.database.parent.mkdir(parents=True, exist_ok=True)
    connection = connect(config.database, initialize=False)
    try:
        connection.execute("BEGIN IMMEDIATE")
        stored_version = connection.execute("PRAGMA user_version").fetchone()[0]
        if stored_version != SCHEMA_VERSION:
            _reset_schema(connection)
        else:
            _create_schema(connection)
        if rebuild and stored_version == SCHEMA_VERSION:
            generation_row = connection.execute("SELECT value FROM metadata WHERE key = 'generation'").fetchone()
            _reset_schema(connection, generation_row["value"] if generation_row else None)
        generation_row = connection.execute("SELECT value FROM metadata WHERE key = 'generation'").fetchone()
        old_generation = int(generation_row["value"]) if generation_row else 0
        generation = old_generation + 1
        existing = {row["canonical_path"]: row for row in connection.execute("SELECT * FROM files")}
        observed_source_hashes: dict[str, str] = {}
        for canonical, (root, path) in found.items():
            stat = path.stat()
            old = existing.get(canonical)
            raw_bytes = path.read_bytes()
            content = raw_bytes.decode("utf-8")
            document = split_markdown(canonical, content, config.section_bytes, raw_bytes)
            observed_source_hashes[canonical] = document.content_hash
            if old and old["root_id"] == root.id and old["mtime_ns"] == stat.st_mtime_ns and old["content_hash"] == document.content_hash:
                connection.execute("UPDATE sections SET index_generation = ? WHERE canonical_path = ?",
                                   (generation, canonical))
                continue
            _remove_file(connection, canonical)
            for section in document.sections:
                connection.execute(
                    """INSERT INTO sections
                       (section_id, canonical_path, heading, heading_path, line_start,
                        line_end, root_id, text, content_hash, index_generation,
                        indexed_at, note_type, updated_date, created_date, links)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (section.section_id, canonical, section.heading, "\n".join(section.heading_path), section.line_start,
                     section.line_end, root.id, section.text, document.content_hash, generation, time.time(),
                     document.frontmatter.note_type, document.frontmatter.updated, document.frontmatter.created,
                     json.dumps(explicit_links(content))),
                )
                connection.execute("INSERT INTO sections_fts(section_id, heading, text) VALUES (?, ?, ?)",
                                   (section.section_id, section.heading or "", section.text))
            connection.execute("INSERT INTO files VALUES (?, ?, ?, ?)", (canonical, root.id, stat.st_mtime_ns, document.content_hash))
        current = set(found)
        for canonical in set(existing) - current:
            _remove_file(connection, canonical)
        semantic_state = SEMANTIC_STATE_DISABLED
        semantic_reason = SEMANTIC_DISABLED_REASON
        backend_reason = None if encoder is not None else _semantic_backend_reason(config)
        if backend_reason is not None:
            semantic_reason = backend_reason
        else:
            to_encode: list[sqlite3.Row] = []
            try:
                section_rows = connection.execute("SELECT section_id, canonical_path, text, content_hash FROM sections").fetchall()
                stored_embeddings = {
                    row["section_id"]: row
                    for row in connection.execute("SELECT * FROM embeddings").fetchall()
                }
                expected_manifest = manifest_fingerprint()
                to_encode = [
                    row for row in section_rows
                    if (row["section_id"] not in stored_embeddings
                        or stored_embeddings[row["section_id"]]["content_hash"] != row["content_hash"]
                        or stored_embeddings[row["section_id"]]["manifest_fingerprint"] != expected_manifest
                        or stored_embeddings[row["section_id"]]["embedding_version"] != EMBEDDING_VERSION
                        or stored_embeddings[row["section_id"]]["embedding_format_version"] != EMBEDDING_FORMAT_VERSION
                        or len(stored_embeddings[row["section_id"]]["vector"]) != int(MODEL_BUNDLE_MANIFEST["dimension"]) * 4)
                ]
                if to_encode:
                    if encoder is None:
                        encoder = _OnnxEncoder(_resolve_active_bundle(config.bundle_dir))
                    vectors = _encoder_vectors(encoder, [row["text"] for row in to_encode])
                    if len(vectors) != len(to_encode):
                        raise SemanticError("embedding backend returned the wrong number of vectors")
                    for row, vector in zip(to_encode, vectors, strict=True):
                        connection.execute(
                            "INSERT INTO embeddings VALUES (?, ?, ?, ?, ?, ?, ?) "
                            "ON CONFLICT(section_id) DO UPDATE SET generation=excluded.generation, "
                            "content_hash=excluded.content_hash, embedding_version=excluded.embedding_version, "
                            "embedding_format_version=excluded.embedding_format_version, "
                            "manifest_fingerprint=excluded.manifest_fingerprint, vector=excluded.vector",
                            (row["section_id"], generation, row["content_hash"], EMBEDDING_VERSION,
                             EMBEDDING_FORMAT_VERSION, expected_manifest, vector),
                        )
                for row in section_rows:
                    connection.execute(
                        "UPDATE embeddings SET generation = ? WHERE section_id = ?",
                        (generation, row["section_id"]),
                    )
                valid_vectors = connection.execute(
                    """SELECT count(*) FROM sections s JOIN embeddings e ON e.section_id = s.section_id
                       WHERE e.generation = ? AND e.content_hash = s.content_hash
                         AND e.embedding_version = ? AND e.embedding_format_version = ?
                         AND e.manifest_fingerprint = ?""",
                    (generation, EMBEDDING_VERSION, EMBEDDING_FORMAT_VERSION, expected_manifest),
                ).fetchone()[0]
                if valid_vectors != len(section_rows):
                    raise SemanticError("vector table is incomplete")
                semantic_state = SEMANTIC_STATE_READY
                semantic_reason = ""
            except (BundleError, IndexError, OSError, SemanticError, TypeError, ValueError, RuntimeError, sqlite3.DatabaseError):
                for row in to_encode:
                    connection.execute("DELETE FROM embeddings WHERE section_id = ?", (row["section_id"],))
                semantic_state = SEMANTIC_STATE_STALE
                semantic_reason = SEMANTIC_REASON_REFRESH_FAILED
        if not _sources_unchanged(config, observed_source_hashes):
            connection.rollback()
            return old_generation
        connection.execute("INSERT INTO metadata(key, value) VALUES ('generation', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(generation),))
        _publish_generation(connection, generation, semantic_state, semantic_reason)
        connection.commit()
        return generation
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


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


def semantic_search(config: Config, query: str, project: str | None = None, root_id: str | None = None,
                    all_projects: bool = False, limit: int = 20, refresh: set[str] | None = None,
                    encoder: object | None = None) -> list[Candidate]:
    """Use semantic candidates, with lexical fallback on failure."""
    candidates = vector_candidates(config, query, project, root_id, all_projects, limit, refresh, encoder)
    return candidates if candidates else search(config, query, project, root_id, all_projects, limit, refresh)


def search(config: Config, query: str, project: str | None = None, root_id: str | None = None,
           all_projects: bool = False, limit: int = 20, refresh: set[str] | None = None,
           semantic: bool = False) -> list[Candidate]:
    if semantic:
        return semantic_search(config, query, project, root_id, all_projects, limit, refresh)
    if limit < 1:
        raise ValueError("limit must be positive")
    if not config.database.exists():
        return []

    if root_id and not any(root.id == root_id for root in config.roots):
        return []
    global_paths = sorted(str(path) for path in config.global_notes)
    selected_root = next((root for root in config.roots if root.id == root_id), None)
    effective_project = project or (selected_root.project if root_id and not all_projects else None)
    if all_projects:
        scope_root_ids = [root.id for root in config.roots]
    elif effective_project is None:
        scope_root_ids = []
    else:
        scope_root_ids = [root.id for root in config.roots if root.project == effective_project]
    if root_id:
        scope_root_ids = [candidate_id for candidate_id in scope_root_ids if candidate_id == root_id]

    def placeholders(values: list[str]) -> str:
        return ", ".join("?" for _ in values) or "NULL"

    connection = connect(config.database, initialize=False, readonly=True)
    try:
        _require_schema_version(connection)
        fts, tokens = _fts_query(query)
        if not fts:
            return []
        conditions = ["sections_fts MATCH ?"]
        params: list[object] = [fts]
        scope_conditions: list[str] = []
        if scope_root_ids:
            scope_conditions.append(f"s.root_id IN ({placeholders(scope_root_ids)})")
            params.extend(scope_root_ids)
        if global_paths:
            scope_conditions.append(f"s.canonical_path IN ({placeholders(global_paths)})")
            params.extend(global_paths)
        conditions.append("(" + " OR ".join(scope_conditions) + ")" if scope_conditions else "0")
        candidate_limit = min(MAX_CANDIDATES, max(limit * 5, 50))
        rows_by_id: dict[str, sqlite3.Row] = {}
        for lane in (None, "heading", "text"):
            lane_params = list(params)
            lane_params[0] = fts if lane is None else _column_query(lane, fts)
            rows_by_id.update({
                row["section_id"]: row
                for row in connection.execute(
                    "SELECT s.*, bm25(sections_fts) AS fts_score FROM sections s JOIN sections_fts ON sections_fts.section_id = s.section_id WHERE "
                    + " AND ".join(conditions) + " ORDER BY fts_score, s.section_id LIMIT ?",
                    (*lane_params, candidate_limit),
                ).fetchall()
            })
        rows = list(rows_by_id.values())
    finally:
        connection.close()

    results: list[Candidate] = []
    source_cache: dict[str, tuple[Root, Path] | None] = {}
    hash_cache: dict[str, str | None] = {}
    lower_tokens = {token.lower() for token in tokens}
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
    results.sort(key=lambda item: (-item.score, item.section_id))
    return results[:limit]


def excerpt(text: str, max_chars: int = 500) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= max_chars else clean[:max_chars].rstrip() + "..."
