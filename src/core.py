from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import time
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import Config, Root, _contained
from .model_bundle import (
    MODEL_BUNDLE_MANIFEST,
    BundleError,
    _resolve_active_bundle,
    manifest_fingerprint,
    package_version,
    verify_bundle,
)
from .splitter import content_hash, explicit_links, split_markdown

MAX_QUERY_TERMS = 64
MAX_CANDIDATES = 200
SCHEMA_VERSION = 2
EMBEDDING_VERSION = 1
EMBEDDING_FORMAT_VERSION = 1
SEMANTIC_STATE_DISABLED = "disabled"
SEMANTIC_DISABLED_REASON = "EXTRA_MISSING"
SEMANTIC_STATE_READY = "ready"
SEMANTIC_STATE_STALE = "stale"
SEMANTIC_REASON_MANIFEST_CHANGED = "MANIFEST_CHANGED"
SEMANTIC_REASON_VECTOR_TABLE_CORRUPT = "VECTOR_TABLE_CORRUPT"
SEMANTIC_REASON_REFRESH_FAILED = "REFRESH_FAILED"
SEMANTIC_REASON_MODEL_MISSING = "MODEL_MISSING"
SEMANTIC_REASON_MODEL_HASH_MISMATCH = "MODEL_HASH_MISMATCH"
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


class SemanticError(RuntimeError):
    """Raised internally when the optional semantic lane is unavailable."""


def _optional_numpy():
    try:
        import numpy
    except ImportError as exc:
        raise SemanticError(SEMANTIC_DISABLED_REASON) from exc
    return numpy


def _semantic_backend_reason(config: Config) -> str | None:
    if (package_version("numpy") != MODEL_BUNDLE_MANIFEST["numpy_version"]
            or package_version("onnxruntime") != MODEL_BUNDLE_MANIFEST["runtime"]["version"]):
        return SEMANTIC_DISABLED_REASON
    bundle = config.bundle_dir
    if not bundle.exists() and not bundle.is_symlink():
        return SEMANTIC_REASON_MODEL_MISSING
    try:
        verify_bundle(bundle)
    except BundleError as exc:
        if "hash mismatch" in str(exc):
            return SEMANTIC_REASON_MODEL_HASH_MISMATCH
        return SEMANTIC_REASON_MODEL_MISSING
    return None


def _normalize_vector(values: Iterable[float]) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    norm = math.sqrt(sum(value * value for value in vector))
    if not vector or not math.isfinite(norm) or norm == 0:
        raise SemanticError("embedding vector is not finite and non-zero")
    return tuple(value / norm for value in vector)


def _pack_vector(values: Iterable[float]) -> bytes:
    vector = _normalize_vector(values)
    if len(vector) != int(MODEL_BUNDLE_MANIFEST["dimension"]):
        raise SemanticError("embedding dimension does not match the manifest")
    return struct.pack("<" + "f" * len(vector), *vector)


def _unpack_vector(blob: bytes, dimension: int) -> tuple[float, ...]:
    expected_size = dimension * 4
    if len(blob) != expected_size:
        raise SemanticError("vector BLOB has the wrong size")
    return struct.unpack("<" + "f" * dimension, blob)


class _StdlibTokenizer:
    def __init__(self, bundle: Path):
        settings = MODEL_BUNDLE_MANIFEST["tokenizer"]
        tokenizer_files = settings["files"]
        if "tokenizer.json" not in tokenizer_files:
            raise SemanticError("tokenizer manifest does not include tokenizer.json")
        try:
            tokenizer = json.loads((bundle / "tokenizer.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SemanticError("tokenizer.json is unreadable") from exc
        model = tokenizer.get("model")
        normalizer = tokenizer.get("normalizer")
        if not isinstance(model, dict) or model.get("type") != "WordPiece":
            raise SemanticError("tokenizer model is not WordPiece")
        if not isinstance(normalizer, dict) or normalizer.get("type") != "BertNormalizer":
            raise SemanticError("tokenizer normalizer is not BertNormalizer")
        vocabulary = model.get("vocab")
        if not isinstance(vocabulary, dict) or not all(isinstance(key, str) and isinstance(value, int)
                                                       for key, value in vocabulary.items()):
            raise SemanticError("tokenizer vocabulary is invalid")
        self.vocabulary = vocabulary
        self.lower_case = bool(normalizer.get("lowercase", settings["do_lower_case"]))
        self.strip_accents = normalizer.get("strip_accents")
        if self.strip_accents is None:
            self.strip_accents = self.lower_case
        self.clean_text = bool(normalizer.get("clean_text", True))
        self.handle_chinese_chars = bool(normalizer.get("handle_chinese_chars", True))
        self.max_length = int(settings["max_length"])
        self.max_input_chars_per_word = int(model.get("max_input_chars_per_word", 100))
        self.unknown_token = str(model.get("unk_token", settings["special_tokens"]["unk"]))
        self.cls_token = str(settings["special_tokens"]["cls"])
        self.sep_token = str(settings["special_tokens"]["sep"])
        self.special_tokens = {
            item["content"]: item["id"]
            for item in tokenizer.get("added_tokens", [])
            if isinstance(item, dict) and item.get("special") is True
            and isinstance(item.get("content"), str) and isinstance(item.get("id"), int)
        }

    def _wordpiece(self, token: str) -> list[str]:
        if len(token) > self.max_input_chars_per_word:
            return [self.unknown_token]
        if token in self.vocabulary:
            return [token]
        pieces: list[str] = []
        start = 0
        while start < len(token):
            end = len(token)
            match = None
            while start < end:
                candidate = token[start:end]
                if start:
                    candidate = "##" + candidate
                if candidate in self.vocabulary:
                    match = candidate
                    break
                end -= 1
            if match is None:
                return ["[UNK]"]
            pieces.append(match)
            start = end
        return pieces

    @staticmethod
    def _is_control(character: str) -> bool:
        return unicodedata.category(character) in {"Cc", "Cf"} and character not in "\t\n\r"

    @staticmethod
    def _is_whitespace(character: str) -> bool:
        return character in " \t\n\r" or unicodedata.category(character) == "Zs"

    @staticmethod
    def _is_chinese_character(character: str) -> bool:
        codepoint = ord(character)
        return (
            0x4E00 <= codepoint <= 0x9FFF
            or 0x3400 <= codepoint <= 0x4DBF
            or 0x20000 <= codepoint <= 0x2A6DF
            or 0x2A700 <= codepoint <= 0x2B73F
            or 0x2B740 <= codepoint <= 0x2B81F
            or 0x2B820 <= codepoint <= 0x2CEAF
            or 0xF900 <= codepoint <= 0xFAFF
            or 0x2F800 <= codepoint <= 0x2FA1F
        )

    @staticmethod
    def _is_punctuation(character: str) -> bool:
        codepoint = ord(character)
        return (33 <= codepoint <= 47 or 58 <= codepoint <= 64 or 91 <= codepoint <= 96
                or 123 <= codepoint <= 126 or unicodedata.category(character).startswith("P"))

    def _normalize(self, text: str) -> str:
        characters: list[str] = []
        for character in text:
            if self.clean_text and (character == "\x00" or self._is_control(character)):
                continue
            if self.handle_chinese_chars and self._is_chinese_character(character):
                characters.extend((" ", character, " "))
            else:
                characters.append(character)
        text = "".join(characters)
        if self.strip_accents:
            text = unicodedata.normalize("NFD", text)
            text = "".join(character for character in text if unicodedata.category(character) != "Mn")
        if self.lower_case:
            text = text.lower()
        return text

    def _pretokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        current: list[str] = []
        for character in text:
            if self._is_whitespace(character):
                if current:
                    tokens.append("".join(current))
                    current = []
            elif self._is_punctuation(character):
                if current:
                    tokens.append("".join(current))
                    current = []
                tokens.append(character)
            else:
                current.append(character)
        if current:
            tokens.append("".join(current))
        return tokens

    def tokens(self, text: str) -> list[int]:
        body: list[int] = []
        remaining = text
        while remaining:
            matches = [(remaining.find(token), token, token_id) for token, token_id in self.special_tokens.items()
                       if remaining.find(token) >= 0]
            if not matches:
                pieces = [piece for token in self._pretokenize(self._normalize(remaining)) for piece in self._wordpiece(token)]
                body.extend(self.vocabulary.get(piece, self.vocabulary[self.unknown_token]) for piece in pieces)
                break
            position, token, token_id = min(matches, key=lambda item: (item[0], -len(item[1])))
            if position:
                pieces = [piece for chunk in self._pretokenize(self._normalize(remaining[:position]))
                          for piece in self._wordpiece(chunk)]
                body.extend(self.vocabulary.get(piece, self.vocabulary[self.unknown_token]) for piece in pieces)
            body.append(token_id)
            remaining = remaining[position + len(token):]
        body = body[: self.max_length - 2]
        ids = [self.vocabulary[self.cls_token]]
        ids.extend(body)
        ids.append(self.vocabulary[self.sep_token])
        return ids


class _OnnxEncoder:
    def __init__(self, bundle: Path):
        try:
            import onnxruntime
        except ImportError as exc:
            raise SemanticError(SEMANTIC_DISABLED_REASON) from exc
        numpy = _optional_numpy()
        self.numpy = numpy
        self.tokenizer = _StdlibTokenizer(bundle)
        self.session = onnxruntime.InferenceSession(
            str(bundle / "onnx" / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )

    def encode(self, texts: Sequence[str]):
        if not texts:
            return self.numpy.empty((0, int(MODEL_BUNDLE_MANIFEST["dimension"])), dtype=self.numpy.float32)
        encoded = [self.tokenizer.tokens(text) for text in texts]
        width = max(len(item) for item in encoded)
        pad_id = self.tokenizer.vocabulary.get("[PAD]", 0)
        input_ids = self.numpy.asarray([item + [pad_id] * (width - len(item)) for item in encoded], dtype=self.numpy.int64)
        attention = (input_ids != pad_id).astype(self.numpy.int64)
        token_types = self.numpy.zeros_like(input_ids)
        names = {item.name for item in self.session.get_inputs()}
        inputs = {"input_ids": input_ids, "attention_mask": attention, "token_type_ids": token_types}
        outputs = self.session.run(None, {name: value for name, value in inputs.items() if name in names})
        hidden = self.numpy.asarray(outputs[0], dtype=self.numpy.float32)
        if hidden.ndim == 2:
            norms = self.numpy.linalg.norm(hidden, axis=1, keepdims=True)
            return (hidden / self.numpy.maximum(norms, self.numpy.finfo(self.numpy.float32).tiny)).astype(self.numpy.float32)
        mask = attention[..., None].astype(self.numpy.float32)
        pooled = (hidden * mask).sum(axis=1) / self.numpy.maximum(mask.sum(axis=1), 1.0)
        norms = self.numpy.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / self.numpy.maximum(norms, self.numpy.finfo(self.numpy.float32).tiny)).astype(self.numpy.float32)


def _encoder_vectors(encoder: object, texts: Sequence[str]) -> list[bytes]:
    raw = encoder.encode(texts) if hasattr(encoder, "encode") else encoder(texts)
    return [_pack_vector(vector) for vector in raw]


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


def _require_schema_version(connection: sqlite3.Connection) -> None:
    stored_version = connection.execute("PRAGMA user_version").fetchone()[0]
    if stored_version != SCHEMA_VERSION:
        raise ValueError(f"index schema is v{stored_version}; run `pausanias index` to rebuild")


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
    connection.execute("DELETE FROM embeddings WHERE section_id IN (SELECT section_id FROM sections WHERE canonical_path = ?)", (canonical,))
    connection.execute("DELETE FROM sections_fts WHERE section_id IN (SELECT section_id FROM sections WHERE canonical_path = ?)", (canonical,))
    connection.execute("DELETE FROM sections WHERE canonical_path = ?", (canonical,))
    connection.execute("DELETE FROM files WHERE canonical_path = ?", (canonical,))


def _sources_unchanged(found: dict[str, tuple[Root, Path]], observed: dict[str, str]) -> bool:
    for canonical, (_, path) in found.items():
        try:
            if content_hash(path.read_bytes()) != observed.get(canonical):
                return False
        except OSError:
            return False
    return True


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
        if not _sources_unchanged(found, observed_source_hashes):
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


def _scope(config: Config, project: str | None, root_id: str | None, all_projects: bool) -> tuple[list[str], list[str], str | None]:
    if root_id and not any(root.id == root_id for root in config.roots):
        return [], [], None
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
    return scope_root_ids, global_paths, effective_project


def _load_vector_matrix(numpy, rows: Sequence[sqlite3.Row], dimension: int):
    matrix = numpy.empty((len(rows), dimension), dtype=numpy.float32)
    expected_size = dimension * 4
    vector_dtype = numpy.dtype("<f4")
    for row_number, row in enumerate(rows):
        blob = row["vector"]
        if len(blob) != expected_size:
            raise SemanticError("vector BLOB has the wrong size")
        matrix[row_number, :] = numpy.frombuffer(blob, dtype=vector_dtype, count=dimension)
    if rows and not bool(numpy.isfinite(matrix).all()):
        raise SemanticError("vector BLOB contains non-finite values")
    return matrix


def scan_vectors(
    config: Config,
    query_vector: Sequence[float],
    project: str | None = None,
    root_id: str | None = None,
    all_projects: bool = False,
    limit: int = 20,
    refresh: set[str] | None = None,
) -> list[Candidate]:
    """Return bounded, scope-filtered exact cosine candidates for a vector."""
    if limit < 1:
        raise ValueError("limit must be positive")
    if not config.database.exists():
        return []
    scope_root_ids, global_paths, effective_project = _scope(config, project, root_id, all_projects)
    if root_id and not scope_root_ids:
        return []
    try:
        numpy = _optional_numpy()
        normalized_query = _normalize_vector(query_vector)
        if len(normalized_query) != int(MODEL_BUNDLE_MANIFEST["dimension"]):
            raise SemanticError("query embedding dimension does not match the manifest")
        query = numpy.asarray(normalized_query, dtype=numpy.float32)
    except (SemanticError, ValueError, TypeError):
        return []

    def placeholders(values: list[str]) -> str:
        return ", ".join("?" for _ in values) or "NULL"

    connection = connect(config.database, initialize=False, readonly=True)
    try:
        connection.execute("BEGIN")
        _require_schema_version(connection)
        state = dict(connection.execute("SELECT key, value FROM metadata WHERE key IN ('generation', 'semantic_state', 'semantic_generation')").fetchall())
        if state.get("semantic_state") != SEMANTIC_STATE_READY or state.get("generation") != state.get("semantic_generation"):
            return []
        generation = int(state["generation"])
        metadata = connection.execute("SELECT manifest_fingerprint FROM embedding_metadata WHERE generation = ?", (generation,)).fetchone()
        if metadata is None or metadata["manifest_fingerprint"] != manifest_fingerprint():
            return []
        complete_count = connection.execute(
            """SELECT count(*) FROM sections s JOIN embeddings e ON e.section_id = s.section_id
                 WHERE s.index_generation = ? AND e.generation = ?
                   AND e.content_hash = s.content_hash AND e.embedding_version = ?
                 AND e.embedding_format_version = ? AND e.manifest_fingerprint = ?
                 AND length(e.vector) = ?""",
            (generation, generation, EMBEDDING_VERSION, EMBEDDING_FORMAT_VERSION, manifest_fingerprint(),
             int(MODEL_BUNDLE_MANIFEST["dimension"]) * 4),
        ).fetchone()[0]
        section_count = connection.execute("SELECT count(*) FROM sections WHERE index_generation = ?", (generation,)).fetchone()[0]
        if complete_count != section_count:
            return []
        conditions = [
            "s.index_generation = ?",
            "e.generation = ?",
            "e.embedding_version = ?",
            "e.embedding_format_version = ?",
            "e.manifest_fingerprint = ?",
        ]
        params: list[object] = [generation, generation, EMBEDDING_VERSION, EMBEDDING_FORMAT_VERSION, manifest_fingerprint()]
        scope_conditions: list[str] = []
        if scope_root_ids:
            scope_conditions.append(f"s.root_id IN ({placeholders(scope_root_ids)})")
            params.extend(scope_root_ids)
        if global_paths:
            scope_conditions.append(f"s.canonical_path IN ({placeholders(global_paths)})")
            params.extend(global_paths)
        conditions.append("(" + " OR ".join(scope_conditions) + ")" if scope_conditions else "0")
        rows = connection.execute(
            "SELECT s.*, e.vector FROM sections s JOIN embeddings e ON e.section_id = s.section_id WHERE "
            + " AND ".join(conditions),
            params,
        ).fetchall()
        dimension = int(MODEL_BUNDLE_MANIFEST["dimension"])
        try:
            matrix = _load_vector_matrix(numpy, rows, dimension)
            scores = matrix @ query if rows else numpy.empty(0, dtype=numpy.float32)
        except (SemanticError, ValueError, TypeError, RuntimeError):
            return []
    finally:
        connection.close()

    scored: list[tuple[float, sqlite3.Row]] = []
    scored = sorted(
        ((float(score), row) for score, row in zip(scores, rows, strict=True)),
        key=lambda item: (-item[0], item[1]["section_id"]),
    )[: min(MAX_CANDIDATES, max(limit * 5, 50))]

    results: list[Candidate] = []
    source_cache: dict[str, tuple[Root, Path] | None] = {}
    hash_cache: dict[str, str | None] = {}
    for score, row in scored:
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
        if hash_cache[canonical] != row["content_hash"]:
            if refresh is not None:
                refresh.add(canonical)
            continue
        reason = "global-note inclusion, semantic match" if is_global else "semantic match"
        results.append(Candidate(
            row["section_id"], canonical, row["heading"], tuple(filter(None, row["heading_path"].split("\n"))),
            row["line_start"], row["line_end"], current_root.id, current_root.project, row["text"],
            row["content_hash"], row["note_type"], row["updated_date"], row["created_date"], score, reason,
        ))
    return results[:limit]


def vector_candidates(
    config: Config,
    query: str,
    project: str | None = None,
    root_id: str | None = None,
    all_projects: bool = False,
    limit: int = 20,
    refresh: set[str] | None = None,
    encoder: object | None = None,
) -> list[Candidate]:
    """Encode a query and return bounded exact cosine candidates."""
    if encoder is None:
        if _semantic_backend_reason(config) is not None:
            return []
        try:
            encoder = _OnnxEncoder(_resolve_active_bundle(config.bundle_dir))
        except (BundleError, OSError, SemanticError):
            return []
    try:
        vector = _encoder_vectors(encoder, [query])[0]
        return scan_vectors(config, _unpack_vector(vector, int(MODEL_BUNDLE_MANIFEST["dimension"])), project, root_id,
                            all_projects, limit, refresh)
    except (IndexError, SemanticError, ValueError, TypeError, RuntimeError, OSError, sqlite3.DatabaseError):
        return []


def semantic_search(
    config: Config,
    query: str,
    project: str | None = None,
    root_id: str | None = None,
    all_projects: bool = False,
    limit: int = 20,
    refresh: set[str] | None = None,
    encoder: object | None = None,
) -> list[Candidate]:
    """Use semantic candidates when ready, with lexical fallback on failure."""
    candidates = vector_candidates(config, query, project, root_id, all_projects, limit, refresh, encoder)
    if candidates:
        return candidates
    return search(config, query, project, root_id, all_projects, limit, refresh)


def excerpt(text: str, max_chars: int = 500) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= max_chars else clean[:max_chars].rstrip() + "..."
