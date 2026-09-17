from __future__ import annotations

import json
import math
import sqlite3
import struct
import time
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import Config, Root
from .model_bundle import (
    MODEL_BUNDLE_MANIFEST,
    BundleError,
    _resolve_active_bundle,
    manifest_fingerprint,
    package_version,
    verify_bundle,
)
from .splitter import content_hash
from .store import connect_readonly, require_schema_version, safe_source

CANDIDATE_OVERSAMPLE = 5
MIN_CANDIDATES = 50
MAX_CANDIDATES = 200
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
    lexical_rank: int | None = None
    vector_rank: int | None = None
    lexical_score: float | None = None
    vector_score: float | None = None
    fused_score: float | None = None
    lane: str = "lexical"
    guard_reason: str | None = None


class SemanticError(RuntimeError):
    """Raised internally when the optional semantic lane is unavailable."""


def optional_numpy():
    try:
        import numpy
    except ImportError as exc:
        raise SemanticError(SEMANTIC_DISABLED_REASON) from exc
    return numpy


def semantic_backend_reason(config: Config, version_checker=package_version) -> str | None:
    bundle = config.bundle_dir
    if not bundle.exists() and not bundle.is_symlink():
        return SEMANTIC_REASON_MODEL_MISSING
    if (version_checker("numpy") != MODEL_BUNDLE_MANIFEST["numpy_version"]
            or version_checker("onnxruntime") != MODEL_BUNDLE_MANIFEST["runtime"]["version"]):
        return SEMANTIC_DISABLED_REASON
    try:
        verify_bundle(bundle)
    except BundleError as exc:
        if "hash mismatch" in str(exc):
            return SEMANTIC_REASON_MODEL_HASH_MISMATCH
        return SEMANTIC_REASON_MODEL_MISSING
    return None


def normalize_vector(values: Iterable[float]) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    norm = math.sqrt(sum(value * value for value in vector))
    if not vector or not math.isfinite(norm) or norm == 0:
        raise SemanticError("embedding vector is not finite and non-zero")
    return tuple(value / norm for value in vector)


def pack_vector(values: Iterable[float]) -> bytes:
    vector = normalize_vector(values)
    if len(vector) != int(MODEL_BUNDLE_MANIFEST["dimension"]):
        raise SemanticError("embedding dimension does not match the manifest")
    return struct.pack("<" + "f" * len(vector), *vector)


def unpack_vector(blob: bytes, dimension: int) -> tuple[float, ...]:
    expected_size = dimension * 4
    if len(blob) != expected_size:
        raise SemanticError("vector BLOB has the wrong size")
    return struct.unpack("<" + "f" * dimension, blob)


class StdlibTokenizer:
    def __init__(self, bundle: Path):
        try:
            tokenizer = json.loads((bundle / "tokenizer.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SemanticError("tokenizer.json is unreadable") from exc
        if not isinstance(tokenizer, dict):
            raise SemanticError("tokenizer.json must contain an object")

        model = tokenizer.get("model")
        normalizer = tokenizer.get("normalizer")
        pre_tokenizer = tokenizer.get("pre_tokenizer")
        post_processor = tokenizer.get("post_processor")
        if not isinstance(model, dict) or model.get("type") != "WordPiece":
            raise SemanticError("tokenizer model is not WordPiece")
        if not isinstance(normalizer, dict) or normalizer.get("type") != "BertNormalizer":
            raise SemanticError("tokenizer normalizer is not BertNormalizer")
        if not isinstance(pre_tokenizer, dict) or pre_tokenizer.get("type") != "BertPreTokenizer":
            raise SemanticError("tokenizer pre-tokenizer is not BertPreTokenizer")
        if not isinstance(post_processor, dict) or post_processor.get("type") != "TemplateProcessing":
            raise SemanticError("tokenizer post-processor is not TemplateProcessing")

        vocabulary = model.get("vocab")
        if not isinstance(vocabulary, dict) or not all(
            isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool)
            for key, value in vocabulary.items()
        ):
            raise SemanticError("tokenizer vocabulary is invalid")
        self.vocabulary = vocabulary

        self.lower_case = self._bool_setting(normalizer, "lowercase")
        self.clean_text = self._bool_setting(normalizer, "clean_text")
        self.handle_chinese_chars = self._bool_setting(normalizer, "handle_chinese_chars")
        strip_accents = normalizer.get("strip_accents")
        if strip_accents is not None and not isinstance(strip_accents, bool):
            raise SemanticError("tokenizer accent rule is unsupported")
        self.strip_accents = self.lower_case if strip_accents is None else strip_accents

        prefix = model.get("continuing_subword_prefix")
        unknown_token = model.get("unk_token")
        max_input_chars = model.get("max_input_chars_per_word")
        if not isinstance(prefix, str) or not isinstance(unknown_token, str):
            raise SemanticError("tokenizer WordPiece settings are invalid")
        if not isinstance(max_input_chars, int) or isinstance(max_input_chars, bool) or max_input_chars < 1:
            raise SemanticError("tokenizer word length limit is invalid")
        if unknown_token not in vocabulary:
            raise SemanticError("tokenizer unknown token is absent from vocabulary")
        self.continuing_subword_prefix = prefix
        self.unknown_token = unknown_token
        self.max_input_chars_per_word = max_input_chars

        truncation = tokenizer.get("truncation")
        if not isinstance(truncation, dict):
            raise SemanticError("tokenizer truncation settings are missing")
        max_length = truncation.get("max_length")
        strategy = truncation.get("strategy")
        direction = truncation.get("direction", "Right")
        stride = truncation.get("stride", 0)
        if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 1:
            raise SemanticError("tokenizer truncation length is invalid")
        if strategy not in {"LongestFirst", "OnlyFirst"}:
            raise SemanticError(f"tokenizer truncation strategy is unsupported: {strategy}")
        if direction not in {"Left", "Right"}:
            raise SemanticError(f"tokenizer truncation direction is unsupported: {direction}")
        if not isinstance(stride, int) or isinstance(stride, bool) or stride != 0:
            raise SemanticError("tokenizer truncation stride is unsupported")
        self.max_length = max_length
        self.truncation_direction = direction

        self.special_tokens = self._added_special_tokens(tokenizer.get("added_tokens"))
        self.template = self._single_template(post_processor)
        special_count = sum(len(ids) for kind, ids in self.template if kind == "special")
        if special_count >= max_length:
            raise SemanticError("tokenizer special tokens leave no room for input")

    @staticmethod
    def _bool_setting(settings: dict[str, object], name: str) -> bool:
        value = settings.get(name)
        if not isinstance(value, bool):
            raise SemanticError(f"tokenizer {name} rule is invalid")
        return value

    @staticmethod
    def _added_special_tokens(raw_tokens: object) -> dict[str, int]:
        if not isinstance(raw_tokens, list):
            raise SemanticError("tokenizer added tokens are invalid")
        tokens: dict[str, int] = {}
        for item in raw_tokens:
            if not isinstance(item, dict):
                raise SemanticError("tokenizer added token is invalid")
            if item.get("special") is not True:
                continue
            content = item.get("content")
            token_id = item.get("id")
            if not isinstance(content, str) or not isinstance(token_id, int) or isinstance(token_id, bool):
                raise SemanticError("tokenizer special token is invalid")
            tokens[content] = token_id
        return tokens

    @staticmethod
    def _single_template(post_processor: dict[str, object]) -> list[tuple[str, list[int]]]:
        template = post_processor.get("single")
        records = post_processor.get("special_tokens")
        if not isinstance(template, list) or not isinstance(records, dict):
            raise SemanticError("tokenizer single-sequence template is invalid")
        result: list[tuple[str, list[int]]] = []
        sequence_count = 0
        for item in template:
            if not isinstance(item, dict):
                raise SemanticError("tokenizer template item is invalid")
            if "Sequence" in item:
                sequence = item["Sequence"]
                if not isinstance(sequence, dict) or sequence.get("id") != "A" or sequence_count:
                    raise SemanticError("tokenizer single-sequence template is unsupported")
                sequence_count += 1
                result.append(("sequence", []))
                continue
            if "SpecialToken" in item:
                special = item["SpecialToken"]
                if not isinstance(special, dict):
                    raise SemanticError("tokenizer special template item is invalid")
                name = special.get("id")
                record = records.get(name) if isinstance(name, str) else None
                ids = record.get("ids") if isinstance(record, dict) else None
                if not isinstance(ids, list) or not all(isinstance(value, int) and not isinstance(value, bool) for value in ids):
                    raise SemanticError("tokenizer special template ids are invalid")
                result.append(("special", list(ids)))
                continue
            raise SemanticError("tokenizer template rule is unsupported")
        if sequence_count != 1:
            raise SemanticError("tokenizer single-sequence template has no input sequence")
        return result

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
                    candidate = self.continuing_subword_prefix + candidate
                if candidate in self.vocabulary:
                    match = candidate
                    break
                end -= 1
            if match is None:
                return [self.unknown_token]
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

    def _body_tokens(self, text: str) -> list[int]:
        body: list[int] = []
        remaining = text
        while remaining:
            matches = [(remaining.find(token), token, token_id) for token, token_id in self.special_tokens.items()
                       if remaining.find(token) >= 0]
            if not matches:
                pieces = [piece for token in self._pretokenize(self._normalize(remaining)) for piece in self._wordpiece(token)]
                body.extend(self.vocabulary[piece] for piece in pieces)
                break
            position, token, token_id = min(matches, key=lambda item: (item[0], -len(item[1])))
            if position:
                pieces = [piece for chunk in self._pretokenize(self._normalize(remaining[:position]))
                          for piece in self._wordpiece(chunk)]
                body.extend(self.vocabulary[piece] for piece in pieces)
            body.append(token_id)
            remaining = remaining[position + len(token):]
        return body

    def tokens(self, text: str) -> list[int]:
        special_count = sum(len(ids) for kind, ids in self.template if kind == "special")
        capacity = self.max_length - special_count
        body = self._body_tokens(text)
        if len(body) > capacity:
            body = body[:capacity] if self.truncation_direction == "Right" else body[-capacity:]
        result: list[int] = []
        for kind, ids in self.template:
            result.extend(ids if kind == "special" else body)
        return result


class OnnxEncoder:
    def __init__(self, bundle: Path):
        try:
            import onnxruntime
        except ImportError as exc:
            raise SemanticError(SEMANTIC_DISABLED_REASON) from exc
        numpy = optional_numpy()
        self.numpy = numpy
        self.tokenizer = StdlibTokenizer(bundle)
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


def encoder_vectors(encoder: object, texts: Sequence[str]) -> list[bytes]:
    raw = encoder.encode(texts) if hasattr(encoder, "encode") else encoder(texts)
    return [pack_vector(vector) for vector in raw]


def scope(config: Config, project: str | None, root_id: str | None, all_projects: bool) -> tuple[list[str], list[str], str | None]:
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


def load_vector_matrix(numpy, rows: Sequence, dimension: int, refresh: set[str] | None = None):
    valid_rows = []
    vectors = []
    expected_size = dimension * 4
    vector_dtype = numpy.dtype("<f4")
    for row in rows:
        blob = row["vector"]
        try:
            if len(blob) != expected_size:
                raise SemanticError("vector BLOB has the wrong size")
            vector = numpy.frombuffer(blob, dtype=vector_dtype, count=dimension)
            if not bool(numpy.isfinite(vector).all()):
                raise SemanticError("vector BLOB contains non-finite values")
        except (SemanticError, TypeError, ValueError):
            if refresh is not None:
                refresh.add(row["canonical_path"])
            continue
        valid_rows.append(row)
        vectors.append(vector)
    if not vectors:
        return numpy.empty((0, dimension), dtype=numpy.float32), valid_rows
    return numpy.asarray(vectors, dtype=numpy.float32), valid_rows


def scan_vectors(
    config: Config,
    query_vector: Sequence[float],
    project: str | None = None,
    root_id: str | None = None,
    all_projects: bool = False,
    limit: int = 20,
    refresh: set[str] | None = None,
    matrix_cache: dict[tuple[object, ...], tuple[object, list[object]]] | None = None,
    timings: dict[str, float] | None = None,
) -> list[Candidate]:
    """Return bounded, scope-filtered exact cosine candidates for a vector."""
    if timings is not None:
        timings["semantic_available"] = 0.0
    if limit < 1:
        raise ValueError("limit must be positive")
    if not config.database.exists():
        return []
    scope_root_ids, global_paths, effective_project = scope(config, project, root_id, all_projects)
    if root_id and not any(root.id == root_id for root in config.roots):
        return []
    try:
        numpy = optional_numpy()
        normalized_query = normalize_vector(query_vector)
        if len(normalized_query) != int(MODEL_BUNDLE_MANIFEST["dimension"]):
            raise SemanticError("query embedding dimension does not match the manifest")
        query = numpy.asarray(normalized_query, dtype=numpy.float32)
    except (SemanticError, ValueError, TypeError):
        return []

    def placeholders(values: list[str]) -> str:
        return ", ".join("?" for _ in values) or "NULL"

    connection = connect_readonly(config.database)
    try:
        connection.execute("BEGIN")
        require_schema_version(connection)
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
        if timings is not None:
            timings["semantic_available"] = 1.0
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
        cache_key = (generation, tuple(scope_root_ids), tuple(global_paths))
        cached = matrix_cache.get(cache_key) if matrix_cache is not None else None
        if cached is None:
            rows = connection.execute(
                "SELECT s.*, e.vector FROM sections s JOIN embeddings e ON e.section_id = s.section_id WHERE "
                + " AND ".join(conditions),
                params,
            ).fetchall()
        else:
            rows = cached[1]
        dimension = int(MODEL_BUNDLE_MANIFEST["dimension"])
        try:
            if cached is None:
                matrix_started = time.perf_counter()
                matrix, valid_rows = load_vector_matrix(numpy, rows, dimension, refresh)
                if timings is not None:
                    timings["matrix_load_ms"] = max((time.perf_counter() - matrix_started) * 1000.0, 0.000001)
                if matrix_cache is not None:
                    matrix_cache[cache_key] = (matrix, valid_rows)
            else:
                matrix, valid_rows = cached
                if timings is not None:
                    timings["matrix_load_ms"] = 0.000001
            scan_started = time.perf_counter()
            scores = matrix @ query if valid_rows else numpy.empty(0, dtype=numpy.float32)
            if timings is not None:
                timings["scan_ms"] = max((time.perf_counter() - scan_started) * 1000.0, 0.000001)
        except (ValueError, TypeError, RuntimeError):
            return []
    finally:
        connection.close()

    scored: list[tuple[float, object]] = []
    for score, row in zip(scores, valid_rows, strict=True):
        numeric_score = float(score)
        if math.isfinite(numeric_score):
            scored.append((numeric_score, row))
        elif refresh is not None:
            refresh.add(row["canonical_path"])
    candidate_limit = min(MAX_CANDIDATES, max(limit * CANDIDATE_OVERSAMPLE, MIN_CANDIDATES))
    scored = sorted(scored, key=lambda item: (-item[0], item[1]["section_id"]))[:candidate_limit]

    results: list[Candidate] = []
    source_cache: dict[str, tuple[Root, Path] | None] = {}
    hash_cache: dict[str, str | None] = {}
    for score, row in scored:
        canonical = row["canonical_path"]
        if canonical not in source_cache:
            source_cache[canonical] = safe_source(config, canonical)
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
            None, len(results) + 1, None, score, None, "semantic", None,
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
    matrix_cache: dict[tuple[object, ...], tuple[object, list[object]]] | None = None,
    timings: dict[str, float] | None = None,
) -> list[Candidate]:
    """Encode a query and return bounded exact cosine candidates."""
    if encoder is None:
        if semantic_backend_reason(config) is not None:
            return []
        try:
            encoder = OnnxEncoder(_resolve_active_bundle(config.bundle_dir))
        except (BundleError, OSError, SemanticError):
            return []
    try:
        encode_started = time.perf_counter()
        vector = encoder_vectors(encoder, [query])[0]
        if timings is not None:
            timings["encode_ms"] = max((time.perf_counter() - encode_started) * 1000.0, 0.000001)
        return scan_vectors(config, unpack_vector(vector, int(MODEL_BUNDLE_MANIFEST["dimension"])), project, root_id,
                            all_projects, limit, refresh, matrix_cache, timings)
    except (IndexError, SemanticError, ValueError, TypeError, RuntimeError, OSError, sqlite3.DatabaseError):
        return []
