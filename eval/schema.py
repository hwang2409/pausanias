"""Dependency-free result and checkpoint types for evaluation runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import os
import tempfile
from typing import Any


SCHEMA_VERSION = "pausanias.eval.v1"
CHECKPOINT_VERSION = "pausanias.eval.checkpoint.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(raw_path, path)
    except BaseException:
        try:
            os.unlink(raw_path)
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True)
class RetrievalResult:
    rank: int
    section_id: str
    source_path: str
    root_id: str
    project: str
    heading: str | None
    line_start: int
    line_end: int
    excerpt: str
    content_hash: str
    score: float
    reason: str


@dataclass(frozen=True)
class PromptMetadata:
    prompt_version: str | None = None
    template_hash: str | None = None
    slot_names: tuple[str, ...] = ()
    provider: str | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class CutoffOutcome:
    retrieved_count: int
    relevant_count: int
    recall: float | None
    precision: float | None
    mrr: float | None
    abstention_correct: bool | None
    forbidden_sources: int
    score: float
    passed: bool
    generated_answer: str | None = None
    judgment: str | None = None
    reason: str | None = None
    model: str | None = None
    error: str | None = None
    prompt_metadata: dict[str, PromptMetadata] = field(default_factory=dict)


@dataclass(frozen=True)
class Evaluation:
    case_id: str
    category: str
    query: str
    expected_sources: tuple[str, ...]
    ground_truth: str | None
    retrieval_results: tuple[RetrievalResult, ...]
    search_latency_ms: float
    cutoff_outcomes: dict[str, CutoffOutcome]
    score: float
    failure_reason: str | None = None


@dataclass(frozen=True)
class Metadata:
    benchmark: str
    run_id: str
    dataset: dict[str, Any]
    corpus_fingerprint: str | None
    index_generation: int | None
    case_set_fingerprint: str | None
    prompt_version: str | None
    prompt_fixture_version: str | None
    run_fingerprint: str
    answerer_prompt_hash: str | None
    judge_prompt_hash: str | None
    config: dict[str, Any]
    started_at: str
    finished_at: str


@dataclass(frozen=True)
class Metrics:
    overall_accuracy: float
    overall_avg_score: float
    total: int
    correct: int
    errors: int
    by_category: dict[str, Any]
    by_cutoff: dict[str, Any]
    latency_ms: dict[str, Any]
    deterministic_gates: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UnifiedResult:
    metadata: Metadata
    metrics: Metrics
    evaluations: tuple[Evaluation, ...]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))

    def write(self, path: Path) -> None:
        atomic_write_json(path, self.to_dict())


@dataclass(frozen=True)
class Checkpoint:
    stage: str
    run_id: str
    dataset_fingerprint: str | None
    corpus_fingerprint: str | None
    index_generation: int | None
    config: dict[str, Any]
    status: str
    started_at: str
    finished_at: str | None
    output: dict[str, Any]
    case_id: str | None = None
    checkpoint_version: str = CHECKPOINT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))

    def write(self, path: Path) -> None:
        atomic_write_json(path, self.to_dict())


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def nearest_rank(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def latency_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "p50_ms": nearest_rank(values, 0.50),
        "p95_ms": nearest_rank(values, 0.95),
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value
