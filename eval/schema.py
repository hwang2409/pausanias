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
    baselines: dict[str, Any] = field(default_factory=dict)


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


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _valid_number(value: object, *, minimum: float | None = None, maximum: float | None = None) -> bool:
    if not _finite_number(value):
        return False
    number = float(value)
    return (minimum is None or number >= minimum) and (maximum is None or number <= maximum)


def retrieval_result_from_dict(value: object) -> RetrievalResult:
    if not isinstance(value, dict):
        raise ValueError("checkpoint contains invalid retrieval results")
    required = {
        "rank", "section_id", "source_path", "root_id", "project", "heading",
        "line_start", "line_end", "excerpt", "content_hash", "score", "reason",
    }
    if set(value) != required:
        raise ValueError("checkpoint contains invalid retrieval results")
    if (
        not isinstance(value["rank"], int) or isinstance(value["rank"], bool) or value["rank"] < 1
        or not all(isinstance(value[key], str) for key in ("section_id", "source_path", "root_id", "project", "excerpt", "content_hash", "reason"))
        or (value["heading"] is not None and not isinstance(value["heading"], str))
        or not isinstance(value["line_start"], int) or isinstance(value["line_start"], bool) or value["line_start"] < 1
        or not isinstance(value["line_end"], int) or isinstance(value["line_end"], bool) or value["line_end"] < value["line_start"]
        or not _valid_number(value["score"])
    ):
        raise ValueError("checkpoint contains invalid retrieval results")
    return RetrievalResult(**value)


def _prompt_metadata_from_dict(value: object) -> PromptMetadata:
    if not isinstance(value, dict):
        raise ValueError("checkpoint contains invalid prompt metadata")
    required = {
        "prompt_version", "template_hash", "slot_names", "provider", "model",
        "prompt_tokens", "completion_tokens",
    }
    if set(value) != required:
        raise ValueError("checkpoint contains invalid prompt metadata")
    if (
        value["prompt_version"] is not None and not isinstance(value["prompt_version"], str)
        or value["template_hash"] is not None and not isinstance(value["template_hash"], str)
        or not isinstance(value["slot_names"], list) or not all(isinstance(slot, str) for slot in value["slot_names"])
        or value["provider"] is not None and not isinstance(value["provider"], str)
        or value["model"] is not None and not isinstance(value["model"], str)
        or value["prompt_tokens"] is not None and (not isinstance(value["prompt_tokens"], int) or isinstance(value["prompt_tokens"], bool) or value["prompt_tokens"] < 0)
        or value["completion_tokens"] is not None and (not isinstance(value["completion_tokens"], int) or isinstance(value["completion_tokens"], bool) or value["completion_tokens"] < 0)
    ):
        raise ValueError("checkpoint contains invalid prompt metadata")
    return PromptMetadata(**{**value, "slot_names": tuple(value["slot_names"])})


def _cutoff_outcome_from_dict(value: object) -> CutoffOutcome:
    if not isinstance(value, dict):
        raise ValueError("checkpoint contains invalid cutoff outcomes")
    required = {
        "retrieved_count", "relevant_count", "recall", "precision", "mrr",
        "abstention_correct", "forbidden_sources", "score", "passed",
        "generated_answer", "judgment", "reason", "model", "error", "prompt_metadata",
    }
    if set(value) != required:
        raise ValueError("checkpoint contains invalid cutoff outcomes")
    if (
        not isinstance(value["retrieved_count"], int) or isinstance(value["retrieved_count"], bool) or value["retrieved_count"] < 0
        or not isinstance(value["relevant_count"], int) or isinstance(value["relevant_count"], bool) or value["relevant_count"] < 0
        or value["recall"] is not None and not _valid_number(value["recall"], minimum=0, maximum=1)
        or value["precision"] is not None and not _valid_number(value["precision"], minimum=0, maximum=1)
        or value["mrr"] is not None and not _valid_number(value["mrr"], minimum=0, maximum=1)
        or value["abstention_correct"] is not None and not isinstance(value["abstention_correct"], bool)
        or not isinstance(value["forbidden_sources"], int) or isinstance(value["forbidden_sources"], bool) or value["forbidden_sources"] < 0
        or not _valid_number(value["score"], minimum=0, maximum=1)
        or not isinstance(value["passed"], bool)
        or value["generated_answer"] is not None and not isinstance(value["generated_answer"], str)
        or value["judgment"] is not None and not isinstance(value["judgment"], str)
        or value["reason"] is not None and not isinstance(value["reason"], str)
        or value["model"] is not None and not isinstance(value["model"], str)
        or value["error"] is not None and not isinstance(value["error"], str)
        or not isinstance(value["prompt_metadata"], dict)
    ):
        raise ValueError("checkpoint contains invalid cutoff outcomes")
    metadata = {str(name): _prompt_metadata_from_dict(details) for name, details in value["prompt_metadata"].items()}
    return CutoffOutcome(**{**value, "prompt_metadata": metadata})


def evaluation_from_dict(value: object) -> Evaluation:
    if not isinstance(value, dict):
        raise ValueError("checkpoint contains invalid evaluation output")
    raw_outcomes = value.get("cutoff_outcomes")
    raw_results = value.get("retrieval_results")
    required = {
        "case_id", "category", "query", "expected_sources", "ground_truth",
        "retrieval_results", "search_latency_ms", "cutoff_outcomes", "score", "failure_reason",
    }
    if set(value) != required or not isinstance(raw_outcomes, dict) or not isinstance(raw_results, list):
        raise ValueError("checkpoint contains invalid evaluation output")
    if (
        not isinstance(value["case_id"], str) or not value["case_id"]
        or not isinstance(value["category"], str) or not value["category"]
        or not isinstance(value["query"], str) or not value["query"]
        or not isinstance(value["expected_sources"], list) or not all(isinstance(item, str) and item for item in value["expected_sources"])
        or value["ground_truth"] is not None and not isinstance(value["ground_truth"], str)
        or not _valid_number(value["search_latency_ms"], minimum=0)
        or not _valid_number(value["score"], minimum=0, maximum=1)
        or value["failure_reason"] is not None and not isinstance(value["failure_reason"], str)
    ):
        raise ValueError("checkpoint contains invalid evaluation output")
    return Evaluation(
        case_id=value["case_id"],
        category=value["category"],
        query=value["query"],
        expected_sources=tuple(value["expected_sources"]),
        ground_truth=value["ground_truth"],
        retrieval_results=tuple(retrieval_result_from_dict(item) for item in raw_results),
        search_latency_ms=float(value["search_latency_ms"]),
        cutoff_outcomes={str(key): _cutoff_outcome_from_dict(item) for key, item in raw_outcomes.items()},
        score=float(value["score"]),
        failure_reason=value["failure_reason"],
    )


def checkpoint_from_dict(value: object) -> Checkpoint:
    if not isinstance(value, dict):
        raise ValueError("checkpoint must contain an object")
    required = {
        "checkpoint_version", "stage", "run_id", "case_id", "dataset_fingerprint",
        "corpus_fingerprint", "index_generation", "config", "status", "started_at",
        "finished_at", "output",
    }
    if set(value) != required:
        raise ValueError("checkpoint contains invalid fields")
    if (
        value["checkpoint_version"] != CHECKPOINT_VERSION
        or not isinstance(value["stage"], str)
        or not isinstance(value["run_id"], str) or not value["run_id"]
        or value["case_id"] is not None and (not isinstance(value["case_id"], str) or not value["case_id"])
        or value["dataset_fingerprint"] is not None and not isinstance(value["dataset_fingerprint"], str)
        or value["corpus_fingerprint"] is not None and not isinstance(value["corpus_fingerprint"], str)
        or value["index_generation"] is not None and (not isinstance(value["index_generation"], int) or isinstance(value["index_generation"], bool) or value["index_generation"] < 1)
        or not isinstance(value["config"], dict)
        or not isinstance(value["status"], str)
        or not isinstance(value["started_at"], str)
        or value["finished_at"] is not None and not isinstance(value["finished_at"], str)
        or not isinstance(value["output"], dict)
    ):
        raise ValueError("checkpoint contains invalid fields")
    stage = value["stage"]
    if stage not in {"ingest", "search", "evaluate"}:
        raise ValueError("checkpoint contains invalid stage")
    output = value["output"]
    if stage == "ingest":
        files = output.get("files")
        if (
            not isinstance(files, list)
            or not all(isinstance(item, dict) for item in files)
            or not isinstance(output.get("index_generation"), int)
            or isinstance(output["index_generation"], bool)
            or output["index_generation"] < 1
        ):
            raise ValueError("checkpoint contains invalid ingest output")
    elif stage == "search":
        if (
            not isinstance(output.get("case_id"), str)
            or not isinstance(output.get("query"), str) or not output["query"]
            or not _valid_number(output.get("search_latency_ms"), minimum=0)
            or not isinstance(output.get("retrieval_results"), list)
            or not isinstance(output.get("retrieval_fingerprint"), str) or not output["retrieval_fingerprint"]
        ):
            raise ValueError("checkpoint contains invalid search output")
        if value["case_id"] != output["case_id"]:
            raise ValueError("checkpoint contains invalid search output")
        for item in output["retrieval_results"]:
            retrieval_result_from_dict(item)
    elif stage == "evaluate":
        evaluation = evaluation_from_dict(output)
        if value["case_id"] != evaluation.case_id:
            raise ValueError("checkpoint contains invalid evaluation output")
    return Checkpoint(
        stage=stage,
        run_id=value["run_id"],
        dataset_fingerprint=value["dataset_fingerprint"],
        corpus_fingerprint=value["corpus_fingerprint"],
        index_generation=value["index_generation"],
        config=value["config"],
        status=value["status"],
        started_at=value["started_at"],
        finished_at=value["finished_at"],
        output=output,
        case_id=value["case_id"],
    )


def read_checkpoint(
    path: Path,
    *,
    stage: str,
    run_id: str,
    config: dict[str, Any] | None = None,
    case_id: str | None = None,
    dataset_fingerprint: str | None = None,
    corpus_fingerprint: str | None = None,
    index_generation: int | None = None,
    cutoffs: tuple[int, ...] | None = None,
) -> Checkpoint | None:
    try:
        checkpoint = checkpoint_from_dict(read_json(path))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        checkpoint.stage != stage
        or checkpoint.run_id != run_id
        or checkpoint.status != "complete"
        or config is not None and checkpoint.config != config
        or case_id is not None and checkpoint.case_id != case_id
        or dataset_fingerprint is not None and checkpoint.dataset_fingerprint != dataset_fingerprint
        or corpus_fingerprint is not None and checkpoint.corpus_fingerprint != corpus_fingerprint
        or index_generation is not None and checkpoint.index_generation != index_generation
    ):
        return None
    if cutoffs is not None:
        output = checkpoint.output
        outcomes = output.get("cutoff_outcomes")
        if not isinstance(outcomes, dict) or {str(key) for key in outcomes} != {str(cutoff) for cutoff in cutoffs}:
            return None
    return checkpoint


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
