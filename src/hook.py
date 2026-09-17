"""The short-lived adapter path used by hooks and the benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing
import time
from pathlib import Path

from .config import Config
from .core import (
    BALANCED_ADMISSION,
    Candidate,
    RELATIVE_SEMANTIC_SCORE_FLOOR,
    SEMANTIC_SCORE_FLOOR,
    TICKET_ID_CROSS_REFERENCE_FILTER,
    search,
)
from .worker import ADAPTER_DEADLINE_MS, HookMetrics, WorkerClient, WorkerError, ensure_worker


@dataclass(frozen=True)
class HookResponse:
    candidates: list[Candidate]
    metrics: HookMetrics


def _candidate(value: dict[str, object]) -> Candidate:
    heading_path = value.get("heading_path", [])
    if not isinstance(heading_path, list) or not all(isinstance(item, str) for item in heading_path):
        raise ValueError("worker candidate heading path is invalid")
    required = (
        "section_id", "canonical_path", "line_start", "line_end", "root_id", "project_scope",
        "text", "content_hash", "score", "reason",
    )
    if any(key not in value for key in required):
        raise ValueError("worker candidate is incomplete")
    heading = value.get("heading")
    if heading is not None and not isinstance(heading, str):
        raise ValueError("worker candidate heading is invalid")
    optional = ("note_type", "updated_date", "created_date")
    if any(value.get(key) is not None and not isinstance(value.get(key), str) for key in optional):
        raise ValueError("worker candidate metadata is invalid")
    return Candidate(
        str(value["section_id"]), str(value["canonical_path"]), heading, tuple(heading_path),
        int(value["line_start"]), int(value["line_end"]), str(value["root_id"]),
        str(value["project_scope"]), str(value["text"]), str(value["content_hash"]),
        value.get("note_type"), value.get("updated_date"), value.get("created_date"),
        float(value["score"]), str(value["reason"]),
        value.get("lexical_rank"), value.get("vector_rank"),
        value.get("lexical_score"), value.get("vector_score"), value.get("fused_score"),
        str(value.get("lane", "lexical")), value.get("guard_reason"),
    )


def _lexical_search_child(connection, config: Config, query: str, project: str | None,
                          root_id: str | None, all_projects: bool, limit: int) -> None:
    try:
        connection.send(search(config, query, project, root_id, all_projects, limit))
    except Exception as exc:
        connection.send(exc)
    finally:
        connection.close()


def _lexical_search_until(config: Config, query: str, project: str | None, root_id: str | None,
                          all_projects: bool, limit: int, deadline: float | None) -> list[Candidate]:
    if deadline is not None and time.perf_counter() >= deadline:
        return []
    start_method = "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
    context = multiprocessing.get_context(start_method)
    parent, child = context.Pipe(False)
    process = context.Process(
        target=_lexical_search_child,
        args=(child, config, query, project, root_id, all_projects, limit),
        daemon=True,
    )
    process.start()
    child.close()
    try:
        remaining = None if deadline is None else max(deadline - time.perf_counter(), 0.0)
        if remaining == 0.0 or not parent.poll(remaining):
            process.kill()
            process.join(timeout=0)
            return []
        value = parent.recv()
        return value if isinstance(value, list) and all(isinstance(item, Candidate) for item in value) else []
    finally:
        parent.close()
        if process.is_alive():
            process.kill()
        process.join(timeout=0)


def _fallback(config: Config, query: str, project: str | None, root_id: str | None,
              all_projects: bool, limit: int, started: float,
              worker_startup_ms: float = 0.000001, deadline: float | None = None,
              disabled_reason: str | None = None) -> HookResponse:
    fallback_started = time.perf_counter()
    candidates = _lexical_search_until(config, query, project, root_id, all_projects, limit, deadline)
    fallback_ms = max((time.perf_counter() - fallback_started) * 1000.0, 0.000001)
    return HookResponse(candidates, HookMetrics(
        worker_startup_ms=worker_startup_ms,
        fallback_ms=fallback_ms,
        hook_total_ms=max((time.perf_counter() - started) * 1000.0, 0.000001),
        fallback=True,
        cache_state="fallback",
        disabled_reason=disabled_reason,
    ))


def run_hook(
    config: Config,
    config_path: str | Path,
    query: str,
    project: str | None = None,
    root_id: str | None = None,
    all_projects: bool = False,
    limit: int = 20,
    deadline_ms: float = ADAPTER_DEADLINE_MS,
    semantic_score_floor: float | None = SEMANTIC_SCORE_FLOOR,
    relative_semantic_score_floor: float | None = RELATIVE_SEMANTIC_SCORE_FLOOR,
    ticket_id_cross_reference_filter: bool = TICKET_ID_CROSS_REFERENCE_FILTER,
    balanced_admission: bool = BALANCED_ADMISSION,
) -> HookResponse:
    """Run one semantic request through a persistent worker or lexical fallback."""
    if deadline_ms <= 0:
        raise ValueError("deadline must be positive")
    started = time.perf_counter()
    deadline = started + deadline_ms / 1000.0
    startup = ensure_worker(config, Path(config_path), deadline)
    if startup is None:
        return _fallback(config, query, project, root_id, all_projects, limit, started, deadline=deadline)
    paths, worker_startup_ms = startup
    try:
        response = WorkerClient(paths).query({
            "query": query,
            "project": project,
            "root_id": root_id,
            "all_projects": all_projects,
            "limit": limit,
            "deadline_ms": deadline_ms,
            "semantic_score_floor": semantic_score_floor,
            "relative_semantic_score_floor": relative_semantic_score_floor,
            "ticket_id_cross_reference_filter": ticket_id_cross_reference_filter,
            "balanced_admission": balanced_admission,
        }, deadline)
        items = response.get("items")
        raw_metrics = response.get("metrics")
        if not isinstance(items, list) or not isinstance(raw_metrics, dict):
            raise ValueError("worker response is incomplete")
        if not all(isinstance(item, dict) for item in items):
            raise ValueError("worker candidates are invalid")
        candidates = [_candidate(item) for item in items]
        status = raw_metrics.get("status", response.get("status", "ok"))
        if status == "semantic_failure" and not bool(raw_metrics.get("fallback", False)):
            raise WorkerError("worker reported semantic failure without fallback")
        metrics = HookMetrics(
            worker_startup_ms=worker_startup_ms,
            model_load_ms=float(raw_metrics.get("model_load_ms", 0.000001)),
            matrix_load_ms=float(raw_metrics.get("matrix_load_ms", 0.000001)),
            encode_ms=float(raw_metrics.get("encode_ms", 0.000001)),
            scan_ms=float(raw_metrics.get("scan_ms", 0.000001)),
            fallback_ms=float(raw_metrics.get("fallback_ms", 0.000001)),
            fts_search_ms=float(raw_metrics.get("fts_search_ms", 0.000001)),
            hybrid_overhead_ms=float(raw_metrics.get("hybrid_overhead_ms", 0.000001)),
            hook_total_ms=max((time.perf_counter() - started) * 1000.0, 0.000001),
            fallback=bool(raw_metrics.get("fallback", False)),
            cache_state=str(raw_metrics.get("cache_state", "unknown")),
            disabled_reason=(str(raw_metrics["disabled_reason"])
                             if raw_metrics.get("disabled_reason") is not None else None),
        )
        return HookResponse(candidates, metrics)
    except (OSError, TimeoutError, ValueError, TypeError, OverflowError, WorkerError):
        return _fallback(config, query, project, root_id, all_projects, limit, started,
                         worker_startup_ms, deadline)
