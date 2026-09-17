"""Retrieval metrics and diagnostics for the LOCOMO benchmark."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from eval.schema import CutoffOutcome, Evaluation, Metrics, RetrievalResult, latency_summary

COMPARABLE_CATEGORIES = ("single-hop", "multi-hop", "temporal", "open-domain")
REQUIRED_RETRIEVAL_DIAGNOSTIC_FIELDS = frozenset(
    {"fallback", "failure_reason", "retrieval_mode", "semantic_state"}
)


def _target_matches(result: RetrievalResult, target: dict[str, object]) -> bool:
    return result.source_path == target["path"] and result.line_start <= int(target["line_start"]) and result.line_end >= int(target["line_end"])


def retrieval_outcome(results: tuple[RetrievalResult, ...], targets: list[dict[str, object]], cutoff: int) -> tuple[int, float, float, float]:
    selected = results[:cutoff]
    matched_targets = {target["evidence_id"] for target in targets if any(_target_matches(result, target) for result in selected)}
    matched_results = {result.section_id for result in selected if any(_target_matches(result, target) for target in targets)}
    relevant = len(matched_targets)
    recall = relevant / len(targets) if targets else 0.0
    precision = len(matched_results) / len(selected) if selected else 0.0
    mrr = 1.0 / next((result.rank for result in selected if result.section_id in matched_results), 1_000_000)
    if not matched_results:
        mrr = 0.0
    return relevant, recall, precision, mrr


def has_complete_retrieval_diagnostics(value: object) -> bool:
    return isinstance(value, dict) and REQUIRED_RETRIEVAL_DIAGNOSTIC_FIELDS <= value.keys()


def _retrieval_group_metrics(outcomes: list[CutoffOutcome]) -> dict[str, Any]:
    return {
        "total": len(outcomes),
        "relevant_count": sum(value.relevant_count for value in outcomes),
        "recall": sum(value.recall or 0.0 for value in outcomes) / len(outcomes) if outcomes else 0.0,
        "precision": sum(value.precision or 0.0 for value in outcomes) / len(outcomes) if outcomes else 0.0,
        "mrr": sum(value.mrr or 0.0 for value in outcomes) / len(outcomes) if outcomes else 0.0,
    }


def _retrieval_diagnostics(records: list[dict[str, object]]) -> dict[str, Any]:
    modes: defaultdict[str, int] = defaultdict(int)
    states: defaultdict[str, int] = defaultdict(int)
    failures: defaultdict[str, int] = defaultdict(int)
    fallback_count = 0
    for record in records:
        mode = record.get("retrieval_mode")
        state = record.get("semantic_state")
        failure = record.get("failure_reason")
        if isinstance(mode, str):
            modes[mode] += 1
        if isinstance(state, str):
            states[state] += 1
        if isinstance(failure, str) and failure:
            failures[failure] += 1
        fallback_count += bool(record.get("fallback", False))
    return {
        "searches": len(records),
        "fallback_count": fallback_count,
        "retrieval_mode_counts": dict(sorted(modes.items())),
        "semantic_state_counts": dict(sorted(states.items())),
        "failure_reason_counts": dict(sorted(failures.items())),
    }


def _predict_metrics(
    evaluations: list[Evaluation],
    cutoffs: tuple[int, ...],
    diagnostics: list[dict[str, object]],
) -> Metrics:
    by_category = {}
    for category in COMPARABLE_CATEGORIES:
        outcomes = [
            item.cutoff_outcomes[str(max(cutoffs))]
            for item in evaluations
            if item.category == category
        ]
        by_category[category] = _retrieval_group_metrics(outcomes)

    by_cutoff = {}
    for cutoff in cutoffs:
        outcomes = [item.cutoff_outcomes[str(cutoff)] for item in evaluations]
        by_cutoff[str(cutoff)] = {
            "cutoff": cutoff,
            "overall": _retrieval_group_metrics(outcomes),
            "by_category": {
                category: _retrieval_group_metrics([
                    item.cutoff_outcomes[str(cutoff)]
                    for item in evaluations
                    if item.category == category
                ])
                for category in COMPARABLE_CATEGORIES
            },
        }

    latencies = [item.search_latency_ms for item in evaluations]
    by_category_latency = {
        category: _predict_latency_summary([
            item.search_latency_ms for item in evaluations if item.category == category
        ])
        for category in COMPARABLE_CATEGORIES
    }
    return Metrics(
        overall_accuracy=0.0,
        overall_avg_score=0.0,
        total=len(evaluations),
        correct=0,
        errors=0,
        by_category=by_category,
        by_cutoff=by_cutoff,
        latency_ms={"overall": _predict_latency_summary(latencies), "by_category": by_category_latency},
        baselines={"retrieval_diagnostics": _retrieval_diagnostics(diagnostics)},
    )


def _predict_latency_summary(values: list[float]) -> dict[str, float | int]:
    summary = latency_summary(values)
    summary["max_ms"] = max(values) if values else 0.0
    return summary
