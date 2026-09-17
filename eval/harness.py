"""Three-stage deterministic harness for the internal golden case set."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from pausanias.config import Config, Root, load_config
from pausanias.core import (
    BALANCED_ADMISSION,
    RELATIVE_SEMANTIC_SCORE_FLOOR,
    SEMANTIC_SCORE_FLOOR,
    TICKET_ID_CROSS_REFERENCE_FILTER,
    SYNONYM_EXPANSION,
    Candidate,
    fusion_diagnostics,
    index,
    search,
    semantic_search,
)
from pausanias.hook import run_hook
from pausanias.worker import stop_worker

from .run import (
    CASES_PATH,
    CONFIG_PATH,
    CORPUS_DIR,
    Case,
    _temporarily_deleted,
    load_case_set,
    load_thresholds,
)
from .schema import (
    Checkpoint,
    CutoffOutcome,
    Evaluation,
    Metadata,
    Metrics,
    RetrievalResult,
    UnifiedResult,
    fingerprint,
    latency_summary,
    read_checkpoint,
    read_json,
    utc_now,
)

DEFAULT_CUTOFFS = (1, 2, 4, 8)
DEFAULT_TOP_K = 8
BENCHMARK = "internal"
LOCK_PATH = Path(__file__).with_name("cases.lock")
THRESHOLD_KEYS = ("recall_at_4", "precision_at_4", "mrr", "abstention_accuracy", "forbidden_violations")
ABLATION_SPECS = {
    "semantic_score_floor_disabled": {
        "semantic_score_floor": None,
        "policy": "semantic_score_floor",
    },
    "relative_semantic_score_floor_disabled": {
        "relative_semantic_score_floor": None,
        "policy": "relative_semantic_score_floor",
    },
    "ticket_id_cross_reference_filter_disabled": {
        "ticket_id_cross_reference_filter": False,
        "policy": "ticket_id_cross_reference_filter",
    },
    "balanced_admission_disabled": {
        "balanced_admission": False,
        "policy": "balanced_admission",
    },
    "synonym_expansion_disabled": {
        "synonym_expansion": False,
        "policy": "synonym_expansion",
    },
}


def case_set_fingerprint(path: Path = CASES_PATH) -> str:
    with path.open(encoding="utf-8") as handle:
        return fingerprint(json.load(handle))


def verify_case_lock(cases_path: Path = CASES_PATH, lock_path: Path = LOCK_PATH) -> str:
    lock = read_json(lock_path)
    actual = case_set_fingerprint(cases_path)
    if lock.get("case_set_fingerprint") != actual:
        raise ValueError("case file does not match eval/cases.lock")
    return actual


def _runtime_config(source: Config, corpus_dir: Path, runtime_corpus: Path, database: Path) -> Config:
    source_corpus = corpus_dir.resolve()

    def map_path(path: Path) -> Path:
        return (runtime_corpus / path.resolve().relative_to(source_corpus)).resolve()

    roots = tuple(Root(root.id, map_path(root.path), root.project, root.excludes) for root in source.roots)
    global_notes = frozenset(map_path(path) for path in source.global_notes)
    return Config(
        roots, global_notes, source.private_paths, database, source.section_bytes,
        source.semantic_bundle, source.synonym_table,
    )


def _runtime_config_path(
    source: Config,
    corpus_dir: Path,
    runtime_corpus: Path,
    database: Path,
    path: Path,
) -> Path:
    runtime = _runtime_config(source, corpus_dir, runtime_corpus, database)
    lines = [
        f"database = {json.dumps(str(runtime.database))}",
        f"private_paths = {json.dumps(list(runtime.private_paths))}",
        f"global_notes = {json.dumps(sorted(str(note) for note in runtime.global_notes))}",
        f"semantic_bundle = {json.dumps(str(runtime.semantic_bundle))}",
    ]
    if runtime.synonym_table is not None:
        lines.append(f"synonym_table = {json.dumps(str(runtime.synonym_table))}")
    for root in runtime.roots:
        lines.extend((
            "",
            "[[roots]]",
            f"id = {json.dumps(root.id)}",
            f"project = {json.dumps(root.project)}",
            f"path = {json.dumps(str(root.path))}",
            f"exclude = {json.dumps(list(root.excludes))}",
        ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _corpus_fingerprint(root: Path) -> str:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append({"path": path.relative_to(root).as_posix(), "sha256": _sha256(path.read_bytes())})
    return fingerprint(files)


def _sha256(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def _file_records(root: Path) -> list[dict[str, str]]:
    return [
        {"path": path.relative_to(root).as_posix(), "sha256": _sha256(path.read_bytes())}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _index_generation(database: Path) -> int:
    with sqlite3.connect(database) as connection:
        row = connection.execute("SELECT value FROM metadata WHERE key = 'generation'").fetchone()
    if row is None:
        raise ValueError("index generation is missing")
    return int(row[0])


def _checkpoint_config(
    top_k: int,
    cutoffs: tuple[int, ...],
    case_lock: str,
    retrieval_mode: str = "fused",
    config: Config | None = None,
) -> dict[str, Any]:
    values = {
        "retrieval_mode": retrieval_mode,
        "prompt_mode": "internal-deterministic",
        "top_k": top_k,
        "cutoffs": list(cutoffs),
        "answerer_model": None,
        "answerer_provider": None,
        "judge_model": None,
        "judge_provider": None,
        "case_set_fingerprint": case_lock,
        "synonym_table": fusion_diagnostics(config)["synonym_table"],
    }
    if retrieval_mode == "fused":
        values["fusion"] = fusion_diagnostics(config)
    return values


def _candidate_result(candidate: Candidate, corpus: Path, rank: int) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        section_id=candidate.section_id,
        source_path=str(Path(candidate.canonical_path).resolve().relative_to(corpus.resolve())).replace("\\", "/"),
        root_id=candidate.root_id,
        project=candidate.project_scope,
        heading=candidate.heading,
        line_start=candidate.line_start,
        line_end=candidate.line_end,
        excerpt=candidate.text,
        content_hash=candidate.content_hash,
        score=candidate.score,
        reason=candidate.reason,
    )


def _cutoff(case: Case, results: tuple[RetrievalResult, ...], cutoff: int) -> CutoffOutcome:
    selected = results[:cutoff]
    expected = set(case.expected_source_paths)
    found = {item.source_path for item in selected}
    relevant = len(found & expected)
    positions = [item.rank for item in selected if item.source_path in expected]
    forbidden = len(found & set(case.forbidden_paths))
    if expected:
        recall = relevant / len(expected)
        precision = relevant / len(selected) if selected else 0.0
        mrr = 1.0 / positions[0] if positions else 0.0
        score = 0.0 if forbidden else recall
        abstention_correct = None
    else:
        recall = precision = mrr = None
        abstention_correct = not selected and forbidden == 0
        score = 1.0 if abstention_correct else 0.0
    return CutoffOutcome(
        retrieved_count=len(selected),
        relevant_count=relevant,
        recall=recall,
        precision=precision,
        mrr=mrr,
        abstention_correct=abstention_correct,
        forbidden_sources=forbidden,
        score=score,
        passed=score >= 0.5,
    )


def _evaluation(case: Case, result: dict[str, Any], cutoffs: tuple[int, ...]) -> Evaluation:
    retrieval = tuple(RetrievalResult(**item) for item in result["retrieval_results"])
    outcomes = {str(cutoff): _cutoff(case, retrieval, cutoff) for cutoff in cutoffs}
    score = outcomes[str(max(cutoffs))].score
    return Evaluation(
        case_id=case.id,
        category=case.category,
        query=case.query,
        expected_sources=case.expected_source_paths,
        ground_truth=None,
        retrieval_results=retrieval,
        search_latency_ms=float(result["search_latency_ms"]),
        cutoff_outcomes=outcomes,
        score=score,
    )


def _group_metrics(evaluations: list[Evaluation]) -> dict[str, Any]:
    total = len(evaluations)
    correct = sum(item.score >= 0.5 for item in evaluations)
    return {
        "total": total,
        "correct": correct,
        "accuracy": 100 * correct / total if total else 0.0,
        "avg_score": 100 * sum(item.score for item in evaluations) / total if total else 0.0,
        "errors": sum(item.failure_reason is not None for item in evaluations),
    }


def _outcome_metrics(outcomes: list[CutoffOutcome]) -> dict[str, Any]:
    return {
        "total": len(outcomes),
        "correct": sum(value.score >= 0.5 for value in outcomes),
        "accuracy": 100 * sum(value.score >= 0.5 for value in outcomes) / len(outcomes) if outcomes else 0.0,
        "avg_score": 100 * sum(value.score for value in outcomes) / len(outcomes) if outcomes else 0.0,
        "errors": sum(value.error is not None for value in outcomes),
    }


def _category_gate(
    outcomes: list[CutoffOutcome],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    retrieval = [value for value in outcomes if value.recall is not None]
    abstention = [value for value in outcomes if value.abstention_correct is not None]
    measured = {
        "recall_at_4": sum(value.recall for value in retrieval) / len(retrieval) if retrieval else 1.0,
        "precision_at_4": sum(value.precision for value in retrieval) / len(retrieval) if retrieval else 1.0,
        "mrr": sum(value.mrr for value in retrieval) / len(retrieval) if retrieval else 1.0,
        "abstention_accuracy": sum(value.abstention_correct for value in abstention) / len(abstention) if abstention else 1.0,
        "forbidden_violations": sum(value.forbidden_sources for value in outcomes),
    }
    passed = (
        measured["recall_at_4"] >= thresholds["recall_at_4"]
        and measured["precision_at_4"] >= thresholds["precision_at_4"]
        and measured["mrr"] >= thresholds["mrr"]
        and measured["abstention_accuracy"] >= thresholds["abstention_accuracy"]
        and measured["forbidden_violations"] <= thresholds["forbidden_violations"]
    )
    return {**measured, "thresholds": thresholds, "passed": passed}


def _metrics(evaluations: list[Evaluation], cutoffs: tuple[int, ...], thresholds: dict[str, Any]) -> Metrics:
    by_category = {category: _group_metrics([item for item in evaluations if item.category == category]) for category in sorted({item.category for item in evaluations})}
    by_cutoff: dict[str, Any] = {}
    gates: dict[str, Any] | None = None
    for cutoff in cutoffs:
        outcomes = [item.cutoff_outcomes[str(cutoff)] for item in evaluations]
        groups: dict[str, Any] = {}
        for category in sorted({item.category for item in evaluations}):
            selected = [item for item in evaluations if item.category == category]
            values = [item.cutoff_outcomes[str(cutoff)] for item in selected]
            groups[category] = {
                "total": len(values),
                "correct": sum(value.score >= 0.5 for value in values),
                "accuracy": 100 * sum(value.score >= 0.5 for value in values) / len(values) if values else 0.0,
                "avg_score": 100 * sum(value.score for value in values) / len(values) if values else 0.0,
                "errors": sum(value.error is not None for value in values),
            }
        by_cutoff[str(cutoff)] = {
            "cutoff": cutoff,
            "overall": _outcome_metrics(outcomes),
            "by_category": groups,
        }
        if cutoff == 4:
            retrieval = [value for value in outcomes if value.recall is not None]
            abstention = [value for value in outcomes if value.abstention_correct is not None]
            gates = {
                "recall": sum(value.recall for value in retrieval) / len(retrieval) if retrieval else 1.0,
                "precision": sum(value.precision for value in retrieval) / len(retrieval) if retrieval else 1.0,
                "mrr": sum(value.mrr for value in retrieval) / len(retrieval) if retrieval else 1.0,
                "abstention_accuracy": sum(value.abstention_correct for value in abstention) / len(abstention) if abstention else 1.0,
                "forbidden_violations": sum(value.forbidden_sources for value in outcomes),
            }
    if gates is None:
        raise ValueError("cutoffs must include 4 for deterministic gates")
    category_thresholds = thresholds.get("categories", {})
    category_gates = {}
    for category in sorted({item.category for item in evaluations}):
        category_values = [item.cutoff_outcomes["4"] for item in evaluations if item.category == category]
        category_gate_thresholds = category_thresholds.get(category, thresholds)
        category_gates[category] = _category_gate(category_values, category_gate_thresholds)
    deterministic = {
        "by_cutoff": {
            "4": {
                **gates,
                "thresholds": {
                    "recall": thresholds["recall_at_4"],
                    "precision": thresholds["precision_at_4"],
                    "mrr": thresholds["mrr"],
                    "abstention_accuracy": thresholds["abstention_accuracy"],
                    "forbidden_violations": thresholds["forbidden_violations"],
                },
                "passed": (
                    gates["recall"] >= thresholds["recall_at_4"]
                    and gates["precision"] >= thresholds["precision_at_4"]
                    and gates["mrr"] >= thresholds["mrr"]
                    and gates["abstention_accuracy"] >= thresholds["abstention_accuracy"]
                    and gates["forbidden_violations"] <= thresholds["forbidden_violations"]
                    and all(category["passed"] for category in category_gates.values())
                ),
            }
        },
        "by_category": category_gates,
    }
    latencies = [item.search_latency_ms for item in evaluations]
    by_category_latency = {
        category: latency_summary([item.search_latency_ms for item in evaluations if item.category == category])
        for category in sorted({item.category for item in evaluations})
    }
    total = len(evaluations)
    correct = sum(item.score >= 0.5 for item in evaluations)
    return Metrics(
        overall_accuracy=100 * correct / total if total else 0.0,
        overall_avg_score=100 * sum(item.score for item in evaluations) / total if total else 0.0,
        total=total,
        correct=correct,
        errors=sum(item.failure_reason is not None for item in evaluations),
        by_category=by_category,
        by_cutoff=by_cutoff,
        latency_ms={"overall": latency_summary(latencies), "by_category": by_category_latency},
        deterministic_gates=deterministic,
    )


def run_internal(
    *,
    run_id: str,
    results_dir: Path,
    cases_path: Path = CASES_PATH,
    config_path: Path = CONFIG_PATH,
    corpus_dir: Path = CORPUS_DIR,
    top_k: int = DEFAULT_TOP_K,
    cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
    resume: bool = False,
    predict_only: bool = False,
    evaluate_only: bool = False,
    retrieval_mode: str = "fused",
    through_worker: bool = False,
) -> UnifiedResult | None:
    if predict_only and evaluate_only:
        raise ValueError("--predict-only and --evaluate-only cannot be combined")
    if top_k < max(cutoffs) or sorted(cutoffs) != list(cutoffs):
        raise ValueError("top-k must include sorted cutoff values")
    if retrieval_mode not in {"lexical", "fused"}:
        raise ValueError("retrieval mode must be lexical or fused")
    case_lock = verify_case_lock(cases_path)
    case_set = load_case_set(cases_path)
    cases = list(case_set.cases)
    source_config = load_config(config_path)
    config_values = _checkpoint_config(top_k, cutoffs, case_lock, retrieval_mode, source_config)
    run_root = results_dir / BENCHMARK / run_id
    workspace = run_root / "workspace"
    runtime_corpus = workspace / "corpus"
    database = workspace / "index.sqlite3"
    ingest_path = run_root / "checkpoints" / "ingest.json"
    search_dir = run_root / "checkpoints" / "search"
    search_dir.mkdir(parents=True, exist_ok=True)
    if not runtime_corpus.exists():
        runtime_corpus.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(corpus_dir, runtime_corpus)
    corpus_hash = _corpus_fingerprint(runtime_corpus)
    config = _runtime_config(source_config, corpus_dir, runtime_corpus, database)
    worker_config_path = (
        _runtime_config_path(source_config, corpus_dir, runtime_corpus, database, workspace / "worker-config.toml")
        if through_worker else None
    )
    reuse_checkpoints = resume or evaluate_only
    ingest = read_checkpoint(
        ingest_path,
        stage="ingest",
        run_id=run_id,
        config=config_values,
        dataset_fingerprint=case_lock,
        corpus_fingerprint=corpus_hash,
    ) if reuse_checkpoints else None
    if evaluate_only and ingest is None:
        raise ValueError("evaluation requires a complete ingest checkpoint")
    if ingest is None:
        started = utc_now()
        index(config, rebuild=True)
        generation = _index_generation(database)
        ingest = Checkpoint(
            stage="ingest",
            run_id=run_id,
            dataset_fingerprint=case_lock,
            corpus_fingerprint=corpus_hash,
            index_generation=generation,
            config=config_values,
            status="complete",
            started_at=started,
            finished_at=utc_now(),
            output={"files": _file_records(runtime_corpus), "index_generation": generation},
        )
        ingest.write(ingest_path)
    generation = int(ingest.index_generation or 0)
    if cases:
        _run_search(config, cases[0], top_k, retrieval_mode, config_path=worker_config_path)
    evaluations: list[Evaluation] = []
    baseline_evaluations: dict[str, list[Evaluation]] = {
        name: [] for name in ABLATION_SPECS
    }
    for case in cases:
        checkpoint_path = search_dir / f"{case.id}.json"
        checkpoint = read_checkpoint(
            checkpoint_path,
            stage="search",
            run_id=run_id,
            config=config_values,
            case_id=case.id,
            dataset_fingerprint=case_lock,
            corpus_fingerprint=corpus_hash,
            index_generation=generation,
        ) if reuse_checkpoints else None
        if evaluate_only and checkpoint is None:
            raise ValueError(f"evaluation requires a complete search checkpoint for {case.id}")
        if checkpoint is not None:
            raw = checkpoint.output
        else:
            raw = _search_case(
                config,
                runtime_corpus,
                case,
                top_k,
                retrieval_mode,
                worker_config_path,
            )
            Checkpoint(
                stage="search",
                run_id=run_id,
                case_id=case.id,
                dataset_fingerprint=case_lock,
                corpus_fingerprint=corpus_hash,
                index_generation=generation,
                config=config_values,
                status="complete",
                started_at=utc_now(),
                finished_at=utc_now(),
                output=raw,
            ).write(checkpoint_path)
        evaluations.append(_evaluation(case, raw, cutoffs))
        if retrieval_mode == "fused":
            for name, options in ABLATION_SPECS.items():
                baseline_evaluations[name].append(_evaluation(
                    case,
                    _search_case(
                        config,
                        runtime_corpus,
                        case,
                        top_k,
                        retrieval_mode,
                        worker_config_path,
                        **{key: value for key, value in options.items() if key != "policy"},
                    ),
                    cutoffs,
                ))
    if through_worker:
        stop_worker(config.database)
    if predict_only:
        print(f"internal predict-only: {len(evaluations)} search checkpoints; case-set fingerprint {case_lock}")
        return None
    if evaluate_only and len(evaluations) != len(cases):
        raise ValueError("evaluation requires complete search checkpoints")
    thresholds = load_thresholds()
    if retrieval_mode == "fused" and "fused_categories" in thresholds:
        thresholds = {**thresholds, "categories": thresholds["fused_categories"]}
    metrics = _metrics(evaluations, cutoffs, thresholds)
    if any(baseline_evaluations.values()):
        current_gate = metrics.deterministic_gates["by_cutoff"]["4"]
        baselines: dict[str, Any] = {}
        for name, options in ABLATION_SPECS.items():
            baseline = _metrics(baseline_evaluations[name], cutoffs, thresholds)
            baseline_gate = baseline.deterministic_gates["by_cutoff"]["4"]
            baselines[name] = {
                "disabled_policy": options["policy"],
                "recall_at_4": baseline_gate["recall"],
                "precision_at_4": baseline_gate["precision"],
                "mrr": baseline_gate["mrr"],
                "abstention_accuracy": baseline_gate["abstention_accuracy"],
                "forbidden_violations": baseline_gate["forbidden_violations"],
                "delta_from_active": {
                    "recall_at_4": current_gate["recall"] - baseline_gate["recall"],
                    "precision_at_4": current_gate["precision"] - baseline_gate["precision"],
                    "mrr": current_gate["mrr"] - baseline_gate["mrr"],
                    "abstention_accuracy": current_gate["abstention_accuracy"] - baseline_gate["abstention_accuracy"],
                    "forbidden_violations": current_gate["forbidden_violations"] - baseline_gate["forbidden_violations"],
                },
                "by_category": baseline.deterministic_gates["by_category"],
            }
        metrics = replace(metrics, baselines=baselines)
    started_at = ingest.started_at
    metadata = Metadata(
        benchmark=BENCHMARK,
        run_id=run_id,
        dataset={"name": "internal-golden", "version": "cases.lock", "fingerprint": case_lock},
        corpus_fingerprint=corpus_hash,
        index_generation=generation,
        case_set_fingerprint=case_lock,
        prompt_version=None,
        prompt_fixture_version=None,
        run_fingerprint=fingerprint({"benchmark": BENCHMARK, "run_id": run_id, "config": config_values, "corpus": corpus_hash}),
        answerer_prompt_hash=None,
        judge_prompt_hash=None,
        config=config_values,
        started_at=started_at,
        finished_at=utc_now(),
    )
    result = UnifiedResult(metadata=metadata, metrics=metrics, evaluations=tuple(evaluations))
    result.write(run_root / "run.json")
    print(f"internal evaluation: {metrics.overall_accuracy:.2f}% accuracy; case-set fingerprint {case_lock}")
    print(json.dumps(metrics.deterministic_gates, sort_keys=True))
    if metrics.baselines:
        print(json.dumps({"baselines": metrics.baselines}, sort_keys=True))
    if not metrics.deterministic_gates["by_cutoff"]["4"]["passed"]:
        raise ValueError("internal deterministic gates failed")
    return result


def _run_search(
    config: Config,
    case: Case,
    top_k: int,
    retrieval_mode: str,
    refresh: set[str] | None = None,
    *,
    config_path: Path | None = None,
    semantic_score_floor: float | None = SEMANTIC_SCORE_FLOOR,
    relative_semantic_score_floor: float | None = RELATIVE_SEMANTIC_SCORE_FLOOR,
    ticket_id_cross_reference_filter: bool = TICKET_ID_CROSS_REFERENCE_FILTER,
    balanced_admission: bool = BALANCED_ADMISSION,
    synonym_expansion: bool = SYNONYM_EXPANSION,
) -> list[Candidate]:
    if config_path is not None and retrieval_mode == "fused":
        return run_hook(
            config,
            config_path,
            case.query,
            project=case.scope.get("project"),
            root_id=case.scope.get("root"),
            all_projects=bool(case.scope.get("all_projects", False)),
            limit=top_k,
            semantic_score_floor=semantic_score_floor,
            relative_semantic_score_floor=relative_semantic_score_floor,
            ticket_id_cross_reference_filter=ticket_id_cross_reference_filter,
            balanced_admission=balanced_admission,
            synonym_expansion=synonym_expansion,
        ).candidates
    if retrieval_mode == "fused":
        return semantic_search(
            config,
            case.query,
            project=case.scope.get("project"),
            root_id=case.scope.get("root"),
            all_projects=bool(case.scope.get("all_projects", False)),
            limit=top_k,
            refresh=refresh,
            semantic_score_floor=semantic_score_floor,
            relative_semantic_score_floor=relative_semantic_score_floor,
            ticket_id_cross_reference_filter=ticket_id_cross_reference_filter,
            balanced_admission=balanced_admission,
            synonym_expansion=synonym_expansion,
        )
    return search(
        config,
        case.query,
        project=case.scope.get("project"),
        root_id=case.scope.get("root"),
        all_projects=bool(case.scope.get("all_projects", False)),
        limit=top_k,
        refresh=refresh,
        semantic=False,
        synonym_expansion=synonym_expansion,
    )


def _search_case(
    config: Config,
    runtime_corpus: Path,
    case: Case,
    top_k: int,
    retrieval_mode: str,
    worker_config_path: Path | None,
    *,
    semantic_score_floor: float | None = SEMANTIC_SCORE_FLOOR,
    relative_semantic_score_floor: float | None = RELATIVE_SEMANTIC_SCORE_FLOOR,
    ticket_id_cross_reference_filter: bool = TICKET_ID_CROSS_REFERENCE_FILTER,
    balanced_admission: bool = BALANCED_ADMISSION,
    synonym_expansion: bool = SYNONYM_EXPANSION,
) -> dict[str, Any]:
    with _temporarily_deleted(runtime_corpus, case.delete_sources):
        refresh: set[str] = set()
        started = time.perf_counter()
        candidates = _run_search(
            config,
            case,
            top_k,
            retrieval_mode,
            refresh,
            config_path=worker_config_path,
            semantic_score_floor=semantic_score_floor,
            relative_semantic_score_floor=relative_semantic_score_floor,
            ticket_id_cross_reference_filter=ticket_id_cross_reference_filter,
            balanced_admission=balanced_admission,
            synonym_expansion=synonym_expansion,
        )
        elapsed = (time.perf_counter() - started) * 1000
    return {
        "case_id": case.id,
        "query": case.query,
        "search_latency_ms": elapsed,
        "retrieval_fingerprint": fingerprint([candidate.section_id for candidate in candidates]),
        "retrieval_results": [
            asdict(_candidate_result(candidate, runtime_corpus, rank))
            for rank, candidate in enumerate(candidates, 1)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="latest")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--cutoffs", default="1,2,4,8")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--retrieval-mode", choices=("lexical", "fused"), default="fused")
    parser.add_argument("--through-worker", action="store_true", help="run retrieval through the real hook worker")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        cutoffs = tuple(int(item) for item in args.cutoffs.split(","))
        run_internal(
            run_id=args.run_id,
            results_dir=args.results_dir,
            cases_path=args.cases,
            config_path=args.config,
            corpus_dir=args.corpus,
            top_k=args.top_k,
            cutoffs=cutoffs,
            resume=args.resume,
            predict_only=args.predict_only,
            evaluate_only=args.evaluate_only,
            retrieval_mode=args.retrieval_mode,
            through_worker=args.through_worker,
        )
        return 0
    except (OSError, ValueError, TypeError, sqlite3.DatabaseError) as exc:
        print(f"eval harness: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
