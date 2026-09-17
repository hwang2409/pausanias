"""Three-stage deterministic harness for the internal golden case set."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time
from typing import Any

from pausanias.config import Config, Root, load_config
from pausanias.core import Candidate, index, search

from .run import CASES_PATH, CONFIG_PATH, CORPUS_DIR, Case, _temporarily_deleted, load_case_set, load_thresholds
from .schema import (
    Checkpoint,
    CutoffOutcome,
    Evaluation,
    Metadata,
    Metrics,
    RetrievalResult,
    UnifiedResult,
    atomic_write_json,
    fingerprint,
    latency_summary,
    read_json,
    utc_now,
)


DEFAULT_CUTOFFS = (1, 2, 4, 8)
DEFAULT_TOP_K = 8
BENCHMARK = "internal"
LOCK_PATH = Path(__file__).with_name("cases.lock")
THRESHOLD_KEYS = ("recall_at_4", "precision_at_4", "mrr", "abstention_accuracy", "forbidden_violations")


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
    return Config(roots, global_notes, source.private_paths, database, source.section_bytes)


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


def _checkpoint_config(top_k: int, cutoffs: tuple[int, ...], case_lock: str) -> dict[str, Any]:
    return {
        "retrieval_mode": "lexical",
        "prompt_mode": "internal-deterministic",
        "top_k": top_k,
        "cutoffs": list(cutoffs),
        "answerer_model": None,
        "answerer_provider": None,
        "judge_model": None,
        "judge_provider": None,
        "case_set_fingerprint": case_lock,
    }


def _valid_checkpoint(path: Path, stage: str, run_id: str, config: dict[str, Any], *, case_id: str | None = None) -> dict[str, Any] | None:
    try:
        value = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        value.get("checkpoint_version") != "pausanias.eval.checkpoint.v1"
        or value.get("stage") != stage
        or value.get("run_id") != run_id
        or value.get("config") != config
        or value.get("status") != "complete"
        or case_id is not None and value.get("case_id") != case_id
    ):
        return None
    required = {"dataset_fingerprint", "corpus_fingerprint", "index_generation", "started_at", "finished_at", "output"}
    if not required <= value.keys() or not isinstance(value["output"], dict):
        return None
    output = value["output"]
    if stage == "ingest" and (not isinstance(output.get("files"), list) or not output.get("index_generation")):
        return None
    if stage == "search" and (
        output.get("case_id") != case_id
        or not isinstance(output.get("query"), str)
        or not isinstance(output.get("search_latency_ms"), (int, float))
        or not isinstance(output.get("retrieval_results"), list)
        or not isinstance(output.get("retrieval_fingerprint"), str)
    ):
        return None
    return value


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


def _metrics(evaluations: list[Evaluation], cutoffs: tuple[int, ...], thresholds: dict[str, float]) -> Metrics:
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
            "overall": _group_metrics(evaluations),
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
                ),
            }
        }
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
) -> UnifiedResult | None:
    if predict_only and evaluate_only:
        raise ValueError("--predict-only and --evaluate-only cannot be combined")
    if top_k < max(cutoffs) or sorted(cutoffs) != list(cutoffs):
        raise ValueError("top-k must include sorted cutoff values")
    case_lock = verify_case_lock(cases_path)
    case_set = load_case_set(cases_path)
    cases = list(case_set.cases)
    config_values = _checkpoint_config(top_k, cutoffs, case_lock)
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
    source_config = load_config(config_path)
    config = _runtime_config(source_config, corpus_dir, runtime_corpus, database)
    reuse_checkpoints = resume or evaluate_only
    ingest = _valid_checkpoint(ingest_path, "ingest", run_id, config_values) if reuse_checkpoints else None
    if ingest is not None and (ingest["dataset_fingerprint"] != case_lock or ingest["corpus_fingerprint"] != corpus_hash):
        ingest = None
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
        ).to_dict()
        atomic_write_json(ingest_path, ingest)
    generation = int(ingest["index_generation"])
    evaluations: list[Evaluation] = []
    for case in cases:
        checkpoint_path = search_dir / f"{case.id}.json"
        checkpoint = _valid_checkpoint(checkpoint_path, "search", run_id, config_values, case_id=case.id) if reuse_checkpoints else None
        if evaluate_only and checkpoint is None:
            raise ValueError(f"evaluation requires a complete search checkpoint for {case.id}")
        if checkpoint is not None and checkpoint["corpus_fingerprint"] == corpus_hash and checkpoint["index_generation"] == generation:
            raw = checkpoint["output"]
        else:
            with _temporarily_deleted(runtime_corpus, case.delete_sources):
                refresh: set[str] = set()
                started = time.perf_counter()
                candidates = search(
                    config,
                    case.query,
                    project=case.scope.get("project"),
                    root_id=case.scope.get("root"),
                    all_projects=bool(case.scope.get("all_projects", False)),
                    limit=top_k,
                    refresh=refresh,
                )
                elapsed = (time.perf_counter() - started) * 1000
            raw = {
                "case_id": case.id,
                "query": case.query,
                "search_latency_ms": elapsed,
                "retrieval_fingerprint": fingerprint([candidate.section_id for candidate in candidates]),
                "retrieval_results": [asdict(_candidate_result(candidate, runtime_corpus, rank)) for rank, candidate in enumerate(candidates, 1)],
            }
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
    if predict_only:
        print(f"internal predict-only: {len(evaluations)} search checkpoints; case-set fingerprint {case_lock}")
        return None
    if evaluate_only and len(evaluations) != len(cases):
        raise ValueError("evaluation requires complete search checkpoints")
    thresholds = load_thresholds()
    metrics = _metrics(evaluations, cutoffs, thresholds)
    started_at = ingest["started_at"]
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
    if not metrics.deterministic_gates["by_cutoff"]["4"]["passed"]:
        raise ValueError("internal deterministic gates failed")
    return result


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
        )
        return 0
    except (OSError, ValueError, TypeError, sqlite3.DatabaseError) as exc:
        print(f"eval harness: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
