"""Run the local retrieval-accuracy evaluation corpus."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pausanias.config import Config, Root, load_config
from pausanias.core import Candidate, index, search

EVAL_DIR = Path(__file__).resolve().parent
CORPUS_DIR = EVAL_DIR / "corpus"
CONFIG_PATH = EVAL_DIR / "corpus.toml"
CASES_PATH = EVAL_DIR / "cases.json"
THRESHOLDS_PATH = EVAL_DIR / "thresholds.toml"
K = 4
CASE_STATUSES = frozenset(("DRAFT", "REVIEWED"))


@dataclass(frozen=True)
class Case:
    id: str
    query: str
    scope: dict[str, Any]
    expected_source_paths: tuple[str, ...]
    forbidden_paths: tuple[str, ...]
    abstain: bool
    rationale: str
    delete_sources: tuple[str, ...] = ()
    category: str = "standard"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Case":
        required = ("id", "query", "scope", "expected_source_paths", "forbidden_paths", "abstain", "rationale")
        missing = [key for key in required if key not in value]
        if missing:
            case_id = value.get("id", "<unknown>")
            raise ValueError(f"case {case_id} is missing fields: {', '.join(missing)}")
        case_id = value["id"]
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("case id must be a non-empty string")
        if not isinstance(value["query"], str):
            raise ValueError(f"case {case_id} query must be a string")
        if not isinstance(value["rationale"], str):
            raise ValueError(f"case {case_id} rationale must be a string")
        if not isinstance(value["scope"], dict):
            raise ValueError(f"case {case_id} scope must be an object")
        for field in ("expected_source_paths", "forbidden_paths"):
            if not isinstance(value[field], list) or not all(isinstance(item, str) for item in value[field]):
                raise ValueError(f"case {case_id} {field} must be an array of paths")
        delete_sources = value.get("delete_sources", [])
        if not isinstance(delete_sources, list) or not all(isinstance(item, str) for item in delete_sources):
            raise ValueError(f"case {case_id} delete_sources must be an array of paths")
        if not isinstance(value["abstain"], bool):
            raise ValueError(f"case {case_id} abstain must be boolean")
        category = value.get("category", "standard")
        if not isinstance(category, str) or not category:
            raise ValueError(f"case {case_id} category must be a non-empty string")
        return cls(
            id=case_id,
            query=value["query"],
            scope=value["scope"],
            expected_source_paths=tuple(value["expected_source_paths"]),
            forbidden_paths=tuple(value["forbidden_paths"]),
            abstain=value["abstain"],
            rationale=value["rationale"],
            delete_sources=tuple(delete_sources),
            category=category,
        )


@dataclass(frozen=True)
class CaseScore:
    case: Case
    result_paths: tuple[str, ...]
    recall_at_k: float | None
    precision_at_k: float | None
    reciprocal_rank: float | None
    abstention_correct: bool
    forbidden_paths: tuple[str, ...]
    latency_ms: float


@dataclass(frozen=True)
class CaseSet:
    status: str
    cases: tuple[Case, ...]


def load_case_set(path: Path = CASES_PATH) -> CaseSet:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("case file must contain an object")
    status = payload.get("status")
    if status not in CASE_STATUSES:
        raise ValueError("case set status must be exactly DRAFT or REVIEWED")
    raw_cases = payload.get("cases", [])
    if not isinstance(raw_cases, list):
        raise ValueError("cases must be an array")
    cases = []
    for item in raw_cases:
        if not isinstance(item, dict):
            raise ValueError("each case must be an object")
        cases.append(Case.from_dict(item))
    if len(cases) < 40:
        raise ValueError("evaluation requires at least 40 cases")
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case ids must be unique")
    return CaseSet(status, tuple(cases))


def load_cases(path: Path = CASES_PATH) -> list[Case]:
    return list(load_case_set(path).cases)


def load_thresholds(path: Path = THRESHOLDS_PATH) -> dict[str, Any]:
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    values = document.get("thresholds", {})
    required = ("recall_at_4", "precision_at_4", "mrr", "abstention_accuracy", "forbidden_violations")
    if any(key not in values for key in required):
        raise ValueError("thresholds must define all aggregate metrics")
    categories = values.get("categories", {})
    if not isinstance(categories, dict):
        raise ValueError("threshold categories must be a table")
    parsed_categories: dict[str, dict[str, float]] = {}
    for category, category_values in categories.items():
        if not isinstance(category, str) or not isinstance(category_values, dict):
            raise ValueError("each threshold category must be a table")
        if any(key not in category_values for key in required):
            raise ValueError(f"threshold category {category} must define all metrics")
        parsed_categories[category] = {key: float(category_values[key]) for key in required}
    fused_categories = values.get("fused_categories", {})
    if not isinstance(fused_categories, dict):
        raise ValueError("fused threshold categories must be a table")
    parsed_fused_categories: dict[str, dict[str, float]] = {}
    for category, category_values in fused_categories.items():
        if not isinstance(category, str) or not isinstance(category_values, dict):
            raise ValueError("each fused threshold category must be a table")
        if any(key not in category_values for key in required):
            raise ValueError(f"fused threshold category {category} must define all metrics")
        parsed_fused_categories[category] = {key: float(category_values[key]) for key in required}
    result = {
        **{key: float(values[key]) for key in required},
        "categories": parsed_categories,
        "fused_categories": parsed_fused_categories,
    }
    arc2 = document.get("arc2", {})
    if not isinstance(arc2, dict):
        raise ValueError("arc2 thresholds must be a table")
    arc2_keys = (
        "semantic_recall_at_4",
        "paraphrase_recall_at_4",
        "held_out_recall_at_4",
        "verbatim_recall_at_4",
        "abstention_accuracy",
        "forbidden_violations",
        "warm_hybrid_p95_ms",
    )
    if arc2:
        if any(key not in arc2 for key in arc2_keys):
            raise ValueError("arc2 thresholds must define all hypotheses")
        result["arc2"] = {key: float(arc2[key]) for key in arc2_keys}
    return result


def _runtime_config(source: Config, corpus_dir: Path, runtime_corpus: Path, database: Path) -> Config:
    source_corpus = corpus_dir.resolve()

    def map_path(path: Path) -> Path:
        return (runtime_corpus / path.resolve().relative_to(source_corpus)).resolve()

    roots = tuple(Root(root.id, map_path(root.path), root.project, root.excludes) for root in source.roots)
    global_notes = frozenset(map_path(path) for path in source.global_notes)
    return Config(roots, global_notes, source.private_paths, database, source.section_bytes)


def _delete_path(runtime_corpus: Path, value: str) -> Path:
    corpus = runtime_corpus.resolve()
    requested = Path(value).expanduser()
    candidate = requested if requested.is_absolute() else corpus / requested
    try:
        path = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"delete source does not exist: {value}") from exc
    try:
        path.relative_to(corpus)
    except ValueError as exc:
        raise ValueError(f"delete source is outside evaluation corpus: {value}") from exc
    if path == corpus or not path.is_file():
        raise ValueError(f"delete source is not a file in evaluation corpus: {value}")
    return path


@contextmanager
def _temporarily_deleted(runtime_corpus: Path, source_paths: tuple[str, ...]):
    deleted: list[tuple[Path, bytes]] = []
    try:
        for source_path in source_paths:
            path = _delete_path(runtime_corpus, source_path)
            deleted.append((path, path.read_bytes()))
            path.unlink()
        yield
    finally:
        for path, content in deleted:
            path.write_bytes(content)


def _relative_path(runtime_corpus: Path, value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        return value
    return path.resolve().relative_to(runtime_corpus.resolve()).as_posix()


def score_case(case: Case, results: list[Candidate], runtime_corpus: Path, elapsed_ms: float, k: int = K) -> CaseScore:
    top_results = results[:k]
    result_paths = tuple(_relative_path(runtime_corpus, result.canonical_path) for result in top_results)
    expected = set(case.expected_source_paths)
    relevant_positions = [position for position, path in enumerate(result_paths, start=1) if path in expected]
    forbidden = tuple(sorted(set(result_paths).intersection(case.forbidden_paths)))
    if expected:
        relevant_count = len(set(result_paths).intersection(expected))
        recall = relevant_count / len(expected)
        precision = relevant_count / len(top_results) if top_results else 0.0
        reciprocal_rank = 1.0 / relevant_positions[0] if relevant_positions else 0.0
    else:
        recall = precision = reciprocal_rank = None
    return CaseScore(
        case=case,
        result_paths=result_paths,
        recall_at_k=recall,
        precision_at_k=precision,
        reciprocal_rank=reciprocal_rank,
        abstention_correct=(not top_results) == case.abstain,
        forbidden_paths=forbidden,
        latency_ms=elapsed_ms,
    )


def _aggregate(scores: list[CaseScore]) -> dict[str, float | int]:
    retrieval_scores = [score for score in scores if score.recall_at_k is not None]
    abstention_scores = [score for score in scores if score.case.abstain]

    def average(name: str) -> float:
        values = [getattr(score, name) for score in retrieval_scores]
        return sum(values) / len(values) if values else 1.0

    return {
        "cases": len(scores),
        "retrieval_cases": len(retrieval_scores),
        "abstention_cases": len(abstention_scores),
        "recall_at_4": average("recall_at_k"),
        "precision_at_4": average("precision_at_k"),
        "mrr": average("reciprocal_rank"),
        "abstention_accuracy": (
            sum(score.abstention_correct for score in abstention_scores) / len(abstention_scores)
            if abstention_scores
            else 1.0
        ),
        "forbidden_violations": sum(len(score.forbidden_paths) for score in scores),
        "average_latency_ms": sum(score.latency_ms for score in scores) / len(scores) if scores else 0.0,
    }


def _category_aggregates(scores: list[CaseScore]) -> dict[str, dict[str, float | int]]:
    categories = sorted({score.case.category for score in scores})
    return {category: _aggregate([score for score in scores if score.case.category == category]) for category in categories}


def run_cases(
    cases: list[Case],
    corpus_dir: Path = CORPUS_DIR,
    config_path: Path = CONFIG_PATH,
) -> tuple[list[CaseScore], dict[str, float | int]]:
    source_config = load_config(config_path)
    with tempfile.TemporaryDirectory(prefix="pausanias-eval-") as temporary:
        temporary_root = Path(temporary)
        runtime_corpus = temporary_root / "corpus"
        shutil.copytree(corpus_dir, runtime_corpus)
        config = _runtime_config(source_config, corpus_dir, runtime_corpus, temporary_root / "index.sqlite3")
        index(config, rebuild=True)
        scores: list[CaseScore] = []
        for case in cases:
            with _temporarily_deleted(runtime_corpus, case.delete_sources):
                started = time.perf_counter()
                results = search(
                    config,
                    case.query,
                    project=case.scope.get("project"),
                    root_id=case.scope.get("root"),
                    all_projects=bool(case.scope.get("all_projects", False)),
                    limit=K,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                scores.append(score_case(case, results, runtime_corpus, elapsed_ms))
        return scores, _aggregate(scores)


def _threshold_failures(aggregate: dict[str, float | int], thresholds: dict[str, float]) -> list[str]:
    failures: list[str] = []
    for metric in ("recall_at_4", "precision_at_4", "mrr", "abstention_accuracy"):
        if float(aggregate[metric]) < thresholds[metric]:
            failures.append(f"{metric}={aggregate[metric]:.3f} < {thresholds[metric]:.3f}")
    if int(aggregate["forbidden_violations"]) > thresholds["forbidden_violations"]:
        failures.append(
            f"forbidden_violations={aggregate['forbidden_violations']} > {thresholds['forbidden_violations']:g}"
        )
    return failures


def _case_failures(score: CaseScore) -> list[str]:
    failures: list[str] = []
    if score.case.abstain:
        if not score.abstention_correct:
            failures.append("expected abstention")
    elif score.recall_at_k != 1.0:
        recall = "-" if score.recall_at_k is None else f"{score.recall_at_k:.3f}"
        failures.append(f"missing expected sources (recall@4={recall})")
    if score.forbidden_paths:
        failures.append("returned forbidden sources: " + ", ".join(score.forbidden_paths))
    return failures


def _print_table(
    scores: list[CaseScore],
    aggregate: dict[str, float | int],
    failures: list[str],
    case_set_status: str,
    single_case: bool,
) -> None:
    print(f"case set status: {case_set_status}")
    print("case                                      recall  prec   mrr    abstain  forbidden  results")
    print("-" * 108)
    for score in scores:
        recall = "-" if score.recall_at_k is None else f"{score.recall_at_k:.2f}"
        precision = "-" if score.precision_at_k is None else f"{score.precision_at_k:.2f}"
        mrr = "-" if score.reciprocal_rank is None else f"{score.reciprocal_rank:.2f}"
        abstain = "ok" if score.abstention_correct else "fail"
        forbidden = ",".join(score.forbidden_paths) or "-"
        print(f"{score.case.id:40} {recall:>6} {precision:>6} {mrr:>6} {abstain:>8} {forbidden:>10}  {', '.join(score.result_paths) or '-'}")
    print()
    print(
        "aggregate: "
        f"recall@4={aggregate['recall_at_4']:.3f} "
        f"precision@4={aggregate['precision_at_4']:.3f} "
        f"mrr={aggregate['mrr']:.3f} "
        f"abstention={aggregate['abstention_accuracy']:.3f} ({aggregate['abstention_cases']} cases) "
        f"forbidden={aggregate['forbidden_violations']}"
    )
    if single_case:
        if failures:
            print("case expectation failures: " + "; ".join(failures))
        else:
            print("case expectations: pass (full-run thresholds not applied)")
    elif failures:
        print("threshold failures: " + "; ".join(failures))
    else:
        print("thresholds: pass")


def main(argv: list[str] | None = None) -> int:
    modern_flags = {
        "--predict-only", "--evaluate-only", "--resume", "--run-id", "--results-dir", "--top-k",
        "--cutoffs", "--corpus", "--retrieval-mode", "--through-worker",
    }
    arguments = sys.argv[1:] if argv is None else argv
    if any(flag in modern_flags or flag.startswith("--retrieval-mode=") for flag in arguments):
        from .harness import main as harness_main

        return harness_main(arguments)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable results")
    parser.add_argument("--case", help="run one case by id")
    parser.add_argument("--cases", type=Path, default=CASES_PATH, help="case-set JSON path")
    parser.add_argument("--thresholds", type=Path, default=THRESHOLDS_PATH, help="threshold TOML path")
    args = parser.parse_args(argv)
    try:
        case_set = load_case_set(args.cases)
        cases = list(case_set.cases)
        single_case = bool(args.case)
        if args.case:
            cases = [case for case in cases if case.id == args.case]
            if not cases:
                raise ValueError(f"unknown case: {args.case}")
        scores, aggregate = run_cases(cases)
        failures = _case_failures(scores[0]) if single_case else _threshold_failures(aggregate, load_thresholds(args.thresholds))
        category_aggregates = _category_aggregates(scores)
        if args.json:
            payload = {
                "case_set_status": case_set.status,
                "cases": [
                    {
                        "id": score.case.id,
                        "results": score.result_paths,
                        "recall_at_4": score.recall_at_k,
                        "precision_at_4": score.precision_at_k,
                        "mrr": score.reciprocal_rank,
                        "abstention_correct": score.abstention_correct,
                        "forbidden_paths": score.forbidden_paths,
                        "latency_ms": score.latency_ms,
                        "category": score.case.category,
                    }
                    for score in scores
                ],
                "aggregate": aggregate,
                "category_aggregates": category_aggregates,
                "single_case": single_case,
                "case_failures": failures if single_case else [],
                "threshold_failures": [] if single_case else failures,
            }
            print(json.dumps(payload, indent=2))
        else:
            _print_table(scores, aggregate, failures, case_set.status, single_case)
        return 1 if failures else 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"eval: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
