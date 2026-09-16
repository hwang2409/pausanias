"""Run the local retrieval-accuracy evaluation corpus."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import tomllib
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

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Case":
        required = ("id", "query", "scope", "expected_source_paths", "forbidden_paths", "abstain", "rationale")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"case is missing fields: {', '.join(missing)}")
        if not isinstance(value["scope"], dict):
            raise ValueError(f"case {value['id']} scope must be an object")
        for field in ("expected_source_paths", "forbidden_paths"):
            if not isinstance(value[field], list) or not all(isinstance(item, str) for item in value[field]):
                raise ValueError(f"case {value['id']} {field} must be an array of paths")
        delete_sources = value.get("delete_sources", [])
        if not isinstance(delete_sources, list) or not all(isinstance(item, str) for item in delete_sources):
            raise ValueError(f"case {value['id']} delete_sources must be an array of paths")
        if not isinstance(value["abstain"], bool):
            raise ValueError(f"case {value['id']} abstain must be boolean")
        return cls(
            id=value["id"],
            query=value["query"],
            scope=value["scope"],
            expected_source_paths=tuple(value["expected_source_paths"]),
            forbidden_paths=tuple(value["forbidden_paths"]),
            abstain=value["abstain"],
            rationale=value["rationale"],
            delete_sources=tuple(delete_sources),
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


def load_cases(path: Path = CASES_PATH) -> list[Case]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status", "").startswith("DRAFT") is False:
        raise ValueError("cases must remain marked DRAFT until human review")
    cases = [Case.from_dict(item) for item in payload.get("cases", [])]
    if len(cases) < 40:
        raise ValueError("evaluation requires at least 40 cases")
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case ids must be unique")
    return cases


def load_thresholds(path: Path = THRESHOLDS_PATH) -> dict[str, float]:
    with path.open("rb") as handle:
        values = tomllib.load(handle).get("thresholds", {})
    required = ("recall_at_4", "precision_at_4", "mrr", "abstention_accuracy", "forbidden_violations")
    if any(key not in values for key in required):
        raise ValueError("thresholds must define all aggregate metrics")
    return {key: float(values[key]) for key in required}


def _runtime_config(source: Config, corpus_dir: Path, runtime_corpus: Path, database: Path) -> Config:
    source_corpus = corpus_dir.resolve()

    def map_path(path: Path) -> Path:
        return (runtime_corpus / path.resolve().relative_to(source_corpus)).resolve()

    roots = tuple(Root(root.id, map_path(root.path), root.project, root.excludes) for root in source.roots)
    global_notes = frozenset(map_path(path) for path in source.global_notes)
    return Config(roots, global_notes, source.private_paths, database, source.section_bytes)


def _case_path(runtime_corpus: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else runtime_corpus / path


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
        recall = len(set(result_paths).intersection(expected)) / len(expected)
        precision = len(set(result_paths).intersection(expected)) / k
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

    def average(name: str) -> float:
        values = [getattr(score, name) for score in retrieval_scores]
        return sum(values) / len(values) if values else 1.0

    return {
        "cases": len(scores),
        "retrieval_cases": len(retrieval_scores),
        "recall_at_4": average("recall_at_k"),
        "precision_at_4": average("precision_at_k"),
        "mrr": average("reciprocal_rank"),
        "abstention_accuracy": sum(score.abstention_correct for score in scores) / len(scores) if scores else 1.0,
        "forbidden_violations": sum(len(score.forbidden_paths) for score in scores),
        "average_latency_ms": sum(score.latency_ms for score in scores) / len(scores) if scores else 0.0,
    }


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
            deleted: list[tuple[Path, bytes]] = []
            for source_path in case.delete_sources:
                path = _case_path(runtime_corpus, source_path)
                deleted.append((path, path.read_bytes()))
                path.unlink()
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
            for path, content in deleted:
                path.write_bytes(content)
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


def _print_table(scores: list[CaseScore], aggregate: dict[str, float | int], failures: list[str]) -> None:
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
        f"abstention={aggregate['abstention_accuracy']:.3f} "
        f"forbidden={aggregate['forbidden_violations']}"
    )
    if failures:
        print("threshold failures: " + "; ".join(failures))
    else:
        print("thresholds: pass")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable results")
    parser.add_argument("--case", help="run one case by id")
    args = parser.parse_args(argv)
    try:
        cases = load_cases()
        if args.case:
            cases = [case for case in cases if case.id == args.case]
            if not cases:
                raise ValueError(f"unknown case: {args.case}")
        scores, aggregate = run_cases(cases)
        failures = _threshold_failures(aggregate, load_thresholds())
        if args.json:
            payload = {
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
                    }
                    for score in scores
                ],
                "aggregate": aggregate,
                "threshold_failures": failures,
            }
            print(json.dumps(payload, indent=2))
        else:
            _print_table(scores, aggregate, failures)
        return 1 if failures else 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"eval: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
