from __future__ import annotations

import hashlib
from pathlib import Path

from pausanias.bench import format_report, run_benchmark
from pausanias.synth import generate_corpus


def corpus_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(directory).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def test_synthetic_corpus_is_deterministic(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_corpus(first, file_count=30, seed=41)
    generate_corpus(second, file_count=30, seed=41)

    assert corpus_hash(first) == corpus_hash(second)


def test_benchmark_report_has_positive_metrics():
    report = run_benchmark(file_count=30, query_count=20, seed=41)

    assert report["source"]["file_count"] == 30
    assert report["queries"]["count"] == 20
    assert report["gate"]["enforced"] is False
    assert all(isinstance(value, (int, float)) and value > 0 for value in report["metrics"].values())
    assert set(report["metrics"]) == set(report["checks"])
    assert report["metrics"]["first_query_ms"] > 0
    assert "cold_search_ms" not in report["metrics"]


def test_benchmark_query_modes_have_hits():
    report = run_benchmark(file_count=30, query_count=60, seed=41)

    modes = report["queries"]["modes"]
    assert set(modes) == {"single", "phrase", "identifier", "multi", "scoped", "global", "miss"}
    for mode in set(modes) - {"miss"}:
        assert modes[mode]["hit_count_p50"] > 0
        assert modes[mode]["warm_search_p95_ms"] > 0
    assert modes["miss"]["hit_count_max"] == 0


def test_default_workload_has_representative_hit_count_mix():
    report = run_benchmark()

    mix = report["queries"]["hit_count_mix"]
    assert mix["shares"]["three_or_more"] >= 0.60
    assert mix["shares"]["zero"] <= 0.10
    assert mix["shares"]["one"] <= 0.10


def test_benchmark_gate_is_only_enforced_for_built_in_corpus(tmp_path: Path):
    corpus = tmp_path / "corpus"
    generate_corpus(corpus, file_count=30, seed=41)

    report = run_benchmark(corpus_dir=corpus, seed=41)

    assert report["source"]["file_count"] == 30
    assert report["gate"]["enforced"] is False


def test_targetless_metrics_are_not_reported_as_passed():
    report = run_benchmark(file_count=30, query_count=20, seed=41)

    for name in ("index_build_ms", "incremental_refresh_ms", "warm_search_p50_ms"):
        assert report["checks"][name]["passed"] is None
    assert "n/a" in format_report(report)
