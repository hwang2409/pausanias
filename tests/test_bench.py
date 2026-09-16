from __future__ import annotations

import hashlib
from pathlib import Path

from pausanias.bench import run_benchmark
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
