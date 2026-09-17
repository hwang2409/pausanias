from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

import pausanias.bench as bench
from pausanias.bench import QuerySpec, _run_hook_benchmark, format_report, run_benchmark
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


def test_hook_process_uses_parent_wall_time(monkeypatch, tmp_path: Path):
    payload = {"items": [], "metrics": {"hook_total_ms": 1.0, "fallback": False}}

    def fake_run(*args, **kwargs):
        time.sleep(0.01)
        return subprocess.CompletedProcess(args[0], 0, json.dumps(payload), "")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    query = QuerySpec("alpha")
    result = bench._hook_process(tmp_path / "config.toml", query)
    assert result["metrics"]["hook_internal_ms"] == 1.0
    assert result["metrics"]["hook_total_ms"] >= 10.0


def test_hook_gate_fails_when_parent_budget_is_exceeded(tmp_path: Path, monkeypatch):
    config_path, config = _benchmark_config(tmp_path)
    queries = [QuerySpec("alpha", project="real")]
    metrics = {
        "hook_total_ms": 301.0,
        "worker_startup_ms": 1.0,
        "model_load_ms": 1.0,
        "matrix_load_ms": 1.0,
        "encode_ms": 1.0,
        "scan_ms": 1.0,
        "fallback_ms": 1.0,
        "fts_search_ms": 1.0,
        "hybrid_overhead_ms": 1.0,
        "fallback": False,
        "cache_state": "warm",
    }
    monkeypatch.setattr(bench, "_hook_process", lambda *args: {"metrics": dict(metrics), "items": []})
    monkeypatch.setattr(bench, "stop_worker", lambda *args: None)
    result = _run_hook_benchmark(config_path, config, queries, 1)
    assert result["warm"]["gate"]["passed"] is False


def test_hook_report_uses_index_encoding_throughput(tmp_path: Path, monkeypatch):
    config_path, config = _benchmark_config(tmp_path)
    queries = [QuerySpec("alpha", project="real")]
    metrics = {
        "hook_total_ms": 1.0,
        "worker_startup_ms": 1.0,
        "model_load_ms": 1.0,
        "matrix_load_ms": 1.0,
        "encode_ms": 1.0,
        "scan_ms": 1.0,
        "fallback_ms": 1.0,
        "fts_search_ms": 1.0,
        "hybrid_overhead_ms": 1.0,
        "fallback": False,
        "cache_state": "warm",
    }
    monkeypatch.setattr(bench, "_hook_process", lambda *args: {"metrics": dict(metrics), "items": []})
    monkeypatch.setattr(bench, "stop_worker", lambda *args: None)
    monkeypatch.setattr(
        bench,
        "_read_verified_bundle_manifest",
        lambda *args: {"manifest_fingerprint": "from-bundle", "runtime": {"provider": "test"}},
    )

    result = _run_hook_benchmark(
        config_path, config, queries, 1, {"section_count": 4.0, "elapsed_ms": 200.0},
    )

    assert result["throughput"]["encoded_sections_per_second"] == 20.0
    assert result["model_manifest_fingerprint"] == "from-bundle"
    assert "encoded_sections_per_second" not in result["warm"]["throughput"]
    assert "encoded_sections_per_second" not in result["cold"]["throughput"]


def test_bundle_fingerprint_is_read_from_verified_manifest(tmp_path: Path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = {"format_version": 1, "model_id": "test"}
    stored = {**manifest, "manifest_fingerprint": bench.manifest_fingerprint(manifest)}
    (bundle / "manifest.json").write_text(json.dumps(stored))
    monkeypatch.setattr(bench, "_resolve_active_bundle", lambda path: bundle)
    monkeypatch.setattr(bench, "verify_bundle", lambda path: None)

    assert bench._read_verified_bundle_manifest(tmp_path / "pointer") == stored


def _benchmark_config(tmp_path: Path):
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("# Note\nalpha memory\n")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'database = "{tmp_path / "index.sqlite3"}"\n\n'
        f'[[roots]]\nid = "root"\nproject = "real"\npath = "{root}"\n'
    )
    from pausanias.config import load_config
    from pausanias.core import index
    config = load_config(config_path)
    index(config)
    return config_path, config
