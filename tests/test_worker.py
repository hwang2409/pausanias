from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from pausanias import core
from pausanias import hook
from pausanias.config import load_config
from pausanias.hook import run_hook
from pausanias.worker import (
    PersistentWorker,
    WorkerClient,
    WorkerPaths,
    _read_record,
    _write_record,
    ensure_worker,
    pid_alive,
    socket_ready,
    stop_worker,
    worker_paths,
)


class FakeEncoder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        values = []
        for text in texts:
            vector = [0.0] * 384
            vector[0] = 1.0 if "alpha" in text else 0.0
            vector[1] = 1.0 if "beta" in text else 0.0
            values.append(vector)
        return values


class FailingEncoder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("encode failed")


def make_config(tmp_path: Path):
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("# Note\nalpha memory\n")
    path = tmp_path / "config.toml"
    path.write_text(
        f'database = "{tmp_path / "index.sqlite3"}"\n\n'
        f'[[roots]]\nid = "vault"\nproject = "p"\npath = "{root}"\n'
    )
    return path, load_config(path)


def wait_for_socket(path: Path) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not socket_ready(path):
        time.sleep(0.005)
    assert socket_ready(path)


def test_worker_reuses_encoder_and_matrix(tmp_path: Path):
    pytest.importorskip("numpy")
    config_path, config = make_config(tmp_path)
    core.index(config, encoder=FakeEncoder())
    paths = worker_paths(config.database)
    worker = PersistentWorker(config, paths, idle_seconds=2, encoder=FakeEncoder())
    thread = threading.Thread(target=worker.serve)
    thread.start()
    try:
        wait_for_socket(paths.socket)
        client = WorkerClient(paths)
        deadline = time.perf_counter() + 2
        first = client.query({"query": "alpha paraphrase", "project": "p", "limit": 20}, deadline)
        second = client.query({"query": "alpha paraphrase", "project": "p", "limit": 20}, deadline)
        assert first["items"][0]["heading"] == "Note"
        assert second["metrics"]["matrix_load_ms"] < first["metrics"]["matrix_load_ms"]
        assert second["metrics"]["model_load_ms"] <= first["metrics"]["model_load_ms"]
    finally:
        worker._stopping.set()
        thread.join(timeout=2)


def test_hook_falls_back_lexically_when_model_is_unavailable(tmp_path: Path):
    config_path, config = make_config(tmp_path)
    core.index(config)
    try:
        response = run_hook(config, config_path, "alpha memory", project="p")
        assert [item.heading for item in response.candidates] == ["Note"]
        assert response.metrics.fallback is True
        assert response.metrics.disabled_reason == "MODEL_MISSING"
    finally:
        stop_worker(config.database)


def test_concurrent_startup_launches_one_real_worker(tmp_path: Path):
    config_path, config = make_config(tmp_path)
    core.index(config)
    script = """
import sys, time
from pathlib import Path
from pausanias.config import load_config
from pausanias.worker import ensure_worker
config_path = Path(sys.argv[1])
config = load_config(config_path)
result = ensure_worker(config, config_path, time.perf_counter() + 2)
print(result is not None, flush=True)
"""
    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository)
    processes = [subprocess.Popen(
        [sys.executable, "-c", script, str(config_path)],
        cwd=repository, env=environment, stdout=subprocess.PIPE, text=True,
    ) for _ in range(2)]
    try:
        outputs = [process.communicate(timeout=4)[0].strip() for process in processes]
        assert outputs == ["True", "True"]
        record = _read_record(worker_paths(config.database).lock)
        assert isinstance(record.get("pid"), int)
        assert pid_alive(record["pid"])
    finally:
        stop_worker(config.database)


def test_waiter_timeout_returns_lexical_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path, config = make_config(tmp_path)
    core.index(config)
    monkeypatch.setattr(hook, "ensure_worker", lambda *args: None)
    started = time.perf_counter()
    response = run_hook(config, config_path, "alpha memory", project="p", deadline_ms=50)
    elapsed = (time.perf_counter() - started) * 1000
    assert response.metrics.fallback is True
    assert elapsed < 150


def test_worker_death_mid_query_recovers_on_next_hook(tmp_path: Path):
    config_path, config = make_config(tmp_path)
    core.index(config)
    paths = worker_paths(config.database)
    try:
        first = run_hook(config, config_path, "alpha memory", project="p")
        assert first.metrics.fallback is True
        assert stop_worker(config.database, paths)
        second = run_hook(config, config_path, "alpha memory", project="p")
        assert second.metrics.fallback is True
        assert [item.heading for item in second.candidates] == ["Note"]
    finally:
        stop_worker(config.database, paths)


def test_live_pid_lock_is_not_treated_as_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path, config = make_config(tmp_path)
    paths = worker_paths(config.database)
    _write_record(paths.lock, {"pid": os.getpid(), "readiness": "starting"})
    monkeypatch.setattr("pausanias.worker.launch_worker", lambda *args: pytest.fail("live worker was replaced"))
    assert ensure_worker(config, config_path, time.perf_counter() + 0.03, paths) is None


def test_dead_pid_lock_is_taken_over(tmp_path: Path):
    config_path, config = make_config(tmp_path)
    core.index(config)
    paths = worker_paths(config.database)
    _write_record(paths.lock, {"pid": 999999, "readiness": "starting"})
    try:
        assert ensure_worker(config, config_path, time.perf_counter() + 2, paths) is not None
        record = _read_record(paths.lock)
        assert record["pid"] != 999999
        assert socket_ready(paths.socket)
    finally:
        stop_worker(config.database, paths)


def test_generation_invalidation_loads_new_matrix_mid_lifecycle(tmp_path: Path):
    config_path, config = make_config(tmp_path)
    core.index(config, encoder=FakeEncoder())
    paths = worker_paths(config.database)
    worker = PersistentWorker(config, paths, idle_seconds=2, encoder=FakeEncoder())
    thread = threading.Thread(target=worker.serve)
    thread.start()
    try:
        wait_for_socket(paths.socket)
        first = WorkerClient(paths).query({"query": "alpha", "project": "p"}, time.perf_counter() + 2)
        note = config.roots[0].path / "note.md"
        note.write_text("# Note\nbeta memory\n")
        core.index(config, encoder=FakeEncoder())
        second = WorkerClient(paths).query({"query": "beta", "project": "p"}, time.perf_counter() + 2)
        assert first["items"][0]["text"] != second["items"][0]["text"]
        assert second["metrics"]["matrix_load_ms"] > 0
    finally:
        worker._stopping.set()
        thread.join(timeout=2)


def test_encode_failure_has_explicit_fallback_state(tmp_path: Path):
    config_path, config = make_config(tmp_path)
    core.index(config)
    worker = PersistentWorker(config, encoder=FailingEncoder())
    response = worker._query({"query": "alpha memory", "project": "p"})
    assert response["status"] == "semantic_failure"
    assert response["metrics"]["fallback"] is True
    assert response["items"]


def test_failure_cache_recovers_and_surfaces_reason(tmp_path: Path):
    config_path, config = make_config(tmp_path)
    core.index(config, encoder=FakeEncoder())
    repaired = False
    calls = 0

    def factory(bundle: Path):
        nonlocal calls
        calls += 1
        if not repaired:
            raise RuntimeError("model loading failed")
        return FakeEncoder()

    worker = PersistentWorker(config, encoder_factory=factory)
    first = worker._query({"query": "alpha", "project": "p"})
    assert first["metrics"]["disabled_reason"] == "model loading failed"
    repaired = True
    second = worker._query({"query": "alpha", "project": "p"})
    assert calls == 2
    assert second["metrics"]["disabled_reason"] is None
    try:
        import numpy  # noqa: F401
    except ImportError:
        assert second["metrics"]["fallback"] is True
    else:
        assert second["metrics"]["fallback"] is False


def test_deadline_cancels_worker_and_returns_before_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path, config = make_config(tmp_path)
    core.index(config, encoder=FakeEncoder())
    paths = worker_paths(config.database)
    cancellation_seen = threading.Event()
    worker: PersistentWorker

    class SlowEncoder:
        def encode(self, texts: list[str]) -> list[list[float]]:
            while True:
                with worker._requests_lock:
                    cancelled = any(event.is_set() for event in worker._requests.values())
                if cancelled:
                    cancellation_seen.set()
                    return FakeEncoder().encode(texts)
                time.sleep(0.002)

    worker = PersistentWorker(config, paths, idle_seconds=2, encoder=SlowEncoder())
    thread = threading.Thread(target=worker.serve)
    thread.start()
    monkeypatch.setattr(hook, "ensure_worker", lambda *args: (paths, 0.1))
    try:
        wait_for_socket(paths.socket)
        started = time.perf_counter()
        response = run_hook(config, config_path, "alpha", project="p", deadline_ms=50)
        assert (time.perf_counter() - started) < 0.15
        assert response.metrics.fallback is True
        assert cancellation_seen.wait(1)
    finally:
        worker._stopping.set()
        thread.join(timeout=2)
