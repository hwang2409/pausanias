from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from pausanias import core
from pausanias.config import load_config
from pausanias.hook import run_hook
from pausanias.worker import PersistentWorker, WorkerClient, WorkerPaths, socket_ready, stop_worker, worker_paths


class FakeEncoder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        values = []
        for text in texts:
            vector = [0.0] * 384
            vector[0] = 1.0 if "alpha" in text else 0.0
            vector[1] = 1.0 if "beta" in text else 0.0
            values.append(vector)
        return values


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
    finally:
        stop_worker(config.database)
