from __future__ import annotations

import os
import importlib.util
import json
import signal
import socket
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
    WorkerError,
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


def make_config(tmp_path: Path, semantic_bundle: Path | None = None):
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("# Note\nalpha memory\n")
    path = tmp_path / "config.toml"
    config_text = f'database = "{tmp_path / "index.sqlite3"}"\n\n'
    if semantic_bundle is not None:
        config_text += f'semantic_bundle = "{semantic_bundle}"\n\n'
    path.write_text(config_text + f'[[roots]]\nid = "vault"\nproject = "p"\npath = "{root}"\n')
    return path, load_config(path)


def semantic_extra_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("numpy", "onnxruntime"))


def wait_for_socket(path: Path) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not socket_ready(path):
        time.sleep(0.005)
    assert socket_ready(path)


def worker_pids(config_path: Path) -> list[int]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,command="], capture_output=True, text=True, check=True,
    )
    pids = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2 and "-m pausanias.worker" in fields[1] and str(config_path) in fields[1]:
            pids.append(int(fields[0]))
    return pids


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
        assert first["metrics"]["retrieval_mode"] == "fused"
        assert second["metrics"]["matrix_load_ms"] < first["metrics"]["matrix_load_ms"]
        assert second["metrics"]["model_load_ms"] <= first["metrics"]["model_load_ms"]
    finally:
        worker._stopping.set()
        thread.join(timeout=2)


def test_worker_applies_configured_synonym_table(tmp_path: Path):
    config_path, config = make_config(tmp_path)
    table_path = tmp_path / "synonyms.toml"
    table_path.write_text('version = 1\n\n[terms]\nmemory = ["recall"]\n')
    config = type(config)(
        config.roots, config.global_notes, config.private_paths, config.database,
        config.section_bytes, config.semantic_bundle, table_path,
    )
    core.index(config, encoder=FakeEncoder())
    worker = PersistentWorker(config, encoder=FakeEncoder())

    response = worker._query({"query": "recall", "project": "p"})

    assert response["items"][0]["heading"] == "Note"


def test_live_worker_reloads_changed_synonym_config(tmp_path: Path):
    config_path, config = make_config(tmp_path, semantic_bundle=tmp_path / "missing-bundle")
    core.index(config)
    paths = worker_paths(config.database)
    table_path = tmp_path / "synonyms.toml"
    try:
        first = ensure_worker(config, config_path, time.perf_counter() + 2, paths)
        assert first is not None
        first_response = WorkerClient(first[0]).query(
            {"query": "recall", "project": "p"}, time.perf_counter() + 2,
        )
        assert first_response["items"] == []

        table_path.write_text('version = 1\n\n[terms]\nmemory = ["recall"]\n')
        config_path.write_text(
            f'database = "{config.database}"\nsynonym_table = "{table_path}"\n\n'
            f'[[roots]]\nid = "vault"\nproject = "p"\npath = "{config.roots[0].path}"\n'
        )
        reloaded_config = load_config(config_path)
        second = ensure_worker(reloaded_config, config_path, time.perf_counter() + 2, paths)
        assert second is not None
        second_response = WorkerClient(second[0]).query(
            {"query": "recall", "project": "p"}, time.perf_counter() + 2,
        )
        assert second_response["items"][0]["heading"] == "Note"
    finally:
        stop_worker(config.database, paths)


def test_hook_falls_back_lexically_when_model_is_unavailable(tmp_path: Path):
    bundle = tmp_path / "invalid-bundle"
    bundle.mkdir()
    config_path, config = make_config(tmp_path, semantic_bundle=bundle)
    core.index(config)
    try:
        response = run_hook(config, config_path, "alpha memory", project="p")
        assert [item.heading for item in response.candidates] == ["Note"]
        assert response.metrics.fallback is True
        assert response.metrics.retrieval_mode == "lexical"
        expected_reason = "MODEL_MISSING" if semantic_extra_available() else "EXTRA_MISSING"
        assert response.metrics.disabled_reason == expected_reason
        assert response.metrics.semantic_state == "disabled"
        assert response.metrics.failure_reason == expected_reason
        assert response.metrics.status == "semantic_disabled"
    finally:
        stop_worker(config.database)


def test_worker_rejects_non_boolean_relaxed_matching(tmp_path: Path):
    _, config = make_config(tmp_path)
    worker = PersistentWorker(config)

    with pytest.raises(WorkerError, match="relaxed matching policy is invalid"):
        worker._query({"query": "alpha", "relaxed_matching": "false"})


def test_hook_and_worker_can_disable_relaxed_matching(tmp_path: Path):
    config_path, config = make_config(tmp_path)
    root = config.roots[0].path
    note = root / "note.md"
    note.write_text("# Vault Conventions\n\n## Frontmatter\n\nEvery note starts with a type and an updated date.\n")
    core.index(config)
    query = "what are the vault conventions for frontmatter?"

    direct = run_hook(config, config_path, query, project="p",
                      retrieval_mode="lexical", relaxed_matching=False)
    worker = PersistentWorker(config)
    response = worker._query({"query": query, "project": "p", "relaxed_matching": False})

    assert direct.candidates == []
    assert response["items"] == []


def test_hook_explicit_lexical_mode_skips_semantic_worker(tmp_path: Path):
    config_path, config = make_config(tmp_path)
    core.index(config)

    response = run_hook(config, config_path, "alpha memory", project="p", retrieval_mode="lexical")

    assert [item.heading for item in response.candidates] == ["Note"]
    assert response.metrics.fallback is False
    assert response.metrics.cache_state == "lexical"
    assert response.metrics.retrieval_mode == "lexical"


def test_hook_surfaces_stale_state_and_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path, config = make_config(tmp_path)
    core.index(config, encoder=FakeEncoder())
    note = config.roots[0].path / "note.md"
    note.write_text("# Note\nnew alpha\n")

    class FailingEncoder:
        def encode(self, texts: list[str]):
            raise RuntimeError("refresh failed")

    core.index(config, encoder=FailingEncoder())
    paths = worker_paths(config.database)
    worker = PersistentWorker(config, paths, idle_seconds=2, encoder=FakeEncoder())
    thread = threading.Thread(target=worker.serve)
    thread.start()
    monkeypatch.setattr(hook, "ensure_worker", lambda *args: (paths, 0.1))
    try:
        wait_for_socket(paths.socket)
        response = run_hook(config, config_path, "new alpha", project="p")
        assert response.metrics.semantic_state == "stale"
        assert response.metrics.failure_reason == "REFRESH_FAILED"
        assert response.metrics.status == "semantic_failure"
        assert response.metrics.fallback is True
    finally:
        worker._stopping.set()
        thread.join(timeout=2)


def test_worker_abstains_when_a_matching_source_is_deleted(tmp_path: Path):
    pytest.importorskip("numpy")
    config_path, config = make_config(tmp_path)
    core.index(config, encoder=FakeEncoder())
    note = config.roots[0].path / "note.md"
    worker = PersistentWorker(config, encoder=FakeEncoder())
    original = note.read_bytes()
    try:
        note.unlink()
        response = worker._query({"query": "alpha memory", "project": "p"})
        assert response["items"] == []
    finally:
        note.write_bytes(original)


def test_worker_paths_use_protocol_versioned_namespace(tmp_path: Path):
    _, config = make_config(tmp_path)
    paths = worker_paths(config.database)

    assert paths.socket.name.endswith("-v2.sock")
    assert paths.lock.name.endswith("-v2.lock")
    assert paths.record.name.endswith("-v2.lock.record")


def test_worker_apis_reject_unversioned_paths_without_connecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    import pausanias.worker as worker_module

    config_path, config = make_config(tmp_path)
    legacy_paths = WorkerPaths(tmp_path / "legacy.sock", tmp_path / "legacy.lock")

    def unexpected_connection(*_args, **_kwargs):
        raise AssertionError("unversioned worker path was accessed")

    monkeypatch.setattr(worker_module, "_worker_request", unexpected_connection)
    monkeypatch.setattr(worker_module, "socket_ready", unexpected_connection)

    assert ensure_worker(config, config_path, time.perf_counter() + 1, legacy_paths) is None
    assert stop_worker(config.database, legacy_paths) is False
    with pytest.raises(WorkerError, match="protocol namespace"):
        WorkerClient(legacy_paths)
    with pytest.raises(WorkerError, match="protocol namespace"):
        PersistentWorker(config, legacy_paths)


def test_upgrade_ignores_real_legacy_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    legacy_tree_value = os.environ.get("PAUSANIAS_MAIN_TREE")
    if legacy_tree_value is None:
        pytest.skip("set PAUSANIAS_MAIN_TREE to run the cross-version worker test")
    legacy_tree = Path(legacy_tree_value)
    if not (legacy_tree / "src/worker.py").exists():
        pytest.skip("PAUSANIAS_MAIN_TREE does not contain the legacy worker")

    import pausanias.worker as worker_module

    config_path, config = make_config(tmp_path, semantic_bundle=tmp_path / "missing-bundle")
    core.index(config)
    digest = worker_module.hashlib.sha256(
        str(config.database.expanduser().resolve()).encode()
    ).hexdigest()[:20]
    parent = Path(worker_module.tempfile.gettempdir())
    legacy_paths = WorkerPaths(
        parent / f"pausanias-{digest}.sock", parent / f"pausanias-{digest}.lock",
    )
    connection_count = tmp_path / "legacy-connections"
    script = """
import importlib.util
import sys
from pathlib import Path

legacy_tree = Path(sys.argv[5]).resolve()
package_root = legacy_tree / "src"
spec = importlib.util.spec_from_file_location(
    "pausanias", package_root / "__init__.py",
    submodule_search_locations=[str(package_root)],
)
assert spec is not None and spec.loader is not None
package = importlib.util.module_from_spec(spec)
sys.modules["pausanias"] = package
spec.loader.exec_module(package)

from pausanias.config import load_config
from pausanias.worker import PersistentWorker, WorkerPaths

config_path = Path(sys.argv[1])
paths = WorkerPaths(Path(sys.argv[2]), Path(sys.argv[3]))
connection_count = Path(sys.argv[4])
worker = PersistentWorker(load_config(config_path), paths, idle_seconds=0.5)
original = worker._serve_connection

def counted(connection):
    try:
        count = int(connection_count.read_text()) if connection_count.exists() else 0
        connection_count.write_text(str(count + 1))
    finally:
        original(connection)

worker._serve_connection = counted
worker.serve()
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(legacy_tree)
    legacy = subprocess.Popen(
        [sys.executable, "-c", script, str(config_path), str(legacy_paths.socket),
         str(legacy_paths.lock), str(connection_count), str(legacy_tree)],
        cwd=legacy_tree, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    signalled: list[int] = []
    real_kill = worker_module.os.kill

    def record_signal(pid: int, signum: int) -> None:
        if pid == legacy.pid and signum != 0:
            signalled.append(signum)
        real_kill(pid, signum)

    monkeypatch.setattr(worker_module.os, "kill", record_signal)
    try:
        deadline = time.monotonic() + 2
        while not legacy_paths.socket.exists() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert legacy_paths.socket.exists()

        startup = ensure_worker(config, config_path, time.perf_counter() + 2)
        assert startup is not None
        current_paths, _ = startup
        assert current_paths != legacy_paths
        assert socket_ready(current_paths.socket)
        observed_connections = connection_count.read_text() if connection_count.exists() else "0"
        assert observed_connections == "0"
        assert signalled == []

        legacy.wait(timeout=2)
        assert legacy.returncode == 0
        assert signalled == []

        response = run_hook(config, config_path, "alpha memory", project="p")
        assert [item.heading for item in response.candidates] == ["Note"]
        assert response.metrics.status == "semantic_disabled"
        assert _read_record(current_paths.record)["pid"] != legacy.pid
    finally:
        stop_worker(config.database)
        if legacy.poll() is None:
            real_kill(legacy.pid, signal.SIGTERM)
            legacy.wait(timeout=2)
        legacy_paths.socket.unlink(missing_ok=True)
        legacy_paths.lock.unlink(missing_ok=True)


def test_concurrent_startup_launches_one_real_worker(tmp_path: Path):
    config_path, config = make_config(tmp_path)
    core.index(config)
    script = """
import os, sys, time
from pathlib import Path
from pausanias.config import load_config
import pausanias.worker as worker_module
from pausanias.worker import ensure_worker
config_path = Path(sys.argv[1])
config = load_config(config_path)
result = ensure_worker(config, config_path, time.perf_counter() + 4)
print(result is not None, flush=True)
"""
    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository)
    processes = [subprocess.Popen(
        [sys.executable, "-c", script, str(config_path)],
        cwd=repository, env=environment, stdout=subprocess.PIPE, text=True,
    ) for _ in range(32)]
    try:
        outputs = [process.communicate(timeout=8)[0].strip() for process in processes]
        assert outputs == ["True"] * 32
        paths = worker_paths(config.database)
        pids = worker_pids(config_path)
        assert len(pids) == 1
        assert all(pid_alive(pid) for pid in pids)
        assert socket_ready(paths.socket)
    finally:
        stop_worker(config.database)
        for pid in worker_pids(config_path):
            try:
                os.kill(pid, 15)
            except ProcessLookupError:
                pass


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
    bundle = tmp_path / "invalid-bundle"
    bundle.mkdir()
    config_path, config = make_config(tmp_path, semantic_bundle=bundle)
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


def test_ensure_rejects_foreign_socket_with_forged_ready_record(tmp_path: Path):
    import pausanias.worker as worker_module

    config_path, config = make_config(tmp_path)
    paths = worker_paths(config.database)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(paths.socket))
    server.listen(4)
    server.settimeout(0.05)
    stopping = threading.Event()

    def serve_foreign_socket() -> None:
        while not stopping.is_set():
            try:
                connection, _ = server.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with connection:
                try:
                    data = connection.recv(65536)
                    if data:
                        connection.sendall(
                            json.dumps({"pid": os.getpid(), "launch_nonce": "foreign"}).encode()
                            + b"\n"
                        )
                except OSError:
                    pass

    thread = threading.Thread(target=serve_foreign_socket)
    thread.start()
    _write_record(paths.record, {
        "config_fingerprint": worker_module._config_fingerprint(config),
        "launch_nonce": "forged", "pid": 999999, "readiness": "ready",
        "socket_path": str(paths.socket),
    })
    try:
        assert ensure_worker(config, config_path, time.perf_counter() + 0.1, paths) is None
    finally:
        stopping.set()
        server.close()
        thread.join(timeout=2)
        paths.socket.unlink(missing_ok=True)
        paths.record.unlink(missing_ok=True)


def test_stop_worker_never_signals_unverified_record_pid(tmp_path: Path):
    _, config = make_config(tmp_path)
    paths = worker_paths(config.database)
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    _write_record(paths.record, {
        "launch_nonce": "stale", "pid": sleeper.pid, "readiness": "ready",
        "socket_path": str(paths.socket),
    })
    try:
        assert stop_worker(config.database, paths, timeout=0.05) is False
        assert sleeper.poll() is None
        assert not paths.record.exists()
    finally:
        if sleeper.poll() is None:
            sleeper.terminate()
        sleeper.wait(timeout=2)


def test_sigterm_worker_cleans_socket_and_record(tmp_path: Path):
    config_path, config = make_config(tmp_path)
    core.index(config)
    paths = worker_paths(config.database)
    result = ensure_worker(config, config_path, time.perf_counter() + 2, paths)
    assert result is not None
    record = _read_record(paths.record)
    pid = record["pid"]
    assert isinstance(pid, int)
    try:
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 2
        while pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not pid_alive(pid)
        assert not paths.socket.exists()
        assert not paths.record.exists()
    finally:
        if pid_alive(pid):
            os.kill(pid, signal.SIGTERM)


def test_record_is_private_under_permissive_umask(tmp_path: Path):
    path = tmp_path / "record"
    previous = os.umask(0o022)
    try:
        _write_record(path, {"pid": os.getpid()})
    finally:
        os.umask(previous)
    assert path.stat().st_mode & 0o777 == 0o600


def test_unverified_live_pid_record_is_replaced_without_signalling_it(tmp_path: Path):
    config_path, config = make_config(tmp_path)
    core.index(config)
    paths = worker_paths(config.database)
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    _write_record(paths.record, {
        "pid": sleeper.pid, "launch_nonce": "unverified", "readiness": "starting",
    })
    try:
        assert ensure_worker(config, config_path, time.perf_counter() + 2, paths) is not None
        assert sleeper.poll() is None
        assert _read_record(paths.record)["pid"] != sleeper.pid
    finally:
        stop_worker(config.database, paths)
        if sleeper.poll() is None:
            sleeper.terminate()
        sleeper.wait(timeout=2)


def test_dead_pid_lock_is_taken_over(tmp_path: Path):
    config_path, config = make_config(tmp_path)
    core.index(config)
    paths = worker_paths(config.database)
    _write_record(paths.record, {"pid": 999999, "readiness": "starting"})
    try:
        assert ensure_worker(config, config_path, time.perf_counter() + 2, paths) is not None
        record = _read_record(paths.record)
        assert record["pid"] != 999999
        assert socket_ready(paths.socket)
    finally:
        stop_worker(config.database, paths)


def test_socket_is_not_connectable_before_complete_record_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _, config = make_config(tmp_path)
    paths = worker_paths(config.database)
    worker = PersistentWorker(config, paths, idle_seconds=2)
    publish_started = threading.Event()
    release_publish = threading.Event()
    original_publish = worker._publish

    def delayed_publish(state: str, pid: int) -> None:
        publish_started.set()
        assert release_publish.wait(2)
        original_publish(state, pid)

    monkeypatch.setattr(worker, "_publish", delayed_publish)
    thread = threading.Thread(target=worker.serve)
    thread.start()
    try:
        assert publish_started.wait(2)
        assert not socket_ready(paths.socket)
        assert _read_record(getattr(paths, "record", paths.lock)) == {}
        release_publish.set()
        wait_for_socket(paths.socket)
        record = _read_record(getattr(paths, "record", paths.lock))
        assert record["pid"] == os.getpid()
        assert isinstance(record.get("launch_nonce"), str) and record["launch_nonce"]
        assert record["readiness"] == "ready"
    finally:
        release_publish.set()
        worker._stopping.set()
        thread.join(timeout=2)


def test_record_reader_never_observes_truncate_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    import pausanias.worker as worker_module

    path = tmp_path / "record"
    old = {"pid": 1, "readiness": "ready"}
    new = {"pid": 42, "readiness": "ready"}
    _write_record(path, old)
    write_paused = threading.Event()
    release_write = threading.Event()

    if hasattr(worker_module, "_write_record_fd"):
        original_ftruncate = worker_module.os.ftruncate

        def delayed_ftruncate(fd: int, length: int) -> None:
            original_ftruncate(fd, length)
            write_paused.set()
            assert release_write.wait(2)

        monkeypatch.setattr(worker_module.os, "ftruncate", delayed_ftruncate)
        fd = os.open(path, os.O_RDWR)

        def publish() -> None:
            try:
                worker_module._write_record_fd(fd, new)
            finally:
                os.close(fd)
    else:
        original_replace = worker_module.os.replace

        def delayed_replace(source, destination) -> None:
            write_paused.set()
            assert release_write.wait(2)
            original_replace(source, destination)

        monkeypatch.setattr(worker_module.os, "replace", delayed_replace)

        def publish() -> None:
            worker_module._write_record(path, new)

    writer = threading.Thread(target=publish)
    writer.start()
    assert write_paused.wait(2)
    observed = _read_record(path)
    release_write.set()
    writer.join(timeout=2)
    assert observed == old
    assert not writer.is_alive()
    assert _read_record(path) == new


def test_record_reader_never_sees_partial_atomic_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import pausanias.worker as worker_module

    path = tmp_path / "record"
    old = {"pid": 1, "readiness": "ready"}
    new = {"pid": 42, "readiness": "ready"}
    _write_record(path, old)
    replace_started = threading.Event()
    release_replace = threading.Event()
    original_replace = worker_module.os.replace

    def delayed_replace(source, destination):
        replace_started.set()
        assert release_replace.wait(2)
        original_replace(source, destination)

    monkeypatch.setattr(worker_module.os, "replace", delayed_replace)
    writer = threading.Thread(target=worker_module._write_record, args=(path, new))
    writer.start()
    assert replace_started.wait(2)
    assert _read_record(path) == old
    release_replace.set()
    writer.join(timeout=2)
    assert not writer.is_alive()
    assert _read_record(path) == new


def test_hung_lifecycle_writer_does_not_block_ensure_worker(tmp_path: Path):
    import fcntl

    config_path, config = make_config(tmp_path)
    paths = worker_paths(config.database)
    paths.lock.touch()
    fd = os.open(paths.lock, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        started = time.perf_counter()
        assert ensure_worker(config, config_path, started + 0.1, paths) is None
        assert time.perf_counter() - started < 0.5
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_stop_worker_has_finite_timeout_with_hung_lifecycle_writer(tmp_path: Path):
    import fcntl

    _, config = make_config(tmp_path)
    paths = worker_paths(config.database)
    paths.lock.touch()
    fd = os.open(paths.lock, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        started = time.perf_counter()
        finished = threading.Event()
        result: list[bool] = []

        def stop():
            result.append(stop_worker(config.database, paths))
            finished.set()

        thread = threading.Thread(target=stop)
        thread.start()
        assert finished.wait(0.2)
        assert time.perf_counter() - started < 0.5
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        thread.join(timeout=1)


def test_old_format_lock_content_does_not_break_startup(tmp_path: Path):
    config_path, config = make_config(tmp_path)
    core.index(config)
    paths = worker_paths(config.database)
    paths.lock.write_text('{"pid": 999999, "readiness": "starting"}')
    try:
        assert ensure_worker(config, config_path, time.perf_counter() + 2, paths) is not None
        assert _read_record(paths.record)["pid"] != 999999
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


def test_cancel_before_registration_is_retained(tmp_path: Path):
    _, config = make_config(tmp_path)
    worker = PersistentWorker(config)

    assert worker._cancel_request("request-before-registration") is False
    event = worker._register_request("request-before-registration")

    assert event.is_set()
    assert worker._cancellation_tombstones == {}
