"""Persistent semantic worker and its startup-lock lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable
import uuid

from .config import Config, load_config
from .core import search
from .model_bundle import BundleError, MODEL_BUNDLE_MANIFEST, _resolve_active_bundle
from .vectors import Candidate, OnnxEncoder, SemanticError, encoder_vectors, scan_vectors, semantic_backend_reason, unpack_vector


ADAPTER_DEADLINE_MS = 750.0
DEFAULT_IDLE_SECONDS = 30 * 60
SOCKET_TIMEOUT_SECONDS = 0.05


class WorkerError(RuntimeError):
    """Raised when a worker cannot serve a request."""


@dataclass(frozen=True)
class WorkerPaths:
    socket: Path
    lock: Path


@dataclass(frozen=True)
class HookMetrics:
    worker_startup_ms: float = 0.000001
    model_load_ms: float = 0.000001
    matrix_load_ms: float = 0.000001
    encode_ms: float = 0.000001
    scan_ms: float = 0.000001
    fallback_ms: float = 0.000001
    fts_search_ms: float = 0.000001
    hybrid_overhead_ms: float = 0.000001
    hook_total_ms: float = 0.000001
    fallback: bool = False
    cache_state: str = "cold"


def worker_paths(database: Path) -> WorkerPaths:
    identity = str(database.expanduser().resolve())
    digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
    parent = Path(tempfile.gettempdir())
    return WorkerPaths(parent / f"pausanias-{digest}.sock", parent / f"pausanias-{digest}.lock")


def _positive_ms(started: float) -> float:
    return max((time.perf_counter() - started) * 1000.0, 0.000001)


def _read_record(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_record(path: Path, record: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def pid_alive(pid: int) -> bool:
    if pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def socket_ready(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(SOCKET_TIMEOUT_SECONDS)
            client.connect(str(path))
        return True
    except OSError:
        return False


def stop_worker(database: Path, paths: WorkerPaths | None = None) -> bool:
    worker_paths_value = paths or worker_paths(database)
    record = _read_record(worker_paths_value.lock)
    pid = record.get("pid")
    stopped = False
    if isinstance(pid, int) and pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
            stopped = True
        except OSError:
            pass
    try:
        worker_paths_value.socket.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass
    return stopped


class PersistentWorker:
    """Serve semantic requests until idle timeout or explicit shutdown."""

    def __init__(self, config: Config, paths: WorkerPaths | None = None,
                 encoder_factory: Callable[[Path], object] | None = None,
                 idle_seconds: float = DEFAULT_IDLE_SECONDS,
                 encoder: object | None = None):
        self.config = config
        self.paths = paths or worker_paths(config.database)
        self.encoder_factory = encoder_factory or OnnxEncoder
        self._injected_encoder_factory = encoder_factory is not None or encoder is not None
        self.idle_seconds = idle_seconds
        self._encoder = encoder
        self._encoder_error: str | None = None
        self._encoder_lock = threading.Lock()
        self._matrix_cache: dict[tuple[object, ...], tuple[object, list[object]]] = {}
        self._matrix_lock = threading.Lock()
        self._started = time.perf_counter()
        self._stopping = threading.Event()

    def _lock_fd(self) -> int | None:
        value = os.environ.get("PAUSANIAS_WORKER_LOCK_FD")
        return int(value) if value is not None else None

    def _publish(self, state: str, pid: int) -> None:
        _write_record(self.paths.lock, {
            "database": str(self.config.database), "pid": pid,
            "launch_nonce": os.environ.get("PAUSANIAS_WORKER_NONCE", ""),
            "start_time": self._started, "socket_path": str(self.paths.socket),
            "readiness": state,
        })

    def _load_encoder(self) -> tuple[object | None, float]:
        if self._encoder_error is not None:
            return None, 0.000001
        if self._encoder is not None:
            return self._encoder, 0.000001
        with self._encoder_lock:
            if self._encoder_error is not None:
                return None, 0.000001
            if self._encoder is not None:
                return self._encoder, 0.000001
            started = time.perf_counter()
            reason = None if self._injected_encoder_factory else semantic_backend_reason(self.config)
            if reason is not None:
                self._encoder_error = reason
                return None, _positive_ms(started)
            try:
                bundle = self.config.bundle_dir if self._injected_encoder_factory else _resolve_active_bundle(self.config.bundle_dir)
                self._encoder = self.encoder_factory(bundle)
            except (BundleError, OSError, SemanticError, RuntimeError, ValueError, TypeError) as exc:
                self._encoder_error = str(exc)
                return None, _positive_ms(started)
            return self._encoder, _positive_ms(started)

    @staticmethod
    def _candidate_payload(candidate: Candidate) -> dict[str, object]:
        return {
            "section_id": candidate.section_id, "canonical_path": candidate.canonical_path,
            "heading": candidate.heading, "heading_path": list(candidate.heading_path),
            "line_start": candidate.line_start, "line_end": candidate.line_end,
            "root_id": candidate.root_id, "project_scope": candidate.project_scope,
            "text": candidate.text, "content_hash": candidate.content_hash,
            "note_type": candidate.note_type, "updated_date": candidate.updated_date,
            "created_date": candidate.created_date, "score": candidate.score,
            "reason": candidate.reason,
        }

    def _query(self, request: dict[str, object]) -> dict[str, object]:
        started = time.perf_counter()
        query = request.get("query")
        if not isinstance(query, str):
            raise WorkerError("worker query must be a string")
        project = request.get("project")
        root_id = request.get("root_id")
        all_projects = request.get("all_projects", False)
        limit = request.get("limit", 20)
        if project is not None and not isinstance(project, str):
            raise WorkerError("worker project must be a string or null")
        if root_id is not None and not isinstance(root_id, str):
            raise WorkerError("worker root_id must be a string or null")
        if not isinstance(all_projects, bool) or not isinstance(limit, int):
            raise WorkerError("worker scope values are invalid")
        encoder, model_load_ms = self._load_encoder()
        timings: dict[str, float] = {}
        fallback = encoder is None
        candidates: list[Candidate] = []
        if encoder is not None:
            encode_started = time.perf_counter()
            try:
                vector = encoder_vectors(encoder, [query])[0]
                timings["encode_ms"] = _positive_ms(encode_started)
                with self._matrix_lock:
                    try:
                        candidates = scan_vectors(
                            self.config, unpack_vector(vector, int(MODEL_BUNDLE_MANIFEST["dimension"])),
                            project, root_id, all_projects, limit, timings=timings,
                            matrix_cache=self._matrix_cache,
                        )
                        if not timings.get("semantic_available"):
                            fallback = True
                    except (SemanticError, ValueError, TypeError, RuntimeError, OSError):
                        self._matrix_cache.clear()
                        try:
                            candidates = scan_vectors(
                                self.config, unpack_vector(vector, int(MODEL_BUNDLE_MANIFEST["dimension"])),
                                project, root_id, all_projects, limit, timings=timings,
                                matrix_cache=self._matrix_cache,
                            )
                        except (SemanticError, ValueError, TypeError, RuntimeError, OSError):
                            candidates = []
                            fallback = True
                        else:
                            fallback = not bool(timings.get("semantic_available"))
                    generations = {key[0] for key in self._matrix_cache}
                    if len(generations) > 1:
                        newest = max(generations)
                        self._matrix_cache = {key: value for key, value in self._matrix_cache.items() if key[0] == newest}
            except (IndexError, SemanticError, ValueError, TypeError, RuntimeError, OSError):
                candidates = []
        if fallback:
            fallback_started = time.perf_counter()
            candidates = search(self.config, query, project, root_id, all_projects, limit)
            timings["fallback_ms"] = _positive_ms(fallback_started)
        timings.setdefault("encode_ms", 0.000001)
        timings.setdefault("matrix_load_ms", 0.000001)
        timings.setdefault("scan_ms", 0.000001)
        return {
            "items": [self._candidate_payload(candidate) for candidate in candidates],
            "metrics": {
                "worker_startup_ms": 0.000001, "model_load_ms": model_load_ms,
                "matrix_load_ms": timings["matrix_load_ms"], "encode_ms": timings["encode_ms"],
                "scan_ms": timings["scan_ms"], "fallback_ms": timings.get("fallback_ms", 0.000001),
                "fts_search_ms": timings.get("fallback_ms", 0.000001) if fallback else 0.000001,
                "hybrid_overhead_ms": 0.000001,
                "hook_total_ms": _positive_ms(started), "fallback": fallback,
                "cache_state": "warm" if self._matrix_cache else "cold",
                "disabled_reason": self._encoder_error,
            },
        }

    def _serve_connection(self, connection: socket.socket) -> None:
        try:
            connection.settimeout(ADAPTER_DEADLINE_MS / 1000.0)
            data = b""
            while not data.endswith(b"\n") and len(data) < 1_000_000:
                chunk = connection.recv(65536)
                if not chunk:
                    break
                data += chunk
            response = self._query(json.loads(data.decode("utf-8")))
        except (OSError, ValueError, TypeError, json.JSONDecodeError, WorkerError) as exc:
            response = {"error": str(exc)}
        try:
            connection.sendall(json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n")
        except OSError:
            pass
        finally:
            connection.close()

    def serve(self) -> None:
        self.paths.socket.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.paths.socket.unlink()
        except FileNotFoundError:
            pass
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.paths.socket))
        os.chmod(self.paths.socket, 0o600)
        server.listen(32)
        self._publish("ready", os.getpid())
        lock_fd = self._lock_fd()
        if lock_fd is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        server.settimeout(0.25)
        last_request = time.monotonic()
        try:
            while not self._stopping.is_set():
                if time.monotonic() - last_request >= self.idle_seconds:
                    break
                try:
                    connection, _ = server.accept()
                except TimeoutError:
                    continue
                last_request = time.monotonic()
                threading.Thread(target=self._serve_connection, args=(connection,), daemon=True).start()
        finally:
            self._stopping.set()
            server.close()
            try:
                self.paths.socket.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass


def _acquire_lock(path: Path) -> int | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def launch_worker(config_path: Path, paths: WorkerPaths, lock_fd: int,
                  database: Path | None = None) -> subprocess.Popen:
    nonce = uuid.uuid4().hex
    _write_record(paths.lock, {
        "database": str(database or paths.lock.parent), "pid": 0, "launch_nonce": nonce,
        "start_time": time.time(), "socket_path": str(paths.socket), "readiness": "starting",
    })
    environment = os.environ.copy()
    environment["PAUSANIAS_WORKER_LOCK_FD"] = str(lock_fd)
    environment["PAUSANIAS_WORKER_NONCE"] = nonce
    process = subprocess.Popen(
        [sys.executable, "-m", "pausanias.worker", "--serve", "--config", str(config_path),
         "--socket", str(paths.socket), "--lock", str(paths.lock)],
        env=environment, pass_fds=(lock_fd,), close_fds=True,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    record = _read_record(paths.lock)
    record["pid"] = process.pid
    _write_record(paths.lock, record)
    return process


def ensure_worker(config: Config, config_path: Path, deadline: float,
                  paths: WorkerPaths | None = None) -> tuple[WorkerPaths, float] | None:
    worker_paths_value = paths or worker_paths(config.database)
    started = time.perf_counter()
    if socket_ready(worker_paths_value.socket):
        return worker_paths_value, _positive_ms(started)
    lock_fd = _acquire_lock(worker_paths_value.lock)
    if lock_fd is not None:
        record = _read_record(worker_paths_value.lock)
        pid = record.get("pid")
        if isinstance(pid, int) and pid_alive(pid) and not socket_ready(worker_paths_value.socket):
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        else:
            try:
                launch_worker(config_path, worker_paths_value, lock_fd, config.database)
            except OSError:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
                return None
            os.close(lock_fd)
    while time.perf_counter() < deadline:
        if socket_ready(worker_paths_value.socket):
            return worker_paths_value, _positive_ms(started)
        time.sleep(min(0.005, max(deadline - time.perf_counter(), 0.0)))
    return None


class WorkerClient:
    def __init__(self, paths: WorkerPaths):
        self.paths = paths

    def query(self, request: dict[str, object], deadline: float) -> dict[str, object]:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("worker request exceeded adapter deadline")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(remaining)
            connection.connect(str(self.paths.socket))
            connection.sendall(json.dumps(request).encode("utf-8") + b"\n")
            data = b""
            while not data.endswith(b"\n") and len(data) < 2_000_000:
                chunk = connection.recv(65536)
                if not chunk:
                    break
                data += chunk
        response = json.loads(data.decode("utf-8"))
        if not isinstance(response, dict):
            raise WorkerError("worker response must be an object")
        if "error" in response:
            raise WorkerError(str(response["error"]))
        return response


def main(argv: list[str] | None = None) -> int:
    arguments = list(argv if argv is not None else sys.argv[1:])
    if "--serve" not in arguments:
        return 2
    config_path = Path(arguments[arguments.index("--config") + 1])
    socket_path = Path(arguments[arguments.index("--socket") + 1])
    lock_path = Path(arguments[arguments.index("--lock") + 1])
    PersistentWorker(load_config(config_path), WorkerPaths(socket_path, lock_path)).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
