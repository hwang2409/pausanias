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
from .core import (
    BALANCED_ADMISSION,
    RELATIVE_SEMANTIC_SCORE_FLOOR,
    SEMANTIC_SCORE_FLOOR,
    TICKET_ID_CROSS_REFERENCE_FILTER,
    SYNONYM_EXPANSION,
    semantic_index_state,
    search,
    semantic_search,
)
from .model_bundle import BundleError, MODEL_BUNDLE_MANIFEST, _resolve_active_bundle
from .synonyms import SynonymTableError, load_synonym_table
from .vectors import Candidate, OnnxEncoder, SemanticError, semantic_backend_reason


ADAPTER_DEADLINE_MS = 750.0
DEFAULT_IDLE_SECONDS = 30 * 60
SOCKET_TIMEOUT_SECONDS = 0.05
CANCELLATION_TOMBSTONE_SECONDS = 5.0


class WorkerError(RuntimeError):
    """Raised when a worker cannot serve a request."""


class RequestCancelled(WorkerError):
    """Raised when the client cancels an in-flight request."""


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
    disabled_reason: str | None = None
    failure_reason: str | None = None
    semantic_state: str = "unknown"
    status: str = "ok"
    retrieval_mode: str = "lexical"


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


def _write_record_fd(fd: int, record: dict[str, object]) -> None:
    """Update the lock record without replacing its flocked inode."""
    payload = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    view = memoryview(payload)
    while view:
        view = view[os.write(fd, view):]
    os.fsync(fd)


def _config_fingerprint(config: Config) -> str:
    try:
        table = load_synonym_table(config.synonym_table)
        synonym = {
            "path": str(config.synonym_table) if config.synonym_table is not None else None,
            "fingerprint": table.fingerprint,
        }
    except SynonymTableError as exc:
        synonym = {
            "path": str(config.synonym_table) if config.synonym_table is not None else None,
            "error": str(exc),
        }
    payload = {
        "roots": [
            {"id": root.id, "path": str(root.path), "project": root.project, "excludes": root.excludes}
            for root in config.roots
        ],
        "global_notes": sorted(str(path) for path in config.global_notes),
        "private_paths": config.private_paths,
        "database": str(config.database),
        "section_bytes": config.section_bytes,
        "semantic_bundle": str(config.semantic_bundle),
        "synonym_table": synonym,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pid_alive(pid: int) -> bool:
    if pid < 1:
        return False
    try:
        waited, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        waited = 0
    if waited == pid:
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
        wait_deadline = time.monotonic() + 1.0
        while pid_alive(pid) and time.monotonic() < wait_deadline:
            time.sleep(0.005)
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
        self._encoder_failure_token: tuple[object, ...] | None = None
        self._encoder_lock = threading.Lock()
        self._matrix_cache: dict[tuple[object, ...], tuple[object, list[object]]] = {}
        self._matrix_lock = threading.Lock()
        self._requests: dict[str, threading.Event] = {}
        self._cancellation_tombstones: dict[str, float] = {}
        self._requests_lock = threading.Lock()
        self._started = time.perf_counter()
        self._stopping = threading.Event()

    def _lock_fd(self) -> int | None:
        value = os.environ.get("PAUSANIAS_WORKER_LOCK_FD")
        return int(value) if value is not None else None

    def _publish(self, state: str, pid: int) -> None:
        record = {
            "database": str(self.config.database), "pid": pid,
            "launch_nonce": os.environ.get("PAUSANIAS_WORKER_NONCE", ""),
            "start_time": self._started, "socket_path": str(self.paths.socket),
            "readiness": state, "config_fingerprint": _config_fingerprint(self.config),
        }
        lock_fd = self._lock_fd()
        if lock_fd is None:
            _write_record(self.paths.lock, record)
        else:
            _write_record_fd(lock_fd, record)

    def _backend_failure_token(self) -> tuple[object, ...]:
        """Track state that can repair a cached model-load failure."""
        try:
            reason = semantic_backend_reason(self.config)
        except (OSError, RuntimeError, TypeError, ValueError):
            reason = "MODEL_STATE_UNREADABLE"
        paths = [self.config.bundle_dir, self.config.bundle_dir / "manifest.json",
                 self.config.bundle_dir / "onnx/model.onnx"]
        stats: list[object] = []
        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                stats.append((str(path), None))
            else:
                stats.append((str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns))
        return (reason, *stats)

    def _load_encoder(self) -> tuple[object | None, float]:
        if self._encoder_error is not None:
            if (not self._injected_encoder_factory
                    and self._encoder_failure_token == self._backend_failure_token()):
                return None, 0.000001
            self._encoder_error = None
            self._encoder_failure_token = None
        if self._encoder is not None:
            return self._encoder, 0.000001
        with self._encoder_lock:
            if self._encoder_error is not None:
                if (not self._injected_encoder_factory
                        and self._encoder_failure_token == self._backend_failure_token()):
                    return None, 0.000001
                self._encoder_error = None
                self._encoder_failure_token = None
            if self._encoder is not None:
                return self._encoder, 0.000001
            started = time.perf_counter()
            reason = None if self._injected_encoder_factory else semantic_backend_reason(self.config)
            if reason is not None:
                self._encoder_error = reason
                self._encoder_failure_token = self._backend_failure_token()
                return None, _positive_ms(started)
            try:
                bundle = self.config.bundle_dir if self._injected_encoder_factory else _resolve_active_bundle(self.config.bundle_dir)
                self._encoder = self.encoder_factory(bundle)
            except (BundleError, OSError, SemanticError, RuntimeError, ValueError, TypeError) as exc:
                self._encoder_error = str(exc)
                self._encoder_failure_token = self._backend_failure_token()
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
            "reason": candidate.reason, "lexical_rank": candidate.lexical_rank,
            "vector_rank": candidate.vector_rank, "lexical_score": candidate.lexical_score,
            "vector_score": candidate.vector_score, "fused_score": candidate.fused_score,
            "lane": candidate.lane, "guard_reason": candidate.guard_reason,
        }

    @staticmethod
    def _check_cancelled(cancelled: threading.Event | None) -> None:
        if cancelled is not None and cancelled.is_set():
            raise RequestCancelled("worker request cancelled")

    def _query(self, request: dict[str, object], cancelled: threading.Event | None = None) -> dict[str, object]:
        started = time.perf_counter()
        query = request.get("query")
        if not isinstance(query, str):
            raise WorkerError("worker query must be a string")
        project = request.get("project")
        root_id = request.get("root_id")
        all_projects = request.get("all_projects", False)
        limit = request.get("limit", 20)
        semantic_score_floor = request.get("semantic_score_floor", SEMANTIC_SCORE_FLOOR)
        relative_semantic_score_floor = request.get(
            "relative_semantic_score_floor", RELATIVE_SEMANTIC_SCORE_FLOOR,
        )
        ticket_id_cross_reference_filter = request.get(
            "ticket_id_cross_reference_filter", TICKET_ID_CROSS_REFERENCE_FILTER,
        )
        balanced_admission = request.get("balanced_admission", BALANCED_ADMISSION)
        synonym_expansion = request.get("synonym_expansion", SYNONYM_EXPANSION)
        if project is not None and not isinstance(project, str):
            raise WorkerError("worker project must be a string or null")
        if root_id is not None and not isinstance(root_id, str):
            raise WorkerError("worker root_id must be a string or null")
        if not isinstance(all_projects, bool) or not isinstance(limit, int):
            raise WorkerError("worker scope values are invalid")
        if (semantic_score_floor is not None
                and (not isinstance(semantic_score_floor, (int, float))
                     or isinstance(semantic_score_floor, bool)
                     or not 0.0 <= float(semantic_score_floor) <= 1.0)):
            raise WorkerError("worker semantic score floor is invalid")
        if (relative_semantic_score_floor is not None
                and (not isinstance(relative_semantic_score_floor, (int, float))
                     or isinstance(relative_semantic_score_floor, bool)
                     or not 0.0 <= float(relative_semantic_score_floor) <= 1.0)):
            raise WorkerError("worker relative semantic score floor is invalid")
        if not isinstance(ticket_id_cross_reference_filter, bool):
            raise WorkerError("worker ticket ID filter is invalid")
        if not isinstance(balanced_admission, bool):
            raise WorkerError("worker balanced admission policy is invalid")
        if not isinstance(synonym_expansion, bool):
            raise WorkerError("worker synonym expansion policy is invalid")
        self._check_cancelled(cancelled)
        semantic_state, semantic_reason = semantic_index_state(self.config)
        encoder, model_load_ms = self._load_encoder()
        timings: dict[str, float] = {}
        fallback = encoder is None
        semantic_failure = False
        failure_reason: str | None = None
        candidates: list[Candidate] = []
        if encoder is not None:
            while not self._matrix_lock.acquire(timeout=0.01):
                self._check_cancelled(cancelled)
            try:
                try:
                    candidates = semantic_search(
                        self.config, query, project, root_id, all_projects, limit,
                        encoder=encoder, matrix_cache=self._matrix_cache, timings=timings,
                        semantic_score_floor=(float(semantic_score_floor)
                                              if semantic_score_floor is not None else None),
                        relative_semantic_score_floor=(float(relative_semantic_score_floor)
                                                      if relative_semantic_score_floor is not None else None),
                        ticket_id_cross_reference_filter=ticket_id_cross_reference_filter,
                        balanced_admission=balanced_admission,
                        synonym_expansion=synonym_expansion,
                    )
                    self._check_cancelled(cancelled)
                    if not timings.get("semantic_available"):
                        fallback = True
                        semantic_failure = True
                        failure_reason = str(
                            timings.get("semantic_reason") or semantic_reason or "semantic backend unavailable"
                        )
                        semantic_state = str(timings.get("semantic_state", semantic_state))
                    generations = {key[0] for key in self._matrix_cache}
                    if len(generations) > 1:
                        newest = max(generations)
                        self._matrix_cache = {
                            key: value for key, value in self._matrix_cache.items() if key[0] == newest
                        }
                except (IndexError, SemanticError, ValueError, TypeError, RuntimeError, OSError) as exc:
                    self._matrix_cache.clear()
                    candidates = []
                    fallback = True
                    semantic_failure = True
                    failure_reason = str(exc) or "semantic query failed"
            except RequestCancelled:
                raise
            finally:
                self._matrix_lock.release()
        if fallback:
            self._check_cancelled(cancelled)
            fallback_started = time.perf_counter()
            candidates = search(
                self.config, query, project, root_id, all_projects, limit,
                semantic=False, synonym_expansion=synonym_expansion,
            )
            self._check_cancelled(cancelled)
            timings["fallback_ms"] = _positive_ms(fallback_started)
        if encoder is None:
            failure_reason = self._encoder_error or semantic_reason
        status = "semantic_failure" if semantic_failure else ("semantic_disabled" if encoder is None else "ok")
        timings.setdefault("encode_ms", 0.000001)
        timings.setdefault("matrix_load_ms", 0.000001)
        timings.setdefault("scan_ms", 0.000001)
        return {
            "status": "semantic_failure" if semantic_failure else ("semantic_disabled" if encoder is None else "ok"),
            "failure_reason": failure_reason,
            "items": [self._candidate_payload(candidate) for candidate in candidates],
            "metrics": {
                "worker_startup_ms": 0.000001, "model_load_ms": model_load_ms,
                "matrix_load_ms": timings["matrix_load_ms"], "encode_ms": timings["encode_ms"],
                "scan_ms": timings["scan_ms"], "fallback_ms": timings.get("fallback_ms", 0.000001),
                "fts_search_ms": timings.get("fts_search_ms", 0.000001),
                "hybrid_overhead_ms": timings.get("hybrid_overhead_ms", 0.000001),
                "hook_total_ms": _positive_ms(started), "fallback": fallback,
                "cache_state": "disabled" if self._encoder_error else ("warm" if self._matrix_cache else "cold"),
                "disabled_reason": self._encoder_error,
                "failure_reason": failure_reason,
                "semantic_state": semantic_state,
                "status": status,
                "retrieval_mode": "lexical" if fallback else "fused",
            },
        }

    def _serve_connection(self, connection: socket.socket) -> None:
        request_id: str | None = None
        cancelled: threading.Event | None = None
        try:
            connection.settimeout(ADAPTER_DEADLINE_MS / 1000.0)
            data = b""
            while not data.endswith(b"\n") and len(data) < 1_000_000:
                chunk = connection.recv(65536)
                if not chunk:
                    break
                data += chunk
            request = json.loads(data.decode("utf-8"))
            if not isinstance(request, dict):
                raise WorkerError("worker request must be an object")
            if request.get("command") == "cancel":
                target = request.get("request_id")
                cancelled_request = self._cancel_request(target)
                response = {"cancelled": cancelled_request}
            else:
                request_id = request.get("request_id") if isinstance(request.get("request_id"), str) else uuid.uuid4().hex
                cancelled = self._register_request(request_id)
                response = self._query(request, cancelled)
        except RequestCancelled:
            response = {"cancelled": True}
        except (OSError, ValueError, TypeError, json.JSONDecodeError, WorkerError) as exc:
            response = {"error": str(exc)}
        finally:
            if request_id is not None:
                with self._requests_lock:
                    self._requests.pop(request_id, None)
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
                    self._prune_cancellation_tombstones()
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

    def _cancel_request(self, request_id: object) -> bool:
        if not isinstance(request_id, str):
            return False
        with self._requests_lock:
            self._prune_cancellation_tombstones_locked(time.monotonic())
            event = self._requests.get(request_id)
            if event is None:
                self._cancellation_tombstones[request_id] = time.monotonic()
                return False
            event.set()
        return True

    def _register_request(self, request_id: str) -> threading.Event:
        event = threading.Event()
        with self._requests_lock:
            now = time.monotonic()
            self._prune_cancellation_tombstones_locked(now)
            if request_id in self._cancellation_tombstones:
                event.set()
                del self._cancellation_tombstones[request_id]
            self._requests[request_id] = event
        return event

    def _prune_cancellation_tombstones(self) -> None:
        with self._requests_lock:
            self._prune_cancellation_tombstones_locked(time.monotonic())

    def _prune_cancellation_tombstones_locked(self, now: float) -> None:
        expired = [
            request_id
            for request_id, cancelled_at in self._cancellation_tombstones.items()
            if now - cancelled_at >= CANCELLATION_TOMBSTONE_SECONDS
        ]
        for request_id in expired:
            del self._cancellation_tombstones[request_id]


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
                  database: Path | None = None,
                  config_fingerprint_value: str | None = None) -> subprocess.Popen:
    nonce = uuid.uuid4().hex
    initial_record = {
        "database": str(database or paths.lock.parent), "pid": 0, "launch_nonce": nonce,
        "start_time": time.time(), "socket_path": str(paths.socket), "readiness": "starting",
        "config_fingerprint": config_fingerprint_value,
    }
    _write_record_fd(lock_fd, initial_record)
    environment = os.environ.copy()
    environment["PAUSANIAS_WORKER_LOCK_FD"] = str(lock_fd)
    environment["PAUSANIAS_WORKER_NONCE"] = nonce
    process = subprocess.Popen(
        [sys.executable, "-m", "pausanias.worker", "--serve", "--config", str(config_path),
         "--socket", str(paths.socket), "--lock", str(paths.lock)],
        env=environment, pass_fds=(lock_fd,), close_fds=True,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    record = _read_record(paths.lock) or initial_record
    record["pid"] = process.pid
    _write_record_fd(lock_fd, record)
    return process


def ensure_worker(config: Config, config_path: Path, deadline: float,
                  paths: WorkerPaths | None = None) -> tuple[WorkerPaths, float] | None:
    worker_paths_value = paths or worker_paths(config.database)
    started = time.perf_counter()
    expected_fingerprint = _config_fingerprint(config)
    if socket_ready(worker_paths_value.socket):
        record = _read_record(worker_paths_value.lock)
        if record.get("config_fingerprint") == expected_fingerprint:
            return worker_paths_value, _positive_ms(started)
        stop_worker(config.database, worker_paths_value)
    lock_fd = _acquire_lock(worker_paths_value.lock)
    if lock_fd is not None:
        record = _read_record(worker_paths_value.lock)
        pid = record.get("pid")
        if isinstance(pid, int) and pid_alive(pid) and not socket_ready(worker_paths_value.socket):
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        else:
            try:
                launch_worker(
                    config_path, worker_paths_value, lock_fd, config.database, expected_fingerprint,
                )
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
        request = {**request, "request_id": uuid.uuid4().hex}
        request_id = request["request_id"]
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("worker request exceeded adapter deadline")
        try:
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
        except (TimeoutError, socket.timeout):
            self.cancel(str(request_id))
            raise TimeoutError("worker request exceeded adapter deadline")
        response = json.loads(data.decode("utf-8"))
        if not isinstance(response, dict):
            raise WorkerError("worker response must be an object")
        if "error" in response:
            raise WorkerError(str(response["error"]))
        return response

    def cancel(self, request_id: str) -> bool:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(0.005)
                connection.connect(str(self.paths.socket))
                connection.sendall(json.dumps({"command": "cancel", "request_id": request_id}).encode("utf-8") + b"\n")
            return True
        except OSError:
            return False


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
