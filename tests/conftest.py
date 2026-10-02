from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
import sys
import time

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pausanias.config import load_config
from pausanias.worker import stop_worker


def _worker_commands() -> list[tuple[int, Path]]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,command="], capture_output=True, text=True, check=True,
    )
    workers = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or "-m pausanias.worker" not in fields[1]:
            continue
        try:
            arguments = shlex.split(fields[1])
            config = Path(arguments[arguments.index("--config") + 1])
            workers.append((int(fields[0]), config))
        except (ValueError, IndexError):
            continue
    return workers


def _stop_workers_under(base: Path) -> list[tuple[int, Path]]:
    workers = [(pid, config) for pid, config in _worker_commands()
               if config.is_relative_to(base)]
    for _, config in workers:
        try:
            stop_worker(load_config(config).database)
        except (OSError, ValueError, KeyError):
            pass
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        remaining = [(pid, config) for pid, config in _worker_commands()
                     if config.is_relative_to(base)]
        if not remaining:
            return []
        time.sleep(0.02)
    return remaining


@pytest.fixture(autouse=True)
def stop_workers_after_test(request: pytest.FixtureRequest):
    tmp_path = request.getfixturevalue("tmp_path")
    yield
    remaining = _stop_workers_under(tmp_path)
    assert not remaining, f"test workers leaked: {remaining}"


@pytest.fixture(scope="session", autouse=True)
def stop_workers_after_session(request: pytest.FixtureRequest):
    yield
    base = request.config._tmp_path_factory.getbasetemp()
    remaining = _stop_workers_under(base)
    assert not remaining, f"test workers leaked at session end: {remaining}"
