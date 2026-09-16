"""Performance benchmark for the local indexing and retrieval path."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
import time

from .config import Config, load_config
from .core import index, search
from .splitter import split_markdown
from .synth import generate_corpus


IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9]*-\d+")
WORD = re.compile(r"[A-Za-z][A-Za-z0-9]{2,}")
STOP_WORDS = {
    "this", "that", "with", "from", "into", "for", "and", "the", "are",
    "use", "local", "note", "project", "source", "retrieval", "context",
}


@dataclass(frozen=True)
class QuerySpec:
    query: str
    mode: str = "single"
    project: str | None = None
    root_id: str | None = None
    all_projects: bool = False


@dataclass(frozen=True)
class Document:
    path: Path
    project: str
    root_id: str
    indexed_texts: tuple[str, ...]
    global_note: bool


def _markdown_files(directory: Path) -> list[Path]:
    paths: list[Path] = []
    for current, dirnames, filenames in os.walk(directory, followlinks=False):
        dirnames.sort()
        for filename in sorted(filenames):
            path = Path(current) / filename
            if path.suffix.lower() in {".md", ".markdown"} and path.is_file():
                paths.append(path)
    return paths


def _write_config(
    location: Path,
    source: Path,
    database: Path,
    synthetic: bool,
) -> tuple[Path, int]:
    source = source.resolve()
    if synthetic:
        candidates = [path for path in sorted(source.iterdir()) if path.is_dir()]
        roots = [path for path in candidates if _markdown_files(path)]
        if not roots:
            roots = [source]
    else:
        roots = [source]

    root_entries = []
    for number, root in enumerate(roots):
        root_entries.append((f"root-{number}", root.name if synthetic else "real", root))
    global_notes = [path for path in _markdown_files(source) if path.name.startswith("global-")]
    lines = [f"database = {json.dumps(str(database))}"]
    if global_notes:
        notes = ", ".join(json.dumps(str(path)) for path in global_notes)
        lines += [f"global_notes = [{notes}]"]
    for root_id, project, root in root_entries:
        lines += [
            "",
            "[[roots]]",
            f"id = {json.dumps(root_id)}",
            f"project = {json.dumps(project)}",
            f"path = {json.dumps(str(root))}",
        ]
    config_path = location / "config.toml"
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path, len(_markdown_files(source))


def _root_for(config: Config, path: Path) -> tuple[str, str]:
    root = config.root_for(path.resolve())
    if root is None:
        raise ValueError(f"benchmark file is outside configured roots: {path}")
    return root.id, root.project


def _documents(config: Config, source: Path) -> list[Document]:
    documents: list[Document] = []
    for path in _markdown_files(source):
        root_id, project = _root_for(config, path)
        text = path.read_text(encoding="utf-8")
        split = split_markdown(str(path), text, config.section_bytes)
        documents.append(
            Document(
                path,
                project,
                root_id,
                tuple(f"{section.heading or ''}\n{section.text}" for section in split.sections),
                path.name.startswith("global-"),
            )
        )
    if not documents:
        raise ValueError(f"no Markdown files found in {source}")
    return documents


def _tokens(text: str) -> list[str]:
    identifiers = IDENTIFIER.findall(text)
    words = [word.lower() for word in WORD.findall(text) if word.lower() not in STOP_WORDS]
    result: list[str] = []
    seen: set[str] = set()
    for token in identifiers + words:
        if token.lower() not in seen:
            result.append(token)
            seen.add(token.lower())
    return result


def _phrase_tokens(text: str) -> tuple[str, str] | None:
    matches = list(WORD.finditer(text))
    for left, right in zip(matches, matches[1:]):
        left_word = left.group().lower()
        right_word = right.group().lower()
        if left_word not in STOP_WORDS and right_word not in STOP_WORDS:
            return left_word, right_word
    return None


def _indexed_tokens(document: Document) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for text in document.indexed_texts:
        for token in _tokens(text):
            if token.lower() not in seen:
                tokens.append(token)
                seen.add(token.lower())
    return tokens


def _phrase_for(document: Document) -> tuple[str, str] | None:
    for text in document.indexed_texts:
        phrase = _phrase_tokens(text)
        if phrase is not None:
            return phrase
    return None


def _multi_for(document: Document) -> list[str]:
    for text in document.indexed_texts:
        tokens = _tokens(text)
        if len(tokens) >= 2:
            return tokens[:3]
    return []


def _make_queries(documents: list[Document], count: int, seed: int) -> list[QuerySpec]:
    if count < 1:
        raise ValueError("query count must be positive")
    rng = random.Random(seed + 1)
    queries: list[QuerySpec] = []
    global_documents = [document for document in documents if document.global_note]
    modes = ("single", "phrase", "identifier", "multi", "scoped", "global")
    for number in range(count):
        if number % 10 == 9:
            document = documents[rng.randrange(len(documents))]
            queries.append(QuerySpec(
                f"pausanias-benchmark-miss-{seed}-{number}",
                "miss",
                project=document.project,
            ))
            continue
        mode = modes[number % len(modes)]
        document = documents[rng.randrange(len(documents))]
        tokens = _indexed_tokens(document) or [document.path.stem]
        words = [token for token in tokens if not IDENTIFIER.fullmatch(token)] or tokens
        identifier = next((token for token in tokens if IDENTIFIER.fullmatch(token)), tokens[0])
        phrase = _phrase_for(document)
        multi = _multi_for(document)
        if mode == "phrase" and phrase is not None:
            query = f'"{phrase[0]} {phrase[1]}"'
        elif mode == "identifier":
            query = identifier
        elif mode == "multi" and multi:
            query = " ".join(multi)
        elif mode == "global" and global_documents:
            global_document = global_documents[rng.randrange(len(global_documents))]
            global_tokens = _indexed_tokens(global_document)
            query = global_tokens[0] if global_tokens else global_document.path.stem
            queries.append(QuerySpec(query, "global", all_projects=True))
            continue
        else:
            query = words[0]
        if mode == "scoped":
            queries.append(QuerySpec(query, mode, project=document.project, root_id=document.root_id))
        else:
            queries.append(QuerySpec(query, mode, project=document.project))
    return queries


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[position]


def _positive_milliseconds(seconds: float) -> float:
    return max(seconds * 1000.0, 0.000001)


def _check(value: float, target: float | None) -> dict[str, float | bool | None]:
    if target is None:
        return {"passed": None, "headroom_multiple": None, "target_ms": None}
    return {
        "passed": value <= target,
        "headroom_multiple": target / value,
        "target_ms": target,
    }


def run_benchmark(
    *,
    file_count: int = 2000,
    query_count: int = 200,
    seed: int = 0,
    corpus_dir: str | Path | None = None,
    real_dir: str | Path | None = None,
) -> dict:
    """Run the benchmark and return a JSON-serializable report."""

    if corpus_dir is not None and real_dir is not None:
        raise ValueError("--corpus-dir and --real cannot be used together")
    if file_count < 1:
        raise ValueError("file count must be positive")
    if query_count < 1:
        raise ValueError("query count must be positive")

    with tempfile.TemporaryDirectory(prefix="pausanias-bench-") as temporary:
        workspace = Path(temporary)
        synthetic = real_dir is None
        generated = False
        if real_dir is not None:
            source = Path(real_dir).expanduser().resolve(strict=True)
            if not source.is_dir():
                raise ValueError(f"real benchmark path is not a directory: {real_dir}")
        elif corpus_dir is not None:
            source = Path(corpus_dir).expanduser()
            if not source.exists() or not _markdown_files(source):
                generate_corpus(source, file_count=file_count, seed=seed)
                generated = True
        else:
            source = workspace / "corpus"
            generate_corpus(source, file_count=file_count, seed=seed)
            generated = True

        config_path, actual_file_count = _write_config(workspace, source, workspace / "index.sqlite3", synthetic)
        config = load_config(config_path)
        documents = _documents(config, source)
        queries = _make_queries(documents, query_count, seed)

        started = time.perf_counter()
        index(config, rebuild=True)
        index_build_ms = _positive_milliseconds(time.perf_counter() - started)

        started = time.perf_counter()
        index(config)
        refresh_ms = _positive_milliseconds(time.perf_counter() - started)

        cold_query = queries[0]
        started = time.perf_counter()
        search(config, cold_query.query, cold_query.project, cold_query.root_id, cold_query.all_projects, limit=20)
        cold_search_ms = _positive_milliseconds(time.perf_counter() - started)

        warm_timings: list[float] = []
        hit_counts: dict[str, list[int]] = {}
        mode_timings: dict[str, list[float]] = {}
        for query in queries:
            started = time.perf_counter()
            results = search(config, query.query, query.project, query.root_id, query.all_projects, limit=20)
            elapsed_ms = _positive_milliseconds(time.perf_counter() - started)
            warm_timings.append(elapsed_ms)
            hit_counts.setdefault(query.mode, []).append(len(results))
            mode_timings.setdefault(query.mode, []).append(elapsed_ms)
        warm_p50_ms = _percentile(warm_timings, 0.50)
        warm_p95_ms = _percentile(warm_timings, 0.95)
        warm_max_ms = max(warm_timings)

        query_modes = {}
        for mode, timings in mode_timings.items():
            counts = hit_counts[mode]
            hit_p50 = _percentile(counts, 0.50)
            if mode != "miss" and hit_p50 < 1:
                raise ValueError(f"benchmark query mode {mode} has zero-hit median")
            query_modes[mode] = {
                "count": len(timings),
                "warm_search_p50_ms": _percentile(timings, 0.50),
                "warm_search_p95_ms": _percentile(timings, 0.95),
                "hit_count_p50": hit_p50,
                "hit_count_p95": _percentile(counts, 0.95),
                "hit_count_min": min(counts),
                "hit_count_max": max(counts),
            }

        metrics = {
            "index_build_ms": index_build_ms,
            "incremental_refresh_ms": refresh_ms,
            "cold_search_ms": cold_search_ms,
            "warm_search_p50_ms": warm_p50_ms,
            "warm_search_p95_ms": warm_p95_ms,
            "warm_search_max_ms": warm_max_ms,
        }
        targets = {
            "index_build_ms": None,
            "incremental_refresh_ms": None,
            "cold_search_ms": 750.0,
            "warm_search_p50_ms": None,
            "warm_search_p95_ms": 300.0,
            "warm_search_max_ms": 750.0,
        }
        checks = {name: _check(value, targets[name]) for name, value in metrics.items()}
        default_synthetic = (
            corpus_dir is None
            and real_dir is None
            and file_count == 2000
            and actual_file_count == 2000
            and query_count == 200
        )
        gate_passed = checks["warm_search_p95_ms"]["passed"]
        report = {
            "source": {
                "kind": "synthetic" if synthetic else "real",
                "directory": str(source),
                "generated": generated,
                "file_count": actual_file_count,
                "seed": seed if synthetic else None,
            },
            "queries": {"count": query_count, "modes": query_modes},
            "metrics": metrics,
            "targets": targets,
            "checks": checks,
            "gate": {
                "passed": gate_passed,
                "enforced": default_synthetic,
                "reason": "warm search p95 must be at most 300 ms",
            },
        }
        return report


def format_report(report: dict) -> str:
    lines = [
        f"source: {report['source']['kind']} ({report['source']['file_count']} files)",
        f"queries: {report['queries']['count']}",
        "",
        "metric                         value       target      result  headroom",
        "-----------------------------  ----------  ----------  ------  --------",
    ]
    for name, value in report["metrics"].items():
        check = report["checks"][name]
        target = "-" if check["target_ms"] is None else f"{check['target_ms']:.1f} ms"
        result = "n/a" if check["passed"] is None else ("pass" if check["passed"] else "fail")
        headroom = "-" if check["headroom_multiple"] is None else f"{check['headroom_multiple']:.2f}x"
        lines.append(f"{name:29}  {value:8.2f} ms  {target:10}  {result:6}  {headroom:8}")
    lines += [
        "",
        "query mode                     count   p50 ms   p95 ms   hits p50  hits p95  min  max",
        "-----------------------------  ------  -------  -------  --------  --------  ---  ---",
    ]
    for mode, stats in report["queries"]["modes"].items():
        lines.append(
            f"{mode:29}  {stats['count']:6}  {stats['warm_search_p50_ms']:7.2f}  "
            f"{stats['warm_search_p95_ms']:7.2f}  {stats['hit_count_p50']:8}  "
            f"{stats['hit_count_p95']:8}  {stats['hit_count_min']:3}  {stats['hit_count_max']:3}"
        )
    gate = report["gate"]
    status = "pass" if gate["passed"] else "fail"
    enforcement = " (enforced)" if gate["enforced"] else ""
    lines += ["", f"warm p95 gate: {status}{enforcement}"]
    return "\n".join(lines)
