"""Performance benchmark for the local indexing and retrieval path."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sqlite3
import subprocess
import sys
import tempfile
import time

from .config import Config, load_config
from .core import index, search
from .worker import stop_worker
from .splitter import split_markdown
from .synth import generate_corpus
from .model_bundle import MODEL_BUNDLE_MANIFEST


IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9]*-\d+")
WORD = re.compile(r"[A-Za-z][A-Za-z0-9]{2,}")
CONTENT_WORD = re.compile(r"(?<![A-Za-z0-9-])[A-Za-z][A-Za-z0-9]{2,}(?![A-Za-z0-9-])")
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
class TopicQuery:
    query: str
    hit_documents: int


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
    bundle_dir: str | Path | None = None,
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
    if bundle_dir is not None:
        lines.append(f"semantic_bundle = {json.dumps(str(Path(bundle_dir).expanduser().resolve()))}")
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


def _indexed_tokens(document: Document) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for text in document.indexed_texts:
        for token in _tokens(text):
            if token.lower() not in seen:
                tokens.append(token)
                seen.add(token.lower())
    return tokens


def _common_topic_queries(
    documents: list[Document],
    minimum_documents: int,
) -> tuple[list[TopicQuery], list[TopicQuery]]:
    word_documents: dict[str, set[int]] = {}
    pair_documents: dict[tuple[str, str], set[int]] = {}
    for number, document in enumerate(documents):
        document_words: set[str] = set()
        document_pairs: set[tuple[str, str]] = set()
        for text in document.indexed_texts:
            all_words = [match.group().lower() for match in CONTENT_WORD.finditer(text)]
            content_words = [word for word in all_words if word not in STOP_WORDS]
            document_words.update(content_words)
            document_pairs.update(
                (left, right)
                for left, right in zip(all_words, all_words[1:])
                if left not in STOP_WORDS and right not in STOP_WORDS
            )
        for word in document_words:
            word_documents.setdefault(word, set()).add(number)
        for pair in document_pairs:
            pair_documents.setdefault(pair, set()).add(number)

    words = [
        TopicQuery(word, len(hit_documents))
        for word, hit_documents in word_documents.items()
        if len(hit_documents) >= minimum_documents
    ]
    pairs = [
        TopicQuery(" ".join(pair), len(hit_documents))
        for pair, hit_documents in pair_documents.items()
        if len(hit_documents) >= minimum_documents
    ]
    words.sort(key=lambda candidate: (-candidate.hit_documents, candidate.query))
    pairs.sort(key=lambda candidate: (-candidate.hit_documents, candidate.query))
    return words, pairs


WORKLOAD_MODES = (
    ("multi",) * 12
    + ("single",) * 2
    + ("phrase", "identifier")
    + ("scoped",) * 2
    + ("global", "miss")
)


def _pick_topic_query(
    candidates: list[TopicQuery],
    fallback: str,
    rng: random.Random,
) -> str:
    return rng.choice(candidates).query if candidates else fallback


def _make_queries(documents: list[Document], count: int, seed: int) -> list[QuerySpec]:
    if count < 1:
        raise ValueError("query count must be positive")
    rng = random.Random(seed + 1)
    queries: list[QuerySpec] = []
    global_documents = [document for document in documents if document.global_note]
    minimum_documents = max(3, math.ceil(len(documents) * 0.1))
    topic_words, topic_pairs = _common_topic_queries(documents, minimum_documents)
    documents_by_project: dict[str, list[Document]] = {}
    for document in documents:
        documents_by_project.setdefault(document.project, []).append(document)
    project_topics = {
        project: _common_topic_queries(
            project_documents,
            max(3, math.ceil(len(project_documents) * 0.5)),
        )
        for project, project_documents in documents_by_project.items()
    }
    global_words, global_pairs = _common_topic_queries(
        global_documents,
        max(1, math.ceil(len(global_documents) * 0.5)),
    ) if global_documents else ([], [])
    for number in range(count):
        mode = WORKLOAD_MODES[number % len(WORKLOAD_MODES)]
        if mode == "miss":
            document = documents[rng.randrange(len(documents))]
            queries.append(QuerySpec(
                f"pausanias-benchmark-miss-{seed}-{number}",
                "miss",
                project=document.project,
            ))
            continue
        document = documents[rng.randrange(len(documents))]
        tokens = _indexed_tokens(document) or [document.path.stem]
        words = [token for token in tokens if not IDENTIFIER.fullmatch(token)] or tokens
        identifier = next((token for token in tokens if IDENTIFIER.fullmatch(token)), tokens[0])
        fallback_word = words[0].lower()
        project_words, project_pairs = project_topics.get(document.project, (topic_words, topic_pairs))
        if mode == "single":
            query = _pick_topic_query(project_words, fallback_word, rng)
        elif mode == "phrase":
            phrase = _pick_topic_query(project_pairs, fallback_word, rng)
            query = f'"{phrase}"'
        elif mode == "identifier":
            query = identifier
        elif mode in {"multi", "scoped"}:
            query = _pick_topic_query(project_pairs, fallback_word, rng)
        elif mode == "global" and global_documents:
            query = _pick_topic_query(
                global_pairs or global_words,
                _pick_topic_query(topic_words, fallback_word, rng),
                rng,
            )
            queries.append(QuerySpec(query, "global", all_projects=True))
            continue
        else:
            query = _pick_topic_query(topic_words, fallback_word, rng)
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


def _hit_count_mix(hit_counts: list[int]) -> dict[str, dict[str, float] | int]:
    total = len(hit_counts)
    zero = sum(count == 0 for count in hit_counts)
    one = sum(count == 1 for count in hit_counts)
    two = sum(count == 2 for count in hit_counts)
    three_or_more = sum(count >= 3 for count in hit_counts)
    return {
        "total": total,
        "zero": zero,
        "one": one,
        "two": two,
        "three_or_more": three_or_more,
        "shares": {
            "zero": zero / total,
            "one": one / total,
            "two": two / total,
            "three_or_more": three_or_more / total,
        },
    }


def _validate_hit_count_mix(mix: dict[str, dict[str, float] | int]) -> None:
    shares = mix["shares"]
    if shares["three_or_more"] < 0.60:
        raise ValueError("benchmark workload has fewer than 60% multi-hit queries")
    if shares["zero"] > 0.10:
        raise ValueError("benchmark workload has too many miss queries")
    if shares["one"] > 0.10:
        raise ValueError("benchmark workload has too many single-hit queries")


def run_benchmark(
    *,
    file_count: int = 2000,
    query_count: int = 200,
    seed: int = 0,
    corpus_dir: str | Path | None = None,
    real_dir: str | Path | None = None,
    hook_path: bool = False,
    bundle_dir: str | Path | None = None,
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

        config_path, actual_file_count = _write_config(
            workspace, source, workspace / "index.sqlite3", synthetic, bundle_dir,
        )
        config = load_config(config_path)
        documents = _documents(config, source)
        queries = _make_queries(documents, query_count, seed)

        started = time.perf_counter()
        index(config, rebuild=True)
        index_build_ms = _positive_milliseconds(time.perf_counter() - started)

        started = time.perf_counter()
        index(config)
        refresh_ms = _positive_milliseconds(time.perf_counter() - started)

        first_query = queries[0]
        started = time.perf_counter()
        search(config, first_query.query, first_query.project, first_query.root_id, first_query.all_projects, limit=20)
        first_query_ms = _positive_milliseconds(time.perf_counter() - started)

        warm_timings: list[float] = []
        all_hit_counts: list[int] = []
        hit_counts: dict[str, list[int]] = {}
        mode_timings: dict[str, list[float]] = {}
        for query in queries:
            started = time.perf_counter()
            results = search(config, query.query, query.project, query.root_id, query.all_projects, limit=20)
            elapsed_ms = _positive_milliseconds(time.perf_counter() - started)
            warm_timings.append(elapsed_ms)
            result_count = len(results)
            all_hit_counts.append(result_count)
            hit_counts.setdefault(query.mode, []).append(result_count)
            mode_timings.setdefault(query.mode, []).append(elapsed_ms)
        warm_p50_ms = _percentile(warm_timings, 0.50)
        warm_p95_ms = _percentile(warm_timings, 0.95)
        warm_max_ms = max(warm_timings)
        hit_count_mix = _hit_count_mix(all_hit_counts)
        if synthetic:
            _validate_hit_count_mix(hit_count_mix)

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
            "first_query_ms": first_query_ms,
            "warm_search_p50_ms": warm_p50_ms,
            "warm_search_p95_ms": warm_p95_ms,
            "warm_search_max_ms": warm_max_ms,
        }
        targets = {
            "index_build_ms": None,
            "incremental_refresh_ms": None,
            "first_query_ms": 750.0,
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
            "queries": {
                "count": query_count,
                "modes": query_modes,
                "hit_count_mix": hit_count_mix,
            },
            "metrics": metrics,
            "targets": targets,
            "checks": checks,
            "gate": {
                "passed": gate_passed,
                "enforced": default_synthetic,
                "reason": "warm search p95 must be at most 300 ms",
            },
        }
        if hook_path:
            report["hook_path"] = _run_hook_benchmark(config_path, config, queries, query_count)
        return report


def _hook_process(config_path: Path, query: QuerySpec) -> dict:
    command = [
        sys.executable, "-m", "pausanias", "--config", str(config_path), "hook", query.query,
        "--limit", "20", "--json",
    ]
    if query.project is not None:
        command.extend(["--project", query.project])
    if query.root_id is not None:
        command.extend(["--root", query.root_id])
    if query.all_projects:
        command.append("--all-projects")
    environment = os.environ.copy()
    repository = str(Path(__file__).resolve().parents[1])
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = repository if not existing_pythonpath else repository + os.pathsep + existing_pythonpath
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, cwd=repository, env=environment, check=False)
    wall_ms = _positive_milliseconds(time.perf_counter() - started)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "hook process failed")
    value = json.loads(completed.stdout)
    if not isinstance(value, dict) or not isinstance(value.get("metrics"), dict):
        raise RuntimeError("hook process returned an invalid report")
    metrics = value["metrics"]
    metrics["hook_internal_ms"] = metrics.get("hook_total_ms", 0.000001)
    metrics["hook_total_ms"] = wall_ms
    value["candidate_count"] = len(value.get("items", [])) if isinstance(value.get("items"), list) else 0
    metrics["candidate_count"] = value["candidate_count"]
    return value


def _stage_summary(values: list[float]) -> dict[str, float]:
    return {"p50_ms": _percentile(values, 0.50), "p95_ms": _percentile(values, 0.95), "max_ms": max(values)}


def _phase_report(samples: list[dict], corpus_section_count: int = 0) -> dict[str, object]:
    names = (
        "hook_total_ms", "worker_startup_ms", "model_load_ms", "matrix_load_ms",
        "encode_ms", "scan_ms", "fallback_ms", "fts_search_ms", "hybrid_overhead_ms",
    )
    metrics = {
        name: _stage_summary([max(float(sample.get(name, 0.000001)), 0.000001) for sample in samples])
        for name in names
    }
    return {
        "metrics": metrics,
        "fallback_count": sum(bool(sample.get("fallback", False)) for sample in samples),
        "candidate_count": sum(int(sample.get("candidate_count", 0)) for sample in samples),
        "corpus_section_count": corpus_section_count,
        "throughput": {
            "queries_per_second": len(samples) / max(sum(float(sample.get("hook_total_ms", 0.000001)) for sample in samples) / 1000.0, 0.000001),
            "encoded_sections_per_second": corpus_section_count / max(sum(float(sample.get("encode_ms", 0.000001)) for sample in samples) / 1000.0, 0.000001),
        },
        "samples": len(samples),
        "cache_states": sorted({str(sample.get("cache_state", "unknown")) for sample in samples}),
    }


def _run_hook_benchmark(config_path: Path, config: Config, queries: list[QuerySpec], query_count: int) -> dict[str, object]:
    _hook_process(config_path, queries[0])
    warm_samples = [_hook_process(config_path, query)["metrics"] for query in queries]
    cold_samples: list[dict] = []
    for query in queries:
        stop_worker(config.database)
        cold_samples.append(_hook_process(config_path, query)["metrics"])
    with sqlite3.connect(config.database) as connection:
        corpus_section_count = int(connection.execute("SELECT count(*) FROM sections").fetchone()[0])
    warm = _phase_report(warm_samples, corpus_section_count)
    cold = _phase_report(cold_samples, corpus_section_count)
    semantic_enabled = any(not bool(sample.get("fallback", False)) for sample in warm_samples + cold_samples)
    if not semantic_enabled:
        reason = "semantic backend unavailable; lexical gate remains active"
        warm_gate = {"enabled": False, "passed": None, "reason": reason}
        cold_gate = {"enabled": False, "passed": None, "reason": reason}
    else:
        warm_gate = {
            "enabled": True,
            "passed": warm["metrics"]["hook_total_ms"]["p95_ms"] <= 300.0 and warm["fallback_count"] == 0,
            "target_ms": 300.0,
            "fallback_count": warm["fallback_count"],
        }
        cold_gate = {
            "enabled": True,
            "passed": cold["metrics"]["hook_total_ms"]["p95_ms"] <= 750.0 and cold["fallback_count"] == 0,
            "target_ms": 750.0,
            "fallback_count": cold["fallback_count"],
        }
    result = {
        "query_count": query_count,
        "throughput": {
            "warm_queries_per_second": warm["throughput"]["queries_per_second"],
            "cold_queries_per_second": cold["throughput"]["queries_per_second"],
            "warm_encoded_sections_per_second": warm["throughput"]["encoded_sections_per_second"],
            "cold_encoded_sections_per_second": cold["throughput"]["encoded_sections_per_second"],
        },
        "candidate_count": warm["candidate_count"] + cold["candidate_count"],
        "corpus_size": {
            "files": sum(len(_markdown_files(root.path)) for root in config.roots),
            "sections": corpus_section_count,
        },
        "platform": platform.platform(),
        "provider": MODEL_BUNDLE_MANIFEST["runtime"]["provider"],
        "model_manifest": MODEL_BUNDLE_MANIFEST,
        "warm": {**warm, "gate": warm_gate},
        "cold": {**cold, "gate": cold_gate},
        "gates": {"warm_scan_path": warm_gate, "cold_scan_path": cold_gate},
        "passed": warm_gate["passed"] is not False and cold_gate["passed"] is not False,
    }
    stop_worker(config.database)
    return result


def format_report(report: dict) -> str:
    lines = [
        f"source: {report['source']['kind']} ({report['source']['file_count']} files)",
        f"queries: {report['queries']['count']}",
        "first query: warm OS page cache, fresh SQLite connection with an empty page cache",
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
    mix = report["queries"]["hit_count_mix"]
    shares = mix["shares"]
    lines += [
        "",
        "workload hit-count mix: "
        f"0={mix['zero']} ({shares['zero']:.0%}), "
        f"1={mix['one']} ({shares['one']:.0%}), "
        f"2={mix['two']} ({shares['two']:.0%}), "
        f"3+={mix['three_or_more']} ({shares['three_or_more']:.0%})",
    ]
    gate = report["gate"]
    status = "pass" if gate["passed"] else "fail"
    enforcement = " (enforced)" if gate["enforced"] else ""
    lines += ["", f"warm p95 gate: {status}{enforcement}"]
    hook = report.get("hook_path")
    if hook is not None:
        corpus = hook["corpus_size"]
        lines += [
            "",
            f"hook corpus: {corpus['files']} files, {corpus['sections']} sections; candidates={hook['candidate_count']}",
            f"hook platform: {hook['platform']}; provider: {hook['provider']}",
            "hook-path stage metrics (p50 / p95 / max ms)",
        ]
        for phase_name in ("warm", "cold"):
            phase = hook[phase_name]
            lines.append(f"{phase_name} scan path:")
            for name, values in phase["metrics"].items():
                lines.append(
                    f"  {name:20} {values['p50_ms']:8.2f} / {values['p95_ms']:8.2f} / {values['max_ms']:8.2f}"
                )
            phase_gate = phase["gate"]
            result = "disabled" if not phase_gate["enabled"] else ("pass" if phase_gate["passed"] else "fail")
            lines.append(f"  gate: {result}; fallbacks={phase['fallback_count']}")
    return "\n".join(lines)
