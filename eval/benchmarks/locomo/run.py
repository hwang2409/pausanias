"""Offline-first LOCOMO ingest, search, and evaluation runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

from pausanias.config import Config, Root
from pausanias.core import Candidate, index, search
from pausanias.hook import HookMetrics, run_hook
from pausanias.worker import stop_worker

from eval.schema import (
    Checkpoint,
    CutoffOutcome,
    Evaluation,
    Metadata,
    Metrics,
    PromptMetadata,
    RetrievalResult,
    UnifiedResult,
    evaluation_from_dict,
    fingerprint,
    latency_summary,
    read_checkpoint,
    retrieval_result_from_dict,
    utc_now,
)
from eval.benchmarks.locomo.metrics import (
    ABSTENTION_CATEGORY,
    COMPARABLE_CATEGORIES,
    _abstention_metrics,
    abstention_outcome,
    _predict_metrics,
    has_complete_retrieval_diagnostics,
    retrieval_outcome,
)
from eval.vendor.mem0.benchmarks.locomo.prompts import (
    CATEGORIES_TO_EVALUATE,
    CATEGORY_NAMES,
    JUDGE_SYSTEM_PROMPT,
    get_answer_generation_prompt,
    get_judge_prompt,
    get_judge_prompt_with_evidence,
    preprocess_answer,
)

DATASET_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DATASET_COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
DATASET_SOURCE_COMMIT = DATASET_COMMIT
DATASET_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
DATASET_DIR = Path("datasets/locomo")
DATASET_PATH = DATASET_DIR / "locomo10.json"
PROMPT_VERSION = "mem0-locomo-4b61c5d"
PROMPT_FIXTURE_VERSION = "golden-prompts-v1"
DEFAULT_CUTOFFS = (10, 20, 50, 200)
DEFAULT_TOP_K = 200
class LocomoError(RuntimeError):
    """Raised for invalid benchmark input or a failed stage."""


@dataclass(frozen=True)
class RenderedSession:
    path: str
    text: str
    turn_ranges: dict[str, tuple[int, int]]


@dataclass(frozen=True)
class LLMReply:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class StubTransport:
    """Deterministic transport for tests. It never performs network I/O."""

    def __init__(self, responses: list[str | dict[str, object]]):
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def __call__(self, *, provider: str, model: str, system: str, user: str, structured: bool = False) -> LLMReply:
        self.calls.append({"provider": provider, "model": model, "system": system, "user": user, "structured": structured})
        if not self.responses:
            raise LocomoError("stub transport has no response")
        response = self.responses.pop(0)
        if isinstance(response, dict):
            return LLMReply(json.dumps(response))
        return LLMReply(response)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n")


def _session_number(key: str) -> int:
    match = re.fullmatch(r"session_(\d+)", key)
    if match is None:
        raise LocomoError(f"invalid session key: {key}")
    return int(match.group(1))


def _parse_locomo_date(value: str) -> datetime | None:
    cleaned = value.strip().strip("()").strip()
    formats = (
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %d %b, %Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    )
    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt).replace(tzinfo=UTC)
        except ValueError:
            pass
    return None


def _date_key(value: str) -> tuple[int, datetime | str]:
    parsed = _parse_locomo_date(value)
    return (0, parsed) if parsed is not None else (1, value)


def _prompt_created_at(value: str) -> str:
    parsed = _parse_locomo_date(value)
    return parsed.replace(tzinfo=None).isoformat() if parsed is not None else value


def sorted_sessions(conversation: dict[str, object]) -> list[tuple[str, str, list[dict[str, object]]]]:
    sessions = []
    for key, turns in conversation.items():
        if not re.fullmatch(r"session_\d+", key):
            continue
        if not isinstance(turns, list) or not all(isinstance(turn, dict) for turn in turns):
            raise LocomoError(f"{key} must contain turn objects")
        date = _normalize(conversation.get(f"{key}_date_time"))
        sessions.append((key, date, turns))
    return sorted(sessions, key=lambda item: (_date_key(item[1]), _session_number(item[0])))


get_sorted_sessions = sorted_sessions


def render_session(conversation_index: int, session_key: str, session_date: str, turns: list[dict[str, object]]) -> RenderedSession:
    session_number = _session_number(session_key)
    blocks = [
        f"# conversation {conversation_index:02d}",
        f"## session {session_number:02d} | timestamp: {_normalize(session_date)}",
    ]
    ranges: dict[str, tuple[int, int]] = {}
    for turn in turns:
        dia_id = _normalize(turn.get("dia_id"))
        speaker = _normalize(turn.get("speaker"))
        turn_date = _normalize(turn.get("session_date_time")) or _normalize(session_date)
        text = _normalize(turn.get("text"))
        query = _normalize(turn.get("query"))
        caption = _normalize(turn.get("blip_caption"))
        if query or caption:
            metadata = f"[image query: {query}; caption: {caption}]"
            if text and not text.endswith("\n"):
                text += "\n"
            text += metadata
        heading = f"### turn {dia_id} | speaker: {speaker} | timestamp: {turn_date}"
        blocks.append(f"{heading}\n\n{text}")
        start = sum(len(block.split("\n")) for block in blocks[:-1]) + len(blocks[:-1]) + 1
        end = start + 1 + len(text.split("\n"))
        ranges[dia_id] = (start, end)
    content = "\n\n".join(blocks) + "\n"
    path = f"conversation-{conversation_index:02d}--session-{session_number:02d}.md"
    return RenderedSession(path, content, ranges)


def evidence_id(path: str, line_start: int, line_end: int) -> str:
    normalized = unicodedata.normalize("NFC", path).replace("\\", "/")
    pure = Path(normalized)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise LocomoError("evidence path must be repository-relative")
    return f"{normalized}#{line_start:08d}-{line_end:08d}"


def _dataset_fingerprint(path: Path) -> str:
    return _sha256(path.read_bytes())


def _validate_dataset(data: object, *, pinned: bool) -> list[dict[str, object]]:
    if not isinstance(data, list) or (pinned and len(data) != 10):
        raise LocomoError("LOCOMO dataset must contain ten conversations")
    conversations: list[dict[str, object]] = []
    counts: defaultdict[int, int] = defaultdict(int)
    for entry in data:
        if not isinstance(entry, dict) or not isinstance(entry.get("conversation"), dict):
            raise LocomoError("each LOCOMO entry must contain a conversation")
        conversations.append(entry)
        for question in entry.get("qa", []):
            if isinstance(question, dict) and isinstance(question.get("category"), int):
                counts[question["category"]] += 1
    if pinned and counts != {1: 282, 2: 321, 3: 96, 4: 841, 5: 446}:
        raise LocomoError(f"unexpected LOCOMO category counts: {dict(counts)}")
    return conversations


def fetch_dataset(path: Path = DATASET_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        data = path.read_bytes()
        if _sha256(data) != DATASET_SHA256:
            raise LocomoError(f"cached LOCOMO dataset hash mismatch: {path}")
        _validate_dataset(json.loads(data), pinned=True)
        return path
    request = Request(DATASET_URL, headers={"User-Agent": "pausanias-locomo/1"})
    data = urlopen(request, timeout=30).read()
    if _sha256(data) != DATASET_SHA256:
        raise LocomoError("downloaded LOCOMO dataset hash mismatch")
    _validate_dataset(json.loads(data), pinned=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return path


def load_dataset(path: Path | None = None) -> tuple[list[dict[str, object]], dict[str, object]]:
    selected = fetch_dataset() if path is None else path
    data = selected.read_bytes()
    entries = _validate_dataset(json.loads(data), pinned=path is None)
    return entries, {
        "name": "locomo10",
        "version": DATASET_COMMIT if path is None else "fixture",
        "url": DATASET_URL if path is None else None,
        "sha256": _sha256(data),
        "fingerprint": _sha256(data),
    }


def _render_entries(entries: list[dict[str, object]], notes: Path) -> dict[tuple[int, str], RenderedSession]:
    rendered: dict[tuple[int, str], RenderedSession] = {}
    for fallback_index, entry in enumerate(entries):
        conversation_index = int(entry.get("_conversation_index", fallback_index))
        conversation = entry["conversation"]
        if not isinstance(conversation, dict):
            raise LocomoError("conversation must be an object")
        for key, date, turns in sorted_sessions(conversation):
            rendered_session = render_session(conversation_index, key, date, turns)
            path = notes / rendered_session.path
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(rendered_session.text, encoding="utf-8", newline="\n")
            os.replace(temporary, path)
            rendered[(conversation_index, key)] = rendered_session
    expected_paths = {session.path for session in rendered.values()}
    for path in notes.glob("*.md"):
        if path.name not in expected_paths:
            path.unlink()
    return rendered


def _evidence_targets(entry: dict[str, object], conversation_index: int, rendered: dict[tuple[int, str], RenderedSession]) -> dict[str, list[dict[str, object]]]:
    conversation = entry["conversation"]
    if not isinstance(conversation, dict):
        raise LocomoError("conversation must be an object")
    by_dia: dict[str, tuple[str, tuple[int, int]]] = {}
    for session_key, _, turns in sorted_sessions(conversation):
        session = rendered[(conversation_index, session_key)]
        for turn in turns:
            dia_id = _normalize(turn.get("dia_id"))
            if dia_id in session.turn_ranges:
                by_dia[dia_id] = (session.path, session.turn_ranges[dia_id])
    result: dict[str, list[dict[str, object]]] = {}
    for question in entry.get("qa", []):
        if not isinstance(question, dict) or question.get("category") not in CATEGORIES_TO_EVALUATE:
            continue
        targets = []
        for reference in question.get("evidence", []):
            if reference not in by_dia:
                continue
            path, (start, end) = by_dia[reference]
            targets.append({"evidence_id": evidence_id(path, start, end), "path": path, "line_start": start, "line_end": end})
        result[str(question.get("question"))] = sorted(targets, key=lambda item: item["evidence_id"].encode("utf-8"))
    return result


def _candidate_result(candidate: Candidate, notes: Path, rank: int) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        section_id=candidate.section_id,
        source_path=Path(candidate.canonical_path).resolve().relative_to(notes.resolve()).as_posix(),
        root_id=candidate.root_id,
        project=candidate.project_scope,
        heading=candidate.heading,
        line_start=candidate.line_start,
        line_end=candidate.line_end,
        excerpt=candidate.text,
        content_hash=candidate.content_hash,
        score=candidate.score,
        reason=candidate.reason,
    )


def _prompt_metadata(prompt: str, provider: str, model: str, reply: LLMReply | None, slots: tuple[str, ...]) -> PromptMetadata:
    return PromptMetadata(
        prompt_version=PROMPT_VERSION,
        template_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        slot_names=slots,
        provider=provider,
        model=model,
        prompt_tokens=reply.prompt_tokens if reply else None,
        completion_tokens=reply.completion_tokens if reply else None,
    )


def build_answer_prompt(question: str, results: list[dict[str, object]], reference_date: str | None, user_profile: dict[str, object] | None) -> str:
    return get_answer_generation_prompt(question, results, reference_date=reference_date, user_profile=user_profile)


def build_judge_prompt(category: int, question: str, answer: str, response: str, evidence_context: str | None = None) -> str:
    if evidence_context:
        return get_judge_prompt_with_evidence(category, question, answer, response, evidence_context)
    return get_judge_prompt(category, question, answer, response)


def _extract_answer(text: str) -> str:
    return text.rsplit("ANSWER:", 1)[-1].strip() if "ANSWER:" in text else text


def _call_transport(transport: object, *, provider: str, model: str, system: str, user: str, structured: bool = False) -> LLMReply:
    reply = transport(provider=provider, model=model, system=system, user=user, structured=structured)
    if isinstance(reply, LLMReply):
        return reply
    if isinstance(reply, str):
        return LLMReply(reply)
    raise LocomoError("LLM transport returned an invalid response")


def _required_key(provider: str) -> bool:
    if provider == "azure":
        return bool(os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"))
    return bool(os.environ.get({"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider, "")))


def _http_transport(*, provider: str, model: str, system: str, user: str, structured: bool = False) -> LLMReply:
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if provider == "openai":
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
    elif provider == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"}
        body = {"model": model, "max_tokens": 4096, "system": system, "messages": [{"role": "user", "content": user}]}
    elif provider == "azure":
        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
        url = f"{endpoint}/openai/deployments/{model}/chat/completions?api-version=2024-02-15-preview"
        headers = {"api-key": os.environ["AZURE_OPENAI_API_KEY"]}
    else:
        raise LocomoError(f"real provider adapter is not configured for {provider}")
    if structured and provider in {"openai", "azure"}:
        body["response_format"] = {"type": "json_object"}
    request = Request(url, data=json.dumps(body).encode(), headers={**headers, "Content-Type": "application/json"}, method="POST")
    response = json.loads(urlopen(request, timeout=120).read())
    if provider == "anthropic":
        choice = response["content"][0]["text"]
        usage = response.get("usage", {})
        return LLMReply(choice, usage.get("input_tokens"), usage.get("output_tokens"))
    choice = response["choices"][0]["message"]["content"]
    usage = response.get("usage", {})
    return LLMReply(choice, usage.get("prompt_tokens"), usage.get("completion_tokens"))


def _call_with_retries(transport: object, **kwargs: object) -> LLMReply:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            return _call_transport(transport, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt < 4 and not isinstance(transport, StubTransport):
                time.sleep((attempt + 1) * 2)
    if last_error is None:
        raise LocomoError("LLM call failed without an error")
    raise last_error


def _judge_with_retries(
    transport: object,
    *,
    provider: str,
    model: str,
    user: str,
) -> tuple[LLMReply, dict[str, object]]:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            reply = _call_transport(transport, provider=provider, model=model, system=JUDGE_SYSTEM_PROMPT, user=user, structured=True)
            payload = json.loads(reply.text)
            if not isinstance(payload, dict) or str(payload.get("label", "")).upper() not in {"CORRECT", "WRONG"}:
                raise ValueError("invalid judge label")
            return reply, payload
        except Exception as exc:
            last_error = exc
            if attempt < 4 and not isinstance(transport, StubTransport):
                time.sleep((attempt + 1) * 2)
    if last_error is None:
        raise LocomoError("judge call failed without an error")
    raise last_error


def _question_records(
    entries: list[dict[str, object]],
    rendered: dict[tuple[int, str], RenderedSession],
    *,
    abstention: bool = False,
) -> list[dict[str, object]]:
    records = []
    expected_category = {5} if abstention else set(CATEGORIES_TO_EVALUATE)
    for fallback_index, entry in enumerate(entries):
        conversation_index = int(entry.get("_conversation_index", fallback_index))
        targets = {} if abstention else _evidence_targets(entry, conversation_index, rendered)
        for question_index, question in enumerate(entry.get("qa", [])):
            category = question.get("category") if isinstance(question, dict) else None
            if category not in expected_category:
                continue
            records.append({
                "case_id": f"conv{conversation_index}_q{question_index}",
                "conversation_index": conversation_index,
                "question_index": question_index,
                "question": _normalize(question.get("question")),
                "category": CATEGORY_NAMES[int(question["category"])],
                "category_number": int(question["category"]),
                "ground_truth": None if abstention else _normalize(question.get("answer")),
                "targets": targets.get(_normalize(question.get("question")), []),
                "evidence": question.get("evidence", []),
                "sessions": sorted_sessions(entry["conversation"]),
            })
    return records


def _reference_date(entry: dict[str, object]) -> str | None:
    conversation = entry["conversation"]
    if not isinstance(conversation, dict):
        raise LocomoError("conversation must be an object")
    sessions = sorted_sessions(conversation)
    return sessions[-1][1] if sessions else None


def _conversation_scope(
    entry: dict[str, object],
    conversation_index: int,
    notes: Path,
    workspace: Path,
) -> tuple[Path, Config]:
    scope = workspace / "scopes" / f"conversation-{conversation_index:02d}"
    scope.mkdir(parents=True, exist_ok=True)
    conversation = entry["conversation"]
    if not isinstance(conversation, dict):
        raise LocomoError("conversation must be an object")
    expected_paths = {
        f"conversation-{conversation_index:02d}--session-{_session_number(session_key):02d}.md"
        for session_key, _, _ in sorted_sessions(conversation)
    }
    for path in scope.glob("*.md"):
        if path.name not in expected_paths:
            path.unlink()
    for session_key, _, _ in sorted_sessions(conversation):
        source = notes / f"conversation-{conversation_index:02d}--session-{_session_number(session_key):02d}.md"
        destination = scope / source.name
        if not destination.exists() or destination.read_bytes() != source.read_bytes():
            temporary = destination.with_suffix(".tmp")
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
    config = Config(
        (Root(f"locomo-{conversation_index}", scope.resolve(), f"locomo-{conversation_index}", ()),),
        frozenset(),
        (),
        (scope / "index.sqlite3").resolve(),
        12000,
        Path("~/.cache/pausanias/models/all-MiniLM-L6-v2").expanduser(),
    )
    return scope, config


def _scope_corpus_fingerprint(scope: Path) -> str:
    return fingerprint([
        {"path": path.name, "sha256": _sha256(path.read_bytes())}
        for path in sorted(scope.glob("*.md"))
    ])


def _index_generation(database: Path) -> int:
    with sqlite3.connect(database) as connection:
        row = connection.execute("SELECT value FROM metadata WHERE key = 'generation'").fetchone()
    if row is None:
        raise LocomoError(f"index generation is missing: {database}")
    try:
        generation = int(row[0])
    except (TypeError, ValueError) as exc:
        raise LocomoError(f"index generation is invalid: {database}") from exc
    if generation < 1:
        raise LocomoError(f"index generation is invalid: {database}")
    return generation


def _worker_config_path(config: Config, path: Path) -> Path:
    lines = [
        f"database = {json.dumps(str(config.database))}",
        f"private_paths = {json.dumps(list(config.private_paths))}",
        f"global_notes = {json.dumps(sorted(str(note) for note in config.global_notes))}",
        f"section_bytes = {config.section_bytes}",
        f"semantic_bundle = {json.dumps(str(config.semantic_bundle))}",
    ]
    for root in config.roots:
        lines.extend((
            "",
            "[[roots]]",
            f"id = {json.dumps(root.id)}",
            f"project = {json.dumps(root.project)}",
            f"path = {json.dumps(str(root.path))}",
            f"exclude = {json.dumps(list(root.excludes))}",
        ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _search_candidates(
    config: Config,
    query: str,
    top_k: int,
    retrieval_mode: str,
    worker_config_path: Path | None,
    hook_equivalent: bool = False,
) -> tuple[list[Candidate], dict[str, object] | None]:
    if retrieval_mode == "fused" or hook_equivalent:
        if worker_config_path is None:
            raise LocomoError("fused retrieval requires a worker config")
        try:
            response = run_hook(
                config,
                worker_config_path,
                query,
                project=config.roots[0].project,
                limit=top_k,
                retrieval_mode=retrieval_mode,
            )
        except Exception as exc:
            if not hook_equivalent:
                raise
            return [], asdict(HookMetrics(
                failure_reason=f"{type(exc).__name__}: {exc}",
                status="error",
                retrieval_mode=retrieval_mode,
            ))
        return response.candidates, asdict(response.metrics)
    return search(config, query, project=config.roots[0].project, limit=top_k, semantic=False), None


def _search_record(
    record: dict[str, object],
    config: Config,
    notes: Path,
    top_k: int,
    retrieval_mode: str = "lexical",
    worker_config_path: Path | None = None,
    hook_equivalent: bool = False,
) -> dict[str, object]:
    source_dates = {
        f"conversation-{int(record['conversation_index']):02d}--session-{_session_number(session_key):02d}.md": _prompt_created_at(date)
        for session_key, date, _ in record["sessions"]
    }
    evidence_lines = []
    for target in record["targets"]:
        lines = notes.joinpath(str(target["path"])).read_text(encoding="utf-8").splitlines()
        excerpt = "\n".join(lines[int(target["line_start"]) - 1:int(target["line_end"])])
        evidence_lines.append(f"[{target['evidence_id']}] {excerpt}")
    start = time.perf_counter()
    if hook_equivalent:
        candidates, diagnostics = _search_candidates(
            config, str(record["question"]), top_k, retrieval_mode,
            worker_config_path, hook_equivalent=True,
        )
    else:
        candidates, diagnostics = _search_candidates(
            config, str(record["question"]), top_k, retrieval_mode, worker_config_path,
        )
    elapsed = (time.perf_counter() - start) * 1000
    return {
        **record,
        "query": record["question"],
        "search_latency_ms": elapsed,
        "retrieval_fingerprint": fingerprint([candidate.section_id for candidate in candidates]),
        "retrieval_results": [asdict(_candidate_result(candidate, notes, rank)) for rank, candidate in enumerate(candidates, 1)],
        "reference_date": record["sessions"][-1][1] if record["sessions"] else None,
        "source_dates": source_dates,
        "evidence_context": "\n".join(evidence_lines),
        **({"retrieval_diagnostics": diagnostics} if diagnostics is not None else {}),
    }


def _common_cutoff(
    record: dict[str, object],
    results: tuple[RetrievalResult, ...],
    cutoff: int,
    *,
    answerer_model: str,
    answerer_provider: str,
    judge_model: str,
    judge_provider: str,
    with_evidence: bool,
    user_profile: dict[str, object] | None,
    transport: object,
) -> CutoffOutcome:
    targets = list(record["targets"])
    relevant, recall, precision, mrr = retrieval_outcome(results, targets, cutoff)
    selected = results[:cutoff]
    source_dates = record.get("source_dates", {})
    answer_results = [
        {"memory": item.excerpt, "created_at": str(source_dates.get(item.source_path, "")), "section_id": item.section_id}
        for item in selected[:200]
    ]
    answer_prompt = build_answer_prompt(str(record["question"]), answer_results, record.get("reference_date"), user_profile)
    answer_meta: PromptMetadata
    judge_meta: PromptMetadata
    try:
        answer_reply = _call_with_retries(transport, provider=answerer_provider, model=answerer_model, system="", user=answer_prompt)
        generated = _extract_answer(answer_reply.text)
        answer_meta = _prompt_metadata(answer_prompt, answerer_provider, answerer_model, answer_reply, ("reference_date", "memories", "question"))
    except Exception:
        answer_meta = _prompt_metadata(answer_prompt, answerer_provider, answerer_model, None, ("reference_date", "memories", "question"))
        return CutoffOutcome(len(selected), relevant, recall, precision, mrr, None, 0, 0.0, False, None, "ERROR", None, answerer_model, "answerer_error", {"answerer": answer_meta})
    evidence_context = str(record.get("evidence_context", "")) if with_evidence else None
    judge_prompt = build_judge_prompt(int(record["category_number"]), str(record["question"]), preprocess_answer(int(record["category_number"]), str(record["ground_truth"])), generated, evidence_context)
    try:
        judge_reply, payload = _judge_with_retries(transport, provider=judge_provider, model=judge_model, user=judge_prompt)
        judgment = str(payload["label"]).upper()
        score = 1.0 if judgment == "CORRECT" else 0.0
        reason = str(payload.get("reasoning", ""))
        judge_meta = _prompt_metadata(judge_prompt, judge_provider, judge_model, judge_reply, ("question", "answer", "response"))
        return CutoffOutcome(len(selected), relevant, recall, precision, mrr, None, 0, score, score >= 0.5, generated, judgment, reason, answerer_model, None, {"answerer": answer_meta, "judge": judge_meta})
    except Exception:
        judge_meta = _prompt_metadata(judge_prompt, judge_provider, judge_model, None, ("question", "answer", "response"))
        return CutoffOutcome(len(selected), relevant, recall, precision, mrr, None, 0, 0.0, False, generated, "ERROR", None, answerer_model, "judge_error", {"answerer": answer_meta, "judge": judge_meta})


def _common_metrics(evaluations: list[Evaluation], cutoffs: tuple[int, ...]) -> Metrics:
    categories = list(COMPARABLE_CATEGORIES)
    by_category = {}
    primary = max(cutoffs)
    for category in categories:
        selected = [item for item in evaluations if item.category == category]
        outcomes = [item.cutoff_outcomes[str(primary)] for item in selected]
        errors = sum(outcome.error is not None for outcome in outcomes)
        by_category[category] = {"total": len(outcomes), "correct": sum(outcome.score >= 0.5 for outcome in outcomes), "accuracy": 100 * sum(outcome.score >= 0.5 for outcome in outcomes) / len(outcomes) if outcomes else 0.0, "avg_score": 100 * sum(outcome.score for outcome in outcomes) / len(outcomes) if outcomes else 0.0, "errors": errors}
    by_cutoff = {}
    for cutoff in cutoffs:
        groups = {}
        for category in categories:
            outcomes = [item.cutoff_outcomes[str(cutoff)] for item in evaluations if item.category == category]
            groups[category] = {"total": len(outcomes), "correct": sum(outcome.score >= 0.5 for outcome in outcomes), "accuracy": 100 * sum(outcome.score >= 0.5 for outcome in outcomes) / len(outcomes) if outcomes else 0.0, "avg_score": 100 * sum(outcome.score for outcome in outcomes) / len(outcomes) if outcomes else 0.0, "errors": sum(outcome.error is not None for outcome in outcomes)}
        outcomes = [item.cutoff_outcomes[str(cutoff)] for item in evaluations]
        by_cutoff[str(cutoff)] = {"cutoff": cutoff, "overall": {"total": len(outcomes), "correct": sum(outcome.score >= 0.5 for outcome in outcomes), "accuracy": 100 * sum(outcome.score >= 0.5 for outcome in outcomes) / len(outcomes) if outcomes else 0.0, "avg_score": 100 * sum(outcome.score for outcome in outcomes) / len(outcomes) if outcomes else 0.0, "errors": sum(outcome.error is not None for outcome in outcomes)}, "by_category": groups}
    primary_outcomes = [item.cutoff_outcomes[str(primary)] for item in evaluations]
    total = len(primary_outcomes)
    errors = sum(outcome.error is not None for outcome in primary_outcomes)
    return Metrics(
        overall_accuracy=100 * sum(outcome.score >= 0.5 for outcome in primary_outcomes) / total if total else 0.0,
        overall_avg_score=100 * sum(outcome.score for outcome in primary_outcomes) / total if total else 0.0,
        total=total,
        correct=sum(outcome.score >= 0.5 for outcome in primary_outcomes),
        errors=errors,
        by_category=by_category,
        by_cutoff=by_cutoff,
        latency_ms={"overall": latency_summary([item.search_latency_ms for item in evaluations]), "by_category": {category: latency_summary([item.search_latency_ms for item in evaluations if item.category == category]) for category in categories}},
    )


def run_locomo(
    *,
    run_id: str,
    dataset_path: Path | None = None,
    results_dir: Path = Path("results"),
    conversations: tuple[int, ...] | None = None,
    top_k: int = DEFAULT_TOP_K,
    cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
    answerer_model: str = "gpt-4o-mini",
    judge_model: str = "gpt-4o-mini",
    provider: str = "openai",
    judge_provider: str | None = None,
    with_evidence: bool = False,
    user_profile: dict[str, object] | None = None,
    predict_only: bool = False,
    evaluate_only: bool = False,
    resume: bool = False,
    transport: object | None = None,
    retrieval_mode: str = "lexical",
    abstention: bool = False,
) -> UnifiedResult | None:
    if predict_only and evaluate_only:
        raise ValueError("--predict-only and --evaluate-only cannot be combined")
    if abstention and (predict_only or evaluate_only):
        raise ValueError("--abstention runs retrieval-only and cannot use stage flags")
    if top_k < max(cutoffs) or not cutoffs:
        raise ValueError("top-k must include all cutoffs")
    if retrieval_mode not in {"lexical", "fused"}:
        raise ValueError("retrieval mode must be lexical or fused")
    selected_judge_provider = judge_provider or provider
    if not predict_only and not abstention and transport is None and (not _required_key(provider) or not _required_key(selected_judge_provider)):
        raise LocomoError("answerer and judge API keys are required for evaluation")
    auto_dataset = dataset_path is None
    if auto_dataset and predict_only:
        dataset_path = fetch_dataset()
    entries, dataset = load_dataset(None if auto_dataset else dataset_path)
    selected = set(range(len(entries))) if conversations is None else set(conversations)
    entries = [dict(entry, _conversation_index=index) for index, entry in enumerate(entries) if index in selected]
    run_root = results_dir / "locomo" / run_id
    workspace = run_root / "workspace"
    notes = workspace / "notes"
    search_dir = run_root / "checkpoints" / "search"
    evaluate_dir = run_root / "checkpoints" / "evaluate"
    ingest_path = run_root / "checkpoints" / "ingest.json"
    search_dir.mkdir(parents=True, exist_ok=True)
    evaluate_dir.mkdir(parents=True, exist_ok=True)
    rendered = _render_entries(entries, notes)
    records = _question_records(entries, rendered, abstention=abstention)
    corpus_hash = fingerprint([{ "path": path.relative_to(notes).as_posix(), "sha256": _sha256(path.read_bytes()) } for path in sorted(notes.glob("*.md"))]) if notes.exists() else None
    retrieval_config = {"retrieval_mode": retrieval_mode, "top_k": top_k, "cutoffs": list(cutoffs), "conversations": sorted(selected)}
    if abstention:
        retrieval_config["evaluation_mode"] = "abstention"
    model_config = {
        "prompt_mode": f"answerer-{'profile' if user_profile else 'no-profile'}+judge-{'with-evidence' if with_evidence else 'without-evidence'}",
        "prompt_version": PROMPT_VERSION,
        "prompt_fixture_version": PROMPT_FIXTURE_VERSION,
        "profile_fingerprint": fingerprint(user_profile) if user_profile is not None else None,
        "answerer_model": answerer_model,
        "answerer_provider": provider,
        "judge_model": judge_model,
        "judge_provider": judge_provider or provider,
    }
    config_values = {**retrieval_config, **model_config, "retrieval_config": retrieval_config, "model_config": model_config}
    checkpoint_config = config_values
    candidate = None
    existing_ingest = None
    if resume or evaluate_only:
        candidate = read_checkpoint(ingest_path, stage="ingest", run_id=run_id, dataset_fingerprint=dataset["fingerprint"], corpus_fingerprint=corpus_hash)
        candidate_config = candidate.config if candidate is not None else None
        candidate_retrieval_config = candidate_config.get("retrieval_config") if candidate_config is not None else None
        candidate_model_config = candidate_config.get("model_config") if candidate_config is not None else None
        prompt_config_matches = (
            isinstance(candidate_model_config, dict)
            and candidate_model_config.get("prompt_mode") == model_config["prompt_mode"]
            and candidate_model_config.get("prompt_version") == model_config["prompt_version"]
            and candidate_model_config.get("prompt_fixture_version") == model_config["prompt_fixture_version"]
            and candidate_model_config.get("profile_fingerprint") == model_config["profile_fingerprint"]
        )
        retrieval_config_matches = candidate_retrieval_config == retrieval_config
        config_matches = retrieval_config_matches and (candidate_config == config_values or ((evaluate_only or resume) and transport is not None and prompt_config_matches))
        if candidate is not None and config_matches:
            existing_ingest = candidate
            checkpoint_config = candidate_config
    candidate_is_complete = candidate is not None
    if existing_ingest is None and evaluate_only and not candidate_is_complete:
        raise LocomoError("evaluation requires a complete ingest checkpoint")
    scope_configs = {
        int(entry["_conversation_index"]): _conversation_scope(entry, int(entry["_conversation_index"]), notes, workspace)
        for entry in entries
    }
    worker_config_paths = {
        conversation_index: _worker_config_path(config, scope / "worker-config.toml")
        for conversation_index, (scope, config) in scope_configs.items()
    } if retrieval_mode == "fused" or abstention else {}
    scope_indexes: dict[int, dict[str, object]] = {}
    if existing_ingest is None:
        for conversation_index, (scope, scope_config) in scope_configs.items():
            scope_indexes[conversation_index] = {
                "corpus_fingerprint": _scope_corpus_fingerprint(scope),
                "index_generation": index(scope_config, rebuild=True),
            }
        ingest_generation = next(iter(scope_indexes.values()), {}).get("index_generation")
        ingest = Checkpoint(
            "ingest", run_id, dataset["fingerprint"], corpus_hash, ingest_generation, config_values,
            "complete", utc_now(), utc_now(),
            {
                "files": [{"path": path.name, "sha256": _sha256(path.read_bytes())} for path in sorted(notes.glob("*.md"))],
                "scope_indexes": {str(index): metadata for index, metadata in scope_indexes.items()},
                "index_generation": ingest_generation,
            },
        )
        ingest.write(ingest_path)
    else:
        for conversation_index, (scope, scope_config) in scope_configs.items():
            if not scope_config.database.exists():
                raise LocomoError("evaluation requires complete conversation indexes")
            scope_indexes[conversation_index] = {
                "corpus_fingerprint": _scope_corpus_fingerprint(scope),
                "index_generation": _index_generation(scope_config.database),
            }
        ingest = existing_ingest
    generation = int(ingest.index_generation or 0)
    if records:
        first_scope = scope_configs[int(records[0]["conversation_index"])]
        search_kwargs = {"hook_equivalent": True} if abstention else {}
        _search_candidates(
            first_scope[1], str(records[0]["question"]), top_k, retrieval_mode,
            worker_config_paths.get(int(records[0]["conversation_index"])),
            **search_kwargs,
        )
    evaluations: list[Evaluation] = []
    diagnostic_records: list[dict[str, object]] = []
    abstention_records: list[dict[str, object]] = []
    for record in records:
        scope_notes, scope_config = scope_configs[int(record["conversation_index"])]
        scope_metadata = scope_indexes[int(record["conversation_index"])]
        scope_corpus_hash = str(scope_metadata["corpus_fingerprint"])
        scope_generation = int(scope_metadata["index_generation"])
        checkpoint_path = search_dir / f"{record['case_id']}.json"
        checkpoint = read_checkpoint(
            checkpoint_path,
            stage="search",
            run_id=run_id,
            config=checkpoint_config,
            case_id=str(record["case_id"]),
            dataset_fingerprint=str(dataset["fingerprint"]),
            corpus_fingerprint=scope_corpus_hash,
            index_generation=scope_generation,
        ) if (resume or evaluate_only) else None
        if (
            checkpoint is not None
            and predict_only
            and resume
            and retrieval_mode == "fused"
            and not has_complete_retrieval_diagnostics(checkpoint.output.get("retrieval_diagnostics"))
        ):
            checkpoint = None
        if checkpoint is None:
            if evaluate_only and existing_ingest is not None:
                raise LocomoError(f"missing search checkpoint: {record['case_id']}")
            raw = _search_record(
                record, scope_config, scope_notes, top_k, retrieval_mode,
                worker_config_paths.get(int(record["conversation_index"])),
                hook_equivalent=abstention,
            )
            Checkpoint("search", run_id, dataset["fingerprint"], scope_corpus_hash, scope_generation, config_values, "complete", utc_now(), utc_now(), raw, str(record["case_id"])).write(checkpoint_path)
        else:
            raw = checkpoint.output
        if abstention:
            abstention_records.append(raw)
        raw_diagnostics = raw.get("retrieval_diagnostics")
        if isinstance(raw_diagnostics, dict):
            diagnostic_records.append(raw_diagnostics)
        elif retrieval_mode == "lexical":
            diagnostic_records.append({"retrieval_mode": "lexical", "fallback": False})
        retrieval = tuple(retrieval_result_from_dict(item) for item in raw["retrieval_results"])
        evaluation_path = evaluate_dir / f"{record['case_id']}.json"
        if not predict_only:
            evaluation_checkpoint = read_checkpoint(
                evaluation_path,
                stage="evaluate",
                run_id=run_id,
                config=config_values,
                case_id=str(record["case_id"]),
                dataset_fingerprint=str(dataset["fingerprint"]),
                corpus_fingerprint=scope_corpus_hash,
                index_generation=scope_generation,
                cutoffs=cutoffs,
            ) if (resume or evaluate_only) else None
            if evaluation_checkpoint is not None:
                evaluations.append(evaluation_from_dict(evaluation_checkpoint.output))
                continue
        outcomes: dict[str, CutoffOutcome] = {}
        if predict_only or abstention:
            for cutoff in cutoffs:
                if abstention:
                    outcomes[str(cutoff)] = abstention_outcome(
                        retrieval, cutoff, raw.get("retrieval_diagnostics")
                    )
                else:
                    relevant, recall, precision, mrr = retrieval_outcome(retrieval, list(raw["targets"]), cutoff)
                    outcomes[str(cutoff)] = CutoffOutcome(
                        retrieved_count=len(retrieval[:cutoff]),
                        relevant_count=relevant,
                        recall=recall,
                        precision=precision,
                        mrr=mrr,
                        abstention_correct=None,
                        forbidden_sources=0,
                        score=0.0,
                        passed=False,
                    )
        else:
            if transport is None:
                transport = _http_transport
            for cutoff in cutoffs:
                outcomes[str(cutoff)] = _common_cutoff(raw, retrieval, cutoff, answerer_model=answerer_model, answerer_provider=provider, judge_model=judge_model, judge_provider=selected_judge_provider, with_evidence=with_evidence, user_profile=user_profile, transport=transport)
        primary_outcome = outcomes[str(max(cutoffs))]
        evaluation = Evaluation(str(raw["case_id"]), str(raw["category"]), str(raw["question"]), tuple(target["evidence_id"] for target in raw["targets"]), raw["ground_truth"], retrieval, float(raw["search_latency_ms"]), outcomes, primary_outcome.score, primary_outcome.error)
        evaluations.append(evaluation)
        if not predict_only:
            Checkpoint(
                stage="evaluate",
                run_id=run_id,
                case_id=str(record["case_id"]),
                dataset_fingerprint=dataset["fingerprint"],
                corpus_fingerprint=scope_corpus_hash,
                index_generation=scope_generation,
                config=config_values,
                status="complete",
                started_at=utc_now(),
                finished_at=utc_now(),
                output=asdict(evaluation),
            ).write(evaluation_path)
    if retrieval_mode == "fused":
        for _, scope_config in scope_configs.values():
            stop_worker(scope_config.database)
    if predict_only:
        predict_config = {
            **retrieval_config,
            "prompt_mode": None,
            "prompt_version": None,
            "prompt_fixture_version": None,
            "profile_fingerprint": None,
            "answerer_model": None,
            "answerer_provider": None,
            "judge_model": None,
            "judge_provider": None,
        }
        metrics = _predict_metrics(evaluations, cutoffs, diagnostic_records)
        metadata = Metadata(
            "locomo",
            run_id,
            dataset,
            corpus_hash,
            generation,
            None,
            None,
            None,
            fingerprint({"dataset": dataset, "config": predict_config, "corpus": corpus_hash}),
            None,
            None,
            predict_config,
            ingest.started_at,
            utc_now(),
        )
        result = UnifiedResult(metadata, metrics, tuple(evaluations))
        result.write(run_root / "run.json")
        print(f"locomo predict-only: {len(evaluations)} search checkpoints; dataset fingerprint {dataset['fingerprint']}")
        return result
    if abstention:
        metrics = _abstention_metrics(evaluations, abstention_records)
        metadata = Metadata(
            "locomo",
            run_id,
            dataset,
            corpus_hash,
            generation,
            None,
            None,
            None,
            fingerprint({"dataset": dataset, "config": config_values, "corpus": corpus_hash}),
            None,
            None,
            {**config_values, "evaluation_mode": "abstention", "category": ABSTENTION_CATEGORY},
            ingest.started_at,
            utc_now(),
        )
        result = UnifiedResult(metadata, metrics, tuple(evaluations))
        result.write(run_root / "run.json")
        print(f"locomo abstention: {metrics.abstention['false_injection_rate']:.2%} false injection rate across {metrics.total} questions")
        return result
    metrics = _common_metrics(evaluations, cutoffs)
    prompt_hashes = [outcome.prompt_metadata for evaluation in evaluations for outcome in evaluation.cutoff_outcomes.values()]
    answer_hash = fingerprint([metadata["answerer"].template_hash for metadata in prompt_hashes if "answerer" in metadata]) if prompt_hashes else None
    judge_hash = fingerprint([metadata["judge"].template_hash for metadata in prompt_hashes if "judge" in metadata]) if prompt_hashes else None
    metadata = Metadata("locomo", run_id, dataset, corpus_hash, generation, None, PROMPT_VERSION, PROMPT_FIXTURE_VERSION, fingerprint({"dataset": dataset, "config": config_values, "corpus": corpus_hash}), answer_hash, judge_hash, {**config_values, "answerer_model": answerer_model, "answerer_provider": provider, "judge_model": judge_model, "judge_provider": judge_provider or provider}, ingest.started_at, utc_now())
    result = UnifiedResult(metadata, metrics, tuple(evaluations))
    result.write(run_root / "run.json")
    print(f"locomo evaluation: {metrics.overall_accuracy:.2f}% accuracy across {metrics.total} questions")
    return result


def _profile(value: str | None) -> dict[str, object] | None:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("user profile must be a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-id", default="latest")
    parser.add_argument("--conversations")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--top-k-cutoffs", default="10,20,50,200")
    parser.add_argument("--answerer-model", default="gpt-4o-mini")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--judge-provider")
    parser.add_argument("--with-evidence", action="store_true")
    parser.add_argument("--user-profile")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retrieval-mode", choices=("fused", "lexical"), default="lexical")
    parser.add_argument("--abstention", action="store_true", help="evaluate category-5 adversarial questions")
    args = parser.parse_args(argv)
    try:
        conversations = tuple(int(item) for item in args.conversations.split(",")) if args.conversations else None
        cutoffs = tuple(int(item) for item in args.top_k_cutoffs.split(","))
        run_locomo(run_id=args.run_id, dataset_path=args.dataset_path, results_dir=args.results_dir, conversations=conversations, top_k=args.top_k, cutoffs=cutoffs, answerer_model=args.answerer_model, judge_model=args.judge_model, provider=args.provider, judge_provider=args.judge_provider, with_evidence=args.with_evidence, user_profile=_profile(args.user_profile), predict_only=args.predict_only, evaluate_only=args.evaluate_only, resume=args.resume, retrieval_mode=args.retrieval_mode, abstention=args.abstention)
        return 0
    except (OSError, ValueError, TypeError, KeyError, LocomoError) as exc:
        print(f"locomo: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
