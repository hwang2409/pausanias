"""Generate deterministic, vault-like Markdown corpora for benchmarks."""

from __future__ import annotations

from pathlib import Path
import random


PROJECTS = ("phoebe", "pausanias", "orion", "atlas")
NOTE_TYPES = ("decision", "research", "project", "meeting", "runbook")
TOPICS = (
    "retrieval latency",
    "source boundaries",
    "context budgets",
    "index refresh",
    "adapter deadlines",
    "evidence packets",
    "search ranking",
    "project scope",
)
WORDS = (
    "local", "bounded", "source", "evidence", "memory", "query", "section",
    "project", "reader", "index", "fresh", "stable", "review", "decision",
    "context", "packet", "latency", "scope", "adapter", "model", "note",
    "change", "verify", "safe", "current", "history", "signal", "result",
)


def _paragraph(rng: random.Random, words: int) -> str:
    selected = [rng.choice(WORDS) for _ in range(words)]
    selected[0] = selected[0].capitalize()
    return " ".join(selected) + "."


def _note_content(
    rng: random.Random,
    project: str,
    identifier: str,
    note_type: str,
    topic: str,
    updated: str,
    created: str,
    large: bool,
    global_note: bool,
) -> str:
    visibility = "global" if global_note else "project"
    sections = [
        f"# {topic.title()} ({identifier})\n\n"
        f"This {note_type} note records {topic} for the {project} project. "
        f"It is a {visibility} reference for later retrieval.\n",
        "## Decision\n\n"
        f"Use a local, bounded approach for {topic}. Preserve the source path "
        "and verify the current file before using retrieved evidence.\n",
        "## Context\n\n"
        + _paragraph(rng, 44 if large else rng.randint(38, 72))
        + "\n\n"
        + _paragraph(rng, 44 if large else rng.randint(38, 72))
        + "\n",
        "## Evidence\n\n"
        + _paragraph(rng, 44 if large else rng.randint(38, 72))
        + "\n",
    ]
    if large:
        sections.append(
            "## Extended notes\n\n"
            + "\n\n".join(_paragraph(rng, 72) for _ in range(45))
            + "\n"
        )
    return (
        "---\n"
        f"type: {note_type}\n"
        f"updated: {updated}\n"
        f"created: {created}\n"
        f"project: {project}\n"
        f"visibility: {visibility}\n"
        "---\n\n"
        + "\n".join(sections)
    )


def generate_corpus(
    directory: str | Path,
    file_count: int = 2000,
    seed: int = 0,
) -> list[Path]:
    """Write a deterministic synthetic corpus and return its Markdown paths."""

    if file_count < 1:
        raise ValueError("file_count must be positive")
    target = Path(directory).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    large_every = max(100, file_count // 50)
    generated: list[Path] = []

    for number in range(file_count):
        project = PROJECTS[number % len(PROJECTS)]
        project_dir = target / project
        project_dir.mkdir(parents=True, exist_ok=True)
        identifier = f"{project[:4].upper()}-{1000 + number}"
        global_note = number % max(20, file_count // 20 or 1) == 0
        note_name = "global" if global_note else "note"
        path = project_dir / f"{note_name}-{number:05d}.md"
        month = number % 12 + 1
        day = number % 28 + 1
        content = _note_content(
            rng,
            project,
            identifier,
            NOTE_TYPES[number % len(NOTE_TYPES)],
            TOPICS[number % len(TOPICS)],
            f"2026-{month:02d}-{day:02d}",
            f"2025-{month:02d}-{day:02d}",
            number % large_every == 0,
            global_note,
        )
        path.write_text(content, encoding="utf-8")
        generated.append(path)
    return generated
