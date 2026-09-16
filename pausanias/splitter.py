from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re


@dataclass(frozen=True)
class Frontmatter:
    note_type: str | None = None
    updated: str | None = None
    created: str | None = None


@dataclass(frozen=True)
class Section:
    section_id: str
    heading: str | None
    heading_path: tuple[str, ...]
    line_start: int
    line_end: int
    text: str


@dataclass(frozen=True)
class SplitDocument:
    frontmatter: Frontmatter
    sections: tuple[Section, ...]
    content_hash: str


HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
LINK = re.compile(r"!?(?:\[[^\]]*\]\(([^)\s]+)[^)]*\)|\[\[([^|\]#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\])")


def parse_frontmatter(lines: list[str]) -> tuple[Frontmatter, int]:
    if not lines or lines[0].strip() != "---":
        return Frontmatter(), 0
    end = next((index for index in range(1, len(lines)) if lines[index].strip() in {"---", "..."}), None)
    if end is None:
        return Frontmatter(), 0
    values: dict[str, str] = {}
    for line in lines[1:end]:
        key, separator, value = line.partition(":")
        if separator and key.strip() in {"type", "note_type", "updated", "created"}:
            clean = value.strip().strip('"\'')
            if clean and clean.lower() not in {"null", "none", "~"}:
                values[key.strip()] = clean
    return Frontmatter(values.get("type", values.get("note_type")), values.get("updated"), values.get("created")), end + 1


def _split_body(lines: list[str], start: int, end: int, max_bytes: int) -> list[tuple[int, int, str]]:
    body = lines[start:end]
    full = "".join(body).strip()
    if len(full.encode("utf-8")) <= max_bytes:
        return [(start, end, full)] if full else []

    chunks: list[tuple[int, int, str]] = []
    chunk_start = start
    current: list[str] = []
    current_bytes = 0
    for index in range(start, end):
        line = lines[index]
        if len(line.encode("utf-8")) > max_bytes:
            if current:
                text = "".join(current).strip()
                if text:
                    chunks.append((chunk_start, index, text))
                current = []
                current_bytes = 0
            piece: list[str] = []
            piece_bytes = 0
            for character in line:
                character_bytes = len(character.encode("utf-8"))
                if piece and piece_bytes + character_bytes > max_bytes:
                    chunks.append((index, index + 1, "".join(piece)))
                    piece = []
                    piece_bytes = 0
                piece.append(character)
                piece_bytes += character_bytes
            if piece:
                chunks.append((index, index + 1, "".join(piece).strip()))
            chunk_start = index + 1
            continue
        line_bytes = len(line.encode("utf-8"))
        if current and current_bytes + line_bytes > max_bytes:
            text = "".join(current).strip()
            if text:
                chunks.append((chunk_start, index, text))
            chunk_start = index
            current = []
            current_bytes = 0
        current.append(line)
        current_bytes += line_bytes
    text = "".join(current).strip()
    if text:
        chunks.append((chunk_start, end, text))
    return chunks


def split_markdown(path: str, content: str, max_bytes: int = 12000) -> SplitDocument:
    lines = content.splitlines(keepends=True)
    frontmatter, content_start = parse_frontmatter(lines)
    headings: list[tuple[int, int, str]] = []
    for index in range(content_start, len(lines)):
        match = HEADING.match(lines[index].rstrip("\r\n"))
        if match:
            headings.append((index, len(match.group(1)), match.group(2).strip()))

    boundaries: list[tuple[int, int, tuple[str, ...], str | None, int]] = []
    stack: list[tuple[int, str]] = []
    section_start = content_start
    for index, level, heading in headings:
        if index > section_start:
            ancestry = tuple(item[1] for item in stack)
            text_start = section_start + 1 if stack else section_start
            boundaries.append((text_start, index, ancestry, stack[-1][1] if stack else None, section_start))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
        section_start = index
    if section_start < len(lines) or not boundaries:
        ancestry = tuple(item[1] for item in stack)
        text_start = section_start + 1 if stack else section_start
        boundaries.append((text_start, len(lines), ancestry, stack[-1][1] if stack else None, section_start))

    sections: list[Section] = []
    canonical = str(path)
    occurrences: dict[str, int] = {}
    for body_start, body_end, ancestry, heading, display_start in boundaries:
        parts = _split_body(lines, body_start, body_end, max_bytes)
        if not parts and heading is not None:
            parts = [(body_start, body_end, "")]
        heading_key = " > ".join(ancestry) or "(document)"
        occurrences[heading_key] = occurrences.get(heading_key, 0) + 1
        occurrence = occurrences[heading_key]
        for part_index, (part_start, part_end, text) in enumerate(parts):
            identity = heading_key if occurrence == 1 else f"{heading_key}~{occurrence}"
            suffix = f"#chunk-{part_index + 1}" if len(parts) > 1 else ""
            section_id = hashlib.sha256(f"{canonical}\0{identity}{suffix}".encode()).hexdigest()
            line_start = display_start + 1 if part_index == 0 else part_start + 1
            sections.append(Section(section_id, heading, ancestry, line_start, part_end, text))
    return SplitDocument(frontmatter, tuple(sections), hashlib.sha256(content.encode("utf-8")).hexdigest())


def explicit_links(content: str) -> list[str]:
    result: list[str] = []
    for match in LINK.finditer(content):
        value = match.group(1) or match.group(2)
        if value and value not in result:
            result.append(value)
    return result
