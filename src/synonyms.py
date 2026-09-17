"""Versioned, operator-maintained lexical synonym tables."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import tomllib


MAX_SYNONYM_ALIASES = 4
MAX_SYNONYM_ENTRIES = 128
MAX_SYNONYM_VARIANTS = 16
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9]*-\d+|[\w]+", flags=re.UNICODE)
TICKET_PATTERN = re.compile(r"[a-z][a-z0-9]*-\d+", flags=re.IGNORECASE)


class SynonymTableError(ValueError):
    """Raised when an operator synonym table is invalid."""


@dataclass(frozen=True)
class SynonymTable:
    version: int | None
    fingerprint: str | None
    groups: tuple[tuple[str, tuple[str, ...]], ...]

    def alternatives(self, token: str) -> tuple[str, ...]:
        normalized = token.casefold()
        for canonical, aliases in self.groups:
            forms = (canonical, *aliases)
            if normalized in forms:
                return tuple(form for form in forms if form != normalized)
        return ()


EMPTY_TABLE = SynonymTable(None, None, ())


def _ordinary_term(value: str, field: str) -> str:
    normalized = " ".join(value.strip().casefold().split())
    tokens = TOKEN_PATTERN.findall(normalized)
    if len(tokens) != 1 or tokens[0] != normalized or TICKET_PATTERN.fullmatch(normalized):
        raise SynonymTableError(f"{field} must be one ordinary lexical term")
    return normalized


def _fingerprint(version: int, groups: tuple[tuple[str, tuple[str, ...]], ...]) -> str:
    payload = {"version": version, "terms": {key: list(value) for key, value in groups}}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_synonym_table(path: Path | None) -> SynonymTable:
    if path is None or not path.exists():
        return EMPTY_TABLE
    try:
        with path.open("rb") as handle:
            document = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise SynonymTableError(f"cannot read synonym table {path}: {exc}") from exc
    version = document.get("version")
    terms = document.get("terms")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise SynonymTableError("synonym table version must be a positive integer")
    if not isinstance(terms, dict):
        raise SynonymTableError("synonym table terms must be a table")
    if len(terms) > MAX_SYNONYM_ENTRIES:
        raise SynonymTableError(f"synonym table cannot exceed {MAX_SYNONYM_ENTRIES} terms")
    groups: list[tuple[str, tuple[str, ...]]] = []
    seen: set[str] = set()
    for raw_canonical, raw_aliases in terms.items():
        if not isinstance(raw_canonical, str):
            raise SynonymTableError("synonym table term names must be strings")
        canonical = _ordinary_term(raw_canonical, "canonical term")
        if canonical in seen:
            raise SynonymTableError(f"duplicate synonym term: {canonical}")
        if not isinstance(raw_aliases, list) or not all(isinstance(alias, str) for alias in raw_aliases):
            raise SynonymTableError(f"aliases for {canonical} must be an array of strings")
        if len(raw_aliases) > MAX_SYNONYM_ALIASES:
            raise SynonymTableError(
                f"aliases for {canonical} cannot exceed {MAX_SYNONYM_ALIASES} entries"
            )
        aliases = tuple(_ordinary_term(alias, "synonym alias") for alias in raw_aliases)
        if len(set(aliases)) != len(aliases) or canonical in aliases:
            raise SynonymTableError(f"synonym forms for {canonical} must be unique")
        forms = (canonical, *aliases)
        overlap = seen.intersection(forms)
        if overlap:
            raise SynonymTableError(f"synonym form is used by multiple terms: {sorted(overlap)[0]}")
        seen.update(forms)
        groups.append((canonical, aliases))
    ordered = tuple(sorted(groups))
    return SynonymTable(version, _fingerprint(version, ordered), ordered)


def table_metadata(path: Path | None) -> dict[str, object]:
    table = load_synonym_table(path)
    return {
        "configured": path is not None,
        "path": str(path) if path is not None else None,
        "version": table.version,
        "fingerprint": table.fingerprint,
        "entry_count": len(table.groups),
    }


def query_variants(query: str, table: SynonymTable) -> tuple[str, ...]:
    if not table.groups:
        return ()
    quoted_ranges = [(match.start(), match.end()) for match in re.finditer(r'"[^"\n]*"', query)]
    occurrences = [
        match for match in TOKEN_PATTERN.finditer(query)
        if not any(start <= match.start() < end for start, end in quoted_ranges)
        and not TICKET_PATTERN.fullmatch(match.group())
    ]
    variants: list[str] = []
    seen: set[str] = set()
    for match in occurrences:
        for alternative in table.alternatives(match.group()):
            variant = query[:match.start()] + alternative + query[match.end():]
            normalized = " ".join(variant.casefold().split())
            if normalized not in seen:
                seen.add(normalized)
                variants.append(variant)
            if len(variants) >= MAX_SYNONYM_VARIANTS:
                return tuple(variants)
    return tuple(variants)
