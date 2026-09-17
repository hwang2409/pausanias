from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import fnmatch
import os
import tomllib


class ConfigError(ValueError):
    """Raised when the configuration is missing or unsafe."""


@dataclass(frozen=True)
class Root:
    id: str
    path: Path
    project: str
    excludes: tuple[str, ...]


@dataclass(frozen=True)
class Config:
    roots: tuple[Root, ...]
    global_notes: frozenset[Path]
    private_paths: tuple[str, ...]
    database: Path
    section_bytes: int = 12000
    semantic_bundle: Path = Path("~/.cache/pausanias/models/all-MiniLM-L6-v2")
    synonym_table: Path | None = None

    @property
    def bundle_dir(self) -> Path:
        return self.semantic_bundle

    def root_for(self, path: Path) -> Root | None:
        resolved = _resolve(path, strict=False)
        for root in self.roots:
            if _contained(resolved, root.path):
                return root
        return None

    def is_global(self, path: Path) -> bool:
        return _resolve(path, strict=False) in self.global_notes

    def is_excluded(self, path: Path, root: Root) -> bool:
        resolved = _resolve(path, strict=False)
        if not _contained(resolved, root.path):
            return True
        relative_parts = resolved.relative_to(root.path).parts
        for end in range(1, len(relative_parts) + 1):
            ancestor = root.path.joinpath(*relative_parts[:end])
            relative = ancestor.relative_to(root.path).as_posix()
            if _credential_name(ancestor.name):
                return True
            for private in self.private_paths:
                private_path = Path(private).expanduser()
                if private_path.is_absolute() and _resolve(private_path, strict=False) == ancestor:
                    return True
                if (fnmatch.fnmatch(str(ancestor), private)
                        or fnmatch.fnmatch(relative, private)
                        or fnmatch.fnmatch(ancestor.name, private)):
                    return True
            if any(fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(ancestor.name, pattern)
                   for pattern in root.excludes):
                return True
        return False


def _resolve(path: Path, *, strict: bool) -> Path:
    try:
        return path.expanduser().resolve(strict=strict)
    except OSError as exc:
        raise ConfigError(f"cannot resolve path {path}: {exc}") from exc


def _contained(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _credential_name(name: str) -> bool:
    return name.startswith(".env")


def _read_root_entries(raw: dict) -> list[dict]:
    roots = raw.get("roots", [])
    if not isinstance(roots, list):
        raise ConfigError("roots must be an array of tables")
    return roots


def _global_paths(raw: dict) -> list[str]:
    values = raw.get("global_notes", [])
    if not isinstance(values, list):
        raise ConfigError("global_notes must be an array of paths")
    result: list[str] = []
    for value in values:
        if isinstance(value, str):
            result.append(value)
        elif isinstance(value, dict) and isinstance(value.get("path"), str):
            result.append(value["path"])
        else:
            raise ConfigError("each global note must be a path")
    for entry in raw.get("roots", []):
        if isinstance(entry, dict) and "global_notes" in entry:
            nested = entry["global_notes"]
            if not isinstance(nested, list) or not all(isinstance(item, str) for item in nested):
                raise ConfigError("root global_notes must be an array of paths")
            result.extend(nested)
    return result


def load_config(path: str | Path) -> Config:
    config_path = Path(path).expanduser()
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"cannot read config {config_path}: {exc}") from exc

    roots: list[Root] = []
    seen_ids: set[str] = set()
    for entry in _read_root_entries(raw):
        if not isinstance(entry, dict):
            raise ConfigError("each root must be a table")
        root_id = entry.get("id")
        project = entry.get("project", entry.get("project_scope"))
        root_path = entry.get("path")
        if not all(isinstance(value, str) and value for value in (root_id, project, root_path)):
            raise ConfigError("each root needs non-empty id, path, and project")
        if root_id in seen_ids:
            raise ConfigError(f"duplicate root id: {root_id}")
        seen_ids.add(root_id)
        root_candidate = Path(root_path).expanduser()
        if not root_candidate.is_absolute():
            root_candidate = config_path.parent / root_candidate
        resolved = _resolve(root_candidate, strict=True)
        if not resolved.is_dir():
            raise ConfigError(f"root is not a directory: {root_path}")
        if any(_contained(resolved, existing.path) or _contained(existing.path, resolved)
               for existing in roots):
            raise ConfigError(f"root overlaps another configured root: {root_path}")
        excludes = entry.get("exclude", entry.get("exclude_globs", entry.get("excludes", [])))
        if not isinstance(excludes, list) or not all(isinstance(item, str) for item in excludes):
            raise ConfigError(f"excludes for root {root_id} must be an array of globs")
        roots.append(Root(root_id, resolved, project, tuple(excludes)))

    if not roots:
        raise ConfigError("config must declare at least one root")

    private_paths = raw.get("private_paths", raw.get("private", []))
    if not isinstance(private_paths, list) or not all(isinstance(item, str) for item in private_paths):
        raise ConfigError("private_paths must be an array")

    provisional = Config(tuple(roots), frozenset(), tuple(private_paths), Path("."), 12000)
    global_notes: set[Path] = set()
    for raw_note in _global_paths(raw):
        note_candidate = Path(raw_note).expanduser()
        if not note_candidate.is_absolute():
            note_candidate = config_path.parent / note_candidate
        note = _resolve(note_candidate, strict=True)
        root = next((item for item in roots if _contained(note, item.path)), None)
        if root is None or not note.is_file() or note.suffix.lower() not in {".md", ".markdown"}:
            raise ConfigError(f"global note is not an allowed markdown file: {raw_note}")
        if provisional.is_excluded(note, root):
            raise ConfigError(f"global note is excluded: {raw_note}")
        global_notes.add(note)

    database_raw = raw.get("database", raw.get("index", {}).get("database") if isinstance(raw.get("index"), dict) else None)
    if isinstance(database_raw, str):
        database_candidate = Path(database_raw).expanduser()
        if not database_candidate.is_absolute():
            database_candidate = config_path.parent / database_candidate
        database = _resolve(database_candidate, strict=False)
    else:
        database = config_path.with_name("index.sqlite3").resolve()
    section_bytes = raw.get("section_bytes", 12000)
    if not isinstance(section_bytes, int) or section_bytes < 1:
        raise ConfigError("section_bytes must be a positive integer")
    semantic = raw.get("semantic", {})
    if not isinstance(semantic, dict):
        raise ConfigError("semantic must be a table")
    bundle_raw = semantic.get(
        "bundle_dir",
        raw.get("semantic_bundle", raw.get("model_bundle", "~/.cache/pausanias/models/all-MiniLM-L6-v2")),
    )
    if not isinstance(bundle_raw, str) or not bundle_raw:
        raise ConfigError("semantic.bundle_dir must be a non-empty path")
    bundle_candidate = Path(bundle_raw).expanduser()
    if not bundle_candidate.is_absolute():
        bundle_candidate = config_path.parent / bundle_candidate
    bundle_dir = _resolve(bundle_candidate, strict=False)
    synonym_raw = raw.get("synonym_table")
    if synonym_raw is not None and (not isinstance(synonym_raw, str) or not synonym_raw):
        raise ConfigError("synonym_table must be a non-empty path")
    synonym_table = None
    if synonym_raw is not None:
        synonym_candidate = Path(synonym_raw).expanduser()
        if not synonym_candidate.is_absolute():
            synonym_candidate = config_path.parent / synonym_candidate
        synonym_table = _resolve(synonym_candidate, strict=False)
    return Config(tuple(roots), frozenset(global_notes), tuple(private_paths), database, section_bytes, bundle_dir, synonym_table)


def default_config_path() -> Path:
    return Path(os.environ.get("PAUSANIAS_CONFIG", "~/.config/pausanias/config.toml")).expanduser()
