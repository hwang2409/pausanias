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
        if _credential_name(resolved.name):
            return True
        for private in self.private_paths:
            private_path = Path(private).expanduser()
            if private_path.is_absolute() and _resolve(private_path, strict=False) == resolved:
                return True
            relative = resolved.relative_to(root.path).as_posix()
            if (fnmatch.fnmatch(str(resolved), private) or fnmatch.fnmatch(relative, private)
                    or fnmatch.fnmatch(resolved.name, private)):
                return True
        relative = resolved.relative_to(root.path).as_posix()
        return any(fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(resolved.name, pattern)
                   for pattern in root.excludes)


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
    return Config(tuple(roots), frozenset(global_notes), tuple(private_paths), database, section_bytes)


def default_config_path() -> Path:
    return Path(os.environ.get("PAUSANIAS_CONFIG", "~/.config/pausanias/config.toml")).expanduser()
