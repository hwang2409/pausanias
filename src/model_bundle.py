"""Fetch and verify the optional semantic retrieval model bundle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Mapping
from urllib.request import urlopen
import uuid


LOGGER = logging.getLogger(__name__)


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ONNXRUNTIME_VERSION = "1.30.0"
NUMPY_VERSION = "2.3.3"
MODEL_LICENSE_NOTICE = """all-MiniLM-L6-v2 is distributed under the Apache License 2.0.
Source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
License: https://www.apache.org/licenses/LICENSE-2.0
"""
ONNXRUNTIME_LICENSE_NOTICE = """ONNX Runtime is distributed under the MIT License.
Source: https://github.com/microsoft/onnxruntime
License: https://github.com/microsoft/onnxruntime/blob/main/LICENSE
"""


MODEL_BUNDLE_MANIFEST = {
    "format_version": 1,
    "model_id": MODEL_ID,
    "model_version": MODEL_REVISION,
    "model_revision": MODEL_REVISION,
    "tokenizer_version": MODEL_REVISION,
    "model_license": "Apache-2.0",
    "runtime": {"name": "onnxruntime", "version": ONNXRUNTIME_VERSION, "provider": "CPUExecutionProvider"},
    "numpy_version": NUMPY_VERSION,
    "dimension": 384,
    "dtype": "float32",
    "pooling": "mean",
    "normalization": "l2",
    "metric": "cosine",
    "tokenizer": {
        "name": "BertWordPiece",
        "version": MODEL_REVISION,
        "files": ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"],
        "do_lower_case": True,
        "padding": "longest",
        "truncation": "longest_first",
        "max_length": 256,
        "special_tokens": {"cls": "[CLS]", "sep": "[SEP]", "pad": "[PAD]", "unk": "[UNK]"},
    },
    "files": [
        {
            "path": "onnx/model.onnx",
            "url": f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/onnx/model.onnx",
            "size": 90405214,
            "sha256": "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
        },
        {
            "path": "config.json",
            "url": f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/config.json",
            "size": 612,
            "sha256": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
        },
        {
            "path": "tokenizer.json",
            "url": f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/tokenizer.json",
            "size": 466247,
            "sha256": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
        },
        {
            "path": "tokenizer_config.json",
            "url": f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/tokenizer_config.json",
            "size": 350,
            "sha256": "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b",
        },
        {
            "path": "special_tokens_map.json",
            "url": f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/special_tokens_map.json",
            "size": 112,
            "sha256": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
        },
        {
            "path": "vocab.txt",
            "url": f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/vocab.txt",
            "size": 231508,
            "sha256": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
        },
    ],
    "licenses": [
        {
            "name": "all-MiniLM-L6-v2 model and tokenizer",
            "spdx_id": "Apache-2.0",
            "file": "LICENSES/all-MiniLM-L6-v2.txt",
            "notice": MODEL_LICENSE_NOTICE,
        },
        {
            "name": "ONNX Runtime",
            "spdx_id": "MIT",
            "file": "LICENSES/onnxruntime.txt",
            "notice": ONNXRUNTIME_LICENSE_NOTICE,
        },
    ],
}


class BundleError(ValueError):
    """Raised when a model bundle cannot be fetched or verified."""


@dataclass(frozen=True)
class BundleFile:
    path: str
    url: str
    size: int
    sha256: str


def manifest_fingerprint(manifest: Mapping[str, object] = MODEL_BUNDLE_MANIFEST) -> str:
    body = {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _relative_path(raw_path: object, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise BundleError(f"{label} must be a non-empty relative path")
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise BundleError(f"{label} must stay inside the bundle: {raw_path}")
    return path


def _files(manifest: Mapping[str, object]) -> tuple[BundleFile, ...]:
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise BundleError("manifest files must be a non-empty array")
    result: list[BundleFile] = []
    seen: set[str] = set()
    for raw_file in raw_files:
        if not isinstance(raw_file, dict):
            raise BundleError("each manifest file must be a table")
        path = _relative_path(raw_file.get("path"), "manifest file path")
        path_text = path.as_posix()
        if path_text in seen:
            raise BundleError(f"duplicate manifest file: {path_text}")
        seen.add(path_text)
        url = raw_file.get("url")
        size = raw_file.get("size")
        sha256 = raw_file.get("sha256")
        if not isinstance(url, str) or not url:
            raise BundleError(f"manifest URL missing for {path_text}")
        if not isinstance(size, int) or size < 0:
            raise BundleError(f"manifest size invalid for {path_text}")
        if not isinstance(sha256, str) or len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
            raise BundleError(f"manifest SHA-256 invalid for {path_text}")
        result.append(BundleFile(path_text, url, size, sha256))
    return tuple(result)


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _check_file(root: Path, expected: BundleFile) -> None:
    path = root / expected.path
    current = root
    for part in Path(expected.path).parts[:-1]:
        current /= part
        if current.is_symlink():
            raise BundleError(f"bundle file uses a symlinked directory: {expected.path}")
    if path.is_symlink() or not path.is_file():
        raise BundleError(f"bundle file missing: {expected.path}")
    size, digest = _hash_file(path)
    if size != expected.size:
        raise BundleError(f"bundle file size mismatch: {expected.path} (expected {expected.size}, got {size})")
    if digest != expected.sha256:
        raise BundleError(f"bundle file hash mismatch: {expected.path} (expected {expected.sha256}, got {digest})")


def _license_files(manifest: Mapping[str, object]) -> tuple[BundleFile, ...]:
    records = manifest.get("licenses", [])
    if not isinstance(records, list):
        raise BundleError("manifest licenses must be an array")
    result: list[BundleFile] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise BundleError("each manifest license must be a table")
        path = _relative_path(record.get("file"), "license file")
        path_text = path.as_posix()
        if path_text in seen:
            raise BundleError(f"duplicate manifest license: {path_text}")
        seen.add(path_text)
        content = _license_content(record).encode()
        result.append(BundleFile(path_text, "", len(content), hashlib.sha256(content).hexdigest()))
    return tuple(result)


def _verify_bundle_root(root: Path, manifest: Mapping[str, object]) -> None:
    """Verify every declared bundle file and the installed manifest."""
    if not root.is_dir():
        raise BundleError(f"model bundle is not a directory: {root}")
    manifest_path = root / "manifest.json"
    try:
        if manifest_path.is_symlink():
            raise OSError("manifest is a symlink")
        stored = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        raise BundleError(f"bundle manifest is unreadable: {manifest_path}") from exc
    if not isinstance(stored, dict):
        raise BundleError("bundle manifest does not match the pinned manifest")
    expected_fingerprint = manifest_fingerprint(manifest)
    stored_body = {key: value for key, value in stored.items() if key != "manifest_fingerprint"}
    if (stored.get("manifest_fingerprint") != expected_fingerprint
            or manifest_fingerprint(stored_body) != expected_fingerprint):
        raise BundleError("bundle manifest does not match the pinned manifest")
    for expected in _files(manifest):
        _check_file(root, expected)
    for expected in _license_files(manifest):
        _check_file(root, expected)


def _versions_dir(pointer: Path) -> Path:
    return pointer.parent / "bundles" / pointer.name


def _migration_pointer(destination: Path) -> Path:
    return destination.parent / f".{destination.name}.migration-pointer"


def _resolve_pointer(pointer: Path, versions: Path) -> Path:
    if pointer.is_symlink():
        raise BundleError(f"active model bundle pointer is a symlink: {pointer}")
    try:
        stored = json.loads(pointer.read_text())
    except (OSError, ValueError) as exc:
        raise BundleError(f"active model bundle pointer is unreadable: {pointer}") from exc
    if not isinstance(stored, dict) or stored.get("format_version") != 1:
        raise BundleError(f"active model bundle pointer is invalid: {pointer}")
    version = stored.get("version")
    if not isinstance(version, str):
        raise BundleError(f"active model bundle pointer has no version: {pointer}")
    version_path = _relative_path(version, "active bundle version")
    if version_path.parent != Path("."):
        raise BundleError(f"active model bundle pointer has an invalid version: {pointer}")
    root = versions / version_path
    if root.is_symlink() or not root.is_dir():
        raise BundleError(f"active model bundle is missing: {root}")
    return root


def _resolve_active_bundle(pointer: Path) -> Path:
    pointer = Path(pointer)
    if pointer.is_symlink():
        raise BundleError(f"active model bundle pointer is a symlink: {pointer}")
    versions = _versions_dir(pointer)
    migration_pointer = _migration_pointer(pointer)
    if migration_pointer.is_symlink():
        raise BundleError(f"active model bundle pointer is a symlink: {migration_pointer}")
    errors: list[BundleError] = []
    for candidate in (pointer, migration_pointer):
        if candidate.is_file():
            try:
                return _resolve_pointer(candidate, versions)
            except BundleError as exc:
                errors.append(exc)
    if pointer.is_dir():
        return pointer
    if errors:
        raise errors[0]
    raise BundleError(f"active model bundle pointer is missing or not a file: {pointer}")


def _remove_legacy_bundle(destination: Path) -> bool:
    try:
        shutil.rmtree(destination)
    except OSError as exc:
        LOGGER.warning("could not remove legacy model bundle %s: %s", destination, exc)
    return not destination.exists()


def verify_bundle(bundle_dir: str | Path, manifest: Mapping[str, object] = MODEL_BUNDLE_MANIFEST) -> None:
    """Resolve the active pointer and verify every declared bundle file."""
    root = _resolve_active_bundle(Path(bundle_dir).expanduser())
    _verify_bundle_root(root, manifest)


def _license_content(record: dict) -> str:
    notice = record.get("notice")
    if isinstance(notice, str) and notice:
        return notice if notice.endswith("\n") else notice + "\n"
    spdx_id = record.get("spdx_id", "unknown")
    return f"License: {spdx_id}\n"


def _write_licenses(root: Path, manifest: Mapping[str, object]) -> None:
    for expected, record in zip(_license_files(manifest), manifest.get("licenses", []), strict=True):
        target = root / expected.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_license_content(record))


def _download(url: str, target: Path) -> None:
    try:
        with urlopen(url) as response, target.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    except OSError as exc:
        raise BundleError(f"could not download {url}: {exc}") from exc


def fetch_bundle(
    bundle_dir: str | Path,
    manifest: Mapping[str, object] = MODEL_BUNDLE_MANIFEST,
    downloader: Callable[[str, Path], None] = _download,
) -> Path:
    """Download, verify, and atomically publish a model bundle."""
    expected_files = _files(manifest)
    destination = Path(bundle_dir).expanduser()
    if destination.exists() or destination.is_symlink():
        try:
            active = _resolve_active_bundle(destination)
            _verify_bundle_root(active, manifest)
        except BundleError:
            pass
        else:
            if not (destination.is_dir() and active == destination):
                migration_pointer = _migration_pointer(destination)
                if destination.is_dir() and migration_pointer.is_file():
                    if _remove_legacy_bundle(destination):
                        try:
                            os.replace(migration_pointer, destination)
                        except OSError as exc:
                            LOGGER.warning("could not finalize model bundle migration %s: %s", destination, exc)
                return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    versions = _versions_dir(destination)
    versions.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=versions))
    version_dir = versions / f"{manifest_fingerprint(manifest)}-{uuid.uuid4().hex}"
    pointer = destination.parent / f".{destination.name}.pointer-{uuid.uuid4().hex}"
    published = False
    try:
        for expected in expected_files:
            target = staging / expected.path
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                downloader(expected.url, target)
            except BundleError:
                raise
            except OSError as exc:
                raise BundleError(f"could not download {expected.path}: {exc}") from exc
            _check_file(staging, expected)
        _write_licenses(staging, manifest)
        (staging / "manifest.json").write_text(json.dumps(
            {**manifest, "manifest_fingerprint": manifest_fingerprint(manifest)},
            indent=2,
            sort_keys=True,
        ) + "\n")
        _verify_bundle_root(staging, manifest)
        staging.rename(version_dir)
        legacy = destination.is_dir() and not destination.is_symlink()
        active_pointer = _migration_pointer(destination) if legacy else destination
        pointer.write_text(json.dumps({"format_version": 1, "version": version_dir.name}) + "\n")
        os.replace(pointer, active_pointer)
        published = True
        if legacy:
            if not _remove_legacy_bundle(destination):
                return destination
            os.replace(active_pointer, destination)
        return destination
    except OSError as exc:
        raise BundleError(f"could not install model bundle: {exc}") from exc
    except Exception as exc:
        if isinstance(exc, BundleError):
            raise
        raise BundleError(f"could not install model bundle: {exc}") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        pointer.unlink(missing_ok=True)
        if not published:
            shutil.rmtree(version_dir, ignore_errors=True)


def license_records(bundle_dir: str | Path | None = None) -> list[dict]:
    """Return the pinned license records or the verified bundle records."""
    if bundle_dir is None:
        manifest: Mapping[str, object] = MODEL_BUNDLE_MANIFEST
    else:
        root = _resolve_active_bundle(Path(bundle_dir).expanduser())
        _verify_bundle_root(root, MODEL_BUNDLE_MANIFEST)
        try:
            loaded = json.loads((root / "manifest.json").read_text())
        except (OSError, ValueError) as exc:
            raise BundleError(f"bundle manifest is unreadable: {root / 'manifest.json'}") from exc
        if not isinstance(loaded, dict):
            raise BundleError("bundle manifest does not contain license records")
        manifest = loaded
    records = manifest.get("licenses", [])
    return [record for record in records if isinstance(record, dict)]


def package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
