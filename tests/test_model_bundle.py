from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pausanias.cli import main
from pausanias.config import load_config
import pausanias.model_bundle as model_bundle
from pausanias.model_bundle import BundleError, fetch_bundle, manifest_fingerprint, verify_bundle


def fixture_manifest(source: Path) -> dict:
    payload = (source / "model.bin").read_bytes()
    return {
        "format_version": 1,
        "model_id": "fixture/model",
        "model_version": "fixture-revision",
        "files": [{
            "path": "model.bin",
            "url": (source / "model.bin").as_uri(),
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }],
        "licenses": [{
            "name": "fixture model",
            "spdx_id": "Apache-2.0",
            "file": "LICENSES/model.txt",
            "notice": "fixture license\n",
        }],
    }


def test_fixture_bundle_is_verified_and_idempotent(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.bin").write_bytes(b"known model bytes")
    manifest = fixture_manifest(source)
    destination = tmp_path / "bundle"
    downloads: list[str] = []

    def downloader(url: str, target: Path) -> None:
        downloads.append(url)
        target.write_bytes(Path(url.removeprefix("file://")).read_bytes())

    assert fetch_bundle(destination, manifest, downloader) == destination
    verify_bundle(destination, manifest)
    assert (destination / "LICENSES/model.txt").read_text() == "fixture license\n"
    assert (destination / "manifest.json").read_text()
    assert fetch_bundle(destination, manifest, downloader) == destination
    assert len(downloads) == 1


def test_hash_mismatch_names_file_and_does_not_publish(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.bin").write_bytes(b"unexpected bytes")
    manifest = fixture_manifest(source)
    manifest["files"][0]["sha256"] = "0" * 64

    with pytest.raises(BundleError, match="model.bin"):
        fetch_bundle(tmp_path / "bundle", manifest)
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle.*"))


def test_failed_replacement_keeps_previous_complete_bundle(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.bin").write_bytes(b"old bytes")
    manifest = fixture_manifest(source)
    destination = tmp_path / "bundle"
    fetch_bundle(destination, manifest)

    changed = tmp_path / "changed"
    changed.mkdir()
    (changed / "model.bin").write_bytes(b"new bytes")
    changed_manifest = fixture_manifest(changed)

    def fail_download(url: str, target: Path) -> None:
        target.write_bytes(b"partial")

    with pytest.raises(BundleError, match="model.bin"):
        fetch_bundle(destination, changed_manifest, fail_download)
    verify_bundle(destination, manifest)
    assert (destination / "model.bin").read_bytes() == b"old bytes"


def test_interrupted_pointer_swap_keeps_active_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.bin").write_bytes(b"old bytes")
    manifest = fixture_manifest(source)
    destination = tmp_path / "bundle"
    fetch_bundle(destination, manifest)

    changed = tmp_path / "changed"
    changed.mkdir()
    (changed / "model.bin").write_bytes(b"new bytes")
    changed_manifest = fixture_manifest(changed)
    real_replace = model_bundle.os.replace

    def interrupt(source_path: str | Path, destination_path: str | Path) -> None:
        if Path(destination_path) == destination:
            raise OSError("interrupted pointer swap")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(model_bundle.os, "replace", interrupt)
    with pytest.raises(BundleError, match="pointer swap"):
        fetch_bundle(destination, changed_manifest)

    verify_bundle(destination, manifest)
    assert (destination / "model.bin").read_bytes() == b"old bytes"


def test_tampered_manifest_body_is_rejected(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.bin").write_bytes(b"known model bytes")
    manifest = fixture_manifest(source)
    destination = tmp_path / "bundle"
    fetch_bundle(destination, manifest)

    stored = json.loads((destination / "manifest.json").read_text())
    stored["model_id"] = "fixture/tampered"
    (destination / "manifest.json").write_text(json.dumps(stored))

    with pytest.raises(BundleError, match="manifest"):
        verify_bundle(destination, manifest)


def test_tampered_license_is_rejected(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.bin").write_bytes(b"known model bytes")
    manifest = fixture_manifest(source)
    destination = tmp_path / "bundle"
    fetch_bundle(destination, manifest)
    (destination / "LICENSES/model.txt").write_text("tampered license\n")

    with pytest.raises(BundleError, match="LICENSES/model.txt"):
        verify_bundle(destination, manifest)


def test_config_reads_semantic_bundle_dir(tmp_path: Path):
    root = tmp_path / "notes"
    root.mkdir()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[semantic]\n"
        'bundle_dir = "bundles/model"\n\n'
        "[[roots]]\n"
        'id = "notes"\nproject = "test"\npath = "notes"\n'
    )
    config = load_config(config_path)
    assert config.bundle_dir == (tmp_path / "bundles/model").resolve()


def test_license_command_works_without_config(capsys):
    assert main(["model", "license"]) == 0
    output = capsys.readouterr().out
    assert "Apache-2.0" in output
    assert "MIT" in output


def test_license_command_uses_configured_bundle(tmp_path: Path, capsys):
    source = tmp_path / "source"
    source.mkdir()
    config = tmp_path / "config.toml"
    config.write_text(
        "[semantic]\n"
        'bundle_dir = "missing-bundle"\n\n'
        "[[roots]]\n"
        'id = "notes"\nproject = "test"\npath = "source"\n'
    )

    assert main(["--config", str(config), "model", "license"]) == 2
    assert "not a directory" in capsys.readouterr().err


def test_status_reports_missing_optional_model(tmp_path: Path, capsys):
    bundle = tmp_path / "missing-bundle"

    assert main(["model", "status", "--bundle-dir", str(bundle)]) == 0
    output = capsys.readouterr().out
    assert "extra installed: no" in output
    assert "bundle present: no" in output
    assert "manifest valid: no" in output
    assert "onnxruntime: installed=missing expected=1.30.0" in output


def test_manifest_fingerprint_is_stable(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.bin").write_bytes(b"bytes")
    manifest = fixture_manifest(source)
    assert manifest_fingerprint(manifest) == manifest_fingerprint(json.loads(json.dumps(manifest)))
