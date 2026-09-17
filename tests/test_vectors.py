import json
import math
import sqlite3
import struct
from pathlib import Path

import pytest
from pausanias import core
from pausanias import vectors
from pausanias.config import load_config


class FakeEncoder:
    def __init__(self):
        self.texts: list[str] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        vectors = []
        for text in texts:
            vector = [0.0] * 384
            vector[0] = 1.0 if "alpha" in text else 0.0
            vector[1] = 1.0 if "beta" in text else 0.0
            vectors.append(vector)
        return vectors


def make_config(tmp_path: Path, roots: list[tuple[str, str, Path]], global_notes: list[Path] | None = None):
    global_notes = global_notes or []
    lines = [f'database = "{tmp_path / "index.sqlite3"}"']
    if global_notes:
        lines.append("global_notes = [" + ", ".join(f'"{path}"' for path in global_notes) + "]")
    for root_id, project, path in roots:
        lines.extend(["", "[[roots]]", f'id = "{root_id}"', f'project = "{project}"', f'path = "{path}"'])
    config_path = tmp_path / "config.toml"
    config_path.write_text("\n".join(lines))
    return load_config(config_path)


def test_pinned_tokenizer_golden_ids_and_truncation(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    fixture = Path(__file__).parent / "fixtures" / "pinned_tokenizer.json"
    (bundle / "tokenizer.json").write_bytes(fixture.read_bytes())
    tokenizer = core._StdlibTokenizer(bundle)

    # These ids come from the vocabulary and TemplateProcessing rules in the pinned fixture.
    assert tokenizer.tokens("Hello, world!") == [101, 200, 203, 201, 202, 102]
    assert tokenizer.tokens("CAFÉ déjà") == [101, 204, 205, 102]
    assert tokenizer.tokens("special [MASK] chars") == [101, 206, 103, 207, 102]
    assert tokenizer.tokens(" ".join(["long"] * 130)) == [101, *([208] * 126), 102]


def test_pinned_tokenizer_rejects_unsupported_truncation_strategy(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    fixture = json.loads((Path(__file__).parent / "fixtures" / "pinned_tokenizer.json").read_text())
    fixture["truncation"]["strategy"] = "OnlySecond"
    (bundle / "tokenizer.json").write_text(json.dumps(fixture))

    with pytest.raises(core.SemanticError, match="truncation strategy is unsupported"):
        core._StdlibTokenizer(bundle)


def test_vector_store_scans_normalized_float32_vectors_with_scope(tmp_path: Path):
    pytest.importorskip("numpy")
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    (one / "alpha.md").write_text("# Alpha\nalpha memory\n")
    (two / "beta.md").write_text("# Beta\nbeta memory\n")
    config = make_config(tmp_path, [("one", "first", one), ("two", "second", two)])
    encoder = FakeEncoder()

    assert core.index(config, encoder=encoder) == 1
    results = core.semantic_search(config, "alpha paraphrase", project="first", encoder=encoder)

    assert results[0].heading == "Alpha"
    assert all(result.project_scope == "first" for result in results)
    connection = sqlite3.connect(config.database)
    try:
        assert connection.execute("SELECT length(vector) FROM embeddings").fetchall() == [(1536,), (1536,)]
        state = dict(connection.execute("SELECT key, value FROM metadata"))
        assert state["semantic_state"] == "ready"
    finally:
        connection.close()


def test_refresh_reembeds_only_changed_sections(tmp_path: Path):
    pytest.importorskip("numpy")
    root = tmp_path / "vault"
    root.mkdir()
    first = root / "first.md"
    second = root / "second.md"
    first.write_text("# First\nalpha\n")
    second.write_text("# Second\nbeta\n")
    config = make_config(tmp_path, [("vault", "p", root)])
    encoder = FakeEncoder()

    core.index(config, encoder=encoder)
    encoder.texts.clear()
    core.index(config, encoder=encoder)
    assert encoder.texts == []
    first.write_text("# First\nchanged alpha\n")
    core.index(config, encoder=encoder)

    assert encoder.texts == ["changed alpha"]


def test_source_race_keeps_previous_generation_active(tmp_path: Path):
    pytest.importorskip("numpy")
    root = tmp_path / "vault"
    root.mkdir()
    note = root / "note.md"
    note.write_text("# Note\nold alpha\n")
    config = make_config(tmp_path, [("vault", "p", root)])
    core.index(config, encoder=FakeEncoder())
    note.write_text("# Note\nnew alpha\n")

    class RacingEncoder(FakeEncoder):
        def encode(self, texts: list[str]) -> list[list[float]]:
            note.write_text("# Note\nchanged during refresh\n")
            return super().encode(texts)

    assert core.index(config, encoder=RacingEncoder()) == 1
    connection = sqlite3.connect(config.database)
    try:
        assert connection.execute("SELECT value FROM metadata WHERE key = 'generation'").fetchone()[0] == "1"
        assert connection.execute("SELECT text FROM sections").fetchone()[0] == "old alpha"
    finally:
        connection.close()


def test_added_source_race_keeps_previous_generation_active(tmp_path: Path):
    pytest.importorskip("numpy")
    root = tmp_path / "vault"
    root.mkdir()
    note = root / "note.md"
    note.write_text("# Note\nold alpha\n")
    config = make_config(tmp_path, [("vault", "p", root)])
    core.index(config, encoder=FakeEncoder())
    note.write_text("# Note\nnew alpha\n")

    class RacingEncoder(FakeEncoder):
        def encode(self, texts: list[str]) -> list[list[float]]:
            (root / "added.md").write_text("# Added\nnew beta\n")
            return super().encode(texts)

    assert core.index(config, encoder=RacingEncoder()) == 1
    connection = sqlite3.connect(config.database)
    try:
        assert connection.execute("SELECT value FROM metadata WHERE key = 'generation'").fetchone()[0] == "1"
        assert connection.execute("SELECT count(*) FROM files").fetchone()[0] == 1
        assert connection.execute("SELECT text FROM sections").fetchone()[0] == "old alpha"
    finally:
        connection.close()


def test_failed_vector_refresh_publishes_lexical_generation_as_stale(tmp_path: Path):
    pytest.importorskip("numpy")
    root = tmp_path / "vault"
    root.mkdir()
    note = root / "note.md"
    note.write_text("# Note\nold alpha\n")
    config = make_config(tmp_path, [("vault", "p", root)])
    core.index(config, encoder=FakeEncoder())
    note.write_text("# Note\nnew alpha\n")

    class FailingEncoder:
        def encode(self, texts: list[str]):
            raise RuntimeError("encoder interrupted")

    assert core.index(config, encoder=FailingEncoder()) == 2
    connection = sqlite3.connect(config.database)
    try:
        state = dict(connection.execute("SELECT key, value FROM metadata"))
        assert state["semantic_state"] == "stale"
        assert state["semantic_reason"] == "REFRESH_FAILED"
    finally:
        connection.close()


def test_semantic_failure_falls_back_to_lexical_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("# Note\nlexical fallback\n")
    config = make_config(tmp_path, [("vault", "p", root)])
    monkeypatch.setattr(core, "package_version", lambda _: None)

    core.index(config)
    results = core.semantic_search(config, "lexical fallback", project="p")

    assert [result.heading for result in results] == ["Note"]
    connection = sqlite3.connect(config.database)
    try:
        assert connection.execute("SELECT value FROM metadata WHERE key = 'semantic_reason'").fetchone()[0] == "MODEL_MISSING"
    finally:
        connection.close()


def test_corrupt_vector_table_falls_back_without_mutating_database(tmp_path: Path):
    pytest.importorskip("numpy")
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("# Note\nalpha\n")
    config = make_config(tmp_path, [("vault", "p", root)])
    encoder = FakeEncoder()
    core.index(config, encoder=encoder)
    connection = sqlite3.connect(config.database)
    try:
        connection.execute("UPDATE embeddings SET vector = ?", (b"bad",))
        connection.commit()
    finally:
        connection.close()
    before = config.database.read_bytes()

    results = core.semantic_search(config, "alpha", project="p", encoder=encoder)

    assert results[0].heading == "Note"
    assert config.database.read_bytes() == before
    connection = sqlite3.connect(config.database)
    try:
        state = dict(connection.execute("SELECT key, value FROM metadata"))
        assert state["semantic_state"] == "ready"
    finally:
        connection.close()


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_vector_falls_back_without_scoring_nan(tmp_path: Path, bad_value: float):
    pytest.importorskip("numpy")
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("# Note\nalpha\n")
    config = make_config(tmp_path, [("vault", "p", root)])
    core.index(config, encoder=FakeEncoder())
    bad_vector = struct.pack("<384f", bad_value, *([0.0] * 383))
    connection = sqlite3.connect(config.database)
    try:
        connection.execute("UPDATE embeddings SET vector = ?", (bad_vector,))
        connection.commit()
        raw_row = connection.execute(
            "SELECT s.canonical_path, e.vector FROM sections s JOIN embeddings e ON e.section_id = s.section_id"
        ).fetchone()
    finally:
        connection.close()

    import numpy
    row = {"canonical_path": raw_row[0], "vector": raw_row[1]}
    refresh: set[str] = set()
    matrix, valid_rows = vectors.load_vector_matrix(numpy, [row], 384, refresh)
    assert matrix.shape == (0, 384)
    assert valid_rows == []
    assert refresh == {str((root / "note.md").resolve())}

    refresh = set()
    results = core.semantic_search(config, "alpha", project="p", refresh=refresh, encoder=FakeEncoder())

    assert [result.heading for result in results] == ["Note"]
    assert refresh == {str((root / "note.md").resolve())}
    assert all(math.isfinite(result.score) for result in results)


def test_default_search_does_not_use_vectors(tmp_path: Path):
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("# Note\nalpha\n")
    config = make_config(tmp_path, [("vault", "p", root)])
    core.index(config)

    assert core.search(config, "alpha", project="p") == core.search(config, "alpha", project="p", semantic=False)
