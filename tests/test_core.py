from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sqlite3
import threading
import time

import pytest

import pausanias.core as core
from pausanias.cli import main
from pausanias.config import Config, ConfigError, load_config
from pausanias.core import connect, index, read_source, search
from pausanias.splitter import explicit_links, split_markdown


def make_config(
    tmp_path: Path,
    roots: list[tuple[str, str, Path]],
    global_notes: list[Path] | None = None,
    private: list[str] | None = None,
    semantic_bundle: Path | None = None,
):
    global_notes = global_notes or []
    private = private or []
    database = tmp_path / "index.sqlite3"
    lines = [f'database = "{database}"']
    if semantic_bundle is not None:
        lines += ["", f'semantic_bundle = "{semantic_bundle}"']
    if global_notes:
        lines += ["", "global_notes = [" + ", ".join(f'"{item}"' for item in global_notes) + "]"]
    if private:
        lines += ["", "private_paths = [" + ", ".join(f'"{item}"' for item in private) + "]"]
    for root_id, project, path in roots:
        lines += ["", "[[roots]]", f'id = "{root_id}"', f'project = "{project}"', f'path = "{path}"']
    config_path = tmp_path / "config.toml"
    config_path.write_text("\n".join(lines))
    return load_config(config_path)


def semantic_extra_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("numpy", "onnxruntime"))


def create_v1_index(database: Path, canonical_path: str, root_id: str) -> None:
    connection = sqlite3.connect(database)
    connection.executescript("""
    CREATE TABLE files (
        canonical_path TEXT PRIMARY KEY,
        root_id TEXT NOT NULL,
        mtime_ns INTEGER NOT NULL,
        content_hash TEXT NOT NULL
    );
    CREATE TABLE sections (
        section_id TEXT PRIMARY KEY,
        canonical_path TEXT NOT NULL,
        heading TEXT,
        heading_path TEXT NOT NULL,
        line_start INTEGER NOT NULL,
        line_end INTEGER NOT NULL,
        root_id TEXT NOT NULL,
        text TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        index_generation INTEGER NOT NULL,
        indexed_at REAL NOT NULL,
        note_type TEXT,
        updated_date TEXT,
        created_date TEXT,
        links TEXT NOT NULL
    );
    CREATE INDEX sections_path_idx ON sections(canonical_path);
    CREATE VIRTUAL TABLE sections_fts USING fts5(section_id UNINDEXED, heading, text);
    CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    connection.execute("PRAGMA user_version = 1")
    connection.execute(
        "INSERT INTO files VALUES (?, ?, ?, ?)",
        (canonical_path, root_id, 1, "old-v1-file-hash"),
    )
    connection.execute(
        """INSERT INTO sections
           (section_id, canonical_path, heading, heading_path, line_start,
            line_end, root_id, text, content_hash, index_generation,
            indexed_at, note_type, updated_date, created_date, links)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "old-v1-section",
            canonical_path,
            "Legacy",
            "Legacy",
            1,
            2,
            root_id,
            "legacy v1 data",
            "old-v1-content-hash",
            4,
            1.0,
            None,
            None,
            None,
            "[]",
        ),
    )
    connection.execute(
        "INSERT INTO sections_fts(section_id, heading, text) VALUES (?, ?, ?)",
        ("old-v1-section", "Legacy", "legacy v1 data"),
    )
    connection.execute("INSERT INTO metadata VALUES ('generation', '4')")
    connection.commit()
    connection.close()


def test_splitter_preserves_ancestry_and_frontmatter():
    document = split_markdown("/notes/a.md", """---
type: decision
updated: 2026-01-02
created: 2025-12-01
---
# Parent
intro
## Child
detail
""")
    assert document.frontmatter.note_type == "decision"
    assert document.frontmatter.updated == "2026-01-02"
    assert document.frontmatter.created == "2025-12-01"
    assert [section.heading_path for section in document.sections] == [("Parent",), ("Parent", "Child")]
    assert document.sections[0].line_start == 6


def test_splitter_keeps_missing_dates_unknown_and_caps_long_sections():
    document = split_markdown("/notes/a.md", "# A\n" + ("x" * 40) + "\n", max_bytes=10)
    assert document.frontmatter.updated is None
    assert document.frontmatter.created is None
    assert len(document.sections) > 1
    assert all(len(section.text.encode()) <= 10 for section in document.sections)
    assert all(section.heading_path == ("A",) for section in document.sections)


def test_splitter_ignores_fenced_headings_and_preserves_trailing_hashes():
    document = split_markdown("/notes/a.md", """# Real
```
# not a heading
```
~~~
# also not a heading
~~~
# C#
# Trailing ###
""")
    assert [section.heading for section in document.sections] == ["Real", "C#", "Trailing"]


def test_splitter_requires_matching_closing_fence_without_info_string():
    document = split_markdown("/notes/a.md", """# Real
```
```lang
# not a heading
~~~
# still not a heading
```
# After
""")
    assert [section.heading for section in document.sections] == ["Real", "After"]

    unclosed = split_markdown("/notes/a.md", """# Real
```
# not a heading
""")
    assert [section.heading for section in unclosed.sections] == ["Real"]


def test_splitter_minimum_byte_cap_keeps_codepoints_intact():
    document = split_markdown("/notes/a.md", "# A\né😀\n", max_bytes=1)
    assert "é" in "".join(section.text for section in document.sections)
    assert "😀" in "".join(section.text for section in document.sections)
    assert all(len(section.text.encode()) <= 4 for section in document.sections)


def test_links_and_index_lifecycle(tmp_path: Path):
    root = tmp_path / "vault"
    root.mkdir()
    note = root / "note.md"
    note.write_text("# Decision\nKeep the old plan. [source](other.md) and [[second]].\n")
    config = make_config(tmp_path, [("vault", "phoebe", root)])
    assert explicit_links(note.read_text()) == ["other.md", "second"]
    generation = index(config)
    connection = connect(config.database)
    row = connection.execute("SELECT * FROM sections").fetchone()
    assert row["index_generation"] == generation
    assert json.loads(row["links"]) == ["other.md", "second"]
    connection.close()
    assert search(config, "old plan", project="phoebe")[0].canonical_path == str(note.resolve())
    before = connect(config.database).execute("SELECT count(*) FROM sections").fetchone()[0]
    second_generation = index(config)
    after = connect(config.database).execute("SELECT count(*) FROM sections").fetchone()[0]
    assert before == after
    assert all(row[0] == second_generation for row in connect(config.database).execute("SELECT index_generation FROM sections"))
    note.write_text("# Decision\nUse the new plan.\n")
    index(config)
    assert not search(config, "old plan", project="phoebe")
    assert search(config, "new plan", project="phoebe")
    note.unlink()
    index(config)
    assert not search(config, "new plan", project="phoebe")


def test_index_creates_missing_database_parent(tmp_path: Path):
    root = tmp_path / "vault"
    root.mkdir()
    note = root / "note.md"
    note.write_text("# First run\nCreate the database parent.\n")
    database = tmp_path / "missing" / "nested" / "index.sqlite3"
    config = make_config(tmp_path, [("vault", "phoebe", root)])
    config = Config(config.roots, config.global_notes, config.private_paths, database, config.section_bytes)

    assert index(config) == 1
    assert database.exists()
    assert search(config, "database parent", project="phoebe")


def test_schema_v2_records_missing_model_generation_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("# Schema\nrecord generation metadata\n")
    bundle = tmp_path / "invalid-bundle"
    bundle.mkdir()
    config = make_config(tmp_path, [("vault", "phoebe", root)], semantic_bundle=bundle)
    monkeypatch.setattr(core, "package_version", lambda name: {
        "numpy": core.MODEL_BUNDLE_MANIFEST["numpy_version"],
        "onnxruntime": core.MODEL_BUNDLE_MANIFEST["runtime"]["version"],
    }[name])

    assert index(config) == 1

    connection = connect(config.database, initialize=False)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        state = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
        assert state["generation"] == "1"
        assert state["semantic_generation"] == "1"
        assert state["semantic_state"] == "disabled"
        assert state["semantic_reason"] == "MODEL_MISSING"

        metadata = connection.execute("SELECT * FROM embedding_metadata").fetchone()
        assert metadata["generation"] == 1
        assert metadata["semantic_state"] == "disabled"
        assert metadata["semantic_reason"] == "MODEL_MISSING"
        assert metadata["model_id"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert metadata["model_hash"]
        assert metadata["tokenizer_fingerprint"]
        assert metadata["runtime_version"] == "1.30.0"
        assert metadata["dimension"] == 384
        assert metadata["embedding_version"] == 1
        assert metadata["embedding_format_version"] == 1
        assert "vector" not in {row[1] for row in connection.execute("PRAGMA table_info(embedding_metadata)")}
    finally:
        connection.close()


def test_tokenizer_fingerprint_includes_artifact_hashes():
    baseline = core._tokenizer_fingerprint(core.MODEL_BUNDLE_MANIFEST)
    modified = json.loads(json.dumps(core.MODEL_BUNDLE_MANIFEST))
    tokenizer_file = next(file for file in modified["files"] if file["path"] == "tokenizer.json")
    tokenizer_file["sha256"] = "0" * 64

    assert core._tokenizer_fingerprint(modified) != baseline


@pytest.mark.parametrize("rebuild", [False, True])
def test_real_v1_index_schema_is_recreated_before_indexing(tmp_path: Path, rebuild: bool):
    root = tmp_path / "vault"
    root.mkdir()
    note = root / "note.md"
    note.write_text("# Decision\nUse the new schema.\n")
    bundle = tmp_path / "invalid-bundle"
    bundle.mkdir()
    config = make_config(tmp_path, [("vault", "phoebe", root)], semantic_bundle=bundle)
    create_v1_index(config.database, str(note.resolve()), "vault")

    generation = index(config, rebuild=rebuild)

    connection = connect(config.database, initialize=False)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(sections)")}
    assert "project_scope" not in columns
    assert "is_global" not in columns
    assert connection.execute("PRAGMA user_version").fetchone()[0] == core.SCHEMA_VERSION
    assert connection.execute("SELECT index_generation FROM sections").fetchone()[0] == generation
    state = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
    assert state["semantic_generation"] == str(generation)
    assert state["semantic_state"] == "disabled"
    expected_reason = "MODEL_MISSING" if semantic_extra_available() else "EXTRA_MISSING"
    assert state["semantic_reason"] == expected_reason
    assert connection.execute("SELECT generation FROM embedding_metadata").fetchone()[0] == generation
    assert connection.execute("SELECT count(*) FROM sections WHERE section_id = 'old-v1-section'").fetchone()[0] == 0
    connection.close()
    assert search(config, "new schema", project="phoebe")


def test_search_rejects_v1_without_mutating_database(tmp_path: Path):
    root = tmp_path / "vault"
    root.mkdir()
    note = root / "note.md"
    note.write_text("# Legacy\nlegacy v1 data\n")
    config = make_config(tmp_path, [("vault", "phoebe", root)])
    create_v1_index(config.database, str(note.resolve()), "vault")
    before = config.database.read_bytes()

    with pytest.raises(ValueError, match=r"index schema is v1; run `pausanias index` to rebuild"):
        search(config, "legacy", project="phoebe")

    assert config.database.read_bytes() == before


def test_search_reads_committed_generation_during_write_transaction(tmp_path: Path):
    root = tmp_path / "vault"
    root.mkdir()
    note = root / "note.md"
    note.write_text("# Committed\nvisible while refresh writes\n")
    config = make_config(tmp_path, [("vault", "phoebe", root)])
    index(config)

    writer = sqlite3.connect(config.database, timeout=0)
    writer.execute("BEGIN IMMEDIATE")
    try:
        started = time.perf_counter()
        results = search(config, "visible", project="phoebe")
        elapsed = time.perf_counter() - started
    finally:
        writer.rollback()
        writer.close()

    assert elapsed < 1
    assert [result.canonical_path for result in results] == [str(note.resolve())]


def test_rebuild_keeps_previous_generation_visible_until_commit(tmp_path: Path, monkeypatch):
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("# Decision\nKeep the old generation visible.\n")
    config = make_config(tmp_path, [("vault", "phoebe", root)])
    assert index(config) == 1

    reset_finished = threading.Event()
    allow_reset = threading.Event()
    original_reset = core._reset_schema

    def paused_reset(connection, generation=None):
        original_reset(connection, generation)
        reset_finished.set()
        assert allow_reset.wait(timeout=5)

    monkeypatch.setattr(core, "_reset_schema", paused_reset)
    worker_errors: list[BaseException] = []

    def rebuild() -> None:
        try:
            assert index(config, rebuild=True) == 2
        except BaseException as exc:
            worker_errors.append(exc)

    worker = threading.Thread(target=rebuild)
    worker.start()
    assert reset_finished.wait(timeout=5)

    reader = sqlite3.connect(config.database)
    try:
        assert reader.execute("SELECT value FROM metadata WHERE key = 'generation'").fetchone()[0] == "1"
        assert reader.execute("SELECT count(*) FROM sections").fetchone()[0] == 1
        assert reader.execute("SELECT count(*) FROM sections_fts").fetchone()[0] == 1
        assert reader.execute("SELECT generation FROM embedding_metadata").fetchone()[0] == 1
    finally:
        reader.close()

    allow_reset.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert worker_errors == []
    connection = connect(config.database, initialize=False)
    try:
        assert connection.execute("SELECT value FROM metadata WHERE key = 'generation'").fetchone()[0] == "2"
        assert connection.execute("SELECT count(*) FROM sections").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM sections_fts").fetchone()[0] == 1
        assert connection.execute("SELECT generation FROM embedding_metadata").fetchone()[0] == 2
    finally:
        connection.close()


def test_rebuild_and_scope_with_global_notes(tmp_path: Path):
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    (one / "private.md").write_text("# One\nsecret project detail\n")
    (two / "global.md").write_text("# Shared\nshared deployment decision\n")
    config = make_config(tmp_path, [("one", "alpha", one), ("two", "beta", two)], [two / "global.md"])
    index(config)
    assert search(config, "secret", project="beta") == []
    assert search(config, "shared", project="alpha")[0].project_scope == "beta"
    assert len(search(config, "secret", all_projects=True)) == 1
    assert index(config, rebuild=True) == 2


def test_scope_change_takes_effect_without_reindex(tmp_path: Path):
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    note = one / "global.md"
    note.write_text("# Shared\nshared scope change\n")
    config = make_config(tmp_path, [("one", "alpha", one), ("two", "beta", two)], [note])
    index(config)
    assert search(config, "scope change", project="beta")

    config_path = tmp_path / "config.toml"
    config_path.write_text("\n".join([
        f'database = "{config.database}"',
        "",
        "[[roots]]",
        f'id = "one"',
        'project = "alpha"',
        f'path = "{one}"',
        "",
        "[[roots]]",
        f'id = "two"',
        'project = "beta"',
        f'path = "{two}"',
    ]))
    current_config = load_config(config_path)
    assert search(current_config, "scope change", project="beta") == []


def test_root_project_change_takes_effect_without_reindex(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    note = root / "note.md"
    note.write_text("# Scope\nroot project change\n")
    config = make_config(tmp_path, [("root", "alpha", root)])
    index(config)

    config_path = tmp_path / "config.toml"
    config_path.write_text("\n".join([
        f'database = "{config.database}"',
        "",
        "[[roots]]",
        'id = "root"',
        'project = "beta"',
        f'path = "{root}"',
    ]))
    current_config = load_config(config_path)
    assert search(current_config, "project change", project="alpha") == []
    assert search(current_config, "project change", project="beta")


def test_crlf_hash_is_verified_from_raw_bytes(tmp_path: Path):
    root = tmp_path / "vault"
    root.mkdir()
    note = root / "note.md"
    note.write_bytes(b"# Terms\r\nold plan\r\n")
    config = make_config(tmp_path, [("vault", "p", root)])
    index(config)
    assert search(config, "old plan", project="p")


def test_adversarial_fts_queries_are_safe(tmp_path: Path):
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("# Terms\nquotes NEAR operators and stars\n")
    config = make_config(tmp_path, [("vault", "p", root)])
    index(config)
    for query in ['"', "*", "NEAR", "(", ")", 'one" OR *']:
        search(config, query, project="p")
    assert search(config, "NEAR", project="p")[0].heading == "Terms"


def test_quoted_query_preserves_phrase_meaning_and_caps_terms(tmp_path: Path):
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("# Terms\nold unrelated plan\n")
    config = make_config(tmp_path, [("vault", "p", root)])
    index(config)
    assert search(config, '"old plan"', project="p") == []
    started = time.perf_counter()
    results = search(config, "term " * 100_000, project="p")
    elapsed = time.perf_counter() - started
    assert len(results) <= 20
    assert elapsed < 1


def test_selection_reason_reports_body_match(tmp_path: Path):
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("# Decision\nbody-only phrase\n")
    config = make_config(tmp_path, [("vault", "p", root)])
    index(config)
    assert search(config, "body-only", project="p")[0].reason == "body match"


def test_selection_reason_does_not_match_heading_prefixes(tmp_path: Path):
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("# Planet\nplan appears in the body\n")
    config = make_config(tmp_path, [("vault", "p", root)])
    index(config)
    result = search(config, "plan", project="p")[0]
    assert result.reason == "body match"


def test_heading_matches_survive_bounded_candidate_lanes(tmp_path: Path):
    root = tmp_path / "vault"
    root.mkdir()
    sections = [f"# Body {index}\nrepeated needle body\n" for index in range(121)]
    sections.append("# Needle\nunrelated text\n")
    (root / "note.md").write_text("".join(sections))
    config = make_config(tmp_path, [("vault", "p", root)])
    index(config)
    result = search(config, "needle", project="p", limit=1)
    assert result[0].heading_path == ("Needle",)
    assert result[0].reason == "heading match"


def test_symlink_escape_is_skipped_in_index_and_read(tmp_path: Path):
    root = tmp_path / "vault"
    outside = tmp_path / "outside.md"
    root.mkdir()
    outside.write_text("# Outside\nprivate material\n")
    link = root / "link.md"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    config = make_config(tmp_path, [("vault", "p", root)])
    index(config)
    assert search(config, "private", project="p") == []
    with pytest.raises(ValueError, match="outside allowed roots"):
        read_source(config, str(link))


def test_read_heading_and_stale_hash(tmp_path: Path):
    root = tmp_path / "vault"
    root.mkdir()
    note = root / "note.md"
    note.write_text("# Parent\nintro\n## Child\nchild text\n")
    config = make_config(tmp_path, [("vault", "p", root)])
    index(config)
    assert "child text" in read_source(config, str(note), heading="Parent > Child")
    assert len(read_source(config, str(note), max_bytes=10).encode()) <= 10
    note.write_text("# Parent\nchanged text\n")
    refresh: set[str] = set()
    assert search(config, "child text", project="p", refresh=refresh) == []
    assert str(note.resolve()) in refresh


def test_search_keeps_valid_results_when_one_match_is_stale(tmp_path: Path):
    root = tmp_path / "vault"
    root.mkdir()
    stale = root / "stale.md"
    valid = root / "valid.md"
    stale.write_text("# Stale\nshared query\n")
    valid.write_text("# Valid\nshared query\n")
    config = make_config(tmp_path, [("vault", "p", root)])
    index(config)
    stale.write_text("# Stale\nchanged content\n")

    refresh: set[str] = set()
    results = search(config, "shared query", project="p", refresh=refresh, limit=2)

    assert [result.heading for result in results] == ["Valid"]
    assert refresh == {str(stale.resolve())}


def test_cli_json_output(tmp_path: Path, capsys):
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("# Choice\nUse sqlite.\n")
    config = make_config(tmp_path, [("vault", "p", root)])
    config_path = tmp_path / "config.toml"
    assert main(["--config", str(config_path), "index"]) == 0
    assert main(["--config", str(config_path), "search", "sqlite", "--project", "p", "--json"]) == 0
    result = json.loads(capsys.readouterr().out.splitlines()[-1])[0]
    assert result["excerpt"] == "Use sqlite."
    assert set(result) == {
        "excerpt", "path", "heading", "line_range", "dates", "score", "reason", "content_hash",
    }


def test_cli_diagnostics_enumerate_fusion_policies(tmp_path: Path, capsys):
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("# Choice\nUse sqlite.\n")
    config = make_config(tmp_path, [("vault", "p", root)])
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'database = "{config.database}"\n\n'
        f'[[roots]]\nid = "vault"\nproject = "p"\npath = "{root}"\n'
    )
    main(["--config", str(config_path), "index"])
    main(["--config", str(config_path), "search", "sqlite", "--project", "p", "--diagnostics", "--json"])

    result = json.loads(capsys.readouterr().out.splitlines()[-1])[0]
    assert set(result["fusion_policies"]["selection_policies"]) == {
        "semantic_score_floor",
        "relative_semantic_score_floor",
        "ticket_id_cross_reference_filter",
        "balanced_admission",
        "synonym_expansion",
    }


def test_synonym_variants_are_conservative_and_versioned(tmp_path: Path):
    from pausanias.synonyms import load_synonym_table, query_variants

    path = tmp_path / "synonyms.toml"
    path.write_text('version = 1\n\n[terms]\ndatabase = ["db"]\n')
    table = load_synonym_table(path)

    assert table.version == 1
    versioned_path = tmp_path / "synonyms-v2.toml"
    versioned_path.write_text('version = 2\n\n[terms]\ndatabase = ["db"]\n')
    versioned_table = load_synonym_table(versioned_path)

    assert table.fingerprint != versioned_table.fingerprint
    assert query_variants('database "database" PAUS-11', table) == ('db "database" PAUS-11',)


def test_synonym_alias_order_does_not_change_lexical_ranking(tmp_path: Path):
    root = tmp_path / "vault"
    root.mkdir()
    (root / "alpha.md").write_text("# Alpha\nsql sql sql\n")
    (root / "beta.md").write_text("# Beta\ndb\n")
    config = make_config(tmp_path, [("vault", "p", root)])
    first_table = tmp_path / "synonyms-first.toml"
    first_table.write_text('version = 1\n\n[terms]\ndatabase = ["db", "sql"]\n')
    second_table = tmp_path / "synonyms-second.toml"
    second_table.write_text('version = 1\n\n[terms]\ndatabase = ["sql", "db"]\n')

    first_config = Config(
        config.roots, config.global_notes, config.private_paths, config.database,
        config.section_bytes, config.semantic_bundle, first_table,
    )
    second_config = Config(
        config.roots, config.global_notes, config.private_paths, config.database,
        config.section_bytes, config.semantic_bundle, second_table,
    )
    index(first_config)

    first = [item.heading for item in search(first_config, "database", project="p")]
    second = [item.heading for item in search(second_config, "database", project="p")]

    assert first[0] == "Alpha"
    assert first == second


def test_synonym_diagnostics_report_effective_state(tmp_path: Path):
    root = tmp_path / "vault"
    root.mkdir()
    config = make_config(tmp_path, [("vault", "p", root)])

    diagnostics = core.fusion_diagnostics(config)

    assert diagnostics["selection_policies"]["synonym_expansion"]["enabled"] is False
    assert diagnostics["selection_policies"]["synonym_expansion"]["runtime_toggle"] is True


def test_synonym_search_uses_alias_without_changing_disabled_output(tmp_path: Path):
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("# Storage\nUse a database for durable state.\n")
    table_path = tmp_path / "synonyms.toml"
    table_path.write_text('version = 1\n\n[terms]\ndatabase = ["db"]\n')
    config = make_config(tmp_path, [("vault", "p", root)])
    synonym_config = Config(
        config.roots, config.global_notes, config.private_paths, config.database,
        config.section_bytes, config.semantic_bundle, table_path,
    )
    index(synonym_config)

    assert search(synonym_config, "db", project="p")[0].heading == "Storage"
    assert search(synonym_config, "db", project="p", synonym_expansion=False) == []
    assert search(synonym_config, "database", project="p")[0].heading == "Storage"


def test_config_rejects_missing_root(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    config_path.write_text('[[roots]]\nid = "x"\nproject = "p"\npath = "/does/not/exist"\n')
    with pytest.raises(ConfigError, match="cannot resolve path"):
        load_config(config_path)


def test_config_rejects_overlapping_roots_after_resolution(tmp_path: Path):
    parent = tmp_path / "parent"
    child = parent / "child"
    parent.mkdir()
    child.mkdir()
    with pytest.raises(ConfigError, match="overlaps"):
        make_config(tmp_path, [("parent", "p1", parent), ("child", "p2", child)])

    alias = tmp_path / "child-alias"
    try:
        alias.symlink_to(child, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ConfigError, match="overlaps"):
        make_config(tmp_path, [("parent", "p1", parent), ("alias", "p2", alias)])


def test_config_excludes_credentials_and_private_globs(tmp_path: Path):
    root = tmp_path / "vault"
    root.mkdir()
    (root / ".env.local").write_text("TOKEN=secret")
    (root / "private").mkdir()
    (root / "private" / "note.md").write_text("private token")
    (root / "public.md").write_text("public token")
    config = make_config(tmp_path, [("vault", "p", root)], private=["private/**"])
    index(config)
    assert search(config, "secret", project="p") == []
    assert search(config, "private", project="p") == []
    assert search(config, "public", project="p")


def test_private_parent_exclusion_applies_to_read(tmp_path: Path):
    root = tmp_path / "vault"
    root.mkdir()
    private = root / "private"
    private.mkdir()
    note = private / "secret.md"
    note.write_text("secret")
    config = make_config(tmp_path, [("vault", "p", root)], private=["private"])
    assert config.is_excluded(private, config.roots[0])
    assert config.is_excluded(note, config.roots[0])
    with pytest.raises(ValueError, match="outside allowed roots"):
        read_source(config, str(note))


def test_cli_root_scope_defaults_to_that_project(tmp_path: Path, capsys):
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    (one / "one.md").write_text("# One\nroot-only needle\n")
    (two / "two.md").write_text("# Two\nother needle\n")
    config = make_config(tmp_path, [("one", "alpha", one), ("two", "beta", two)])
    config_path = tmp_path / "config.toml"

    assert main(["--config", str(config_path), "index"]) == 0
    capsys.readouterr()

    assert main(["--config", str(config_path), "search", "root-only", "--root", "one", "--json"]) == 0
    root_results = json.loads(capsys.readouterr().out)
    assert len(root_results) == 1
    assert root_results[0]["heading"] == ["One"]

    assert main(["--config", str(config_path), "search", "root-only", "--root", "one", "--project", "alpha", "--json"]) == 0
    project_results = json.loads(capsys.readouterr().out)
    assert len(project_results) == 1

    assert main(["--config", str(config_path), "search", "root-only", "--root", "one", "--all-projects", "--json"]) == 0
    all_results = json.loads(capsys.readouterr().out)
    assert len(all_results) == 1


def test_cli_root_scope_includes_global_notes_from_any_root(tmp_path: Path, capsys):
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    (one / "local.md").write_text("# One\nroot-only needle\n")
    inside_global = one / "inside-global.md"
    inside_global.write_text("# Inside shared\ninside global needle\n")
    global_note = two / "global.md"
    global_note.write_text("# Shared\nglobal needle\n")
    config = make_config(tmp_path, [("one", "alpha", one), ("two", "beta", two)], [inside_global, global_note])
    config_path = tmp_path / "config.toml"

    assert main(["--config", str(config_path), "index"]) == 0
    capsys.readouterr()

    assert main(["--config", str(config_path), "search", "global", "--root", "one", "--json"]) == 0
    root_results = json.loads(capsys.readouterr().out)
    assert {result["path"] for result in root_results} == {str(inside_global.resolve()), str(global_note.resolve())}
    assert all(result["reason"].startswith("global-note inclusion") for result in root_results)

    assert main(["--config", str(config_path), "search", "global", "--root", "one", "--all-projects", "--json"]) == 0
    all_results = json.loads(capsys.readouterr().out)
    assert {result["path"] for result in all_results} == {str(inside_global.resolve()), str(global_note.resolve())}

    assert main(["--config", str(config_path), "search", "global", "--root", "one", "--project", "alpha", "--json"]) == 0
    project_results = json.loads(capsys.readouterr().out)
    assert {result["path"] for result in project_results} == {str(inside_global.resolve()), str(global_note.resolve())}


def test_root_project_scope_filters_before_candidate_limit(tmp_path: Path):
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    for number in range(150):
        (alpha / f"alpha-{number}.md").write_text(f"# Alpha {number}\nneedle alpha match\n")
    global_note = beta / "global.md"
    global_note.write_text("# Shared\nneedle global match\n")
    config = make_config(tmp_path, [("a", "alpha", alpha), ("b", "beta", beta)], [global_note])
    index(config)

    results = search(config, "needle", project="beta", root_id="a", limit=1)

    assert [result.canonical_path for result in results] == [str(global_note.resolve())]
    assert search(config, "alpha", project="beta", root_id="a") == []


def test_tied_results_have_stable_order_across_rebuilds(tmp_path: Path, monkeypatch):
    root = tmp_path / "vault"
    root.mkdir()
    for name in ("one.md", "two.md"):
        (root / name).write_text("# Same\nshared tie text\n")
    config = make_config(tmp_path, [("vault", "p", root)])
    found = core._files(config)

    monkeypatch.setattr(core, "_files", lambda _: found)
    index(config, rebuild=True)
    first = [result.section_id for result in search(config, "shared tie")]

    reversed_found = dict(reversed(list(found.items())))
    monkeypatch.setattr(core, "_files", lambda _: reversed_found)
    index(config, rebuild=True)
    second = [result.section_id for result in search(config, "shared tie")]

    assert first == second


def test_selection_reason_matches_heading_tokens_only(tmp_path: Path):
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("# Planet\nplan appears in the body\n")
    config = make_config(tmp_path, [("vault", "p", root)])
    index(config)

    assert search(config, "plan", project="p")[0].reason == "body match"
    assert search(config, "Planet", project="p")[0].reason == "heading match"
