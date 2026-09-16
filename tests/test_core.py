from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

from pausanias.cli import main
from pausanias.config import ConfigError, load_config
from pausanias.core import connect, index, read_source, search
from pausanias.splitter import explicit_links, split_markdown


def make_config(tmp_path: Path, roots: list[tuple[str, str, Path]], global_notes: list[Path] | None = None, private: list[str] | None = None):
    global_notes = global_notes or []
    private = private or []
    database = tmp_path / "index.sqlite3"
    lines = [f'database = "{database}"']
    if global_notes:
        lines += ["", "global_notes = [" + ", ".join(f'"{item}"' for item in global_notes) + "]"]
    if private:
        lines += ["", "private_paths = [" + ", ".join(f'"{item}"' for item in private) + "]"]
    for root_id, project, path in roots:
        lines += ["", "[[roots]]", f'id = "{root_id}"', f'project = "{project}"', f'path = "{path}"']
    config_path = tmp_path / "config.toml"
    config_path.write_text("\n".join(lines))
    return load_config(config_path)


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


def test_cli_json_output(tmp_path: Path, capsys):
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("# Choice\nUse sqlite.\n")
    config = make_config(tmp_path, [("vault", "p", root)])
    config_path = tmp_path / "config.toml"
    assert main(["--config", str(config_path), "index"]) == 0
    assert main(["--config", str(config_path), "search", "sqlite", "--project", "p", "--json"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])[0]["excerpt"] == "Use sqlite."


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
