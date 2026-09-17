from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pausanias import core
from pausanias.config import load_config
from pausanias.vectors import Candidate


class FakeEncoder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * 384
            vector[0] = 1.0 if "semantic" in text.lower() or text.strip().lower() == "paus-4" else 0.0
            vector[1] = 1.0 if "alpha" in text.lower() else 0.0
            vectors.append(vector)
        return vectors


class BalancedEncoder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * 384
            if text.strip().lower() == "needle phrase" or "semantic target" in text.lower():
                vector[0] = 1.0
            else:
                vector[1] = 1.0
            vectors.append(vector)
        return vectors


def _candidate(section_id: str, lexical_rank: int | None = None, vector_rank: int | None = None) -> Candidate:
    return Candidate(
        section_id, f"/{section_id}.md", section_id, (section_id,), 1, 2, "vault", "p",
        section_id, section_id, None, None, None, 0.0, "match", lexical_rank, vector_rank,
        0.0 if lexical_rank is not None else None, 1.0 if vector_rank is not None else None,
        None, "lexical" if lexical_rank is not None else "semantic", None,
    )


def _config(tmp_path: Path):
    root = tmp_path / "vault"
    root.mkdir()
    (root / "lexical.md").write_text("# PAUS-4\nalpha lexical result\n")
    (root / "semantic.md").write_text("# Semantic\nsemantic result\n")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'database = "{tmp_path / "index.sqlite3"}"\n\n'
        f'[[roots]]\nid = "vault"\nproject = "p"\npath = "{root}"\n'
    )
    return load_config(config_path)


def test_fused_search_keeps_exact_identifier_first(tmp_path: Path):
    pytest.importorskip("numpy")
    config = _config(tmp_path)
    encoder = FakeEncoder()
    core.index(config, encoder=encoder)

    results = core.semantic_search(config, "PAUS-4", project="p", limit=2, encoder=encoder)

    assert [result.heading for result in results] == ["PAUS-4", "Semantic"]
    assert results[0].guard_reason == "protected:atom-1"
    assert results[0].lexical_rank == 1
    assert results[0].vector_rank is None
    assert results[0].fused_score == 1 / 61
    assert results[0].lane == "lexical"


def test_fused_search_uses_semantic_candidate_when_fts_misses(tmp_path: Path):
    pytest.importorskip("numpy")
    config = _config(tmp_path)
    encoder = FakeEncoder()
    core.index(config, encoder=encoder)

    results = core.semantic_search(config, "semantic paraphrase", project="p", limit=1, encoder=encoder)

    assert results[0].heading == "Semantic"
    assert results[0].lane == "semantic"
    assert results[0].vector_rank == 1


def test_guard_atoms_normalize_quotes_and_whitespace():
    assert core._guard_atoms('Why did  PAUS-4\tbeat "  Cold\u00a0Path "?') == ["paus-4", "cold path"]


def test_unguarded_fusion_admits_top_semantic_candidate_after_lexical_distractors(tmp_path: Path):
    pytest.importorskip("numpy")
    root = tmp_path / "vault"
    root.mkdir()
    for number in range(50):
        (root / f"distractor-{number}.md").write_text(f"# Distractor {number}\nneedle phrase distractor\n")
    (root / "target.md").write_text("# Target\nsemantic target evidence\n")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'database = "{tmp_path / "index.sqlite3"}"\n\n'
        f'[[roots]]\nid = "vault"\nproject = "p"\npath = "{root}"\n'
    )
    config = load_config(config_path)
    encoder = BalancedEncoder()
    core.index(config, encoder=encoder)

    results = core.semantic_search(config, "needle phrase", project="p", limit=4, encoder=encoder)

    target = next(item for item in results if item.heading == "Target")
    assert target.vector_rank == 1


def test_guarded_fusion_keeps_existing_lexical_order():
    lexical = [_candidate(section_id, rank) for rank, section_id in enumerate(("c", "a", "b", "d"), 1)]
    semantic = [_candidate(section_id, None, rank) for rank, section_id in enumerate(("b", "c", "a", "d"), 1)]
    result = core._fuse_candidates(
        lexical, semantic, {"b": 1, "c": 2, "a": 3, "d": 4},
        {1: {"a", "b", "c", "d"}}, True, 4,
    )

    assert [item.section_id for item in result] == ["c", "a", "b", "d"]


def test_guard_matching_casefolds_indexed_text(tmp_path: Path):
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("# Straße\nThe road decision.\n")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'database = "{tmp_path / "index.sqlite3"}"\n\n'
        f'[[roots]]\nid = "vault"\nproject = "p"\npath = "{root}"\n'
    )
    config = load_config(config_path)
    core.index(config)

    result = core.semantic_search(config, '"STRASSE"', project="p", limit=1, encoder=FakeEncoder())

    assert result[0].guard_reason == "protected:atom-1"


def test_ticket_cross_reference_filter_is_named_and_optional(monkeypatch):
    candidate = replace(
        _candidate("cross-reference", vector_rank=1),
        text="PAUS-4 references PHO-123",
        vector_score=0.9,
        lane="semantic",
    )
    monkeypatch.setattr(
        core,
        "_lexical_candidates",
        lambda *args, **kwargs: ([], {}, {}, True),
    )
    monkeypatch.setattr(
        core,
        "vector_candidates",
        lambda *args, **kwargs: [candidate],
    )

    timings = {"semantic_available": 1.0}
    assert core.semantic_search(object(), "PAUS-4", timings=timings) == []
    assert core.semantic_search(
        object(), "PAUS-4", timings=timings, ticket_id_cross_reference_filter=False,
    )[0].section_id == "cross-reference"


def test_relative_semantic_floor_is_independently_named(monkeypatch):
    lexical = _candidate("lexical", lexical_rank=1)
    semantic = [
        replace(_candidate("weak", vector_rank=1), vector_score=0.69, lane="semantic"),
        replace(_candidate("strong", vector_rank=2), vector_score=0.70, lane="semantic"),
        replace(_candidate("lexical", vector_rank=3), vector_score=0.99, lane="semantic"),
    ]
    monkeypatch.setattr(
        core,
        "_lexical_candidates",
        lambda *args, **kwargs: ([lexical], {"lexical": 1}, {}, False),
    )
    monkeypatch.setattr(core, "vector_candidates", lambda *args, **kwargs: semantic)

    result = core.semantic_search(
        object(), "ordinary", timings={"semantic_available": 1.0}, semantic_score_floor=None,
    )

    assert {item.section_id for item in result} == {"strong", "lexical"}
    assert core.FUSION_DIAGNOSTICS["relative_semantic_score_floor"]["value"] == 0.70
