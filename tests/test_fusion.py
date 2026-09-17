from __future__ import annotations

from pathlib import Path

import pytest

from pausanias import core
from pausanias.config import load_config


class FakeEncoder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * 384
            vector[0] = 1.0 if "semantic" in text.lower() or text.strip().lower() == "paus-4" else 0.0
            vector[1] = 1.0 if "alpha" in text.lower() else 0.0
            vectors.append(vector)
        return vectors


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
