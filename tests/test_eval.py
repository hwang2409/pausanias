from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from eval.run import Case, run_cases, score_case


def test_score_case_computes_recall_precision_and_mrr(tmp_path: Path):
    runtime_corpus = tmp_path / "corpus"
    runtime_corpus.mkdir()
    results = [
        SimpleNamespace(canonical_path=str(runtime_corpus / "a.md")),
        SimpleNamespace(canonical_path=str(runtime_corpus / "noise.md")),
        SimpleNamespace(canonical_path=str(runtime_corpus / "b.md")),
    ]
    case = Case(
        "math",
        "query",
        {},
        ("a.md", "b.md"),
        ("noise.md",),
        False,
        "checks scoring",
    )

    score = score_case(case, results, runtime_corpus, 1.5, k=3)

    assert score.recall_at_k == 1.0
    assert score.precision_at_k == 2 / 3
    assert score.reciprocal_rank == 1.0
    assert score.forbidden_paths == ("noise.md",)
    assert score.abstention_correct is True


def test_runner_executes_three_case_mini_slice(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "alpha.md").write_text("# Alpha\nalpha evidence\n")
    (corpus / "beta.md").write_text("# Beta\nbeta evidence\n")
    (corpus / "noise.md").write_text("# Noise\nunrelated material\n")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'database = "./index.sqlite3"\n\n'
        '[[roots]]\n'
        'id = "mini"\n'
        'project = "mini"\n'
        'path = "corpus"\n'
    )
    cases = [
        Case("alpha", "alpha evidence", {"project": "mini"}, ("alpha.md",), (), False, "hit"),
        Case("beta", "beta evidence", {"project": "mini"}, ("beta.md",), (), False, "hit"),
        Case("empty", "missing memory", {"project": "mini"}, (), (), True, "abstain"),
    ]

    scores, aggregate = run_cases(cases, corpus, config_path)

    assert len(scores) == 3
    assert aggregate["recall_at_4"] == 1.0
    assert aggregate["precision_at_4"] == 0.25
    assert aggregate["mrr"] == 1.0
    assert aggregate["abstention_accuracy"] == 1.0
    assert aggregate["forbidden_violations"] == 0
