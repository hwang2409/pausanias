from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from eval import run as eval_run
from eval.run import Case, load_cases, run_cases, score_case


def _threshold_file(path: Path, *, recall: float = 0.0) -> Path:
    path.write_text(
        "[thresholds]\n"
        f"recall_at_4 = {recall}\n"
        "precision_at_4 = 0.0\n"
        "mrr = 0.0\n"
        "abstention_accuracy = 0.0\n"
        "forbidden_violations = 999\n"
    )
    return path


def test_score_case_precision_changes_when_noise_is_added(tmp_path: Path):
    runtime_corpus = tmp_path / "corpus"
    runtime_corpus.mkdir()

    def result(name: str) -> SimpleNamespace:
        return SimpleNamespace(canonical_path=str(runtime_corpus / name))

    case = Case("math", "query", {}, ("a.md",), ("noise.md",), False, "checks scoring")

    clean = score_case(case, [result("a.md")], runtime_corpus, 1.5, k=4)
    noisy = score_case(case, [result("a.md"), result("noise.md")], runtime_corpus, 1.5, k=4)

    assert clean.precision_at_k == 1.0
    assert noisy.precision_at_k == 0.5
    assert noisy.reciprocal_rank == clean.reciprocal_rank == 1.0
    assert noisy.forbidden_paths == ("noise.md",)
    assert noisy.abstention_correct is True


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
    assert aggregate["precision_at_4"] == 1.0
    assert aggregate["mrr"] == 1.0
    assert aggregate["abstention_accuracy"] == 1.0
    assert aggregate["forbidden_violations"] == 0


def test_abstention_gate_uses_only_abstention_cases(tmp_path: Path):
    runtime_corpus = tmp_path / "corpus"
    runtime_corpus.mkdir()

    def result(name: str) -> SimpleNamespace:
        return SimpleNamespace(canonical_path=str(runtime_corpus / name))

    positive = Case("positive", "query", {}, ("hit.md",), (), False, "hit")
    abstentions = [
        Case("abstain-one", "query", {}, (), (), True, "abstain"),
        Case("abstain-two", "query", {}, (), (), True, "abstain"),
    ]
    scores = [
        score_case(positive, [result("hit.md")], runtime_corpus, 1.0),
        *(score_case(case, [result("hit.md")], runtime_corpus, 1.0) for case in abstentions),
    ]

    aggregate = eval_run._aggregate(scores)
    thresholds = {
        "recall_at_4": 0.0,
        "precision_at_4": 0.0,
        "mrr": 0.0,
        "abstention_accuracy": 1.0,
        "forbidden_violations": 0.0,
    }

    assert aggregate["abstention_cases"] == 2
    assert aggregate["abstention_accuracy"] == 0.0
    assert any("abstention_accuracy=0.000" in failure for failure in eval_run._threshold_failures(aggregate, thresholds))


def test_delete_sources_reject_traversal_and_absolute_paths(tmp_path: Path):
    runtime_corpus = tmp_path / "runtime" / "corpus"
    runtime_corpus.mkdir(parents=True)
    (runtime_corpus / "inside.md").write_text("inside")
    outside = runtime_corpus.parent / "outside.md"
    outside.write_text("outside")

    for value in ("../outside.md", str(outside)):
        with pytest.raises(ValueError, match="outside evaluation corpus"):
            eval_run._delete_path(runtime_corpus, value)

    assert outside.read_text() == "outside"


def test_delete_sources_restore_after_mid_run_exception(tmp_path: Path):
    runtime_corpus = tmp_path / "corpus"
    runtime_corpus.mkdir()
    source = runtime_corpus / "source.md"
    source.write_bytes(b"original")

    with pytest.raises(RuntimeError, match="search failed"):
        with eval_run._temporarily_deleted(runtime_corpus, ("source.md",)):
            assert not source.exists()
            raise RuntimeError("search failed")

    assert source.read_bytes() == b"original"


def test_load_cases_accepts_reviewed_and_rejects_unknown_status(tmp_path: Path):
    payload = json.loads(Path(eval_run.CASES_PATH).read_text())
    cases_path = tmp_path / "cases.json"

    payload["status"] = "REVIEWED"
    cases_path.write_text(json.dumps(payload))
    assert len(load_cases(cases_path)) >= 40

    payload["status"] = "PENDING"
    cases_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="exactly DRAFT or REVIEWED"):
        load_cases(cases_path)


def test_full_harness_reports_all_metrics(capsys):
    assert eval_run.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["case_set_status"] == "DRAFT"
    assert len(payload["cases"]) >= 40
    assert {
        "recall_at_4",
        "precision_at_4",
        "mrr",
        "abstention_accuracy",
        "abstention_cases",
        "forbidden_violations",
    } <= payload["aggregate"].keys()
    assert payload["aggregate"]["abstention_cases"] == 10
    assert {"paraphrase", "held-out"} <= payload["category_aggregates"].keys()


def test_gate_fails_when_threshold_exceeds_measured_performance(tmp_path: Path, capsys):
    thresholds = _threshold_file(tmp_path / "thresholds.toml", recall=2.0)

    assert eval_run.main(["--json", "--thresholds", str(thresholds)]) == 1
    payload = json.loads(capsys.readouterr().out)

    assert any(failure.startswith("recall_at_4=") for failure in payload["threshold_failures"])


def test_harness_enforces_privacy_and_project_scope():
    scores, _ = run_cases(load_cases())
    by_id = {score.case.id: score for score in scores}

    for case_id in ("wrong-project-sandbox", "wrong-project-email", "private-path-abstention", "deleted-email-provider"):
        score = by_id[case_id]
        assert score.result_paths == ()
        assert score.abstention_correct is True
        assert score.forbidden_paths == ()


def test_runner_reports_malformed_case_and_exits_nonzero(tmp_path: Path):
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps({"status": "DRAFT", "cases": [{"id": "broken"}]}))

    result = subprocess.run(
        [sys.executable, "-m", "eval.run", "--cases", str(cases_path)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[1],
    )

    assert result.returncode == 2
    assert "eval: error: case broken is missing fields:" in result.stderr


def test_single_case_validates_its_expectations_without_thresholds(tmp_path: Path, capsys):
    thresholds = _threshold_file(tmp_path / "thresholds.toml", recall=2.0)

    assert eval_run.main(["--case", "exact-pho-123", "--thresholds", str(thresholds)]) == 0
    assert "full-run thresholds not applied" in capsys.readouterr().out
