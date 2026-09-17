from __future__ import annotations

import json
from pathlib import Path

import eval.benchmarks.locomo.run as locomo_runner
from eval.benchmarks.locomo.run import (
    StubTransport,
    _reference_date,
    _scope_corpus_fingerprint,
    build_answer_prompt,
    build_judge_prompt,
    render_session,
    run_locomo,
)
from eval.vendor.mem0.benchmarks.locomo.prompts import get_answer_generation_prompt

FIXTURES = Path(__file__).parent.parent / "eval/fixtures/locomo/golden-prompts-v1"


def test_render_session_matches_the_normative_markdown_shape():
    rendered = render_session(
        0,
        "session_1",
        "1:56 pm on 8 May, 2023",
        [
            {"dia_id": "D1:0", "speaker": "Alex", "text": "Hi there."},
            {"dia_id": "D1:1", "speaker": "Sam", "text": None, "query": "a red kite", "blip_caption": "park"},
        ],
    )
    assert rendered.text == (
        "# conversation 00\n\n"
        "## session 01 | timestamp: 1:56 pm on 8 May, 2023\n\n"
        "### turn D1:0 | speaker: Alex | timestamp: 1:56 pm on 8 May, 2023\n\n"
        "Hi there.\n\n"
        "### turn D1:1 | speaker: Sam | timestamp: 1:56 pm on 8 May, 2023\n\n"
        "[image query: a red kite; caption: park]\n"
    )
    assert rendered.turn_ranges == {"D1:0": (5, 7), "D1:1": (9, 11)}


def test_golden_prompt_fixtures_match_vendored_entry_points():
    for name in ("answerer-basic", "answerer-same-created-at"):
        payload = json.loads((FIXTURES / f"{name}.json").read_text())
        actual = build_answer_prompt(payload["question"], payload["search_results"], payload["reference_date"], payload["user_profile"])
        assert actual == (FIXTURES / f"{name}.txt").read_text()
    for name in ("judge-without-evidence", "judge-with-evidence"):
        payload = json.loads((FIXTURES / f"{name}.json").read_text())
        actual = build_judge_prompt(payload["category"], payload["question"], payload["answer"], payload["response"], payload.get("evidence_context"))
        assert actual == (FIXTURES / f"{name}.txt").read_text().rstrip("\n")


def test_locomo_runner_prompt_matches_vendored_date_fixture(tmp_path: Path):
    fixture = json.loads((FIXTURES / "answerer-locomo-date.json").read_text())
    dataset = json.loads((Path(__file__).parent / "fixtures/locomo/small.json").read_text())
    dataset[0]["qa"] = [dataset[0]["qa"][0]]
    dataset_path = tmp_path / "locomo.json"
    dataset_path.write_text(json.dumps(dataset))

    run_locomo(run_id="date-prompt", dataset_path=dataset_path, results_dir=tmp_path, top_k=1, cutoffs=(1,), predict_only=True)
    transport = StubTransport(["ANSWER: fixture", {"label": "CORRECT", "reasoning": "fixture"}])
    run_locomo(
        run_id="date-prompt",
        dataset_path=dataset_path,
        results_dir=tmp_path,
        top_k=1,
        cutoffs=(1,),
        provider="stub",
        judge_provider="stub",
        evaluate_only=True,
        transport=transport,
    )

    assert transport.calls[0]["user"] == get_answer_generation_prompt(
        fixture["question"], fixture["search_results"], fixture["reference_date"], fixture["user_profile"]
    )
    assert "around 2:00 pm on 2 May, 2023" in transport.calls[0]["user"]
    assert "(Monday, May 01, 2023)" in transport.calls[0]["user"]


def test_locomo_search_and_stub_evaluate_are_resumable(tmp_path: Path):
    dataset = Path(__file__).parent / "fixtures/locomo/small.json"
    run_locomo(
        run_id="fixture",
        dataset_path=dataset,
        results_dir=tmp_path,
        top_k=8,
        cutoffs=(1, 2, 4, 8),
        predict_only=True,
    )
    responses = [item for _ in range(8) for item in ("ANSWER: fixture", {"label": "CORRECT", "reasoning": "fixture"})]
    result = run_locomo(
        run_id="fixture",
        dataset_path=dataset,
        results_dir=tmp_path,
        top_k=8,
        cutoffs=(1, 2, 4, 8),
        answerer_model="stub-answerer",
        judge_model="stub-judge",
        provider="stub",
        judge_provider="stub",
        evaluate_only=True,
        transport=StubTransport(responses),
    )
    assert result is not None
    assert result.metrics.total == 2
    assert result.metrics.errors == 0
    assert result.metadata.prompt_fixture_version == "golden-prompts-v1"


def test_stub_failures_keep_mem0_error_semantics(tmp_path: Path):
    dataset = Path(__file__).parent / "fixtures/locomo/small.json"
    run_locomo(run_id="errors", dataset_path=dataset, results_dir=tmp_path, top_k=8, cutoffs=(1, 2, 4, 8), predict_only=True)
    result = run_locomo(
        run_id="errors",
        dataset_path=dataset,
        results_dir=tmp_path,
        top_k=8,
        cutoffs=(1, 2, 4, 8),
        provider="stub",
        judge_provider="stub",
        evaluate_only=True,
        transport=StubTransport([]),
    )
    assert result is not None
    assert result.metrics.errors == 2
    outcome = result.evaluations[0].cutoff_outcomes["8"]
    assert outcome.judgment == "ERROR"
    assert outcome.error == "answerer_error"
    assert outcome.generated_answer is None


def test_locomo_search_is_scoped_to_each_conversation(tmp_path: Path):
    dataset = json.loads((Path(__file__).parent / "fixtures/locomo/small.json").read_text())
    second = json.loads(json.dumps(dataset[0]))
    second["conversation"]["session_1"][0]["text"] = "The orange telescope is in the observatory."
    second["conversation"]["session_1"][0]["dia_id"] = "D1:0"
    second["conversation"]["session_1"][0]["session_date_time"] = "3:00 pm on 3 May, 2023"
    second["conversation"]["session_1_date_time"] = "3:00 pm on 3 May, 2023"
    second["conversation"].pop("session_2")
    second["conversation"].pop("session_2_date_time")
    second["qa"] = [{"question": "orange telescope", "answer": "orange", "category": 4, "evidence": []}]
    dataset.append(second)
    dataset_path = tmp_path / "locomo.json"
    dataset_path.write_text(json.dumps(dataset))

    run_locomo(run_id="scoped", dataset_path=dataset_path, results_dir=tmp_path, top_k=8, cutoffs=(1,), predict_only=True)

    checkpoint = json.loads((tmp_path / "locomo/scoped/checkpoints/search/conv0_q0.json").read_text())
    assert all(item["source_path"].startswith("conversation-00") for item in checkpoint["output"]["retrieval_results"])
    assert not any("orange telescope" in item["excerpt"] for item in checkpoint["output"]["retrieval_results"])


def test_locomo_scope_sync_removes_deleted_sessions_and_records_scope_metadata(tmp_path: Path):
    dataset_path = Path(__file__).parent / "fixtures/locomo/small.json"
    run_locomo(run_id="scope-reuse", dataset_path=dataset_path, results_dir=tmp_path, top_k=8, cutoffs=(1,), predict_only=True)
    dataset = json.loads(dataset_path.read_text())
    dataset[0]["conversation"].pop("session_2")
    dataset[0]["conversation"].pop("session_2_date_time")
    dataset[0]["qa"] = [dataset[0]["qa"][0]]
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(dataset))

    run_locomo(run_id="scope-reuse", dataset_path=changed_path, results_dir=tmp_path, top_k=8, cutoffs=(1,), predict_only=True)

    root = tmp_path / "locomo/scope-reuse"
    checkpoint = json.loads((root / "checkpoints/search/conv0_q0.json").read_text())
    scope = root / "workspace/scopes/conversation-00"
    assert [path.name for path in scope.glob("*.md")] == ["conversation-00--session-01.md"]
    assert not any("painting landscapes" in item["excerpt"] for item in checkpoint["output"]["retrieval_results"])
    assert checkpoint["index_generation"] == 2
    assert checkpoint["corpus_fingerprint"] == _scope_corpus_fingerprint(scope)


def test_locomo_retrieval_config_mismatch_recomputes_search(tmp_path: Path):
    dataset = Path(__file__).parent / "fixtures/locomo/small.json"
    run_locomo(run_id="retrieval-config", dataset_path=dataset, results_dir=tmp_path, top_k=1, cutoffs=(1,), predict_only=True)
    responses = [item for _ in range(4) for item in ("ANSWER: fixture", {"label": "CORRECT", "reasoning": "fixture"})]
    run_locomo(
        run_id="retrieval-config",
        dataset_path=dataset,
        results_dir=tmp_path,
        top_k=8,
        cutoffs=(1, 8),
        provider="stub",
        judge_provider="stub",
        evaluate_only=True,
        transport=StubTransport(responses),
    )

    checkpoint = json.loads((tmp_path / "locomo/retrieval-config/checkpoints/search/conv0_q0.json").read_text())
    assert checkpoint["config"]["top_k"] == 8
    assert checkpoint["config"]["cutoffs"] == [1, 8]
    assert checkpoint["config"]["retrieval_config"] == {
        "retrieval_mode": "lexical",
        "top_k": 8,
        "cutoffs": [1, 8],
        "conversations": [0],
    }


def test_locomo_fused_and_lexical_smoke_modes(tmp_path: Path, monkeypatch):
    dataset = Path(__file__).parent / "fixtures/locomo/small.json"
    fused_calls = []
    real_run_hook = locomo_runner.run_hook

    def recording_run_hook(*args, **kwargs):
        fused_calls.append((args[1], args[2]))
        return real_run_hook(*args, **kwargs)

    monkeypatch.setattr(locomo_runner, "run_hook", recording_run_hook)
    run_locomo(
        run_id="fused-smoke",
        dataset_path=dataset,
        results_dir=tmp_path,
        top_k=8,
        cutoffs=(1,),
        predict_only=True,
        retrieval_mode="fused",
    )
    fused_checkpoint = json.loads(
        (tmp_path / "locomo/fused-smoke/checkpoints/search/conv0_q0.json").read_text()
    )
    assert fused_checkpoint["config"]["retrieval_mode"] == "fused"
    assert fused_checkpoint["config"]["retrieval_config"]["retrieval_mode"] == "fused"
    assert fused_calls

    run_locomo(
        run_id="lexical-smoke",
        dataset_path=dataset,
        results_dir=tmp_path,
        top_k=8,
        cutoffs=(1,),
        predict_only=True,
        retrieval_mode="lexical",
    )
    lexical_checkpoint = json.loads(
        (tmp_path / "locomo/lexical-smoke/checkpoints/search/conv0_q0.json").read_text()
    )
    assert lexical_checkpoint["config"]["retrieval_mode"] == "lexical"
    assert lexical_checkpoint["config"]["retrieval_config"]["retrieval_mode"] == "lexical"


def test_locomo_corrupt_score_checkpoint_is_recomputed(tmp_path: Path):
    dataset = Path(__file__).parent / "fixtures/locomo/small.json"
    run_locomo(run_id="corrupt-score", dataset_path=dataset, results_dir=tmp_path, top_k=8, cutoffs=(1,), predict_only=True)
    first = StubTransport([item for _ in range(2) for item in ("ANSWER: fixture", {"label": "CORRECT", "reasoning": "fixture"})])
    run_locomo(run_id="corrupt-score", dataset_path=dataset, results_dir=tmp_path, top_k=8, cutoffs=(1,), provider="stub", judge_provider="stub", evaluate_only=True, transport=first)
    path = tmp_path / "locomo/corrupt-score/checkpoints/evaluate/conv0_q0.json"
    value = json.loads(path.read_text())
    value["output"]["cutoff_outcomes"]["1"]["score"] = "not-a-number"
    path.write_text(json.dumps(value))

    second = StubTransport(["ANSWER: fixture", {"label": "CORRECT", "reasoning": "recomputed"}] * 2)
    result = run_locomo(run_id="corrupt-score", dataset_path=dataset, results_dir=tmp_path, top_k=8, cutoffs=(1,), provider="stub", judge_provider="stub", resume=True, transport=second)
    assert result is not None
    assert second.calls


def test_locomo_evaluation_checkpoints_skip_completed_llm_calls(tmp_path: Path):
    dataset = Path(__file__).parent / "fixtures/locomo/small.json"
    run_locomo(run_id="checkpoint", dataset_path=dataset, results_dir=tmp_path, top_k=8, cutoffs=(1,), predict_only=True)
    first_transport = StubTransport(["ANSWER: fixture", {"label": "CORRECT", "reasoning": "fixture"}] * 2)
    run_locomo(run_id="checkpoint", dataset_path=dataset, results_dir=tmp_path, top_k=8, cutoffs=(1,), provider="stub", judge_provider="stub", evaluate_only=True, transport=first_transport)
    second_transport = StubTransport([])
    result = run_locomo(run_id="checkpoint", dataset_path=dataset, results_dir=tmp_path, top_k=8, cutoffs=(1,), provider="stub", judge_provider="stub", resume=True, transport=second_transport)

    assert result is not None
    assert second_transport.calls == []


def test_locomo_profile_change_recomputes_evaluation_checkpoint(tmp_path: Path):
    dataset = Path(__file__).parent / "fixtures/locomo/small.json"
    alice = {"name": "Alice"}
    bob = {"name": "Bob"}
    run_locomo(run_id="profile", dataset_path=dataset, results_dir=tmp_path, top_k=8, cutoffs=(1,), predict_only=True)

    run_locomo(
        run_id="profile",
        dataset_path=dataset,
        results_dir=tmp_path,
        top_k=8,
        cutoffs=(1,),
        provider="stub",
        judge_provider="stub",
        user_profile=alice,
        transport=StubTransport(["ANSWER: Alice", {"label": "CORRECT", "reasoning": "Alice"}] * 2),
    )
    second_transport = StubTransport(["ANSWER: Bob", {"label": "CORRECT", "reasoning": "Bob"}] * 2)
    result = run_locomo(
        run_id="profile",
        dataset_path=dataset,
        results_dir=tmp_path,
        top_k=8,
        cutoffs=(1,),
        provider="stub",
        judge_provider="stub",
        user_profile=bob,
        resume=True,
        transport=second_transport,
    )

    assert result is not None
    assert second_transport.calls
    assert "Bob" in second_transport.calls[0]["user"]


def test_invalid_judge_json_is_retried(tmp_path: Path):
    dataset = Path(__file__).parent / "fixtures/locomo/small.json"
    run_locomo(run_id="judge-retry", dataset_path=dataset, results_dir=tmp_path, top_k=8, cutoffs=(1,), predict_only=True)
    transport = StubTransport([
        "ANSWER: fixture",
        "not json",
        {"label": "CORRECT", "reasoning": "recovered"},
        "ANSWER: fixture",
        {"label": "CORRECT", "reasoning": "fixture"},
    ])
    result = run_locomo(run_id="judge-retry", dataset_path=dataset, results_dir=tmp_path, top_k=8, cutoffs=(1,), provider="stub", judge_provider="stub", evaluate_only=True, transport=transport)

    assert result is not None
    assert result.metrics.errors == 0
    assert len(transport.calls) == 5


def test_locomo_parses_raw_session_dates_for_prompts():
    entry = {
        "conversation": {
            "session_1": [],
            "session_1_date_time": "(1:00 pm on 1 May, 2023)",
            "session_2": [],
            "session_2_date_time": "2:00 pm on 2 May, 2023",
        }
    }

    assert _reference_date(entry) == "2:00 pm on 2 May, 2023"


def test_locomo_records_actual_index_generation_after_rebuild(tmp_path: Path):
    dataset = Path(__file__).parent / "fixtures/locomo/small.json"
    for _ in range(2):
        run_locomo(run_id="generation", dataset_path=dataset, results_dir=tmp_path, top_k=8, cutoffs=(1,), predict_only=True)

    checkpoint = json.loads((tmp_path / "locomo/generation/checkpoints/ingest.json").read_text())
    assert checkpoint["index_generation"] == 2
