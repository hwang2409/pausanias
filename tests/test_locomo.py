from __future__ import annotations

import json
from pathlib import Path

from eval.benchmarks.locomo.run import (
    StubTransport,
    _reference_date,
    build_answer_prompt,
    build_judge_prompt,
    render_session,
    run_locomo,
)

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


def test_locomo_evaluation_checkpoints_skip_completed_llm_calls(tmp_path: Path):
    dataset = Path(__file__).parent / "fixtures/locomo/small.json"
    run_locomo(run_id="checkpoint", dataset_path=dataset, results_dir=tmp_path, top_k=8, cutoffs=(1,), predict_only=True)
    first_transport = StubTransport(["ANSWER: fixture", {"label": "CORRECT", "reasoning": "fixture"}] * 2)
    run_locomo(run_id="checkpoint", dataset_path=dataset, results_dir=tmp_path, top_k=8, cutoffs=(1,), provider="stub", judge_provider="stub", evaluate_only=True, transport=first_transport)
    second_transport = StubTransport([])
    result = run_locomo(run_id="checkpoint", dataset_path=dataset, results_dir=tmp_path, top_k=8, cutoffs=(1,), provider="stub", judge_provider="stub", resume=True, transport=second_transport)

    assert result is not None
    assert second_transport.calls == []


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

    assert _reference_date(entry) == "Tuesday, May 02, 2023"


def test_locomo_records_actual_index_generation_after_rebuild(tmp_path: Path):
    dataset = Path(__file__).parent / "fixtures/locomo/small.json"
    for _ in range(2):
        run_locomo(run_id="generation", dataset_path=dataset, results_dir=tmp_path, top_k=8, cutoffs=(1,), predict_only=True)

    checkpoint = json.loads((tmp_path / "locomo/generation/checkpoints/ingest.json").read_text())
    assert checkpoint["index_generation"] == 2
