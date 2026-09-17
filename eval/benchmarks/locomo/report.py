"""Render the PAUS-15 report from LOCOMO abstention artifacts."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"artifact must contain an object: {path}")
    return value


def _abstention(artifact: dict[str, object]) -> dict[str, object]:
    metrics = artifact.get("metrics")
    if not isinstance(metrics, dict) or not isinstance(metrics.get("abstention"), dict):
        raise ValueError("artifact does not contain abstention metrics")
    return metrics["abstention"]


def _number(value: object) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _percent(value: object) -> str:
    return f"{float(value) * 100:.2f}%"


def _table_row(values: list[object]) -> str:
    return "| " + " | ".join(str(value) for value in values) + " |"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def render(lexical_path: Path, fused_path: Path) -> str:
    artifacts = {"lexical": _load(lexical_path), "fused": _load(fused_path)}
    summaries = {mode: _abstention(artifact) for mode, artifact in artifacts.items()}
    dataset = artifacts["lexical"]["metadata"]["dataset"]
    lexical = summaries["lexical"]
    fused = summaries["fused"]
    lines = [
        "# paus-15 results",
        "",
        "## headline",
        "",
        "this eval measures retrieval-side abstention on LOCOMO category-5 "
        "adversarial questions.",
        "",
        _table_row(["mode", "questions", "abstained", "false injections", "false-injection rate"]),
        _table_row(["---", "---:", "---:", "---:", "---:"]),
    ]
    for mode in ("lexical", "fused"):
        summary = summaries[mode]
        lines.append(_table_row([
            mode,
            summary["total"],
            summary["abstained"],
            summary["false_injections"],
            _percent(summary["false_injection_rate"]),
        ]))

    lines.extend([
        "",
        "the false-injection rate is the primary metric. an abstention is a "
        "selection-policy result with zero injectable candidates.",
        "",
        "## injected result counts",
        "",
        _table_row(["injected results", "lexical questions", "fused questions"]),
        _table_row(["---:", "---:", "---:"]),
    ])
    count_keys = sorted(
        set(lexical["injected_count_distribution"]) | set(fused["injected_count_distribution"]),
        key=int,
    )
    for key in count_keys:
        lines.append(_table_row([
            key,
            lexical["injected_count_distribution"].get(key, 0),
            fused["injected_count_distribution"].get(key, 0),
        ]))

    lines.extend([
        "",
        "## injected score summaries",
        "",
        "scores cover every injected result in a false-injection query. "
        "they support later floor analysis and do not tune this lane.",
        "",
        _table_row(["mode", "results", "min", "p50", "p95", "max", "mean"]),
        _table_row(["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ])
    for mode in ("lexical", "fused"):
        score = summaries[mode]["injected_score_summary"]
        lines.append(_table_row([
            mode,
            score["count"],
            _number(score["min"]),
            _number(score["p50"]),
            _number(score["p95"]),
            _number(score["max"]),
            _number(score["mean"]),
        ]))

    lines.extend([
        "",
        "## per-conversation breakdown",
        "",
        _table_row(["conversation", "lexical total", "lexical false", "lexical rate", "fused total", "fused false", "fused rate"]),
        _table_row(["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ])
    conversation_ids = sorted(
        set(lexical["by_conversation"]) | set(fused["by_conversation"]),
        key=int,
    )
    for conversation_id in conversation_ids:
        lexical_conversation = lexical["by_conversation"].get(conversation_id, {})
        fused_conversation = fused["by_conversation"].get(conversation_id, {})
        lines.append(_table_row([
            conversation_id,
            lexical_conversation.get("total", 0),
            lexical_conversation.get("false_injections", 0),
            _percent(lexical_conversation.get("false_injection_rate", 0.0)),
            fused_conversation.get("total", 0),
            fused_conversation.get("false_injections", 0),
            _percent(fused_conversation.get("false_injection_rate", 0.0)),
        ]))

    lines.extend([
        "",
        "## dataset and environment",
        "",
        f"- dataset: `{dataset['name']}`",
        f"- dataset commit: `{dataset['version']}`",
        f"- dataset sha256: `{dataset['sha256']}`",
        f"- dataset fingerprint: `{dataset['fingerprint']}`",
        f"- runner commit: `{_git_commit()}`",
        f"- python: `{sys.version.split()[0]}`",
        f"- platform: `{platform.platform()}`",
        "- retrieval limit: `top_k=200`",
        "- retrieval modes: `lexical`, `fused`",
        "- network: dataset fetch only; retrieval was local",
        "",
        "## scope note",
        "",
        "this is a retrieval-side abstention eval for conversational adversarial "
        "questions. category 5 has no valid evidence targets, so recall, precision, "
        "and mrr are null in each artifact outcome. this complements the category "
        "1–4 retrieval benchmark in `eval/PAUS-14-RESULTS.md`.",
        "",
        "the results record the current selection policy. this lane does not tune "
        "floors, thresholds, or policies.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lexical", type=Path, required=True)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.write_text(render(args.lexical, args.fused), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
