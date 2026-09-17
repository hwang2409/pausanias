# paus-15 results

regeneration command: `.venv/bin/python -m eval.benchmarks.locomo.report --lexical results/locomo/lexical/run.json --fused results/locomo/fused/run.json --output eval/PAUS-15-RESULTS.md`

## headline

this eval measures retrieval-side abstention on LOCOMO category-5 adversarial questions.

| mode | questions | abstained | errors | false injections | false-injection rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| lexical | 2 | 1 | 1 | 0 | 0.00% |
| fused | 2 | 0 | 0 | 2 | 100.00% |

the false-injection rate is the primary metric. an abstention is a successful selection pass with zero injectable candidates. failed, timed-out, and fallback-degraded selections are errors.

## injected result counts

| injected results | lexical questions | fused questions |
| ---: | ---: | ---: |
| 0 | 1 | 0 |
| 1 | 0 | 1 |
| 2 | 0 | 1 |

## injected score summaries

scores cover every injected result in a false-injection query. they support later floor analysis and do not tune this lane.

| mode | results | min | p50 | p95 | max | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lexical | 0 | 0 | 0 | 0 | 0 | 0 |
| fused | 3 | 0.1 | 0.2 | 0.3 | 0.3 | 0.2 |

## per-conversation breakdown

| conversation | lexical total | lexical errors | lexical false | lexical rate | fused total | fused errors | fused false | fused rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2 | 1 | 0 | 0.00% | 2 | 0 | 2 | 100.00% |

## dataset and environment

- dataset: `fixture`
- dataset commit: `v1`
- dataset sha256: `dataset-sha`
- dataset fingerprint: `dataset-fingerprint`
- retrieval limit: `top_k=200`
- retrieval modes: `lexical`, `fused`
- network: dataset fetch only; retrieval was local

## scope note

this is a retrieval-side abstention eval for conversational adversarial questions. category 5 has no valid evidence targets, so recall, precision, and mrr are null in each artifact outcome. this complements the category 1–4 retrieval benchmark in `eval/PAUS-14-RESULTS.md`.

the results record the current selection policy. this lane does not tune floors, thresholds, or policies.
