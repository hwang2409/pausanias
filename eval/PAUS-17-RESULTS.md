# PAUS-17 relaxed lexical matching results

PAUS-17 adds a relaxed lexical pass after strict all-term matching (see `DESIGN.md`,
"Relaxed lexical matching"). Every number below comes from the same eval commands, run
on the pre-change `main` (`baseline`) and on the final change (`final`). Both runs used
the pinned LOCOMO10 dataset (`79fa87e9…`) with retrieval only.

```bash
python -m eval.run
python -m eval.harness --retrieval-mode {lexical,fused}
python -m eval.benchmarks.locomo.run --retrieval-mode {lexical,fused} --predict-only
python -m eval.benchmarks.locomo.run --retrieval-mode {lexical,fused} --abstention
```

## LOCOMO retrieval, categories 1–4 (1,540 questions)

| mode | run | recall@10 | MRR@10 | recall@50 | recall@200 |
| --- | --- | ---: | ---: | ---: | ---: |
| lexical | baseline | 0.1% | 0.001 | 0.1% | 0.1% |
| lexical | final | 38.9% | 0.363 | 39.4% | 39.4% |
| fused | baseline | 29.8% | 0.161 | 49.8% | 55.5% |
| fused | final | 50.5% | 0.405 | 61.3% | 63.7% |

Recall@10 by category:

| mode | run | multi-hop | temporal | open-domain | single-hop |
| --- | --- | ---: | ---: | ---: | ---: |
| lexical | baseline | 0.0% | 0.0% | 0.3% | 0.1% |
| lexical | final | 13.0% | 52.8% | 7.6% | 45.8% |
| fused | baseline | 14.2% | 32.1% | 12.6% | 36.2% |
| fused | final | 20.5% | 61.9% | 15.8% | 60.1% |

## Internal golden cases (fingerprint `03be0de0…`)

| harness | baseline | final |
| --- | ---: | ---: |
| `eval.run` lexical recall@4 / abstention | 0.673 / 1.000 | 0.735 / 1.000 |
| `eval.harness` lexical accuracy | 72.88% | 77.97% |
| `eval.harness` fused accuracy | 100.00% | 100.00% |

All recorded thresholds pass. No case, threshold, or lock file changed.

## LOCOMO adversarial abstention, category 5 (446 questions)

| mode | baseline false injection | final false injection |
| --- | ---: | ---: |
| lexical | 1.35% | 67.71% |
| fused | 100.00% | 100.00% |

This is the one regression. The baseline lexical lane abstained because it retrieved
almost nothing, not because it told evidence from non-evidence (0.1% recall). The
category 5 questions reuse a conversation's entities with a wrong attribute, so any
term-overlap retrieval returns those turns. Fused retrieval, the default when the
semantic extra is ready, was already at 100% and is unchanged. Retrieval-side
abstention on these questions needs a semantic or answer-level check. Answerers that
read the evidence, such as an LLM told to say when the notes do not answer, remain
responsible for it.

## IDF coverage floor sweep

The count floor alone (`RELAXED_MIN_COVERAGE = 0.5`) failed two internal abstention
cases: `deleted-email-provider`, a near miss without “Postmark”, and
`ambiguous-history-follow-up`, a lone leftover keyword. The two-term minimum fixes the
second. The IDF floor was chosen on LOCOMO, not on the internal set:

| IDF floor | internal recall@4 | internal abstention | LOCOMO fused recall@10 | LOCOMO lexical cat-5 false injection |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 (no floor) | 0.776 | 0.800 | 57.5% | not run |
| 0.40 | 0.735 | 0.900 | not run | not run |
| **0.45** | **0.735** | **1.000** | **50.5%** | **67.71%** |
| 0.50 | 0.694 | 1.000 | 47.5% | 60.31% |

The no-floor row predates the two-term minimum. 0.45 is the loosest floor that keeps
every internal abstention case and has the best LOCOMO recall of the passing floors.

## Latency

On a 1,538-section personal vault, warm, with 30 lexical candidate calls:

| pass | p50 | p95 | max |
| --- | ---: | ---: | ---: |
| strict only | 0.6 ms | 3.6 ms | 3.9 ms |
| strict + relaxed | 5.4 ms | 27.2 ms | 50.9 ms |

The first version tokenized every relaxed candidate body in Python (p95 60 ms, max
159 ms). The final version asks FTS5 for per-term section sets instead.

## Relaxed lexical matching ablation

The fused harness now includes `relaxed_lexical_matching_disabled`:

| ablation | recall@4 | precision@4 | MRR | abstention accuracy | forbidden violations | delta from active (recall, precision, MRR, abstention, forbidden) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| relaxed lexical matching disabled | 1.000 | 0.862 | 0.990 | 1.000 | 0 | (0.000, -0.010, 0.000, 0.000, 0) |

Active fused retrieval remains 100.00% harness accuracy. The disabled-policy row is
based on `python -m eval.harness --retrieval-mode fused`.
