# PAUS-14 LOCOMO retrieval-only results

This run used the pinned LOCOMO10 dataset with all 10 conversations and 1,540
questions from categories 1-4. Category 5 is excluded, as in the existing
LOCOMO runner.

The results are read from these `run.json` artifacts:

- `results/locomo/paus14-lexical/run.json`
- `results/locomo/paus14-fused/run.json`

The tables were generated from these artifacts with a deterministic Python
snippet. It reads each metrics block and formats percentages to three decimals.

## Overall retrieval metrics at cutoff 200

These values are from `metrics.by_cutoff["200"].overall`.

| mode | relevant targets / questions | recall | precision | MRR |
| --- | ---: | ---: | ---: | ---: |
| lexical | 2 / 1,540 | 0.081% | 0.130% | 0.130% |
| fused | 1,157 / 1,540 | 55.505% | 2.946% | 17.238% |

## Per-category retrieval metrics at cutoff 200

The category numbers map as follows: 1 = multi-hop, 2 = temporal, 3 =
open-domain, and 4 = single-hop. These values are from
`metrics.by_cutoff["200"].by_category`.

| category | mode | relevant targets / questions | recall | precision | MRR |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 multi-hop | lexical | 0 / 282 | 0.000% | 0.000% | 0.000% |
| 1 multi-hop | fused | 335 / 282 | 39.111% | 3.722% | 11.071% |
| 2 temporal | lexical | 0 / 321 | 0.000% | 0.000% | 0.000% |
| 2 temporal | fused | 209 / 321 | 57.191% | 3.336% | 20.103% |
| 3 open-domain | lexical | 1 / 96 | 0.260% | 1.042% | 1.042% |
| 3 open-domain | fused | 43 / 96 | 22.986% | 1.834% | 11.134% |
| 4 single-hop | lexical | 1 / 841 | 0.119% | 0.119% | 0.119% |
| 4 single-hop | fused | 570 / 841 | 64.071% | 2.664% | 18.909% |

Relevant targets can exceed question count because one question can have
multiple evidence targets.

## Per-cutoff overall retrieval metrics

These values are from each `metrics.by_cutoff[k].overall` block.

| cutoff | lexical relevant | lexical recall | lexical precision | lexical MRR | fused relevant | fused recall | fused precision | fused MRR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 2 | 0.081% | 0.130% | 0.130% | 579 | 29.830% | 4.235% | 16.059% |
| 20 | 2 | 0.081% | 0.130% | 0.130% | 788 | 39.401% | 3.506% | 16.780% |
| 50 | 2 | 0.081% | 0.130% | 0.130% | 1,027 | 49.822% | 3.025% | 17.155% |
| 200 | 2 | 0.081% | 0.130% | 0.130% | 1,157 | 55.505% | 2.946% | 17.238% |

## Search latency

These values are from `metrics.latency_ms.overall`. The timer covers ranked
retrieval only. It excludes ingest, answer generation, judging, and writes.

| mode | searches | p50 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| lexical | 1,540 | 0.426 ms | 0.712 ms | 1.412 ms |
| fused | 1,540 | 11.702 ms | 14.487 ms | 252.173 ms |

The fused run used `metrics.baselines.retrieval_diagnostics` for operational
counts. It recorded 1,540 fused searches, 1,540 ready semantic states, zero
fallbacks, and zero failure reasons. The lexical run recorded 1,540 lexical
searches and zero fallbacks.

## Dataset and environment

Both runs record this dataset fingerprint in `metadata.dataset.fingerprint` and
this corpus fingerprint in `metadata.corpus_fingerprint`:

- dataset SHA-256: `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`
- dataset commit: `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`
- corpus SHA-256: `24dd317d2ab8ab14e00ea90580386283ed3e2a144433aab1f5feedfa3b98533e`

Environment: Python 3.14.5 on macOS arm64, numpy 2.3.3, and onnxruntime
1.30.0. The model bundle was loaded from the existing local cache. The only
network access was the hash-pinned dataset fetch.

## Scope

These are deterministic retrieval metrics against LOCOMO human evidence
annotations. They are not mem0's LLM-judge answer-accuracy numbers and are not
directly comparable to mem0's published LOCOMO results. This run did not invoke
the answerer or judge stages. The result metadata therefore contains null
answerer and judge model, provider, prompt-version, and prompt-hash fields.
