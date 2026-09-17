# paus-15 results

## headline

this eval measures retrieval-side abstention on LOCOMO category-5 adversarial questions.

| mode | questions | abstained | false injections | false-injection rate |
| --- | ---: | ---: | ---: | ---: |
| lexical | 446 | 440 | 6 | 1.35% |
| fused | 446 | 0 | 446 | 100.00% |

the false-injection rate is the primary metric. an abstention is a selection-policy result with zero injectable candidates.

## injected result counts

| injected results | lexical questions | fused questions |
| ---: | ---: | ---: |
| 0 | 440 | 0 |
| 1 | 6 | 6 |
| 2 | 0 | 1 |
| 3 | 0 | 1 |
| 4 | 0 | 8 |
| 5 | 0 | 5 |
| 6 | 0 | 5 |
| 7 | 0 | 8 |
| 8 | 0 | 12 |
| 9 | 0 | 9 |
| 10 | 0 | 6 |
| 11 | 0 | 10 |
| 12 | 0 | 7 |
| 13 | 0 | 5 |
| 14 | 0 | 7 |
| 15 | 0 | 4 |
| 16 | 0 | 5 |
| 17 | 0 | 12 |
| 18 | 0 | 4 |
| 19 | 0 | 7 |
| 20 | 0 | 6 |
| 21 | 0 | 6 |
| 22 | 0 | 5 |
| 23 | 0 | 4 |
| 24 | 0 | 1 |
| 25 | 0 | 9 |
| 26 | 0 | 4 |
| 27 | 0 | 8 |
| 28 | 0 | 7 |
| 29 | 0 | 8 |
| 30 | 0 | 4 |
| 31 | 0 | 2 |
| 32 | 0 | 6 |
| 33 | 0 | 7 |
| 34 | 0 | 4 |
| 35 | 0 | 4 |
| 36 | 0 | 4 |
| 37 | 0 | 2 |
| 38 | 0 | 8 |
| 39 | 0 | 6 |
| 40 | 0 | 2 |
| 41 | 0 | 3 |
| 42 | 0 | 6 |
| 43 | 0 | 4 |
| 44 | 0 | 4 |
| 45 | 0 | 7 |
| 46 | 0 | 1 |
| 47 | 0 | 2 |
| 48 | 0 | 4 |
| 49 | 0 | 6 |
| 51 | 0 | 5 |
| 52 | 0 | 3 |
| 53 | 0 | 2 |
| 54 | 0 | 3 |
| 55 | 0 | 5 |
| 56 | 0 | 4 |
| 57 | 0 | 5 |
| 58 | 0 | 3 |
| 59 | 0 | 3 |
| 60 | 0 | 3 |
| 61 | 0 | 3 |
| 62 | 0 | 3 |
| 63 | 0 | 2 |
| 64 | 0 | 4 |
| 65 | 0 | 4 |
| 66 | 0 | 7 |
| 67 | 0 | 4 |
| 68 | 0 | 2 |
| 69 | 0 | 4 |
| 70 | 0 | 2 |
| 71 | 0 | 8 |
| 72 | 0 | 4 |
| 73 | 0 | 4 |
| 74 | 0 | 3 |
| 75 | 0 | 4 |
| 76 | 0 | 2 |
| 77 | 0 | 4 |
| 78 | 0 | 3 |
| 79 | 0 | 3 |
| 80 | 0 | 4 |
| 81 | 0 | 6 |
| 82 | 0 | 2 |
| 83 | 0 | 7 |
| 85 | 0 | 3 |
| 86 | 0 | 3 |
| 87 | 0 | 2 |
| 88 | 0 | 3 |
| 89 | 0 | 1 |
| 90 | 0 | 2 |
| 91 | 0 | 2 |
| 92 | 0 | 2 |
| 93 | 0 | 5 |
| 94 | 0 | 1 |
| 95 | 0 | 1 |
| 96 | 0 | 2 |
| 97 | 0 | 1 |
| 98 | 0 | 1 |
| 99 | 0 | 2 |
| 100 | 0 | 3 |
| 102 | 0 | 1 |
| 104 | 0 | 1 |
| 105 | 0 | 2 |
| 107 | 0 | 1 |
| 109 | 0 | 2 |
| 111 | 0 | 1 |
| 112 | 0 | 2 |
| 114 | 0 | 1 |
| 116 | 0 | 1 |
| 119 | 0 | 1 |
| 120 | 0 | 1 |
| 126 | 0 | 1 |
| 134 | 0 | 1 |
| 151 | 0 | 1 |
| 160 | 0 | 1 |
| 167 | 0 | 1 |
| 169 | 0 | 1 |
| 183 | 0 | 1 |

## injected score summaries

scores cover every injected result in a false-injection query. they support later floor analysis and do not tune this lane.

| mode | results | min | p50 | p95 | max | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lexical | 6 | 16.510301 | 22.149234 | 29.125708 | 33.164049 | 23.982639 |
| fused | 20322 | 0.004115 | 0.011364 | 0.015873 | 0.032787 | 0.011346 |

## per-conversation breakdown

| conversation | lexical total | lexical false | lexical rate | fused total | fused false | fused rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 47 | 1 | 2.13% | 47 | 47 | 100.00% |
| 1 | 24 | 1 | 4.17% | 24 | 24 | 100.00% |
| 2 | 41 | 0 | 0.00% | 41 | 41 | 100.00% |
| 3 | 61 | 1 | 1.64% | 61 | 61 | 100.00% |
| 4 | 64 | 1 | 1.56% | 64 | 64 | 100.00% |
| 5 | 35 | 0 | 0.00% | 35 | 35 | 100.00% |
| 6 | 40 | 0 | 0.00% | 40 | 40 | 100.00% |
| 7 | 48 | 1 | 2.08% | 48 | 48 | 100.00% |
| 8 | 40 | 1 | 2.50% | 40 | 40 | 100.00% |
| 9 | 46 | 0 | 0.00% | 46 | 46 | 100.00% |

## dataset and environment

- dataset: `locomo10`
- dataset commit: `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`
- dataset sha256: `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`
- dataset fingerprint: `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`
- runner commit: `151369816c7f179c9fd369371d59ae3ad08508f8`
- python: `3.14.7`
- platform: `macOS-26.6.2-arm64-arm-64bit-Mach-O`
- retrieval limit: `top_k=200`
- retrieval modes: `lexical`, `fused`
- network: dataset fetch only; retrieval was local

## scope note

this is a retrieval-side abstention eval for conversational adversarial questions. category 5 has no valid evidence targets, so recall, precision, and mrr are null in each artifact outcome. this complements the category 1–4 retrieval benchmark in `eval/PAUS-14-RESULTS.md`.

the results record the current selection policy. this lane does not tune floors, thresholds, or policies.
