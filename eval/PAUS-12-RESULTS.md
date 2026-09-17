# PAUS-12 Arc 2 results

## Summary

- operators can review the final Arc 2 accuracy, latency, policy, and corpus-envelope evidence.
- the fused hook path passed all active warm and cold gates on both tested corpora.
- the evidence supports fused retrieval as the default recommendation, with activation still deferred.
- the real LOCOMO run remains unexecuted and requires Henry's explicit trigger.

## Implementation notes

- added the reproducible PAUS-12 evaluation record and all measured tables.
- updated the design status and measured-results section.
- made no runtime default change and made no real LLM calls.

These results were measured on commit `9d7a2c0` with the pinned
`all-MiniLM-L6-v2` bundle, ONNX Runtime 1.30.0, NumPy 2.3.3, and the
case-set fingerprint `03be0de0af3a6bdda282137842889212ff135321eea6268cea774dc7dd5eb3cf`.
No real LLM calls were made.

## Gate reruns

The benchmark ran the real fresh CLI hook process. The current Phase B
report exposes the active scan-path and fused gate names over the same
measurements. Cold means a stopped worker with cleared process caches; the
OS page cache stays warm.

| corpus | samples | files | sections | warm p95 / target | cold p95 / target | scan warm | scan cold | fused warm | fused cold |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| small synthetic | 20 | 30 | 122 | 91.71 / 300 ms | 272.14 / 750 ms | pass | pass | pass | pass |
| vault-scale synthetic | 200 | 2,000 | 8,040 | 165.34 / 300 ms | 366.73 / 750 ms | pass | pass | pass | pass |

All gate samples had zero semantic fallbacks. Scale headroom was 1.81x warm
and 2.05x cold.

### Hook-path stage metrics

Values are p50 / p95 / maximum milliseconds.

#### Small synthetic corpus

| stage | warm p50 | warm p95 | warm max | cold p50 | cold p95 | cold max |
|---|---:|---:|---:|---:|---:|---:|
| hook total | 87.72 | 91.71 | 95.23 | 258.77 | 272.14 | 277.53 |
| worker startup | 0.20 | 0.26 | 0.27 | 62.23 | 69.78 | 69.94 |
| model load | 0.00 | 0.00 | 0.00 | 110.30 | 115.05 | 117.19 |
| matrix load | 0.00 | 0.07 | 0.15 | 0.07 | 0.08 | 0.16 |
| query encode | 1.16 | 1.60 | 1.94 | 1.58 | 1.99 | 2.11 |
| vector scan | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| FTS search | 0.95 | 1.13 | 1.19 | 1.13 | 1.34 | 1.38 |
| hybrid overhead | 0.08 | 0.16 | 0.31 | 0.08 | 0.16 | 0.32 |
| fallback | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

#### Vault-scale synthetic corpus

| stage | warm p50 | warm p95 | warm max | cold p50 | cold p95 | cold max |
|---|---:|---:|---:|---:|---:|---:|
| hook total | 140.56 | 165.34 | 186.83 | 331.08 | 366.73 | 398.63 |
| worker startup | 0.20 | 0.24 | 0.38 | 67.90 | 69.93 | 75.86 |
| model load | 0.00 | 0.00 | 0.00 | 113.08 | 119.30 | 123.72 |
| matrix load | 0.00 | 0.00 | 8.84 | 2.38 | 2.63 | 10.06 |
| query encode | 1.10 | 1.37 | 1.85 | 1.74 | 2.33 | 3.16 |
| vector scan | 0.04 | 0.10 | 0.72 | 0.02 | 0.03 | 0.31 |
| FTS search | 12.67 | 20.70 | 29.69 | 12.77 | 21.08 | 31.12 |
| hybrid overhead | 0.43 | 0.66 | 0.86 | 0.43 | 0.65 | 0.84 |
| fallback | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Arc 2 retrieval evaluation

The internal harness evaluated all 59 locked cases. Fused runs used the real
worker path. Each cell is recall@4 / precision@4 / MRR; abstention rows show
abstention accuracy instead of retrieval metrics.

| category (count) | lexical baseline | fused | fused + synonyms |
|---|---|---|---|
| verbatim (31) | 100% / 100% / 1.000 | 100% / 92.74% / 1.000 | 100% / 92.74% / 1.000 |
| paraphrase (12) | 0% / 0% / 0.000 | 100% / 68.75% / 0.958 | 100% / 68.75% / 0.958 |
| held-out (4) | 0% / 0% / 0.000 | 100% / 81.25% / 1.000 | 100% / 81.25% / 1.000 |
| synonym (2) | 0% / 0% / 0.000 | 0% / 0% / 0.000 | 100% / 100% / 1.000 |
| abstention (6) | 100% abstention | 100% abstention | 100% abstention |
| wrong-project (4) | 100% abstention | 100% abstention | 100% abstention |

At cutoff 4, the aggregate rows were:

| variant | accuracy | recall | precision | MRR | abstention accuracy | forbidden violations |
|---|---:|---:|---:|---:|---:|---:|
| lexical baseline | 69.49% (41/59) | 0.6327 | 0.6327 | 0.6327 | 1.000 | 0 |
| fused | 96.61% (57/59) | 0.9592 | 0.8214 | 0.9490 | 1.000 | 0 |
| fused + synonyms | 100.00% (59/59) | 1.0000 | 0.8622 | 0.9898 | 1.000 | 0 |

The active run's accuracy by cutoff was 96.61% at 1, 100.00% at 2,
100.00% at 4, and 100.00% at 8.

### Arc 2 hypothesis comparison

The recorded values in `eval/thresholds.toml` were not changed.

| hypothesis | recorded threshold | measured value | result |
|---|---:|---:|---|
| semantic recall@4 across paraphrase + held-out | >= 0.75 | 1.0000 | pass |
| paraphrase recall@4 | >= 0.75 | 1.0000 | pass |
| held-out recall@4 | >= 0.75 | 1.0000 | pass |
| verbatim recall@4 | >= 0.98 | 1.0000 | pass |
| abstention accuracy | 1.00 | 1.0000 | pass |
| forbidden-source violations | 0 | 0 | pass |
| warm hybrid p95 at scale | <= 300 ms | 165.34 ms | pass |

### Policy ablation

The active row and every disabled-policy row came from the same fused+
synonyms run. Delta means active minus the ablated value.

| policy state | recall@4 | precision@4 | MRR | abstention accuracy | forbidden | delta recall | delta precision | delta MRR | delta abstention | delta forbidden |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| active | 1.0000 | 0.8622 | 0.9898 | 1.0000 | 0 | — | — | — | — | — |
| balanced admission disabled | 1.0000 | 0.8622 | 0.9898 | 1.0000 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| relative score floor disabled | 1.0000 | 0.7891 | 0.9898 | 1.0000 | 1 | 0.0000 | +0.0731 | 0.0000 | 0.0000 | -1 |
| semantic score floor disabled | 1.0000 | 0.6803 | 0.9898 | 0.1000 | 0 | 0.0000 | +0.1820 | 0.0000 | +0.9000 | 0 |
| synonym expansion disabled | 0.9592 | 0.8214 | 0.9490 | 1.0000 | 0 | +0.0408 | +0.0408 | +0.0408 | 0.0000 | 0 |
| ticket cross-reference filter disabled | 1.0000 | 0.8316 | 0.9898 | 1.0000 | 2 | 0.0000 | +0.0306 | 0.0000 | 0.0000 | -2 |

## Corpus envelope

The scale corpus used seed `0`, 2,000 files, and 8,040 sections. Indexing
encoded all 8,040 sections in 31,138.72 ms at 258.20 sections/second.

| measurement | value |
|---|---:|
| files / sections | 2,000 / 8,040 |
| vector rows | 8,040 |
| vector payload | 12,349,440 bytes (11.78 MiB) |
| SQLite index file | 32,555,008 bytes (31.05 MiB) |
| index build | 33,979.73 ms |
| incremental refresh | 439.95 ms |
| peak process RSS during indexed build | 34,119,532,544 bytes (31.77 GiB) |
| warm hook p95 at scale | 165.34 ms |
| cold hook p95 at scale | 366.73 ms |

The peak RSS is an indexing-process observation on this machine. It is a
residual scale risk, despite the hook-path gates passing. The vector payload
and SQLite file sizes are the persistent query-time footprint.

## LOCOMO readiness

The runner completed ingest, search, and evaluate stages on the committed
fixture `tests/fixtures/locomo/small.json`. The smoke run used 2 questions,
16 stub calls, zero evaluation errors, and wrote a complete `run.json`.

Exact smoke command:

```bash
PYTHONPATH=. .venv/bin/python - <<'PY'
from pathlib import Path
from eval.benchmarks.locomo.run import StubTransport, run_locomo

fixture = Path("tests/fixtures/locomo/small.json")
results = Path(".artifacts/locomo-smoke-results")
run_locomo(run_id="paus12-smoke", dataset_path=fixture, results_dir=results,
           top_k=8, cutoffs=(1, 2, 4, 8), predict_only=True)
responses = [item for _ in range(8)
             for item in ("ANSWER: fixture",
                          {"label": "CORRECT", "reasoning": "fixture"})]
run_locomo(run_id="paus12-smoke", dataset_path=fixture, results_dir=results,
           top_k=8, cutoffs=(1, 2, 4, 8), answerer_model="stub-answerer",
           judge_model="stub-judge", provider="stub", judge_provider="stub",
           evaluate_only=True, transport=StubTransport(responses))
PY
```

The comparable real run uses fused retrieval and has a 12,320-call no-retry
minimum: 1,540 questions, four cutoffs, one answerer call, and one judge call
per cutoff. The retry ceiling is about 61,600 calls when every answerer and
judge call uses all five attempts. The expected practical range is 12,320 to
24,640 calls when retries are rare or limited to one extra attempt. It needs no
calls for the predict-only stage. Henry can trigger it after caching and verifying
`datasets/locomo/locomo10.json`:

```bash
PYTHONPATH=. .venv/bin/python -m eval.benchmarks.locomo.run \
  --dataset-path datasets/locomo/locomo10.json --run-id locomo-full \
  --retrieval-mode fused \
  --top-k 200 --top-k-cutoffs 10,20,50,200 --provider openai \
  --judge-provider openai --answerer-model gpt-4o-mini \
  --judge-model gpt-4o-mini --predict-only
PYTHONPATH=. .venv/bin/python -m eval.benchmarks.locomo.run \
  --dataset-path datasets/locomo/locomo10.json --run-id locomo-full \
  --retrieval-mode fused \
  --top-k 200 --top-k-cutoffs 10,20,50,200 --provider openai \
  --judge-provider openai --answerer-model gpt-4o-mini \
  --judge-model gpt-4o-mini --resume
```

The real run knobs are `--conversations`, `--with-evidence`,
`--user-profile`, `--answerer-model`, `--judge-model`, `--provider`,
`--judge-provider`, `--top-k`, and `--top-k-cutoffs`. The runner requires
provider credentials only for the evaluation stage.

## Recommendation

Recommend fused retrieval as the default runtime path, with the existing
lexical fallback retained. Accuracy rose from 69.49% lexical baseline to
100.00% with fused synonyms. All Arc 2 hypotheses passed. Scale warm and cold
hook p95 stayed below 300 ms and 750 ms, with zero fallbacks. The ablation
shows that score floors and ticket filtering protect abstention and source
precision.

Caution remains warranted. Paraphrase precision is 0.6875, the scale index
build took 33.98 seconds, and peak build RSS reached 31.77 GiB. The result
supports a default fused hook path for the tested envelope. It does not prove
the design's 50,000-section planning envelope or replace a future memory
test. No default was changed in this ticket.

## Verification

- `python3 -m venv .venv && .venv/bin/pip install -q -e '.[dev]'`
- `PYTHONPATH=. .venv/bin/pytest tests -q` — 135 passed in 51.14 seconds.
- `PYTHONPATH=. /tmp/paus12-base-venv/bin/pytest tests -q` — 121 passed, 14 skipped in 10.96 seconds.
- `python3 -m venv .venv && .venv/bin/pip install -q -e '.[dev,semantic]'`
- small and scale `pausanias bench --hook-path` runs — all four gates passed.
- full lexical, fused, and fused+synonyms internal evaluations — active run passed; baselines recorded without changing thresholds.
- LOCOMO fixture smoke — 16 stub calls, zero errors.

Simplicity pass: added one result record and one measured-results section. No
new runtime flags, ranking paths, or duplicate evaluation machinery were added.
