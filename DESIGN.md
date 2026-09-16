# Pausanias: Automatic Memory Context for Agents

Status: proposed design. No implementation exists yet.
Date: 2026-09-16

## Purpose

Pausanias retrieves relevant memory before an agent answers a user. It inserts a small evidence packet into the model's context automatically. The model can then search further or read the original sources.

The problem is discovery: an agent may not search for useful history because it does not know that history exists. Instructions such as “check memory first” also depend on the model following them. Pausanias moves the initial search into the harness lifecycle.

The proposed first application is a local coding assistant using Henry's markdown vault. The retrieval core must not depend on a particular model or harness. Claude Code is the first proposed adapter because its documented prompt hook can return additional context. Other adapters require verified lifecycle support before implementation.

## Design decisions

- Run bounded retrieval before each user turn, outside the main model's decision loop.
- Keep explicit search and source-read commands available for deeper investigation.
- Treat injected memory as evidence, never as instructions or proof of current external state.
- Start with existing markdown notes. Do not automatically extract facts from conversations in the first version.
- Use local full-text search first. Add embeddings only if evaluation shows a useful gain.
- Preserve original sources and expose why each result was selected.
- Return no context when evidence is weak. A filled memory budget is not a goal.

These are implementation proposals, not a commitment to adopt a dependency or replace the existing vault workflow.

## User experience

The user asks: “Why did we choose one sandbox image?”

Before the model receives the turn, Pausanias searches the active project's memory scope. It finds a decision note, then injects a short excerpt, its date, and its path. The agent can answer from that evidence or read the full note.

If the user asks, “Is that still deployed?”, the note supplies background only. The agent must check the deployment or current repository state.

Routine retrieval is silent. An inspection command shows the selected sources, omitted candidates, token cost, latency, and errors for a turn. It should not require the user to read logs during normal work.

## Scope

### First version

- One local user and an explicitly configured set of markdown roots.
- A rebuildable SQLite full-text index.
- A command-line retrieval core and one prompt-hook adapter.
- Automatic excerpt injection, explicit search, and full-source reads.
- Source version tracking, scope filtering, bounded context, and local evaluation.

### Later, only with evidence

- Embeddings and hybrid ranking.
- Model-generated search queries or reranking.
- Conversation-derived claims with evidence and validity intervals.
- Additional harness adapters.

### Non-goals

- Model training or changes to model weights.
- A replacement for repository instructions, task state, or live operational systems.
- A graph database, hosted multi-tenant service, or general workflow engine.
- Automatic edits to source notes or automatic adoption of assistant statements as facts.

## Architecture

The request path is:

1. The adapter receives a user-turn event.
2. It constructs a retrieval request from the prompt and trusted session metadata.
3. The core resolves allowed roots and project scope.
4. It searches the local index and verifies candidate source versions.
5. It filters, ranks, removes duplicates, and packs excerpts within budget.
6. The adapter inserts the packet as additional context before inference.
7. The agent optionally uses explicit search or source-read commands.

Indexing runs separately from the request path. A failed or slow index refresh must not delay every prompt. The same retrieval core serves the hook and explicit search, so their results remain explainable and consistent.

### Components

| Component | Responsibility |
| --- | --- |
| Indexer | Read allowed files, split sections, update the rebuildable index |
| Retriever | Resolve scope, search, validate candidates, rank, and pack |
| Harness adapter | Translate lifecycle input and output; enforce the deadline |
| CLI | Search, inspect sources, rebuild the index, and inspect turn decisions |
| Evaluation runner | Replay test conversations and compare answer quality |

Use Python and its standard-library SQLite support for the initial implementation. Check
FTS5 availability during setup. Use a persistent local semantic worker only for the
optional embedding lane. The worker keeps model and vector caches warm; lexical retrieval
remains usable without it.

## Source and index model

Markdown files remain authoritative. The index contains derived data and can be deleted and rebuilt.

Split files at headings, with a size limit for long sections. Preserve heading ancestry. Do not split merely to force every sentence into a standalone fact; decisions often need their rationale.

Each indexed section contains:

- A stable section identifier, canonical path, heading, and current line range.
- Root identity and configured project scope.
- Text, content hash, index generation, and indexing timestamp.
- Available frontmatter, including note type and declared update date.
- Explicit source links already present in the note.

Keep document update time distinct from event time. A recently edited note may describe an old event. Missing dates remain unknown; the indexer must not invent them.

Use a canonical path and heading identity for section continuity. Use a content hash to detect revisions. Line numbers are display hints, not durable identifiers.

Delete indexed sections when files disappear. Before injection, confirm that each selected file still exists within its allowed root and matches its indexed version. On mismatch, skip that result and schedule a refresh. This prevents stale index text from surviving a source edit or deletion.

## Retrieval contract

The request carries:

- `prompt`: the current user message.
- `recent_user_messages`: an optional bounded history supplied by the adapter.
- `project_id`: configured workspace identity, not a value inferred from retrieved text.
- `session_id`, `turn_id`, and `context_epoch`: identity for retries and context resets.
- `budget_tokens` and `deadline_ms`: hard limits.

The response carries:

- `status`: `ok`, `empty`, `timeout`, or `error`.
- `items`: excerpt, source path, heading, content hash, available dates, and selection reason.
- `rendered_context` and its estimated token count.
- `index_generation` and `trace_id` for inspection.

An error or empty result produces no evidence packet. Internal diagnostics remain outside the model's evidence text.

### Query construction

Start with the current prompt, preserving exact identifiers and quoted terms. Escape user text before constructing an FTS query.

For short follow-ups such as “what about that?”, add bounded recent user messages and the active project. If the adapter cannot supply history, search conservatively or return no results. Do not pretend that the current prompt resolves the reference.

Do not include previously injected packets or generated answers in the default query. Otherwise, a bad memory can cause its own repeated retrieval.

### Scope and ranking

Apply allowed-root and project filters before candidate selection. Project requests can also search explicitly configured global notes. Cross-project search requires an explicit request or configuration; semantic similarity cannot grant access.

Use full-text ranking, exact identifier matches, and heading/title matches. Rank within each query; raw full-text scores are not universal confidence probabilities. Calibrate acceptance rules on the local evaluation set.

Prefer direct decision notes over incidental mentions when the question asks for rationale. Use dates when the question asks about time, but do not treat recent modification as factual authority.

Group overlapping sections from the same document and remove exact duplicate excerpts. Do not merge distinct claims solely because they are similar. When sources disagree, preserve the disagreement and provide both sources if the budget permits.

## Context packet

Use a clearly delimited data block containing:

- A statement that the excerpts are retrieved evidence and may be incomplete or outdated.
- Up to a few excerpts with paths, headings, versions, and available dates.
- A reminder to read original sources for missing details and verify current external state.

Put integration instructions in the adapter configuration, outside retrieved content. Escape delimiters so a note cannot terminate the evidence block. Retrieved instructions, shell commands, and claimed authority remain source text; they do not become permission to act.

Initial budgets are hypotheses for evaluation:

- At most 4 excerpts and 1,200 estimated tokens per user turn, including labels.
- At most 350 estimated tokens per excerpt.
- A 750 ms target for the first query after index refresh with a warm OS page cache, fresh SQLite connection with an empty page cache.
- A 300 ms target for warm local retrieval at p95.
- A 750 ms hard adapter deadline, after which the turn proceeds without memory.

Use the target model's tokenizer when available. Otherwise, apply a conservative byte limit as well as an estimate. Cut at section or sentence boundaries where possible; mark truncation and retain the source pointer.

The first version removes duplicates within a packet and handles duplicate delivery of the same turn idempotently. It does not suppress a source merely because an earlier turn received it. A harness may compact that earlier context without exposing a reliable event. Cross-turn suppression requires verified context-epoch handling; correctness comes before saving those tokens.

## Harness integration and explicit access

For Claude Code, use a synchronous `UserPromptSubmit` command hook to produce `additionalContext`. The hook calls the local CLI and enforces the deadline. Validate the exact installed hook contract before writing configuration.

“Synchronous” here means the packet arrives before inference starts. It does not mean retrieval can wait without a bound. A timeout must terminate the retrieval child, not leave accumulating background processes.

Proposed CLI commands:

- `pausanias context`: read a structured request on stdin and return a packet.
- `pausanias search`: return ranked excerpts and source references for explicit investigation.
- `pausanias read`: return a bounded source section from an allowed root.
- `pausanias index`: rebuild or incrementally refresh the index.
- `pausanias inspect`: explain a recorded retrieval decision.

These are proposed interfaces, not installed commands. The model should receive instructions for search and read during adapter setup. Native shell access is sufficient for the first version; an MCP server is not required.

Use a scheduled incremental index command initially. The hook may request a refresh, but must not rebuild the corpus on the foreground path. Use transactions so readers see a complete index generation, never a partial refresh.

## Failure behavior and data boundaries

- Retrieval failure: continue the user turn without memory and record the failure locally.
- No useful match: inject nothing. Do not fill the packet with low-quality results.
- Changed or removed source: skip it and refresh the index.
- Concurrent refresh: serve a complete readable generation and validate selected sources.
- Repeated hook delivery: return the same packet for the same request identity and source generation.
- Index corruption: disable retrieval, report the diagnostic, and allow an explicit rebuild.

Only index configured roots. Resolve symlinks and enforce containment during indexing and reading. Exclude credentials, environment files, and configured private paths.

Local indexing does not mean local-only disclosure: injected excerpts go to the selected model provider. Source-root configuration must make that boundary explicit. No separate remote embedding or extraction service is needed in the first version.

Store source identifiers, hashes, scores, timing, and selection reasons in normal traces. Keep raw prompts and excerpts out of normal diagnostic logs. Full debug capture is explicit, local, and subject to a retention limit.

## Evaluation

Replay the same multi-turn tasks under four conditions:

1. No memory access.
2. Model-chosen search only.
3. Automatic retrieval only.
4. Automatic retrieval plus model-chosen search.

Hold the model, task, corpus snapshot, generation settings, and total context allowance constant. Record actual tokens and retrieval/tool latency separately. Use repeated trials to account for model variation.

Include questions about prior decisions, exact ticket identifiers, paraphrased concepts, ambiguous follow-ups, changed preferences, conflicting notes, unrelated projects, deleted sources, and questions with no relevant memory. Include adversarial instructions inside notes.

Each case specifies the expected answer, acceptable source evidence, forbidden stale claims, and whether retrieval should abstain. Evaluation requires human-reviewed cases; an LLM judge alone is insufficient for source correctness.

Measure:

- Final answer correctness and supported claims.
- Recall of necessary evidence and irrelevant injected tokens.
- Stale-fact errors, wrong-project results, and appropriate abstention.
- Retrieval latency, added tokens, and explicit follow-up searches.
- Compliance with malicious instructions embedded in retrieved material.

The current case file and all current metric baselines are PROVISIONAL. The 57 cases,
including 12 paraphrase cases and 4 held-out cases, are a DRAFT pending Henry's human
review. After review, keep at least 40 cases spanning the categories above. Treat results
as a pilot, not a statistical guarantee. Choose a larger evaluation size after measuring
baseline variation.

The first release requires no source-boundary violations in the test set, correct deletion handling, and bounded timeout behavior. Adopt automatic retrieval by default only if paired evaluation improves useful recall without increasing unsupported or stale claims. Record thresholds before tuning on the development set, then test on held-out cases.

## Implementation sequence

1. Build the corpus reader, FTS index, source-read command, and explicit search baseline.
2. Build the reviewed evaluation cases and measure model-chosen retrieval.
3. Add packet construction, bounded hook integration, diagnostics, and failure tests.
4. Compare all four conditions. Tune budgets and acceptance rules on development cases only.
5. Add embeddings only if paraphrase failures remain material on held-out cases.
   The semantic retrieval foundation and ticket ladder are specified in the section below.

Do not build automatic memory writing as part of these steps. Existing vault authoring rules remain in effect.

## Semantic retrieval

Arc 1 has a PROVISIONAL lexical baseline. Arc 2 adds a local semantic lane because the
DRAFT results currently show failures on all 16 paraphrase and held-out cases. This
section is a foundation design, not a commitment to ship semantic results before the
evaluation gates pass.

### Recommendation summary

| Area | Recommendation | Runner-up | Why the runner-up loses |
| --- | --- | --- | --- |
| Embedding runtime | Optional ONNX Runtime CPU backend with a pinned model bundle | llama.cpp-style GGUF process | The process boundary and tool version make refresh and reproducibility harder |
| Vector storage | Float32 BLOBs in the same SQLite database, scanned with NumPy | sqlite-vec | An extension adds platform and schema risk before this corpus needs an ANN index |
| Hybrid ranking | Reciprocal rank fusion with an exact lexical guard | Normalized weighted scores | Score calibration is fragile across queries and model versions |
| Query expansion | A small versioned operator synonym table as a lexical complement | Corpus-derived co-occurrence terms | Automatic terms add noise and are harder to audit |
| Refresh | Re-embed only sections whose source or model version changed | Re-embed every indexed section | Full re-embedding wastes the existing source-version contract |
| Evaluation | Separate semantic, lexical, scope, abstention, and latency gates | One aggregate recall threshold | An aggregate can hide a semantic gain or a lexical regression |

All current counts, scores, and latency figures are PROVISIONAL. The evaluation file is
DRAFT until Henry completes human review. The development and held-out case sets must be
reviewed and FROZEN before any ticket tunes embeddings, ranking, synonyms, budgets, or
acceptance thresholds.

### Embedding runtime

**Recommendation:** make semantic retrieval an optional extra using ONNX Runtime's CPU
execution provider and a pinned ONNX export of `sentence-transformers/all-MiniLM-L6-v2`.
The model produces 384-dimensional sentence vectors and is intended for semantic search.
Use the model's exact tokenizer, mean pooling, and L2 normalization. Pin the model files,
tokenizer files, model hash, ONNX Runtime version, and preprocessing contract in a local
manifest. The model card reports 22.7M parameters, so an uncompressed float32 model is
about 90 MB before tokenizer metadata.

The model is fetched only by an explicit setup command. The command verifies a SHA-256
manifest and records the model and tokenizer licenses. Indexing and query execution fail
closed to the lexical lane if the bundle is missing. Neither path downloads a model.
The model is Apache-2.0 licensed. ONNX Runtime is MIT licensed. These licenses do not
replace review of the model's training-data and redistribution terms.

ONNX Runtime has a stable Python API and CPU wheels for the target platforms. Keep it
outside the default dependency set. A persistent local semantic worker owns one model
session and the vector-matrix cache. The hook's short-lived CLI connects to that worker
over a permissioned local socket, so a fresh CLI does not reload the 90 MB model. Batch
section encoding during refresh and encode one query per request.

The main determinism contract is the manifest, not bit-for-bit equality across every CPU.
For the same manifest and inputs, repeated runs on one platform must have a maximum
absolute vector-component difference of `1e-6` and identical ranking. Across supported
platforms, the cosine difference must be at most `1e-4`; otherwise the manifest or
backend is incompatible. Rank-stability fixtures must have at least `0.002` fused-score
gap between adjacent expected results. A gap below `0.001` is a near tie and may reorder,
but its order must still be deterministic by lexical rank, then section ID. Gaps below
`0.002` are not valid evidence for a cross-platform rank-stability claim.

Evaluate macOS arm64, Linux x86_64, and Windows x86_64 separately when those platforms
are supported. Each platform must meet the same 300 ms warm p95 and 750 ms cold p95
limits; do not pool percentiles across platforms or borrow another platform's result.
Record the platform, CPU provider, runtime version, and manifest in every report. If a
platform cannot install the optional backend, mark semantic gates disabled and require
the lexical gates. Store the model fingerprint, tokenizer fingerprint, dimension, dtype,
normalization, and metric with each index generation. Reject mixed model generations
instead of silently comparing incompatible vectors.

**Runner-up:** a llama.cpp-style GGUF embedder. It can use a small quantized artifact and
keep Python's default dependencies at zero. It loses because it needs a separately shipped
binary, a subprocess protocol, model conversion choices, and process startup handling.
Those choices complicate Windows support, refresh errors, query latency, and exact
reproduction. It remains a valid escape hatch if ONNX Runtime misses the warm-query budget.

Apple Core ML or MLX may be faster on Apple Silicon, but they create a platform-specific
backend and a second model conversion path. Shelling out to an existing local tool has the
lowest code cost only when that tool is already present and pinned. Neither is a suitable
default for a portable Python core.

### Vector storage and search

**Recommendation:** store one little-endian float32 vector as a BLOB per indexed section,
then load the in-scope vectors into one NumPy matrix for an exact cosine scan. Normalize
vectors at write time and normalize the query before the scan. Store vectors in a table
keyed by `section_id`, with the source content hash and embedding manifest fingerprint.
Do not store a second external index.

At 384 dimensions, the raw vector costs 1,536 bytes. Ten thousand sections use about 15
MiB of vector data. Fifty thousand use about 73 MiB. One hundred thousand use about 146
MiB, before SQLite and matrix overhead. This makes exact scanning a good fit for the
current thousands-of-chunks corpus. The initial operating envelope is 50,000 sections.

Treat 50,000 sections, a 75 ms p95 vector-scan budget, or a 256 MiB resident vector
matrix as a review trigger. Measure 10,000, 25,000, 50,000, and 100,000 sections.
Brute force stops being the assumed solution at the first failed trigger. These are
planning thresholds, not measured claims. A standard-library `array` fallback may keep
the storage format readable, but it is not the performance path. If NumPy is unavailable,
keep FTS search available and disable the semantic lane with a diagnostic.

Use schema version 2 for the first implementation that adds vectors. The `embeddings`
table participates in the same transaction as `files`, `sections`, `sections_fts`, and
`metadata`. A rebuild drops and recreates all derived tables, including embeddings. An
incremental refresh updates all changed sections and the generation metadata in one
commit. Readers see the prior complete generation until that commit, then the new
complete generation. Never join a vector from one generation to FTS rows from another.

**Runner-up:** sqlite-vec. It keeps vector search inside SQLite and may become useful
after the corpus outgrows exact scans. It is pre-v1, permits breaking changes, and adds
an extension-loading and binary-distribution contract. Its filtering and transaction
behavior would need a separate acceptance test for every supported platform. FAISS is
also rejected for the foundation: it is a strong library, but its index is a second
artifact that weakens the single-file atomic reset contract. Reconsider either option
only after the scale probe crosses the review trigger.

### Hybrid ranking

**Recommendation:** use reciprocal rank fusion (RRF) over two bounded lanes, with a
lexical guard for exact identifiers and quoted phrases.

1. Resolve the allowed root IDs and global-note paths with the existing scope logic.
2. Run FTS, including conservative synonym variants, only against that scope.
3. Scan only vectors from that same scope. Do not let vector similarity widen access.
4. Take bounded top candidates from each lane. Use `C = min(MAX_CANDIDATES,
   max(5 * limit, 50))` as the lane and union bound. For a guarded query, admit the
   lexical candidates first, then vector candidates until the union reaches `C`.
5. For each candidate, add `1 / (60 + rank)` for each lane that returned it. Keep rank 1
   as the best rank. Use deterministic lexical and section-ID tie breakers.
6. Detect a guarded query when the normalized prompt contains an identifier matching
   `[A-Za-z][A-Za-z0-9]*-[0-9]+` or a non-empty quoted phrase. A lexical candidate is
   protected only when it matches that exact identifier or phrase in indexed heading or
   text and also matches the complete primary FTS query. For a compound query, an
   unmatched ordinary term does not become a protected partial hit.
7. Reserve output positions for protected lexical candidates in ascending lexical rank.
   If several protected atoms match, use the best atom rank, then `section_id`. Remove
   reserved candidates before RRF. Fill remaining positions with the fused candidates.
   Thus a vector-only candidate can never rank above a protected lexical candidate, even
   when its vector rank is better. If protected hits exceed `limit`, return their first
   `limit`; if they exceed `C`, the cap still keeps the first `C` lexical hits.
8. Validate source existence and content hash, remove duplicates, and return at most the
   requested `limit`. Apply the cap before validation, but never discard a protected
   lexical candidate in favor of a vector candidate.

RRF avoids treating BM25 and cosine values as comparable confidence scores. The lexical
guard protects the current exact-query behavior while the vector lane supplies recall
for paraphrases. Record lexical rank, vector rank, both raw scores, fused score, and the
guard reason for inspection. These diagnostics do not enter the evidence packet.

**Runner-up:** score normalization plus a weighted sum. It loses because min-max or
z-score ranges change with scope size, query wording, and model version. A weight tuned
on the current corpus can regress on a new project. A vector-only-on-FTS-miss cascade is
simpler but misses semantic results whenever FTS returns plausible distractors. A
reranker adds another model and cost before first-stage recall is proven.

The final result bound and scope rules remain the existing contract. A semantic result
cannot cross a project or root boundary, and `all_projects` remains the only explicit
cross-project escape. A global note remains eligible only through the same configured
global-note rule.

### Query expansion

**Recommendation:** add a small, versioned, operator-owned synonym table as a lexical
complement to embeddings. An entry maps a canonical term to a bounded list of aliases.
Expand only ordinary lexical terms. Preserve ticket identifiers, quoted phrases, and the
64-term query cap. Build query variants locally, run them under the same scope, and merge
them into the lexical lane before RRF. Include the synonym-table fingerprint in the
ranking trace and index metadata when expansion affects indexing.

Do not use network-generated rewrites. Do not expand from retrieved text. Embeddings are
the primary bridge for ordinary paraphrases; synonyms cover stable local vocabulary such
as project names, abbreviations, and known product terms. A missing table means no
expansion, not an error. Add stemming only if a measured case shows a morphology gap;
do not add a lemmatizer dependency for a speculative gain.

**Runner-up:** corpus-derived co-occurrence expansion. It loses because it can turn a
common term into an unrelated query, changes as the corpus changes, and is difficult to
explain to a user. Embedding-neighbor terms duplicate the vector lane and amplify its
errors. A synonym table is smaller, auditable, deterministic, and easy to disable.

### Incremental refresh

**Recommendation:** extend the current source-version contract to vectors and keep one
complete active generation. Re-embed only changed sections, unless the model manifest
changes.

Reuse the current file contract: a section is unchanged only when its canonical path,
root ID, modification time, and file content hash still match. When a file changes,
re-embed every section created from that file because the current hash is file-level.
When a file is removed, delete its sections and vectors in the same transaction. When
the model, tokenizer, pooling, normalization, dimension, or embedding format changes,
mark the whole vector store stale and require a full re-embedding. Do not mix old and new
vectors in one active generation.

Compute changed vectors in batches before taking the final write transaction. Recheck
the source hash before commit. If a source changed during encoding, discard that vector
and leave the old complete generation visible. The next refresh retries it. The commit
must still atomically update the FTS and vector generations.

For `N` sections and `C` changed sections, the expected work is approximately one model
load plus `N` encodes for a full rebuild, or one model load plus `C` encodes for refresh,
plus the SQLite writes and vector-matrix load. Batch throughput is the important index
metric. A no-change refresh should not load the model or touch vector BLOBs. A model
change is intentionally a full rebuild cost.

**Runner-up:** re-embed every indexed section on every refresh. It loses because a normal
source edit would pay the full corpus cost and would make scheduled refreshes less useful.
Use it only as a repair command after manifest or storage corruption.

### Semantic index-state contract

Expose one semantic state for each database: `ready`, `stale`, or `disabled`.

| State | Meaning | Query behavior | Recovery |
| --- | --- | --- | --- |
| `ready(g)` | FTS and vectors are complete for generation `g`, with one matching manifest | Run lexical and semantic lanes against `g` | A source or manifest change moves the state out of `ready` |
| `stale(g, reason)` | FTS has advanced, or vectors are absent or incompatible with the active FTS generation | Run lexical retrieval only; never combine an old vector with new FTS rows | Refresh changed vectors or run a full re-embedding |
| `disabled(reason)` | The optional extra, model bundle, NumPy, schema, or database is unavailable or invalid | Run lexical retrieval only and emit one diagnostic per state change | Install or repair the cause, then rebuild and atomically enter `ready` |

Refresh may advance FTS when the model or optional extra is missing. It commits the new
FTS generation with `stale` or `disabled` metadata. It must not present old vectors as
current semantic results. A temporary encode or source-hash failure leaves the previous
complete generation intact but marks the active state `stale`; a model-manifest change
requires a full re-embedding. State changes, generation IDs, manifest fingerprints, and
reasons commit in the same transaction as their derived rows.

Readers pin one complete generation at query start and check the state before opening the
vector lane. A refresh builds changed vectors off to the side, rechecks source hashes,
then atomically publishes FTS, vectors, and metadata. Cleanup removes superseded vectors
after publication and removes orphaned temporary rows at startup. A failed cleanup does
not make a published generation unreadable. A repair command can set `disabled`, rebuild
all derived tables, and publish `ready`; no query retries a known disabled backend.

### Cold-path architecture

**Recommendation:** use one persistent local semantic worker per configured database.
The hook's short-lived CLI connects to the worker over a permissioned local socket. On a
socket miss, the CLI starts one worker under a lock and waits only until the adapter's
remaining deadline. If the worker is not ready, the CLI returns the lexical result and
leaves the worker to finish for the next turn. This removes repeated model and matrix
loads from normal turns without making semantic retrieval a prerequisite for a turn.

The worker lifecycle is bounded and recoverable:

- Keep one ONNX session per model-manifest fingerprint and one vector matrix per index
  generation plus scope key. Load each lazily on the first request that needs it.
- On refresh, publish the new complete generation first. Drop the old matrix only after
  the swap. Never answer with FTS rows and vectors from different generations.
- Exit after a configurable idle period, with 30 minutes as the initial default. A
  stale socket, crashed worker, or failed load is removed or replaced on the next start.
- A missing extra, model bundle, NumPy, or compatible manifest marks the semantic lane
  disabled. The hook uses lexical retrieval and records the reason; it does not retry
  the same failed load on every turn.
- The worker stores no source text outside the SQLite database and memory. A local socket
  permission check and database ownership check enforce the existing scope boundary.

The existing 750 ms hard adapter deadline remains a safety bound, not a semantic latency
target. The hook must stop waiting at that deadline, terminate its retrieval child, and
continue without memory or with the lexical fallback. A cold semantic request passes only
when the full hook path returns semantic context before the deadline; a timeout is a
fallback and a cold-gate failure.

A bounded probe on this machine produced the following non-binding numbers: macOS
26.6.2, arm64, 15 CPU cores, 52 GB RAM, Python 3.13.13, ONNX Runtime 1.30.0, NumPy
2.5.3, and tokenizers 0.23.2. Loading the 97 MB ONNX file took 70.23 ms. The first
32-token encode took 2.86 ms; warm encode p95 over 50 runs was 2.17 ms. The probe did
not measure worker startup or vector-matrix loading. These are probe-derived observations,
not product baselines or cross-platform promises.

The runner-up is a fresh process with lazy model and matrix loading plus an honest cold
budget. It has less lifecycle code, but it makes every hook turn pay process startup and
risks repeated 90 MB model loads. The persistent worker is the single recommendation
because it makes the measured warm path match normal use while retaining a bounded cold
fallback.

### Performance budgets

**Recommendation:** enforce end-to-end gates on the real hook path, including the fresh
CLI process, worker IPC, query encoding, FTS, vector scan, fusion, source validation, and
packet packing. Do not enforce a semantic gate from an in-process `search()` call alone.

The base design's 750 ms first-query target uses a warm OS page cache and is a lexical
refresh diagnostic. The base 300 ms warm-local target is likewise insufficient for the
semantic lane. The gates below replace those shortcuts for semantic acceptance; the
existing 750 ms adapter deadline remains the hard fallback bound.

`pausanias bench --hook-path` must launch the actual adapter command as a child for every
trial. The warm workload uses a running worker with its session and matrix cache loaded.
The warm semantic gate is total hook-path p95 at or below 300 ms, with no semantic
fallbacks. A separate cold workload stops the worker, clears its session and matrix, and
invokes the same hook path. Its cold semantic gate is total p95 at or below 750 ms, with
every trial completing before the adapter deadline and no fallback. A cold timeout or
fallback fails that gate; the adapter still returns lexical or empty context before its
hard deadline. If the optional backend is unavailable, report the semantic gates as
disabled and enforce the lexical gate instead.

| Stage | Warm p95 planning budget |
| --- | ---: |
| Hook process, config, IPC, and SQLite open | 35 ms |
| Query tokenization and embedding | 50 ms |
| FTS and synonym-lane search | 35 ms |
| Vector matrix lookup and exact scan | 75 ms |
| RRF, dedupe, source validation, and packing | 80 ms |
| Instrumentation and scheduling margin | 25 ms |
| **Total** | **300 ms** |

The warm stage numbers are budgets, not measurements. `pausanias bench` must report p50, p95,
and maximum values for hook total, worker startup, model load, matrix load, query
encoding, FTS search, vector scan, hybrid overhead, and fallback count. It must also
report encoded-section throughput, vector candidate count, corpus section count, worker
cache state, platform, and model manifest. Keep the existing lexical report and gate. A
semantic p95 failure must never be hidden by a fast FTS fallback.

**Runner-up:** retain only the current aggregate FTS timing. It loses because it cannot
show whether encoding, vector scanning, or fusion consumed the budget. Stage metrics are
needed to choose a simpler fix when a gate fails.

### Evaluation extensions

**Recommendation:** extend the existing harness to report fixed category metrics and
regression gates without changing the 57 cases in this ticket. The 12 `paraphrase` and 4
`held-out` cases are the Arc 2 target set. Keep the four held-out cases out of tuning and
report them separately from development cases.

Record these thresholds as hypotheses before implementation tuning:

- semantic recall@4 of at least 0.75 across the 16 paraphrase and held-out cases;
- paraphrase recall@4 of at least 0.75 and held-out recall@4 of at least 0.75;
- standard exact/verbatim recall@4 of at least 0.98, with the Arc 1 1.0 result treated
  as the regression reference;
- abstention accuracy of 1.0 and zero forbidden-source violations;
- warm hybrid p95 at or below 300 ms on the benchmark workload.

The harness must print actual values beside each hypothesis. A failed hypothesis is a
result to record, not a reason to edit the threshold. If the case set changes later,
record the new baseline and rationale before tuning.

Add cases or fixtures for:

- model or tokenizer manifest drift, including a refusal to mix generations;
- a missing model bundle and lexical fallback;
- repeated queries with stable ordering and near-tie handling;
- exact ticket and quoted-term cases where vector results are distractors;
- semantic results that would cross a project or root boundary;
- deleted and changed sources after vector indexing;
- no-relevant-memory queries that are semantically close to common corpus terms;
- ranking regressions on all existing verbatim cases.

**Runner-up:** raise the existing aggregate recall threshold. It loses because the current
aggregate mixes lexical, semantic, and abstention behavior. Category metrics expose both
the desired semantic gain and an exact-match regression. Keep the existing aggregate for
continuity, but do not use it as the Arc 2 decision by itself.

### Dependency policy

**Recommendation:** keep the base package at zero dependencies and put only ONNX Runtime
and NumPy in an optional semantic extra. Treat the model bundle as an explicitly fetched,
hashed data artifact, not as a package dependency.

The default install remains dependency-free. Semantic retrieval is an opt-in extra.
Sizes below are order-of-magnitude planning values; record exact wheel and model sizes
for the pinned versions in the implementation PR.

| Item | Approximate size and transitive weight | Maintenance and license | Decision |
| --- | --- | --- | --- |
| `onnxruntime` CPU | 20–80 MB platform wheel; one native runtime, plus its declared Python dependencies | Active upstream; MIT; platform wheel churn is the main risk | Use as the optional inference backend |
| `numpy` | 15–30 MB platform wheel; native array and BLAS components | Mature upstream; BSD-3-Clause; ABI and wheel coverage need pinning | Use for batched dot products and exact scans |
| `all-MiniLM-L6-v2` bundle | About 90 MB float32 weights, plus tokenizer/config files; no Python transitive dependencies | Apache-2.0 model; model provenance and redistribution terms need recording | Fetch explicitly and pin by hash; do not vendor in the package |
| `tokenizers` | Usually a few MB native wheel, plus Rust ABI/platform variants | Useful but adds a native boundary and another release stream | Do not require it; use a pinned stdlib-compatible tokenizer path first |
| `sentence-transformers` and PyTorch | Hundreds of MB or more with many transitive packages | Large maintenance and security surface; model wrapper is unnecessary | Reject; it exceeds the foundation's dependency budget |
| `sqlite-vec` | Small native extension, but one binary per platform and pre-v1 API | MIT/Apache-2.0; breaking-change and extension-loading risk | Revisit only after exact-scan scale failure |
| `faiss-cpu` | Typically tens to hundreds of MB with native numerical libraries | Mature but heavier; external index lifecycle | Reject for the single-file foundation |

The no-dependency alternative is a shell-managed GGUF tool plus a pure-Python vector
scan. It preserves the default install but loses a stable in-process API and misses the
query budget sooner. The chosen set is the smallest optional set that meets the stated
performance hypothesis. If NumPy or ONNX Runtime cannot be installed, lexical retrieval
must continue to work.

**Runner-up:** keep all Python dependencies at zero by shelling out to a GGUF tool and
scanning vectors in pure Python. It loses the stable in-process API and the performance
hypothesis. This remains the fallback if optional wheels are unavailable on a platform.

### Ticket ladder

The lanes are ordered so that storage, versioning, and evaluation exist before ranking
polish. PAUS-8 lands before the retrieval it judges. No ticket may tune an embedding,
ranking rule, synonym, budget, or threshold until PAUS-8 has completed Henry's human
review and frozen both the development and held-out case sets.

- **PAUS-5:** Define the model bundle manifest, offline fetch, hash verification, license
  record, tokenizer contract, and optional dependency extra. Depends on PAUS-4.
- **PAUS-6:** Add schema version 2, embedding metadata, atomic generation handling, and
  changed-section refresh without changing result ranking. Depends on PAUS-5.
- **PAUS-7:** Add the scope-safe BLOB store, in-memory NumPy matrix, exact cosine scan,
  bounded candidate API, and lexical fallback. Depends on PAUS-6.
- **PAUS-8:** Extend the eval runner with category reports, Arc 2 hypothesis thresholds,
  drift fixtures, and verbatim regression gates. Depends on PAUS-6. Its exit gate is
  human review and freeze of the development and held-out case sets.
- **PAUS-9:** Add report-only benchmark scaffolding for the persistent worker and real
  hook path, including worker startup, model-load, matrix-load, encode, scan, hybrid,
  total latency, and fallback metrics. It does not activate the semantic gate. Depends
  on PAUS-5, PAUS-6, and PAUS-7, and does not depend on PAUS-10.
- **PAUS-10:** Implement scope-preserving RRF, exact lexical guards, deterministic
  tie-breaking, diagnostics, and bounded result packing. Depends on PAUS-7, PAUS-8,
  and PAUS-9.
- **PAUS-11:** Add the versioned operator synonym table and conservative lexical query
  variants. Depends on PAUS-10 and the frozen PAUS-8 fixtures.
- **PAUS-12:** Activate the warm and cold semantic gates, run the full Arc 2 evaluation,
  document measured thresholds and the corpus envelope, and decide whether semantic
  retrieval becomes the default. Depends on PAUS-9, PAUS-10, and PAUS-11.

The verified dependency edges are `PAUS-5 -> PAUS-6 -> PAUS-7`, `PAUS-6 -> PAUS-8`,
`PAUS-5,6,7 -> PAUS-9`, `PAUS-7,8,9 -> PAUS-10`, `PAUS-10 -> PAUS-11`, and
`PAUS-9,10,11 -> PAUS-12`. The cold-path worker belongs to PAUS-9, so its real hook
measurement is available before ranking polish. Gate activation belongs to PAUS-12,
after the complete path exists. These edges contain no cycle.

### References

- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — MIT-licensed cross-platform
  inference runtime.
- [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  — 384 dimensions, 22.7M parameters, and Apache-2.0 model license.
- [llama.cpp embedding discussion](https://github.com/ggml-org/llama.cpp/discussions/7712)
  — GGUF embedding workflow and example artifact sizes.
- [sqlite-vec](https://github.com/asg017/sqlite-vec) — pre-v1 SQLite vector extension.
- [FAISS](https://github.com/facebookresearch/faiss) — exact and approximate vector
  indexes, including its single-machine memory model.
- [MLX](https://github.com/ml-explore/mlx) and [Core ML](https://developer.apple.com/documentation/coreml/)
  — Apple-specific alternatives considered and rejected as the default backend.

## Future memory writing

If conversation extraction becomes necessary, store claims with source evidence rather than unqualified facts. Distinguish a user statement, an assistant assertion, and a confirmed tool result. Track event time, observation time, validity intervals, and explicit replacement links separately.

For example, “the assistant said it booked a flight” cannot establish a confirmed booking. A later correction should preserve the old evidence while excluding the replaced claim from current-state answers.

This extension needs its own write policy, correction behavior, deletion propagation, and evaluation. It should not silently feed generated answers back into trusted memory.

## Alternatives and unresolved choices

**Model-chosen retrieval only:** preserves flexible investigation but depends on the model recognizing a need to search. Keep it as both a baseline and a complementary path.

**Load the whole vault:** avoids retrieval misses for small corpora but consumes context and exposes unrelated information. Use bounded excerpts instead.

**Mem0 as the backend:** a possible later option for extracted conversational memory. It does not replace the hook, scope rules, source policy, or evaluation. The initial markdown use case does not require adopting its storage or extraction pipeline.

Before implementation, confirm the first harness, allowed roots, project-to-note mapping, and whether any source classes must never reach remote models. The proposed defaults are Claude Code, the existing vault, and explicit project scopes. Budget and latency figures remain targets until measured.

## References

- [Claude Code hooks](https://code.claude.com/docs/en/hooks): lifecycle hooks and additional context output. Verify against the installed version during implementation.
- [Mem0 repository](https://github.com/mem0ai/mem0): memory storage and application-controlled retrieval. Managed-platform claims do not establish open-source feature parity.
- [Mem0 implementation](https://github.com/mem0ai/mem0/blob/main/mem0/memory/main.py): extraction and search paths inspected during the preceding discussion; this link tracks a moving branch.
- Existing local research: `~/me/fun/wiki/vault/meta/agent-vault-patterns.md` and `~/me/fun/wiki/vault/tools/harness.md`.

External references informed the design on 2026-09-16. The component boundaries, budgets, release criteria, and implementation sequence above are proposals, not measured capabilities.
