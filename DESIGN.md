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

Use Python and its standard-library SQLite support for the initial implementation. Check FTS5 availability during setup. Avoid adding a daemon or separate database server until measurements justify one.

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
- A 750 ms target for the first query on a fresh SQLite connection after index refresh. This does not clear OS or SQLite page caches.
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

Start with at least 40 reviewed cases spanning the categories above. Treat results as a pilot, not a statistical guarantee. Choose a larger evaluation size after measuring baseline variation.

The first release requires no source-boundary violations in the test set, correct deletion handling, and bounded timeout behavior. Adopt automatic retrieval by default only if paired evaluation improves useful recall without increasing unsupported or stale claims. Record thresholds before tuning on the development set, then test on held-out cases.

## Implementation sequence

1. Build the corpus reader, FTS index, source-read command, and explicit search baseline.
2. Build the reviewed evaluation cases and measure model-chosen retrieval.
3. Add packet construction, bounded hook integration, diagnostics, and failure tests.
4. Compare all four conditions. Tune budgets and acceptance rules on development cases only.
5. Add embeddings only if paraphrase failures remain material on held-out cases.

Do not build automatic memory writing as part of these steps. Existing vault authoring rules remain in effect.

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
