---
type: decision
updated: 2026-09-02
created: 2026-09-02
---
# FTS backend

Use SQLite FTS5 for the first retrieval backend. It keeps the index local,
rebuildable, and easy to inspect without a service dependency.
