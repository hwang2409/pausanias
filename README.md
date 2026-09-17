# Pausanias
Pausanias was a Greek writer whose 10 book guide around Greece still helps modern archaeologists to this day.

## memory core

Pausanias indexes configured Markdown roots in a local SQLite FTS5 database.
It does not edit source notes or send data to a remote service.

Example `config.toml`:

```toml
database = "./index.sqlite3"
private_paths = ["private/**"]
global_notes = ["./shared/decision-log.md"]
synonym_table = "./synonyms.toml"

[[roots]]
id = "vault"
path = "./vault"
project = "phoebe"
exclude = ["archive/**"]
```

Run the indexer and search a project:

```bash
python -m pausanias --config ./config.toml index
python -m pausanias --config ./config.toml search "sandbox image" --project phoebe
python -m pausanias --config ./config.toml search "sandbox image" --project phoebe --retrieval-mode lexical
python -m pausanias --config ./config.toml read ./vault/decision.md --heading "Decision"
```

Use `--rebuild` to recreate the derived database. Use `--all-projects`
only when cross-project search is intentional. Search uses fused retrieval when the
semantic extra is installed and the index is ready. Use `--retrieval-mode lexical` to
force lexical retrieval. Fused retrieval falls back to lexical results when its model,
index, or runtime is unavailable. With `--diagnostics --json`, search returns an
`items` array and a sibling `diagnostics` object, including state and reason for empty
results. A default base-install lexical search keeps its legacy JSON array shape.

The optional synonym table is local and operator-maintained:

```toml
version = 1

[terms]
database = ["db"]
```

Expansion skips ticket identifiers and quoted phrases. It replaces one ordinary term at
a time and stops at 16 variants. A missing table disables expansion.

## optional semantic model

The base install stays dependency-free. Install the semantic extra, then fetch the
pinned model bundle explicitly:

```bash
python -m pip install -e '.[semantic]'
pausanias --config ./config.toml model fetch
pausanias --config ./config.toml model status
pausanias --config ./config.toml model license
```

Set the bundle location in `config.toml` with `[semantic]` and `bundle_dir`. Fetch
downloads only the pinned files, verifies each SHA-256 hash, and publishes the bundle
after all files pass verification. The bundle contains the tokenizer contract and
license notices. `model status` reports the optional extra, bundle, manifest, and
runtime states. No normal command downloads a model.
