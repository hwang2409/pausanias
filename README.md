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
python -m pausanias --config ./config.toml read ./vault/decision.md --heading "Decision"
```

Use `--rebuild` to recreate the derived database. Use `--all-projects`
only when cross-project search is intentional.

## optional semantic model

The base install stays dependency-free. Install the semantic extra, then fetch the
pinned model bundle explicitly:

```bash
python -m pip install -e '.[semantic]'
pausanias --config ./config.toml model fetch
pausanias model license
```

Set the bundle location in `config.toml` with `[semantic]` and `bundle_dir`. Fetch
downloads only the pinned files, verifies each SHA-256 hash, and publishes the bundle
after all files pass verification. The bundle contains the tokenizer contract and
license notices. No normal command downloads a model.
