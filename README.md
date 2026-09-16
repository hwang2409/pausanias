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
