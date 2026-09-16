---
type: decision
updated: 2026-08-18
created: 2026-08-18
---
# Sandbox image

We chose the ubuntu-24.04 sandbox image because it matches the supported
build runners and gives repeatable native package installs. The team rejected
rolling images because image drift made failed builds hard to reproduce.
