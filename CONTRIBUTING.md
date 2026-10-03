# Contributing

Cortex is an experimental runtime. Read [README.md](README.md) for setup and
[PROJECT.md](PROJECT.md) for supported scope. Discuss broad API/backend changes
in an issue before implementation; focused bug reports and fixes are welcome.

Use your own Git identity. Contributions are under this repository's Apache-2.0
license; preserve existing third-party notices. Agent-assisted work follows
[AGENTS.md](AGENTS.md), including review and QA gates.

## Validate a change

Use `uv` and a project-local environment. Every GPU operation needs CPU behavior
and CPU/device comparison coverage. Run the affected tests, then the relevant
backend suite; report hardware/toolchain versions and any skipped tests in the
pull request. CPU tests must work without a GPU. Do not claim a skipped device
test as device validation. Use `uv run --no-sync` to preserve an existing custom
CUDA build when running tests.

For packaging changes, run the source-to-wheel check in
[DISTRIBUTION.md](docs/DISTRIBUTION.md); it verifies notice files and rejects
private files and bytecode caches. Current local wheels are not portable
manylinux releases, and CPU/CUDA variants must remain in separate directories.

## Publish only reviewed evidence

Keep `.env`, keys, local sessions, raw logs, build outputs and history backups
out of Git. Ignore rules are a guardrail, not a secret detector. Use the public
JSON summaries emitted by validation tools, or export an existing report with:

```bash
uv run --no-project python tools/publication_report.py local-report.json --output public-report.json
```

The exporter removes known paths and hardware/process identifiers, not arbitrary
secrets. Inspect the result and scan the complete history with Gitleaks before a
public release. If a credential was ever committed, revoke it; deleting a file
or rewriting history alone cannot revoke credentials or erase other clones.
Report vulnerabilities using [SECURITY.md](SECURITY.md).
