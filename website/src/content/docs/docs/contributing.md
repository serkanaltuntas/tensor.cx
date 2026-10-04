---
title: Contributing
description: Contribute focused fixes, reproducible reports, and CPU-validated backend behavior.
---

tensor.cx is an independent personal project by [Serkan Altuntaş](https://serkan.ai/), licensed
under [Apache 2.0](https://github.com/serkanaltuntas/tensor.cx/blob/main/LICENSE).
The runtime is pre-alpha, and its API and implementation are still evolving.

## Useful contributions

- Small, reproducible bug reports with the expected and actual result.
- Focused fixes with CPU correctness and relevant backend comparison tests.
- Verified documentation improvements and runnable examples.
- Hardware validation that clearly records the tested scope and skipped cases.

Discuss broad API or backend changes before implementation. Start with the
repository's [contribution guide](https://github.com/serkanaltuntas/tensor.cx/blob/main/CONTRIBUTING.md)
and [project scope](https://github.com/serkanaltuntas/tensor.cx/blob/main/PROJECT.md).
They are the canonical development instructions.

## Report a security issue

Follow [SECURITY.md](https://github.com/serkanaltuntas/tensor.cx/blob/main/SECURITY.md)
for private reporting. Avoid putting secrets, personal paths, unreviewed raw
logs, or persistent device identifiers into public issues.

## Website contributions

The website lives alongside the runtime in `website/`. User guides are curated
Markdown/MDX pages; internal project records are not automatically published.
See [the website README](https://github.com/serkanaltuntas/tensor.cx/blob/main/website/README.md)
for local development and verification.

The deployed site's [third-party notices](/third-party-notices.txt) cover its
browser assets. Runtime dependency notices are maintained separately in
[THIRD_PARTY_NOTICES.md](https://github.com/serkanaltuntas/tensor.cx/blob/main/THIRD_PARTY_NOTICES.md).
