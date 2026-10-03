# Security

Cortex Runtime is experimental, pre-alpha software. No version is currently
offered as a security-supported or hardened production release.

Report suspected vulnerabilities privately to **serkan@altuntas.dev**. Include
the affected revision, environment, reproduction steps and impact. Do not post
credentials, private data or an exploit for an unresolved vulnerability in a
public issue. Response times are not guaranteed.

## Trust boundaries

- Python programs, compiler executables and native/PTX/Metal artifacts must be
  trusted. The kernel DSL and ABI validation are **not a sandbox** for hostile
  code. Native artifact loading can execute code with the caller's privileges.
- Compilation explicitly uses external LLVM/CUDA/Apple tools and the caller's
  configured environment. Use trusted toolchains and dependencies.
- Run untrusted contributions in isolated disposable environments. Never expose
  personal files, signing keys or privileged credentials to a GPU CI job.
  Fork pull requests are excluded from the self-hosted CUDA workflow.
- Validation and benchmark summaries remove known local metadata. Raw logs and
  arbitrary strings can still contain secrets; review them before sharing.

See [continuous CUDA validation](docs/CUDA_CONTINUOUS_VALIDATION.md) for runner
isolation requirements, and [distribution](docs/DISTRIBUTION.md) for binary
compatibility limits.
