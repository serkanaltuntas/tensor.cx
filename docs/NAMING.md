# tensor.cx naming and migration

Decision: 2026-10-04. The product brand is **tensor.cx**, an independent personal
project. The former pre-release working name was Cortex Runtime.

| Surface | Current name | Former name |
| --- | --- | --- |
| Product and website | tensor.cx | Cortex Runtime |
| Python distribution | `tensorcx` | `cortex-runtime` |
| Python import | `import tensorcx as cx` | `import cortex_runtime as cx` |
| Native Python extension | `tensorcx._core` | `cortex_runtime._core` |
| Python source | `python/tensorcx/` | `python/cortex_runtime/` |
| C++ namespace / source | `tensorcx` / `cpp/tensorcx/` | `cortex` / `cpp/cortex/` |
| CMake project and targets | `tensorcx`, `tensorcx_*` | `cortex_runtime`, `cortex_*` |
| Build and environment options | `TENSORCX_*` | `CORTEX_*` |
| Local CUDA gate Git option | `tensorcx.cudaLlvmBin` | `cortex.cudaLlvmBin` |
| Generated kernel symbols/manifests | `tensorcx_*`, `tensorcx.cpu.v1`, `tensorcx.cuda.v1` | `cortex_*`, `cortex.cpu.v1`, `cortex.cuda.v1` |

`tensor.cx` is the display/domain spelling. A Python dot denotes a submodule,
so the importable identifier is `tensorcx`, not `tensor.cx` or `tensor`.
Keep the short alias `cx` in examples.

## Existing development environments

This is a pre-release breaking rename. There is no compatibility import for
`cortex_runtime`. Update imports, build flags, CI variables, and test environment
variables together, then rebuild the native extension. Prefer a fresh virtual
environment and a new CMake build directory. To migrate the existing project
environment after activating it:

```bash
uv pip uninstall cortex-runtime
CMAKE_ARGS="-DTENSORCX_ENABLE_METAL=OFF -DTENSORCX_ENABLE_CUDA=OFF" uv pip install -e ".[dev]"
uv run --no-sync python -c "import tensorcx as cx; print(cx.__version__, cx.devices())"
```

Select Metal/CUDA flags using [README.md](../README.md) for accelerator builds.
Old `CORTEX_*` CMake options are rejected to avoid silently changing the selected
backend. Python/tooling environment variables now use `TENSORCX_*`; update
shell profiles and runner configuration. The optional GPU workflow also uses
`TENSORCX_CUDA_CI_ENABLED`, `TENSORCX_CUDA_LLVM_BIN`, and runner label
`tensorcx-cuda-sm52`. It remains gated until its isolated runner is validated.

Regenerate compiled CPU libraries, CUDA PTX and Metal artifacts from source.
Generated symbol and manifest identities changed; artifacts from the former
namespace are not interchangeable with the new runtime. No numerical behavior
or supported backend scope changes are intended by the rename.

## Repository and historical evidence

The existing GitHub repository remains
[`serkanaltuntas/cortex-runtime`](https://github.com/serkanaltuntas/cortex-runtime).
Its URLs and the local checkout directory are not renamed by this migration.
No package has been published or a registry name reserved by changing source
metadata. The website is not deployed by this change.

Live instructions and code examples use the new spelling. Dated acceptance
results still describe their original executions; the naming update does not
retroactively re-run those measurements. Retained JSON/CSV/text reports,
revision IDs, artifact hashes, and old commit messages are preserved. Reports
may contain old package, path, test, and generated-symbol names. The existing
`cortex-public-report-v1` sanitation format identifier stays stable for report
consumers; it is a data format, not the product brand.

## Rename validation — 2026-10-04

Validation used fresh build directories and a fresh CUDA-enabled environment
on the existing Linux x86_64 / sm_52 host:

- Full Python suite with required CUDA and MLIR: 1202 passed, 142 expected
  skips for unavailable Metal or unsupported backend capabilities.
- Native CPU/CUDA contracts: 4/4 passed, also 4/4 with ASan/UBSan.
- CPU and CUDA sdists/wheels installed outside the source checkout: each
  passed without LLVM, with LLVM, and with the GPU hidden; both MLP probes
  passed CPU parity. Wheel metadata and native imports use `tensorcx`;
  neither wheel contains `cortex_runtime`, and the sdist excludes website files.
- Old CMake flags fail with the new spelling in the error; the existing local
  CUDA push-gate setting and development environment were migrated.
- Website type check/build and all four production browser tests passed.
  Shared website Python examples ran with the renamed runtime.

These local checks do not claim new Apple hardware execution or wider NVIDIA
compatibility. The existing CI jobs retain the separate Metal build/device gates.
