# Distribution and release validation

The release check builds an sdist, builds CPU and CUDA wheels **from that
sdist**, and installs each into a separate temporary uv environment. It runs
outside the source checkout with Python `-I`, verifies the package and native
extension paths, rejects editable installs or the wrong backend variant, and
checks runtime dependencies. This complements the operation/ABI test suites;
it does not replace them.

## Validated package scope

Current artifacts are local Linux x86_64 wheels for the exact CPython ABI used
to build them. Nightblade validation uses CPython 3.12, GCC 13 and CUDA 12.4,
targeting GTX 980 Ti (`sm_52`). These are `linux_x86_64` wheels, **not manylinux
wheels**. Do not claim cross-distribution portability or distribute a renamed
manylinux tag without a separate compatibility build and audit. The report
records required GLIBC/GLIBCXX/CXXABI symbol versions and actual shared library
resolution. macOS/Metal packaging and additional NVIDIA architectures are not
validated by this Linux check.

Both variants currently have the same Python package name, version and wheel
filename; they live in separate `cpu/` and `cuda/` output directories. Install
the exact chosen artifact by path. Do not merge those directories or upload
both as interchangeable wheels to a package index. This work validates local
release artifacts; it does not publish a PyPI release.

| Requirement | CPU wheel | CUDA wheel |
| --- | --- | --- |
| Python dependency | NumPy `>=2,<3`, resolved by uv | Same |
| Native dependencies | Host C/C++ runtime libraries | Same, plus CUDA Runtime `libcudart.so.12` and NVIDIA Driver `libcuda.so.1` |
| CUDA toolkit/compiler to use primitives | None | nvcc is only needed to build; runtime libraries and a usable driver/device are needed to execute |
| LLVM to use primitives/MLP | None | None |
| Explicit MLIR compilation | External LLVM 21.1.8; Linux x86_64 CPU slice | External LLVM 21.1.8 and the strict [CUDA environment gate](MLIR_CUDA_INTEGRATION_DECISION.md) |

The CUDA wheel requires `libcuda.so.1` even at import. Hiding a GPU on a host
with the driver libraries installed is distinct from importing on a driverless
host. Use the CPU wheel on hosts without CUDA libraries. Driver stubs used in
compile-only CI are never shipped with the wheel. CUDA/LLVM SDKs, driver
libraries and model weights are not bundled. Native static CUDA kernels are
embedded in the extension; Python compiler modules are included in the wheel.
Examples, validation tools and native source are included in the sdist.

## Reproduce the complete local check

Run from the repository root with uv, GCC 13, CUDA 12.4 and binutils (`ldd`,
`readelf`) available. Use a new output directory for every run:

```bash
CC=gcc-13 CXX=g++-13 CUDACXX=nvcc CUDAHOSTCXX=g++-13 \
  uv run --no-project --python 3.12 python tools/validate_distribution.py \
  --output build/distribution-check \
  --llvm-bin build/mlir-toolchain/root/usr/lib/llvm-21/bin
```

The wrapper uses only Python's standard library and does not install or alter
the developer's editable environment. Build isolation installs the declared
build dependencies; fresh runtime environments install only the selected wheel
and its declared runtime dependencies. Child processes clear `LD_LIBRARY_PATH`
and `LD_PRELOAD`; use installed system libraries or wheel-bundled libraries,
not shell loader overrides. Resolved dependencies from source, temporary or
stub directories are rejected (libraries within the fresh environment are
allowed). Build jobs use two processes to limit
memory pressure. The check records:

- sdist/wheel SHA-256 hashes, members, validator/probe hashes and build settings;
- uv/Python/platform versions and installed dependency versions;
- primitive parity (both axes for reductions/normalizations), device selection,
  copies, input ownership and a compiled local-expression kernel;
- successful ordinary execution without LLVM, with explicit compilation failing;
- hidden-GPU CPU execution, including the CUDA-enabled wheel;
- MLP execution outside the checkout, CPU parity and probability row sums;
- resolved shared libraries, ELF dependencies, search paths and symbol versions.

Use `--variants cpu` for CPU-only hosts, or `--variants cuda` for a required-GPU
run. Omit `--llvm-bin` to check ordinary use without LLVM; the report then does
not claim successful MLIR execution. Required CUDA execution cannot skip or
fall back. The validator refuses an existing output directory, preserves build
and failure logs, and writes `manifest.json` only after every selected check
passes. Failed runs may leave partial artifacts; absence of a manifest means
the run is not accepted. Temporary environments are removed after validation.

To install a validated artifact in a separate environment:

```bash
uv venv build/package-consumer --python 3.12
uv pip install --python build/package-consumer/bin/python \
  build/distribution-check/cuda/cortex_runtime-0.1.0-cp312-cp312-linux_x86_64.whl
```

Use the filename actually produced by your selected Python version. Keep the
manifest with the artifacts and compare its SHA-256 before reuse. Do not infer
that an artifact built elsewhere matches the retained validation report.

## Continuous checks and current evidence

The `CPU source distribution and wheel` CI job runs the same validator with
`--variants cpu`, without LLVM, and retains artifacts/logs for 14 days. It does
not establish real GPU validation; that separate product requirement remains
open. CI wiring is checked locally; a remote passing run must be inspected
before claiming remote validation.

2026-10-03 Nightblade validation passed for both variants. Each fresh install
passed three modes (without LLVM, hidden GPU, with LLVM), and both installed
wheels ran the MLP with CPU parity and normalized probability rows. The
[retained manifest](validation/distribution-nightblade.json) records artifact
hashes, dependencies, all probe results and ELF evidence. The actual artifacts
and detailed logs are retained locally under
`build/distribution-validation-20261003-audited/` (not committed to Git).

Both extensions reference GLIBC 2.43 and GLIBCXX 3.4.31 symbols on this host;
these builds are unsuitable for systems that lack those versions. Neither
extension has an ELF RPATH/RUNPATH. The CPU wheel has no CUDA or LLVM linkage;
the CUDA wheel resolves the documented runtime and driver libraries. These
facts establish the local distribution boundary, not universal portability.

The CPU-only, no-LLVM CI command path also passed locally, with its manifest
under `build/distribution-ci-audited/`. The validator's nine regression tests
passed (`uv run --no-sync pytest -q tests/python/test_distribution_validator.py`):
stderr/JSON separation, failure-log retention, refusal to overwrite existing
evidence, and rejection of contaminated/unresolved/wrong-backend dependencies.
The remote [CPU distribution job for `2281ca9`](https://github.com/serkanaltuntas/cortex-runtime/actions/runs/37118427687/job/111189637894)
also passed and retained its artifact, as verified on 2026-10-03.
That workflow's overall result was failure because the macOS jobs exposed a
hardcoded `/tmp` assumption in a distribution-validator test. The fixture now
uses the platform temporary directory; the production linkage audit is
unchanged. Default and custom-`TMPDIR` local test runs each pass all nine cases.
CPU packaging success does not establish remote GPU execution or portable
CUDA wheels. The full product requirement ledger is
[CUDA_PRODUCT_COMPLETION.md](CUDA_PRODUCT_COMPLETION.md).
