# Portable Linux CPU wheels

The portable build targets **CPython 3.11, 3.12 and 3.13 on Linux x86_64 with
glibc 2.28 or newer**. It produces CPU-only `manylinux_2_28_x86_64` wheels from
one source distribution. CUDA, Metal, LLVM tools, model weights and local caches
are not shipped. This is the first portable-distribution increment, not a
PyPI release, macOS package or universal GPU build.

## Build and validate

On a Linux x86_64 host with Docker and uv, run from the repository root:

```bash
uv run --no-project --python 3.12 python tools/build_portable_wheels.py \
  --output build/portable-cpu
```

The output directory must be new. Use `--python 3.12` to select one supported
ABI, or repeat `--python` to select several; the default builds all three.
The developer's runtime environment is not rebuilt or synchronized.

The pinned cibuildwheel runner uses uv inside the official PyPA manylinux image,
pinned by digest in `pyproject.toml`. The image provides glibc 2.28 and GCC 14;
compiler flags request baseline x86-64 rather than the build host's instruction
set. Builds disable Metal/CUDA and limit parallel compilation to two jobs.
Only the source archive is built; the host filesystem is not mounted at `/host`.
External `CIBW_*` overrides, compiler/loader flags and pytest options are removed
by the wrapper so they cannot silently skip tests or change the accepted target.

For each ABI, cibuildwheel repairs/audits ELF dependencies with auditwheel
(`--only-plat` retains the explicitly requested 2.28 tag),
installs the repaired wheel into a new environment, then runs:

- the complete Python suite, with expected unavailable-device/toolchain skips;
- an isolated CPU probe checking import locations, ordinary operations, DLPack,
  ownership and explicit failure when LLVM is absent;
- the standalone diagnostic module and installed console command;
- the deterministic CPU MLP example with CPU parity.

After the builder tests, each ABI is also installed in a separate pinned
AlmaLinux 8 container with glibc 2.28 and its stock C++ runtime. It has no C++
compiler, CUDA or LLVM. uv downloads the matching interpreter into ephemeral
storage; CPU probes, the MLP and both diagnostic entry points must work there.
Only wheels, matching source-archive probes and the uv executable are mounted
read-only; runtime environments and downloads vanish with the container.

The wrapper checks every expected ABI, archive notices and file policy, repaired
filename/WHEEL metadata, package contents and `auditwheel show`. It writes
`manifest.json` with artifact SHA-256, image identity and checks only after all
selected artifacts and clean-runtime probes pass. Logs and intermediate outputs stay local in the ignored
build directory. A failed run may leave files, but has no completion manifest.
The image and ABI selection are build inputs; a manylinux tag alone is not proof
that an arbitrary renamed wheel is portable.

## Install a validated artifact

There is no package-index release yet. Use the exact reviewed wheel produced
locally or retained by the manual **Portable CPU wheels** Actions workflow:

```bash
uv venv build/cpu-consumer --python 3.12
uv pip install --python build/cpu-consumer/bin/python \
  build/portable-cpu/wheels/tensorcx-0.1.0-cp312-cp312-manylinux_2_28_x86_64.whl
build/cpu-consumer/bin/tensorcx-doctor --json
```

Compare the downloaded artifact against its manifest hash before installing.
Use the matching CPython ABI. NumPy `>=2,<3` is resolved normally by uv; C/C++
runtime libraries remain OS requirements. Linux musl/Alpine, 32-bit x86,
Linux ARM64, free-threaded Python, macOS/Metal and CUDA are outside this matrix.
A CPU wheel installs on an NVIDIA host but does not gain a CUDA backend.
Explicit MLIR compilation still needs separately installed LLVM 21.1.8 and its
existing Linux x86_64 runtime contract; tensor operations need no compiler.

## CI and remaining release work

`.github/workflows/portable-wheels.yml` is manually dispatched and retains only
successful wheels, the source archive and sanitized manifest for 14 days. It
never uploads to PyPI, creates tags or publishes a GitHub release. Ordinary
source/CUDA/Metal build settings are unaffected by the cibuildwheel configuration.

macOS wheels, portable CUDA packaging, versioned releases and package-index
publication remain separate work. The [host-specific validator](DISTRIBUTION.md)
continues to check real CUDA and optional MLIR independently of portability.

Sources: [PyPA manylinux](https://github.com/pypa/manylinux),
[cibuildwheel options](https://cibuildwheel.pypa.io/en/v4.3.1/options/),
[auditwheel](https://github.com/pypa/auditwheel).
