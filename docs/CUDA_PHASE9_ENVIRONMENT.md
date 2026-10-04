# Phase 9 CUDA Environment

> Naming update (2026-10-04): commands, source paths, and symbols in this living
> document use the current tensorcx spelling. Dated results describe runs
> under the former names; they are not new validation runs. For historical
> revisions, use the reverse mapping in [NAMING.md](NAMING.md).

Phase 9 must not start until the CUDA development environment is chosen and
recorded. The goal is to avoid mixing runtime design work with infrastructure
guesswork.

## Decision Gate

Before CUDA backend implementation begins, record one selected environment:

```text
Selected environment: Nightblade local Linux workstation
Date selected: 2026-09-28
CUDA validation status: native kernel compile/run passed before implementation
GPU: NVIDIA GeForce GTX 980 Ti, compute capability 5.2, 6 GiB
OS / image: Ubuntu 26.04.1 LTS, Linux x86_64
C++ compiler: GCC/G++ 13.4.0 (CC=gcc-13, CXX=g++-13)
CUDA toolkit / compilation path: nvcc 12.4.131, -ccbin g++-13, -arch=sm_52
Driver: 580.178.04
Project test result: baseline 206 passed, 129 skipped; CPU CTest 1/1 passed
Validation log: CUDA_PHASE9_VALIDATION.md
```

The previous 2026-06-30 preflight used an Apple Silicon Mac without NVIDIA
hardware. That blocker is superseded by the validated Nightblade environment.
CUDA implementation resumed on 2026-09-28 after this environment passed the gate.
The first implementation slice is discovery/allocation/copy through the existing
backend registry, followed by float32 fill/add/multiply through BackendExecution.

Local preparation uses only the project `.venv`: Python 3.12.14, uv, CMake
4.4.3, Ninja 1.13.2, and nanobind 3.1.0. The versioned system GCC 13 compiler
is selected explicitly; no system compiler links or drivers were changed.

The selected environment must provide:

```text
- NVIDIA GPU with CUDA support
- Linux development shell
- C++20 compiler
- CMake
- uv
- CUDA toolkit with nvcc or a driver-API-compatible compilation path
- Python 3.11+
- ability to run pytest
```

The first CUDA prototype should target a development machine or cloud instance
that can compile and run native CUDA code. macOS is not a CUDA development
target for this phase.

## Recommended Options

### Local Linux Workstation

Use this if a CUDA-capable NVIDIA machine is available locally.

Pros:

```text
- fastest iteration after setup
- stable access to the same GPU
- easy repeated benchmark and debug runs
```

Risks:

```text
- hardware procurement and driver maintenance
- machine setup can distract from runtime work
```

### Cloud GPU Instance

Use this if local CUDA hardware is unavailable. Prefer a simple single-GPU Linux
instance over a managed training platform for the prototype.

Pros:

```text
- no local hardware purchase required
- easy to choose a current NVIDIA GPU
- clean reproducible setup notes
```

Risks:

```text
- cost control
- remote debugging friction
- instance images can drift over time
```

### Remote Development Box

Use this if a known Linux CUDA host is already available through SSH.

Pros:

```text
- low setup overhead when the host is already maintained
- realistic native CUDA environment
```

Risks:

```text
- unclear ownership of drivers/toolkit updates
- possible shared-machine instability
```

## Minimum Validation Commands

Run these before marking the environment ready:

```bash
nvidia-smi
command -v uv
command -v cmake
command -v c++
uv --version
cmake --version
```

Then validate the selected CUDA compilation path. Prefer `nvcc` when available:

```bash
command -v nvcc
nvcc --version
```

If the selected environment uses a Driver API / PTX compilation path without
`nvcc`, record the exact compiler, toolkit, and smoke-test commands in this
document before implementation starts.

Then validate the current project without CUDA changes:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
uv run pytest
NANOBIND_DIR="$(uv run python -c 'import nanobind; print(nanobind.cmake_dir())')"
PYTHON_EXECUTABLE="$(uv run python -c 'import sys; print(sys.executable)')"
cmake -S . -B build/cpp-tests -DTENSORCX_ENABLE_METAL=OFF -DTENSORCX_BUILD_TESTS=ON -Dnanobind_DIR="${NANOBIND_DIR}" -DPython_EXECUTABLE="${PYTHON_EXECUTABLE}"
cmake --build build/cpp-tests --target tensorcx_backend_contract_tests
ctest --test-dir build/cpp-tests --output-on-failure
```

The current Python CPU test suite and C++ backend contract test must pass on the
CUDA host before new CUDA backend work starts. If Metal is unavailable on the
CUDA host, Metal tests must skip cleanly rather than fail.

## Phase 9 Entry Criteria

Phase 9 implementation may begin only after:

```text
1. This document records the selected environment.
2. The minimum validation commands above are captured in the session notes and
   summarized in the Decision Gate fields.
3. `uv pip install -e ".[dev]"` succeeds on the CUDA host.
4. `uv run pytest` succeeds on the CUDA host, with unavailable Metal tests skipped.
5. The selected CUDA compilation path is validated and recorded.
6. The C++ backend contract CTest command succeeds on the CUDA host.
7. The existing public backend registry/routing layer is confirmed on the CUDA
   host; the first CUDA backend-specific task after that is scoped to
   discovery/allocation/copy before kernels.
```

## First CUDA Backend Slice

Once the environment is ready, implement CUDA in this order:

```text
1. Add CUDA backend scaffold under cpp/tensorcx/backends/cuda/.
2. Register cuda through the existing backend registry only when the CUDA runtime is available.
3. Implement device discovery.
4. Implement buffer allocation and host<->device copies.
5. Add Tensor.to("cuda") and Tensor.cpu() round-trip tests.
6. Add fill_f32.
7. Add add_f32 and mul_f32.
8. Enable and extend CUDA capabilities in the existing backend-parametric tests
   so the same Python test body covers cuda.
```

The CUDA backend must use the Phase 8 backend ABI and shared contract validators
where applicable. Do not add CUDA-specific concepts to `cpp/tensorcx/core/`.

## Backend-Parametric Test Harness

The Python test suite already has a backend capability matrix in
`tests/python/conftest.py` and shared parity tests in
`tests/python/test_backend_parity.py`. CUDA declares `copy`,
`tensor_factories_float32`, and `binary_ops_float32` after Phase 9 validation.
Broader CUDA capabilities remain disabled until their implementations pass.

When implementing CUDA, enable capabilities in the matrix only after the matching
backend slice is complete and verified:

```text
copy                       CPU -> CUDA -> CPU round trip
tensor_factories_float32   empty/zeros/ones for float32
binary_ops_float32         add_f32/mul_f32 and shape mismatch errors
tensor_factories_int32     empty/zeros/ones for int32, if implemented later
binary_ops_int32           add_i32/mul_i32, if implemented later
binary_ops_dtype_mismatch  dtype mismatch errors once multiple dtypes exist
unary_float32              exp/gelu/silu, if implemented later
reductions_float32         sum/max/mean for float32, if implemented later
reductions_int32           sum/max for int32, if implemented later
normalization_float32      softmax/rmsnorm/layernorm, if implemented later
```

The immediate Phase 9 target is `copy`, `tensor_factories_float32`, and
`binary_ops_float32`. The same pytest bodies should run for Metal and CUDA once
those CUDA capabilities are declared. Do not enable int32 CUDA capabilities
unless int32 CUDA kernels and factory behavior are implemented intentionally.

On a CUDA host, use strict backend mode after CUDA registration is expected to
exist so a broken backend, missing registration, or missing capability
declaration cannot be hidden by skip behavior:

```bash
TENSORCX_REQUIRE_BACKENDS=cuda \
TENSORCX_REQUIRE_BACKEND_CAPABILITIES=cuda:copy,cuda:tensor_factories_float32,cuda:binary_ops_float32 \
uv run pytest tests/python/test_backend_parity.py -q
```

`TENSORCX_REQUIRE_BACKENDS` accepts a comma-separated backend list, for example
`cuda,metal`. `TENSORCX_REQUIRE_BACKEND_CAPABILITIES` accepts comma-separated
`backend:capability` entries. Required backends and required backend
capabilities fail collection if they are unknown, unavailable, or not declared
in the capability matrix.

## Completion Target

Phase 9 is complete only when:

```text
- device discovery works for device="cuda"
- CUDA is routed through registry/string-keyed public dispatch, not as a third
  ad hoc CPU/Metal branch
- CPU -> CUDA -> CPU copy round-trips pass
- fill_f32, add_f32, and mul_f32 work on CUDA
- CUDA results match CPU references
- the same pytest body runs against both metal and cuda where hardware exists
- CUDA setup and verification are documented
```
