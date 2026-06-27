# Phase 9 CUDA Environment

Phase 9 must not start until the CUDA development environment is chosen and
recorded. The goal is to avoid mixing runtime design work with infrastructure
guesswork.

## Decision Gate

Before CUDA backend implementation begins, record one selected environment:

```text
Selected environment: TBD
Owner: TBD
Date selected: TBD
CUDA validation status: not run
GPU: TBD
OS / image: TBD
C++ compiler: TBD
CUDA toolkit / compilation path: TBD
Project test result: TBD
Validation log: TBD
```

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
cmake -S . -B build/cpp-tests -DCORTEX_ENABLE_METAL=OFF -DCORTEX_BUILD_TESTS=ON -Dnanobind_DIR="${NANOBIND_DIR}" -DPython_EXECUTABLE="${PYTHON_EXECUTABLE}"
cmake --build build/cpp-tests --target cortex_backend_contract_tests
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
7. Public backend registry/routing is the first implementation task; the first
   CUDA backend-specific task after that is scoped to discovery/allocation/copy
   before kernels.
```

## First CUDA Backend Slice

Once the environment is ready, implement CUDA in this order:

```text
1. Add or finish a small backend registry/string-keyed routing layer for public dispatch.
2. Add CUDA backend scaffold under cpp/cortex/backends/cuda/.
3. Register cuda only when the CUDA runtime is available.
4. Implement device discovery.
5. Implement buffer allocation and host<->device copies.
6. Add Tensor.to("cuda") and Tensor.cpu() round-trip tests.
7. Add fill_f32.
8. Add add_f32 and mul_f32.
9. Parametrize the existing CPU/Metal tests so the same Python test body covers cuda.
```

The CUDA backend must use the Phase 8 backend ABI and shared contract validators
where applicable. Do not add CUDA-specific concepts to `cpp/cortex/core/`.

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
