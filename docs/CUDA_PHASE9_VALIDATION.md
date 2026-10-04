# Phase 9 validation — 2026-09-28

Host and toolchain selection: [environment decision](CUDA_PHASE9_ENVIRONMENT.md).
Baseline revision: `3662c9e`.

## Entry evidence (before CUDA implementation)

- `nvidia-smi`: GTX 980 Ti visible, driver 580.178.04.
- `nvcc --version`: 12.4.131; `g++-13`: 13.4.0; `uv`: 0.12.17.
- Host kernel: Linux 7.0.0-34-generic, x86_64.
- `uv venv --python 3.12`; `uv pip install cmake ninja "nanobind>=2.4"`.
- `CC=gcc-13 CXX=g++-13 CMAKE_ARGS="-DCORTEX_ENABLE_METAL=OFF" uv pip install -e ".[dev]"`: passed.
- `uv run pytest -q`: **206 passed, 129 skipped** (unavailable Metal/CUDA/MLIR paths).
- CPU-only CMake configure/build and `uv run ctest --test-dir build/cpp-baseline --output-on-failure`: **1/1 passed**.
- `nvcc -ccbin g++-13 -arch=sm_52 build/cuda-preflight/smoke.cu -o build/cuda-preflight/smoke` and execution: passed. The kernel wrote `42` on device; synchronous device-to-host copy returned `42`.
- Python `Backend` registry and native `BackendRoute` table inspected; CUDA will register through those routes. No third CPU/Metal dispatch branch is needed.

The preflight source/binary and raw logs are local build artifacts, not runtime
sources. This evidence is a functional check, not a performance or long-duration
GPU stability claim. Metal execution requires Apple hardware and was not rerun
on Nightblade.

## Reproduce on the selected host

These reproduction commands use the current tensorcx spelling. The entry and
acceptance records below retain the original names from the dated run; see
[the naming migration](NAMING.md).

The following commands assume the project `.venv` created above. For another
host, choose a matching compiler/toolkit and `CMAKE_CUDA_ARCHITECTURES` value.
CUDA Runtime API and build-time static kernels were selected for this small
prototype; Driver API/PTX generation is not required by Phase 9.

```bash
export CC=gcc-13 CXX=g++-13 CUDACXX=nvcc CUDAHOSTCXX=g++-13
CMAKE_ARGS="-DTENSORCX_ENABLE_METAL=OFF -DTENSORCX_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=52" uv pip install -e ".[dev]"
TENSORCX_REQUIRE_BACKENDS=cuda \
TENSORCX_REQUIRE_BACKEND_CAPABILITIES=cuda:copy,cuda:tensor_factories_float32,cuda:binary_ops_float32 \
uv run pytest -q

uv run cmake -S . -B build/cpp-cuda -G Ninja \
  -DTENSORCX_ENABLE_METAL=OFF -DTENSORCX_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=52 -DTENSORCX_BUILD_TESTS=ON \
  -Dnanobind_DIR="$(uv run python -c 'import nanobind; print(nanobind.cmake_dir())')" \
  -DPython_EXECUTABLE="$(uv run python -c 'import sys; print(sys.executable)')"
uv run cmake --build build/cpp-cuda --target tensorcx_backend_contract_tests tensorcx_cuda_backend_contract_tests
TENSORCX_REQUIRE_CUDA=1 uv run ctest --test-dir build/cpp-cuda --output-on-failure
```

## Acceptance results

- CUDA-enabled editable build: passed with nvcc 12.4.131 / GCC 13.4.0 / `sm_52`.
- Full pytest under required CUDA/capability mode: **280 passed, 152 skipped**.
  CUDA discovery, copies, float32 fill/add/multiply and their shared parity tests
  executed. Remaining skips cover Metal, MLIR tools, and deferred CUDA operations.
- CUDA-enabled CTest with `CORTEX_REQUIRE_CUDA=1`: **2/2 passed**; CUDA contract
  actually executed (the required-device flag prevents a false pass via skip).
- CUDA-disabled rebuild (`CORTEX_ENABLE_CUDA=OFF`) and full pytest:
  **226 passed, 206 skipped**; CPU-only CTest **1/1 passed**.
- CUDA-enabled build with `CUDA_VISIBLE_DEVICES=''`: full pytest
  **226 passed, 206 skipped**; CTest CPU passed and CUDA skipped explicitly.
  Python import, unavailable CUDA errors, and normal CPU work remain functional.
- CUDA+ASan/UBSan configure/build passed. With
  `ASAN_OPTIONS=halt_on_error=1:detect_leaks=1:protect_shadow_gap=0`,
  `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`, and
  `CORTEX_REQUIRE_CUDA=1`, CTest **2/2 passed**. This host's default ASan shadow-gap
  protection prevented CUDA device initialization; the stated option was needed
  for this combined check. Sanitizers instrument C++ host code, not CUDA kernels.
  Normal CUDA acceptance above does not require any ASan setting.
- `git diff --check`: passed.

Coverage includes scalar/empty/multidimensional tensors, 255/256/257-element
launch boundaries, a 262145-element case, CPU parity, exact int32 copy, retained
buffer ownership, concurrent and cold-start execution, unsupported operations,
shape/dtype mismatch, forged/wrong-sized buffers, offsets/strides, and allocation
size overflow. Failed native requests preserve output result slots.

The new `cuda-build` CI job compiles inside a CUDA 12.4 container and checks
CPU fallback on a hosted runner without a GPU. Its remote run is separate from
this local evidence. The existing macOS CI jobs remain responsible for Apple
compile/link checks; no Apple hardware was available for a new Metal run here.
MLIR integration and performance work remain outside this change.

Implementation references: [CUDA Runtime device API](https://docs.nvidia.com/cuda/archive/12.5.1/cuda-runtime-api/group__CUDART__DEVICE.html),
[CMake CUDA architectures](https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html).

## Review and QA

Correctness and architecture reviewers passed after fixing host sanitizer flags
that nvcc rejected. Independent Test QA and Acceptance QA passed; the latter's
minor stale capability-table wording was corrected. No blocking findings remain.
