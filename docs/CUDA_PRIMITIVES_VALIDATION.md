# CUDA float32 primitive extension

Date: 2026-10-03. Extends the completed Phase 9 prototype under the user's
[full CUDA product goal](CUDA_PRODUCT_COMPLETION.md). This is operator coverage,
not completion of compatibility, performance, GPU CI or distribution work.

## Supported behavior

All operations compute on CUDA device 0, use `CudaBackend::execute`, retain
primary-context ownership, run synchronously and publish a new output only on
success. Input tensors remain unchanged. Inputs must be contiguous float32
Cortex CUDA tensors; no implicit CPU fallback or dtype conversion is introduced.

| Operation | Contract |
| --- | --- |
| `matmul` / `@` | Rank-2 `(m,k) @ (k,n)`; `auto` and `custom` use the tiled CUDA kernel; `optimized` and unknown preferences fail explicitly |
| `sum`, `max`, `mean` | Explicit axis, including negative axes; axis is removed; scalar accepts axis 0/-1; empty sum gives zero, empty mean NaN, empty max axis raises |
| `exp`, `gelu`, `silu` | Shape-preserving; GELU uses the same tanh approximation as CPU |
| `softmax` | Stable max-subtracted normalization along any valid axis; shape preserved |
| `rmsnorm`, `layernorm` | Any valid axis, shape preserved, no affine weights; finite non-negative float32-representable epsilon required, even for empty input |

Zero-size matmul outputs do not launch; `k=0` produces zeros. Empty normalization
outputs do not launch. Int32 still supports copies only on CUDA; its arithmetic
and reductions remain explicitly unsupported. Existing static fill/add/multiply
and generated MLIR paths are preserved. The strict sm_52 MLIR toolchain gate
is unchanged; these primitive kernels are ordinary build-time CUDA kernels.

CUDA and CPU agree on NaN propagation, infinity results and constant-row
normalization. LayerNorm with a finite constant row gives zero for positive
epsilon and NaN at epsilon zero, matching CPU. Accumulation uses float32 and
separate add/multiply rounding where applicable. Transcendental comparisons use
project tolerances; this record does not promise bitwise equality across GPUs.

## Implementation

`kernels/primitives.cu` contains a 16x16 shared-memory tiled matmul, grid-stride
unary kernels, and one CUDA lane per independent axis slice. The latter keeps
CPU accumulation order and supports non-final axes without host transfers.
Parallelizing long reductions is a future measured optimization; these kernels
are not claimed to match vendor-library throughput. There is no fast-math mode.
The C++ core's operation types and dispatch contract are unchanged.

Concrete device/buffer/shape/stride/dtype validation and operation-specific
axis, epsilon, rank, contraction-dimension and algorithm checks precede kernel
execution. The native tests exercise every new operation through dispatch,
compare to CPU and verify failed calls preserve the previous output buffer.

## End-to-end example

```bash
uv run python examples/cuda_mlp.py --device cuda
uv run python examples/cuda_mlp.py --device cpu
```

The deterministic workload is `(32,64) @ (64,128) -> LayerNorm -> GELU ->
(32,128) @ (128,10) -> Softmax`. It verifies CPU parity and row probability sums,
then prints median synchronous inference latency after warmup. Transfers and
CPU validation are excluded from that timing; it is a smoke measurement, not
a general performance comparison. Both device choices require their selected
backend explicitly; the CUDA command never silently falls back to CPU.

## Validation

Nightblade GTX 980 Ti, CUDA 12.4/GCC 13, Driver 580.178.04. Validation commands:

```bash
CORTEX_REQUIRE_BACKENDS=cuda CORTEX_REQUIRE_BACKEND_CAPABILITIES=cuda:unary_float32,cuda:reductions_float32,cuda:normalization_float32 uv run pytest tests/python/test_cuda.py tests/python/test_cuda_primitives.py tests/python/test_backend_parity.py -q
CORTEX_LLVM_BIN="$PWD/build/mlir-toolchain/root/usr/lib/llvm-21/bin" CORTEX_REQUIRE_MLIR=1 CORTEX_REQUIRE_MLIR_CUDA=1 CORTEX_REQUIRE_BACKENDS=cuda uv run pytest -q
uv run cmake --build build/cpp-cuda
CORTEX_REQUIRE_CUDA=1 CORTEX_REQUIRE_MLIR_CUDA=1 uv run ctest --test-dir build/cpp-cuda --output-on-failure
```

Run native ASan/UBSan using the existing [Nightblade shadow-gap workaround](CUDA_PHASE9_VALIDATION.md).
Coverage includes matmul tile boundaries/zero dimensions, arbitrary and negative
axes, scalar/empty inputs, constant rows and epsilon boundaries, NaNs/infinities,
subnormals, shape/dtype/device errors, immutable inputs and concurrent inference.
The shared backend capability suite now requires CUDA float32 unary, reduction
and normalization parity instead of skipping those capabilities.

Final local results:

| Configuration | Result |
| --- | --- |
| Focused CUDA primitives | 135 passed |
| Full Python suite, required CUDA capabilities and LLVM | 946 passed, 138 skipped |
| Actual CUDA-off build, LLVM present | 334 passed, 750 skipped |
| Actual CUDA-off build, LLVM absent | 277 passed, 807 skipped |
| Native CTest | 4/4 passed |
| Native ASan/UBSan | 4/4 passed |
| Native CUDA memcheck and racecheck | 0 errors / 0 hazards |
| Python primitive suite under CUDA memcheck | 135 passed, 0 errors |
| Python matmul tests under CUDA racecheck | 19 passed, 0 hazards |
| MLP example on CPU and CUDA | CPU parity and row probability sums passed |

The CUDA-off extension links neither CUDA nor LLVM. The CUDA-enabled package
was restored and focused tests and the MLP example passed again. Run device
instrumentation with the Toolkit's `compute-sanitizer`, for example:

```bash
CORTEX_REQUIRE_BACKENDS=cuda uv run compute-sanitizer --tool memcheck --error-exitcode 1 python -m pytest tests/python/test_cuda_primitives.py -q
CORTEX_REQUIRE_BACKENDS=cuda uv run compute-sanitizer --tool racecheck --error-exitcode 1 python -m pytest tests/python/test_cuda_primitives.py -k matmul -q
```

The initial full parity run caught a changed binary dtype-mismatch error;
the existing message was restored and the final suite above passed. Two code
reviewers and Test/Acceptance QA reviewed this increment. Other GPU architectures,
remote CI and Metal execution are not validated by these Nightblade runs.
