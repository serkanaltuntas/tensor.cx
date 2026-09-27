# Cortex Runtime

Cortex Runtime is the current working name for a Python-first accelerator
runtime and future kernel compiler for tensor computation, starting with Apple
Metal.

Cortex Runtime is currently a research/runtime engineering project, not a
general-purpose machine-learning framework. Its purpose is to build a compact
tensor runtime with a backend-neutral C++ core, a Python API, a mandatory CPU
reference path, and accelerator backends that can be validated operation by
operation. Apple Metal is the first accelerator backend; a small CUDA prototype
is also available. ROCm,
Vulkan/SPIR-V, and MLIR-based lowering are longer-term directions.

The project is useful today as:

- a small Python tensor runtime for CPU and Apple Metal experiments
- a reference implementation for backend abstraction, operation dispatch, and
  CPU-vs-device correctness testing
- a foundation for future custom kernel compilation work

It is intentionally not a PyTorch, JAX, TensorFlow, MLX, Triton, or training
framework replacement. Autograd, distributed training, broad dtype coverage,
broadcasting, async streams, ROCm, and production compiler integration are
not part of the current runtime.

The first milestone was deliberately small:

```text
Python API -> C++20 core -> Metal backend -> static MSL add kernel -> correct result
```

That milestone has been achieved. The current runtime now includes CPU and
Metal tensor operations, matmul, reductions, selected neural-network primitives,
an experimental Metal kernel DSL, a hardened backend execution ABI, and a
completed Phase 10 MLIR decision prototype. Phase 9 CUDA is complete: the
optional backend supports discovery, float32/int32 copies, and float32
fill/add/multiply on a validated NVIDIA host.

## Documentation Map

- [`PROJECT.md`](PROJECT.md): source of truth for project purpose, roadmap,
  phase status, acceptance criteria, and major engineering decisions.
- [`docs/ROADMAP.md`](docs/ROADMAP.md): short status summary, what works today,
  what is intentionally deferred, and next known decisions.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): runtime layers, backend
  boundaries, dispatch model, error taxonomy, and core invariants.
- [`docs/BACKENDS.md`](docs/BACKENDS.md): backend ABI, execution contract,
  buffer ownership, launch metadata, and future backend expectations.
- [`docs/METAL_BACKEND.md`](docs/METAL_BACKEND.md): Apple Metal-specific
  implementation notes and current limitations.
- [`docs/CUDA_PHASE9_ENVIRONMENT.md`](docs/CUDA_PHASE9_ENVIRONMENT.md): CUDA
  environment decision gate and Phase 9 validation requirements.
- [`docs/PHASE_SEQUENCING_DECISION.md`](docs/PHASE_SEQUENCING_DECISION.md):
  rationale and guardrails for completing Phase 10 before Phase 9.
- [`docs/KERNEL_DSL.md`](docs/KERNEL_DSL.md): experimental kernel DSL scope,
  supported subset, and non-goals.
- [`docs/MLIR_DECISION.md`](docs/MLIR_DECISION.md): Phase 10 MLIR decision
  record and prototype result.

## Development Setup

Cortex Runtime is developed first on Apple Silicon macOS. The default build uses
`CORTEX_ENABLE_METAL=AUTO`: it enables Metal when the Apple Metal command-line
tools are available and otherwise falls back to a CPU-only build.

```bash
command -v uv
```

Use `uv` for Python environments and commands:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
uv run pytest
```

On a machine without Metal, or when validating the CPU-only path:

```bash
CMAKE_ARGS="-DCORTEX_ENABLE_METAL=OFF" uv pip install -e ".[dev]"
uv run pytest
```

To require Metal and fail clearly if the command-line tools are missing:

```bash
CMAKE_ARGS="-DCORTEX_ENABLE_METAL=ON" uv pip install -e ".[dev]"
```

Run the basic local benchmark with:

```bash
uv run python benchmarks/bench_elementwise.py
uv run python benchmarks/bench_copy.py
uv run python benchmarks/bench_matmul.py
```

The copy and elementwise benchmarks use the Phase 4 element-count sizes by
default: 1K, 16K, 256K, 1M, and 16M float32 elements. The matmul benchmark uses
matrix triplets such as `16x16x16`, `32x64x16`, and `64x64x64`.

A local Apple Silicon sample run is committed at
`benchmarks/sample_phase4_apple_silicon.txt`. The Phase 5 matmul sample is at
`benchmarks/sample_phase5_matmul_apple_silicon.txt`.

## CUDA prototype setup

CUDA is opt-in (`CORTEX_ENABLE_CUDA=OFF` by default); CPU-only and Apple builds
need no CUDA toolkit. On a CUDA host with a supported C++20 compiler and nvcc:

```bash
CMAKE_ARGS="-DCORTEX_ENABLE_METAL=OFF -DCORTEX_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=52" uv pip install -e ".[dev]"
CORTEX_REQUIRE_BACKENDS=cuda CORTEX_REQUIRE_BACKEND_CAPABILITIES=cuda:copy,cuda:tensor_factories_float32,cuda:binary_ops_float32 uv run pytest
```

`52` is Nightblade's GTX 980 Ti target, validated with CUDA 12.4 and GCC 13.
Select an architecture supported by your GPU and installed toolkit; select
`CC`, `CXX`, and `CUDAHOSTCXX` if the default host compiler is incompatible.
Full Nightblade commands and C++ checks:
[`docs/CUDA_PHASE9_VALIDATION.md`](docs/CUDA_PHASE9_VALIDATION.md).

```python
x = cx.ones((257,), device="cuda")
y = (x + x) * x
print(y.cpu().numpy()[:3])  # [2. 2. 2.]
```

The prototype exposes only device index 0. float32 add/multiply require exact
shape/dtype matches; scalars and empty contiguous tensors work. Copies preserve
float32/int32, but int32 fill/arithmetic, matmul, reductions, activations,
normalization, and generated CUDA kernels are not implemented. Operations are
synchronous and release the Python GIL during execution. CUDA is registered
only when its compiled backend and a usable device are available. CPU remains
the default for constructors; `best_device()` can select CUDA when available.

## Verification

The CUDA build CI job uses a pinned CUDA 12.4 container on a CPU-only hosted
runner. It checks compile/link and unavailable-device fallback; it does not
claim GPU execution. Run strict CUDA tests on a real host as described above.

CPU-only CI runs on GitHub Actions with Metal disabled:

```bash
CMAKE_ARGS="-DCORTEX_ENABLE_METAL=OFF" uv pip install -e ".[dev]"
uv run pytest
NANOBIND_DIR="$(uv run python -c 'import nanobind; print(nanobind.cmake_dir())')"
PYTHON_EXECUTABLE="$(uv run python -c 'import sys; print(sys.executable)')"
cmake -S . -B build/cpp-tests -DCORTEX_ENABLE_METAL=OFF -DCORTEX_BUILD_TESTS=ON -Dnanobind_DIR="${NANOBIND_DIR}" -DPython_EXECUTABLE="${PYTHON_EXECUTABLE}"
cmake --build build/cpp-tests --target cortex_backend_contract_tests
ctest --test-dir build/cpp-tests --output-on-failure
```

GitHub Actions also runs macOS Metal-on build jobs with MPSGraph enabled and
disabled. Hosted runners may still skip runtime Metal tests when a usable Metal
device is unavailable, but those jobs catch Apple framework compile/link breaks.

Local Metal verification should be run on Apple Silicon macOS:

```bash
xcode-select -p
xcrun -sdk macosx --find metal
xcrun -sdk macosx --find metallib
CMAKE_ARGS="-DCORTEX_ENABLE_METAL=ON" uv pip install -e ".[dev]"
uv run pytest
uv run python benchmarks/bench_elementwise.py
uv run python benchmarks/bench_copy.py
uv run python benchmarks/bench_matmul.py
```

To verify that custom Metal matmul works without the optimized primitive path:

```bash
CMAKE_ARGS="-DCORTEX_ENABLE_METAL=ON -DCORTEX_ENABLE_MPSGRAPH=OFF" uv pip install -e ".[dev]"
uv run pytest tests/python/test_matmul.py
```

Use a smaller benchmark smoke test while iterating:

```bash
uv run python benchmarks/bench_elementwise.py --sizes 1024 --repeats 2
uv run python benchmarks/bench_copy.py --sizes 1024 --repeats 2
uv run python benchmarks/bench_matmul.py --sizes 16x16x16 --repeats 2
```

## Current Status

Phase 7 is complete. The experimental kernel DSL under `cx.experimental` can
parse a restricted Python kernel into backend-neutral IR, emit text MSL, compile
that MSL into an in-memory Metal library artifact, validate the generated
function through native Metal library lookup, and launch float32 elementwise
and rowwise-reduction (bounded `for`/accumulator) kernels on Metal, with
`Kernel.reference(...)` as the CPU reference execution path. Phase 8 is complete: the backend ABI, shared
primitive/kernel contract validators, and null backend scaffold exist, and CPU
fill, add/multiply, unary transforms, reductions, and matmul route through
`CpuBackend::execute`. Metal add/multiply, unary transforms, axis/norm
transforms, reductions, matmul, and fill now also route through
`MetalBackend::execute`; the experimental generated-kernel launch path also
enters Metal through `BackendExecution`. Phase 9 is complete on Nightblade:
CUDA discovery, copies, and float32 fill/add/multiply use the existing registries
and execution contract. Setup and validation are in
`docs/CUDA_PHASE9_ENVIRONMENT.md` and `docs/CUDA_PHASE9_VALIDATION.md`.
Phase 10 (MLIR exploration) is complete and
ran ahead of Phase 9 under a documented sequencing exception: the decision
record `docs/MLIR_DECISION.md` answers "yes" — the experimental add kernel
lowers Cortex IR → MLIR → native code and matches the CPU reference
(`experiments/mlir/`) — while runtime integration remains deferred pending a new decision record;
the runtime itself contains no MLIR dependency. Phase 5 provides
CPU reference matmul, a correctness-first custom Metal matmul kernel, and an
optimized Metal primitive path. Phase 6 adds `sum`, `max`, `mean`, `exp`,
`gelu`, `silu`, `softmax`, `rmsnorm`, and `layernorm` on CPU and Metal. The
custom matmul kernel remains available through
`cx.matmul(a, b, backend="custom")`; `backend="optimized"` uses the Apple
optimized primitive path when it is enabled.

```python
import cortex_runtime as cx

print(cx.devices())

x = cx.tensor([1, 2, 3], device="cpu")
y = cx.tensor([4, 5, 6], device="cpu")
z = x + y

print(z.numpy())
# [5 7 9]

device = cx.best_device()
x = cx.ones((1_000_000,), dtype=cx.float32, device=device)
y = cx.ones((1_000_000,), dtype=cx.float32, device=device)
z = x + y

print(z.cpu().numpy()[:5])
# [2. 2. 2. 2. 2.]

a = cx.randn((2, 3), device="metal", seed=1)
b = cx.randn((3, 4), device="metal", seed=2)
c = cx.matmul(a, b)

print(c.shape)
# (2, 4)

r = cx.mean(c, axis=1)
m = c.max(axis=0)
s = cx.softmax(c, axis=1)
rms = cx.rmsnorm(c, axis=1)
ln = cx.layernorm(c, axis=1)
e = cx.gelu(cx.silu(cx.exp(r)))

print(r.cpu().numpy().shape)
# (2,)
print(m.cpu().numpy().shape)
# (4,)
print(e.cpu().numpy().shape)
# (2,)
print(s.cpu().numpy().shape)
# (2, 4)
print(rms.cpu().numpy().shape)
# (2, 4)
print(ln.cpu().numpy().shape)
# (2, 4)
```

### Behavior notes

- Supported dtypes are `float32` and `int32`. Float inputs are narrowed to
  `float32`, so values may lose precision or overflow to `inf`.
- `int32` `add`/`multiply` overflow wraps (defined two's-complement), matching
  NumPy and identical on CPU and Metal.
- `sum`, `max`, and `mean` require an explicit `axis`. Negative axes are
  supported. `sum` over an empty axis returns zeros, `mean` over an empty axis
  returns NaNs, and `max` over an empty axis raises `ValueError`.
- `softmax` requires an explicit `axis`, preserves the input shape, supports
  negative axes, and uses max-subtraction for numerical stability.
- `rmsnorm` requires an explicit `axis`, preserves the input shape, supports
  negative axes, and accepts `eps` with default `1e-5`. It computes
  `x / sqrt(mean(x*x, axis, keepdims=True) + eps)` without affine weights.
- `layernorm` requires an explicit `axis`, preserves the input shape, supports
  negative axes, and accepts `eps` with default `1e-5`. It computes
  `(x - mean) / sqrt(variance + eps)`, with mean and variance taken along
  `axis` as `keepdims=True`, without affine weights.
- `mean`, `exp`, `gelu`, `silu`, `softmax`, `rmsnorm`, and `layernorm`
  currently support `float32` tensors only; `int32` inputs are rejected instead
  of being implicitly cast. `gelu` uses the common tanh approximation.
- `cx.experimental.kernel` is a Phase 7 experimental parser/emitter/compiler
  launch scaffold. It exposes `parse_ir()`, text-only `emit_msl()`, and
  `compile(target="metal")` for an in-memory metallib artifact when Apple Metal
  command-line tools are available. The returned `CompiledKernel` can validate
  Metal library load/function lookup with `validate_metal_function()` when the
  native extension is built with Metal support and a Metal runtime is available,
  and can launch float32 elementwise and rowwise-reduction (bounded
  `for`/accumulator) kernels with `launch(...)`; `Kernel.reference(...)` runs
  the same IR on CPU tensors as the reference path.
  Calling a decorated kernel compiles and launches through the same experimental
  path. Non-empty launches enter the Metal backend through `BackendExecution`;
  zero-thread launches validate the Metal function and return as no-ops. The
  launch path is synchronous, Metal-only, one-output, exact-shape, and still
  under `cx.experimental`.
- Execution is synchronous, but native backend calls release the Python GIL, so
  operations issued from multiple Python threads run concurrently in the native
  layer and stay correct (shared caches are mutex-guarded; see
  `docs/METAL_BACKEND.md`). There is still no async/stream API.

## Naming

```text
Working product name: Cortex Runtime
Python package/import: cortex_runtime
Documentation alias: import cortex_runtime as cx
Python extension module: cortex_runtime._core
C++ source root: cpp/cortex/
C++ namespace: cortex
```
