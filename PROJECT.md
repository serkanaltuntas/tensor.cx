# Cortex Runtime — Project Definition and Roadmap

## 1. Project Summary

**Cortex Runtime** is the current working name for a Python-first accelerator
runtime and kernel compiler for tensor computation. The Python package name is
`cortex_runtime`. This name is provisional and may change later if the project
moves under a broader product or brand.

The first version will be developed on a MacBook Pro with Apple Silicon and will target Apple Metal. The long-term architecture must remain backend-neutral so that CUDA, ROCm, Vulkan/SPIR-V, and MLIR-based compiler paths can be added later.

Cortex Runtime is not intended to start as a full deep learning framework, a PyTorch replacement, or a universal training stack. It should begin as a compact but serious runtime that can allocate tensors, move data, execute GPU operations, and eventually compile user-defined kernels.

The initial product direction is:

> Cortex Runtime is a Python-first accelerator runtime and kernel compiler, starting with Apple Metal and designed for future CUDA, ROCm, Vulkan, and MLIR backends.

The first practical goal is simple:

> From Python, create tensors on Apple GPU, run basic tensor operations through Metal/MPSGraph/custom Metal kernels, and validate correctness against CPU/NumPy.

Current naming:

```text
Working product name: Cortex Runtime
Python package/import: cortex_runtime
Documentation alias: import cortex_runtime as cx
Python extension module: cortex_runtime._core
C++ source root: cpp/cortex/
C++ namespace: cortex
```

---

## Project Status

> Update this block at the end of every phase. It is the single source of truth
> for "where are we." A phase is only `done` when its acceptance criteria pass in
> CI (CPU), on the local Mac (Metal), or on the selected CUDA host (CUDA phases).

```text
Current phase:          Phase 9 — done (2026-09-28, Nightblade CUDA validation)
                        Phase 10 — done (earlier sequencing exception)
Last verified milestone: Phase 9 — CUDA discovery, copies, float32 fill/add/mul
                        through the existing registry and BackendExecution ABI
v0.1 target:            achieved at end of Phase 3
Binding decided:        nanobind (see §5.6)
MLIR decided:           optional CPU-first runtime integration implemented; see
                        docs/MLIR_RUNTIME_INTEGRATION_DECISION.md (2026-09-28)
CUDA MLIR:              guarded float32 add/subtract/multiply implemented on sm_52;
                        docs/MLIR_CUDA_INTEGRATION_DECISION.md (updated 2026-09-29)
Expressions:            bounded CUDA local/nested expressions implemented
                        docs/MLIR_CUDA_EXPRESSIONS_DECISION.md (2026-09-29)
Active user goal:       complete CUDA product list; docs/CUDA_PRODUCT_COMPLETION.md
CUDA primitives:        float32 matmul, reductions, activations and normalization
                        implemented; docs/CUDA_PRIMITIVES_VALIDATION.md
Distribution:           CPU/CUDA sdist and wheel installs locally verified;
                        docs/DISTRIBUTION.md (Nightblade host/ABI limits)
Next implementation:    performance, compatibility and continuous GPU validation
Open scope:             other SM/host versions and broader compiler features
```

Phase checklist:

```text
[x] Phase 0   Project bootstrap
[x] Phase 1   CPU backend
[x] Phase 2   Metal backend foundation
[x] Phase 3   First Metal kernels        <- v0.1 ships here
[x] Phase 4   Runtime polish
[x] Phase 5   MPSGraph matmul
[x] Phase 6   Reductions & NN primitives
[x] Phase 7   Experimental kernel DSL
[x] Phase 8   Backend interface hardening
[x] Phase 9   CUDA prototype             <- Nightblade validated, 2026-09-28
[x] Phase 10  MLIR exploration           <- done, out of order (see below)
```

Phase 10 ran ahead of Phase 9 under the historical sequencing exception in
[`docs/PHASE_SEQUENCING_DECISION.md`](docs/PHASE_SEQUENCING_DECISION.md).
Phase 9 is now complete on the selected Nightblade CUDA host; entry evidence,
commands, results, and validation limits are recorded in
[`docs/CUDA_PHASE9_VALIDATION.md`](docs/CUDA_PHASE9_VALIDATION.md).
Phase 10's [`MLIR decision`](docs/MLIR_DECISION.md) remains a research prototype.
The follow-up [integration decision](docs/MLIR_RUNTIME_INTEGRATION_DECISION.md)
selects the first CPU runtime slice and its acceptance gates. That experimental
slice is implemented and locally validated on Nightblade (Linux x86_64, LLVM
21.1.8); it does not complete a new broad compiler phase. Metal execution and
remote CI results are not claimed by this host validation.

---

## 2. Core Technical Stack

The approved initial stack is:

```text
Python API
C++20 core runtime
Metal-cpp/C++ Apple Metal bridge
MSL custom kernels
nanobind Python binding
uv Python package/environment manager
optional CUDA prototype; future ROCm/MLIR integration
```

Recommended default choice:

```text
Python API: Python 3.11+
Core runtime: C++20
Apple bridge: Metal-cpp/C++ preferred for Metal
GPU kernel language: Metal Shading Language (.metal)
Python binding: nanobind (decided — see §5.6)
Python package manager: uv
Build system: CMake + scikit-build-core
Test framework: pytest + C++ unit tests
Benchmarking: Python benchmark scripts + optional C++ microbenchmarks
```

Use `uv` for Python environments, dependency installation, editable installs,
and Python command execution. Do not use `pip`, `python -m venv`, Poetry, PDM, or
Conda for project workflow commands unless the user explicitly requests it.

Objective-C++ should not be used by default. Prefer Metal-cpp and plain C++ for
the Apple Metal backend whenever possible. If a future Apple API, especially
MPSGraph, cannot be reached cleanly through C++/Metal-cpp, isolate the smallest
possible Objective-C++ shim inside the Apple backend and keep it out of the
backend-neutral C++ core.

Rust and Zig should not be used for the initial runtime core. They may be useful later for CLI tools, cache utilities, or packaging helpers, but the runtime/compiler/backend core should remain C++20-centered.

---

## 3. Development Environment

Initial development target:

```text
Machine: MacBook Pro with Apple Silicon
OS: recent macOS
Compiler: Apple Clang
GPU API: Metal
Preferred Metal host API: Metal-cpp/C++
Primitive tensor graph API: MPSGraph
Kernel language: Metal Shading Language
Python: 3.11 or newer
Build: CMake + scikit-build-core
Binding: nanobind
```

The 2013 Mac Pro / FirePro D700 is not the primary development target. It may be used later for legacy Metal experiments, but the first implementation should target Apple Silicon for faster development, better tooling, and a more realistic current Apple GPU environment.

---

## 4. Product Positioning

Cortex Runtime should be positioned between a low-level GPU API and a high-level ML framework.

It should not initially compete with PyTorch, JAX, TensorFlow, MLX, or Triton directly. Instead, it should begin as a small runtime/compiler project that learns from them.

Useful comparison:

```text
PyTorch:
  Full ML framework, autograd, modules, optimizers, distributed training.

Triton:
  Python-based DSL and compiler for custom GPU kernels.

MLX:
  Apple Silicon-focused machine learning array framework.

Cortex Runtime:
  Python-first accelerator runtime + backend abstraction + future custom kernel compiler.
```

Long-term, Cortex Runtime may include Triton-like features, but the first versions should be narrower.

---

## 5. Design Principles

### 5.1 Python-first, C++ core

The user-facing API should feel Pythonic. The runtime core should be implemented in C++20.

Python should handle:

```text
- user API
- tensor constructors
- device selection
- shape/dtype ergonomics
- test code
- benchmark scripts
- future kernel DSL syntax
```

C++ should handle:

```text
- tensor metadata
- buffer ownership
- backend interfaces
- runtime dispatch
- operation execution
- command scheduling
- error handling
- backend-independent abstractions
```

The Apple backend C++/Metal-cpp layer should handle:

```text
- Metal device access
- Metal command queue
- MTLBuffer management
- Metal shader library compilation/loading
```

If MPSGraph or another Apple framework later requires an API surface that is not
practical from Metal-cpp/C++, use a tiny backend-local bridge only for that
specific integration.

MSL should handle:

```text
- custom GPU kernels
- elementwise operations
- simple reductions
- later fused kernels
```

---

### 5.2 Backend-neutral core

The Cortex Runtime core must not be hardcoded to Metal.

Metal is only the first backend.

The core should speak in terms of abstract devices, buffers, tensors, operations, and execution requests.

Target shape:

```text
Cortex Runtime Python API
  ↓
C++20 Core Runtime
  ↓
Backend Interface
  ├── CPU backend
  ├── Metal backend
  ├── CUDA prototype backend
  ├── future ROCm backend
  ├── future Vulkan/SPIR-V backend
  └── future MLIR backend
```

---

### 5.3 Separate primitive path from custom kernel path

The Metal backend should have two execution paths.

#### Primitive path

Used for known tensor operations such as:

```text
matmul
reduce
possibly convolution later
possibly normalization later
```

On Apple, this path may use MPSGraph or Metal Performance Shaders.

Future equivalents:

```text
Metal: MPSGraph / MPS
CUDA: cuBLAS / cuDNN / custom CUDA
ROCm: rocBLAS / MIOpen
CPU: Eigen / oneDNN / custom C++
```

#### Custom kernel path

Used for project-owned kernels and future user-defined kernels.

On Apple, this path uses:

```text
Cortex Runtime operation or kernel IR
  ↓
MSL code generation or static .metal kernels
  ↓
Metal compute pipeline
  ↓
Metal command buffer execution
```

This separation is important. If the entire runtime becomes an MPSGraph wrapper, Cortex Runtime will not become portable. MPSGraph should be used as an Apple backend optimization, not as the central Cortex Runtime abstraction.

---

### 5.4 CPU reference implementation is mandatory

Every GPU operation must have a CPU reference path.

The CPU backend is not optional. It is needed for:

```text
- correctness tests
- deterministic debugging
- environments without GPU
- comparing output tolerance
- future backend validation
```

For every Metal operation, tests should compare:

```text
CPU result vs Metal result
```

with dtype-specific tolerances.

---

### 5.5 Small working system before ambitious compiler work

The first milestone is not “write a compiler.”

The first milestone is:

```text
Python tensor API
  ↓
C++ core
  ↓
Metal backend
  ↓
elementwise add on GPU
  ↓
correct result copied back to Python
```

Only after that should the project add more compiler-like features.

---

### 5.6 Cross-cutting engineering decisions (decide once, in Phase 1)

These decisions touch every layer. They are cheap to set now and expensive to
retrofit, so they are settled here instead of being discovered during a later
phase. Code written in Phases 1–3 must already obey them.

**Python binding: nanobind (decided).** Smaller, faster, better error messages,
and a native `nb::ndarray` for the NumPy bridge. pybind11 is no longer an
accepted alternative for new binding code.

**Operation dispatch: data-driven, not one virtual method per op.** The
`Backend` interface must NOT grow a method per operation (`add`, `multiply`,
`relu`, …). It exposes a single execution entry point:

```cpp
Status execute(const BackendExecution& execution);
```

`BackendExecution` carries the op class, `OpDesc`, input tensor metadata,
mutable output tensor metadata/result slots, optional launch /
compilation-target metadata for kernel execution, and ordered kernel arguments
for caller-owned kernel buffers and scalar launch parameters. Allocation-style
primitives such as `fill` use output metadata as an allocation descriptor.
`OpDesc` carries an op enum plus attributes, including scalar values for
primitive operations that need them and backend-neutral matmul preference for
auto/custom/optimized algorithm selection; backends switch on the enum. This
keeps Phase 6 (15+ ops) and Phase 8 (primitive vs kernel ops) from becoming a
rewrite. The per-op method sketch in §8.2 is illustrative only and must not be
implemented literally.

**Error handling: `expected`, no exceptions across the Metal boundary.** Core
functions return `Status` or `expected<T, Status>` (use `tl::expected` until
C++23 `std::expected` is available). Metal reports failures via `NS::Error**`,
not C++ exceptions — never let an exception cross the Metal-cpp boundary.
Translate `Status` → Python exception exactly once, at the nanobind layer. This
is a Phase-1 principle, not a Phase-4 task: error paths written exception-free
from the start avoid leaks in the manual-memory Metal layer.

**Memory ownership: RAII everywhere, including Metal handles.**

```text
Buffer  = move-only, reference-counted handle to device memory.
Tensor  = cheap value type: {dtype, shape, strides, offset, device} +
          shared_ptr<Buffer>. Copying a Tensor makes a view, never a data copy.
Metal   = every MTL::/NS:: object held in NS::SharedPtr
          (NS::TransferPtr for owned, NS::RetainPtr for borrowed).
          No raw Metal pointers are stored anywhere.
```

Metal-cpp has no ARC; manual retain/release is the single biggest bug source.
RAII wrapping is mandatory from Phase 2, not deferred.

**NumPy lives only on the Python/binding side.** The C++ core has zero NumPy or
Python knowledge. `cx.tensor(list)` and `Tensor.numpy()` are bridged in the
binding layer via `nb::ndarray`. NumPy must never appear in `cpp/cortex/core/` or
in any backend.

**No broadcasting in v0.1.** Elementwise ops require exact shape AND dtype match.
A mismatch is a clear error — never a silent broadcast or reinterpret.
Broadcasting is a deliberate later feature, not one that leaks in through `add`.

**Shader compilation: build-time `.metallib`, embedded.** `.metal` kernels are
compiled to a `.metallib` at build time via CMake and loaded from the bundle at
runtime. Cortex Runtime MSL string compilation is reserved for Phase 7 (kernel DSL)
only. Decide and wire this in Phase 3 — it affects packaging and distribution.

---

### 5.7 Invariants (check on every change)

A short list to review before any commit. Violating one is a design regression,
not a style nit.

```text
1. Every GPU op has a CPU reference + a CPU-vs-device test before it is "done."
2. No Apple/Metal type appears outside cpp/cortex/backends/metal/.
3. No NumPy/Python type appears in cpp/cortex/core/ or any backend.
4. The core never names a concrete backend; live public selection must stay on
   registry/string-keyed routing, including future CUDA dispatch.
5. Every Metal handle is RAII-wrapped; no manual retain/release calls.
6. Adding an op = OpDesc enum entry + CPU backend impl + test, in that order.
7. A public Python API change ships with its doc + test update in the same change.
```

---

## 6. Initial Scope

### 6.1 In scope for v0.1

Cortex Runtime v0.1 should include:

```text
- Python package named cortex_runtime
- C++20 runtime core
- CPU backend
- Metal backend
- Device discovery
- Tensor object
- Shape and dtype metadata
- GPU buffer allocation
- Host-to-device copy
- Device-to-host copy
- Elementwise add
- Elementwise multiply
- Fill/zero operation
- Basic benchmark script
- pytest correctness tests
```

Minimum Python experience:

```python
import cortex_runtime as cx

x = cx.tensor([1.0, 2.0, 3.0], device="metal")
y = cx.tensor([4.0, 5.0, 6.0], device="metal")

z = x + y

print(z.cpu().numpy())
# expected: [5.0, 7.0, 9.0]
```

---

### 6.2 Out of scope for v0.1

Do not implement these in the first version:

```text
- autograd
- model training
- distributed training
- CUDA backend
- ROCm backend
- TPU backend
- full graph compiler
- ONNX import
- PyTorch replacement
- high-performance matmul from scratch
- quantized LLM inference
- dynamic kernel DSL
```

The first version must prove that the project structure, bindings, runtime abstractions, and Metal execution path work.

---

## 7. Repository Structure

Current repository layout:

```text
README.md
PROJECT.md
AGENTS.md
CLAUDE.md
pyproject.toml
CMakeLists.txt

python/cortex_runtime/
  __init__.py
  device.py
  tensor.py
  experimental.py
  testing.py

cpp/cortex/core/
  backend.h/.cpp
  buffer.h
  device.h
  dtype.h/.cpp
  expected.h
  operation.h/.cpp
  shape.h/.cpp
  status.h/.cpp
  tensor.h

cpp/cortex/backends/
  cpu/
  metal/
    kernels/elementwise.metal
  null/

bindings/python_module.cpp
tests/python/
tests/cpp/
benchmarks/
docs/
cmake/
.github/workflows/
```

The exact file structure can evolve, but the conceptual separation must remain:

```text
python/cortex_runtime/       user-facing API
cpp/cortex/core/             backend-neutral runtime
cpp/cortex/backends/         backend-specific implementations
bindings/                    Python/C++ bridge
tests/                       correctness validation
benchmarks/                  performance checks
docs/                        architecture and development notes
```

---

## 8. Core Runtime Concepts

### 8.1 Device

A device represents a compute target.

Examples:

```text
cpu
metal
cuda    optional Phase 9 prototype (float32 fill/add/mul and buffer copies)
rocm    future
```

Initial API:

```python
cx.devices()
cx.device("cpu")
cx.device("metal")
cx.best_device()
```

`cx.best_device()` should return Metal on Apple Silicon if available, otherwise CPU.

---

### 8.2 Backend

A backend implements device-specific behavior.

Initial backend interface should include:

```cpp
class Backend {
public:
    virtual std::string name() const = 0;

    virtual DeviceList devices() = 0;

    virtual Buffer allocate(size_t nbytes) = 0;

    virtual void copy_host_to_device(Buffer& dst, const void* src, size_t nbytes) = 0;
    virtual void copy_device_to_host(void* dst, const Buffer& src, size_t nbytes) = 0;

    virtual Tensor add(const Tensor& a, const Tensor& b) = 0;
    virtual Tensor multiply(const Tensor& a, const Tensor& b) = 0;
    virtual Tensor fill(Shape shape, DType dtype, Scalar value) = 0;
};
```

This is only a conceptual sketch. The actual implementation can use status/error types, smart pointers, and better ownership rules.

---

### 8.3 Tensor

A tensor contains:

```text
- dtype
- shape
- strides
- device
- buffer reference
- offset
```

Initial supported dtypes:

```text
float32
int32
bool optional
```

Do not start with float16/bfloat16. Add them after the runtime is stable.

Initial supported layout:

```text
contiguous row-major tensors only
```

Strides can exist in metadata, but non-contiguous tensor execution should be out of scope at first.

---

### 8.4 Buffer

A buffer owns or references memory on a device.

CPU buffer:

```text
std::vector<uint8_t> or aligned allocation
```

Metal buffer:

```text
MTLBuffer
```

The C++ core should not expose Apple platform types directly. Metal-specific handles must stay inside the Metal backend.

---

### 8.5 Operation

Implemented operations through Phase 6:

```text
copy
fill
add
multiply
sum
max
mean
exp
gelu
silu
matmul
softmax
rmsnorm
layernorm
```

Future operation candidates:

```text
subtract
divide
relu
```

Operations should first be implemented as direct runtime calls. A graph abstraction can come later.

---

## 9. Metal Backend Design

The Metal backend is the first real accelerator backend.

It must provide:

```text
- Metal device selection
- Metal command queue
- Metal buffer allocation
- Metal library loading
- Metal compute pipeline creation
- command buffer execution
- synchronization
- error reporting
```

### 9.1 Static kernels first

Do not start with dynamic MSL code generation.

Start with static `.metal` kernels checked into the repository.

Example static kernels:

```text
elementwise_add_f32
elementwise_mul_f32
fill_f32
```

Later, add dynamic kernel generation.

---

### 9.2 Matmul: custom MSL first, then MPSGraph

Matmul is the first place the project is tempted to become an MPSGraph wrapper —
exactly the failure mode §5.3 warns against. To defend portability, the custom
kernel path must be proven end-to-end *before* the MPSGraph crutch exists.

Recommended sequence:

```text
v0.1: elementwise kernels using MSL
v0.2: reduce kernels using MSL
v0.3: naive custom MSL matmul (correctness only, not fast)
v0.3: MPSGraph matmul as the optimized primitive path
v0.4: experimental custom tiled matmul using MSL
```

Rationale: the naive MSL matmul is intentionally slow — its job is to prove that
the custom-kernel path can express matmul and pass the CPU comparison. MPSGraph
then provides the *fast* matmul as a primitive-path optimization. Both ship in
v0.3 and benchmark against each other, so the project never has matmul that
*only* exists as an MPSGraph call. Do not skip the naive MSL version; skipping it
is how Cortex Runtime quietly becomes an MPSGraph wrapper.

---

### 9.3 Synchronization

Initial implementation may use simple synchronous execution:

```text
submit command buffer
wait until completed
return result
```

Asynchronous execution can come later.

Future async API:

```python
stream = cx.stream("metal")
with cx.use_stream(stream):
    z = x + y
stream.synchronize()
```

Do not implement streams in v0.1.

---

## 10. Python API

The Python API should be small and stable.

Initial target:

```python
import cortex_runtime as cx

x = cx.tensor([1, 2, 3], dtype=cx.float32, device="metal")
y = cx.ones((3,), dtype=cx.float32, device="metal")
z = x + y

assert z.device == "metal"
print(z.cpu().numpy())
```

Required functions/classes:

```text
cx.tensor(data, dtype=None, device=None)
cx.empty(shape, dtype, device)
cx.zeros(shape, dtype, device)
cx.ones(shape, dtype, device)

Tensor.cpu()
Tensor.numpy()
Tensor.to(device)
Tensor.shape
Tensor.dtype
Tensor.device

cx.devices()
cx.best_device()
```

Operator overloads:

```text
Tensor.__add__
Tensor.__mul__
```

Avoid too many NumPy/PyTorch compatibility features early.

---

## 11. Build System

Use:

```text
pyproject.toml
scikit-build-core
CMake
nanobind
uv
```

Expected developer workflow:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
uv run pytest
```

CMake should compile:

```text
C++20 core
C++20/Metal-cpp Metal backend
MSL kernels if needed
Python extension module
```

The default build may fall back to CPU-only when the Metal command-line
toolchain is unavailable. A build explicitly configured with
`-DCORTEX_ENABLE_METAL=ON` must fail clearly if Metal is unavailable.

---

## 12. Testing Strategy

Testing must be strict from the first commit.

### 12.1 CPU tests

Test:

```text
dtype creation
shape creation
tensor allocation
copy
fill
add
multiply
```

### 12.2 Metal tests

Metal tests should be skipped if Metal is not available.

Test:

```text
device discovery
buffer allocation
host → device copy
device → host copy
fill_f32
add_f32
mul_f32
```

### 12.3 Correctness comparison

Every Metal operation must compare against CPU.

Default tolerances (`cx.testing.assert_allclose` should use these unless an op
overrides them):

```text
float32 elementwise:     rtol=1e-6, atol=1e-6
float32 reductions/sum:  rtol=1e-5, atol=1e-5   (accumulation order differs)
float32 matmul:          rtol=1e-4, atol=1e-4   (FMA + tiling differences)
int32:                   exact equality
```

Reductions and matmul are looser on purpose: GPU and CPU accumulate in different
orders, so bit-exact equality is the wrong test. Exact equality is required only
for integer ops. Boolean tensors are not currently exposed; if added later, they
should use exact equality.

Example:

```python
def test_add_metal_matches_cpu():
    x_cpu = cx.tensor([1, 2, 3], device="cpu", dtype=cx.float32)
    y_cpu = cx.tensor([4, 5, 6], device="cpu", dtype=cx.float32)

    expected = x_cpu + y_cpu

    x_gpu = x_cpu.to("metal")
    y_gpu = y_cpu.to("metal")
    actual = (x_gpu + y_gpu).cpu()

    cx.testing.assert_allclose(actual, expected)
```

---

## 13. Benchmarking Strategy

Do not optimize blindly.

Initial benchmarks:

```text
host-to-device copy bandwidth
device-to-host copy bandwidth
elementwise add throughput
elementwise multiply throughput
CPU vs Metal for different tensor sizes
```

Benchmark sizes:

```text
1K elements
16K elements
256K elements
1M elements
16M elements
```

Early expectation:

```text
Small tensors may be slower on GPU due to launch overhead.
Large tensors should show GPU advantage for simple elementwise work.
```

Benchmark script examples:

```bash
uv run python benchmarks/bench_copy.py
uv run python benchmarks/bench_elementwise.py
```

---

## 14. Roadmap

### Phase 0 — Project bootstrap

Goal:

```text
Create the repository and build skeleton.
```

Tasks:

```text
- Create pyproject.toml
- Create CMakeLists.txt
- Add Python package skeleton
- Add C++ core skeleton
- Add binding skeleton
- Add pytest setup
- Add basic README
- Add docs/ARCHITECTURE.md
```

Acceptance criteria:

```text
uv pip install -e ".[dev]" works
import cortex_runtime works
uv run pytest runs
```

Definition of Done:

```text
- A clean checkout, on a fresh uv-managed venv, builds and imports with the documented
  commands and nothing else.
- `uv run pytest` collects and runs (even with zero real tests) and exits 0.
- CMake configures C++20 + the nanobind extension; the build fails with a clear
  message if the toolchain is missing.
- docs/ARCHITECTURE.md exists and names the layer boundaries from §7.
- The Project Status block is updated to "Phase 0 — done."
```

---

### Phase 1 — CPU backend

Goal:

```text
Implement the minimal tensor runtime on CPU.
```

Tasks:

```text
- DType implementation
- Shape implementation
- Tensor metadata
- CPU buffer
- CPU backend
- tensor creation from Python lists
- tensor to NumPy conversion
- zeros/ones/empty
- add/multiply on CPU
```

Acceptance criteria:

```text
x = cx.tensor([1, 2, 3], device="cpu")
y = cx.tensor([4, 5, 6], device="cpu")
z = x + y
z.numpy() returns [5, 7, 9]
```

Definition of Done:

```text
- add, multiply, fill, zeros, ones, empty work for float32 and int32, 1D.
- Round-trip cx.tensor(list) -> .numpy() is exact for both dtypes.
- Shape or dtype mismatch on add/multiply raises a clear error (no broadcast).
- The Backend dispatch uses OpDesc/execute (§5.6), not per-op methods.
- No Python or NumPy type appears in cpp/cortex/core/ (invariant 3).
- All Phase 1 tests run in CI on the CPU path and pass.
```

---

### Phase 2 — Metal backend foundation

Goal:

```text
Create Metal device, allocate Metal buffers, and copy data.
```

Tasks:

```text
- Metal backend registration
- Metal device discovery
- Metal command queue creation
- Metal buffer allocation
- host-to-device copy
- device-to-host copy
- Tensor.to("metal")
- Tensor.cpu()
```

Acceptance criteria:

```text
x_cpu = cx.tensor([1, 2, 3], device="cpu")
x_gpu = x_cpu.to("metal")
x_back = x_gpu.cpu()
x_back matches x_cpu
```

Definition of Done:

```text
- Host->device->host round-trip is exact for float32 and int32.
- Every Metal handle is held via NS::SharedPtr; no manual retain/release exists
  in the codebase (invariant 5).
- No MTL::/NS:: type appears outside cpp/cortex/backends/metal/ (invariant 2).
- Metal tests skip cleanly (not fail) when Metal is unavailable.
- A 16M-element round-trip runs with no leak across 1000 iterations
  (watch process memory; it must be flat).
```

---

### Phase 3 — First Metal kernels

Goal:

```text
Run custom MSL kernels for simple elementwise operations.
```

Tasks:

```text
- Add elementwise.metal
- Compile/load Metal library
- Create compute pipelines
- Implement add_f32
- Implement mul_f32
- Implement fill_f32
- Add Python tests comparing CPU and Metal
```

Acceptance criteria:

```text
x = cx.tensor([1, 2, 3], device="metal")
y = cx.tensor([4, 5, 6], device="metal")
z = x + y
z.cpu().numpy() matches [5, 7, 9]
```

Definition of Done (this is the v0.1 gate):

```text
- add_f32, mul_f32, fill_f32 run on Metal and match CPU within §12.3 tolerance.
- Kernels load from a build-time .metallib embedded in the extension; no runtime
  MSL string compilation (§5.6).
- The §18 success snippet (1M-element add via best_device) prints [2. 2. 2. 2. 2.].
- Every Metal op has a CPU-vs-Metal test (invariant 1).
- README documents the local Mac setup well enough for a fresh machine.
- Tag v0.1.
```

---

### Phase 4 — Runtime polish

Goal:

```text
Make the first GPU runtime usable and debuggable.
```

Tasks:

```text
- Better error handling
- Better shape validation
- Dtype validation
- Device mismatch errors
- Benchmark scripts
- CI for CPU path
- Local-only Metal test instructions
```

Acceptance criteria:

```text
The project is usable locally on MacBook Pro.
Common mistakes produce clear errors.
Benchmarks produce readable output.
```

Definition of Done:

```text
- Device mismatch, shape mismatch, and dtype mismatch each raise a distinct,
  message-bearing Python exception (translated once at the binding, §5.6).
- CI runs the full CPU path on every push; Metal tests have documented local
  run instructions.
- bench_copy.py and bench_elementwise.py emit a readable table across the §13
  sizes, and a sample run is committed.
- The error taxonomy is listed in docs/ARCHITECTURE.md.
```

---

### Phase 5 — Matmul (custom MSL first, then MPSGraph)

Goal:

```text
Prove matmul on the custom-kernel path, THEN add MPSGraph as the fast path.
```

Tasks:

```text
- Implement a naive custom MSL matmul_f32 (correctness only, not optimized)
- Compare the naive MSL matmul against the CPU reference
- Add 2D tensor support if not already sufficient
- Add MPSGraph integration behind the Metal backend interface
- Implement matmul_f32 through MPSGraph
- Validate shapes for both paths
- Benchmark naive MSL vs MPSGraph vs NumPy/PyTorch/MLX if available
```

Acceptance criteria:

```text
a = cx.randn((M, K), device="metal")
b = cx.randn((K, N), device="metal")
c = cx.matmul(a, b)
c.cpu() matches CPU reference within tolerance
```

Definition of Done:

```text
- Both a custom MSL matmul AND an MPSGraph matmul exist and pass CPU comparison
  (matmul tolerance, §12.3) for non-square M,K,N including a 1xN and Nx1 case.
- A benchmark table reports naive-MSL vs MPSGraph vs reference for at least
  three sizes; the numbers are committed to the repo.
- Removing the MPSGraph path still leaves a working (slow) matmul. This is the
  proof Cortex Runtime is not an MPSGraph wrapper.
```

---

### Phase 6 — Reductions and neural-network primitives

Goal:

```text
Add operations needed for future transformer inference experiments.
```

Candidate operations:

```text
sum
max
mean
exp
softmax
layernorm
rmsnorm
gelu
silu
```

Implementation strategy:

```text
- simple MSL kernels first
- optimize only after benchmarks
- keep CPU references mandatory
```

Progress:

```text
- sum and max are implemented for float32 and int32 on CPU and Metal.
- mean, exp, gelu, silu, softmax, rmsnorm, and layernorm are implemented for
  float32 on CPU and Metal.
- Reductions require an explicit axis, support negative axes, and remove the
  reduced axis from the output shape.
- softmax requires an explicit axis, preserves the input shape, supports
  negative axes, and uses max-subtraction for numerical stability.
- rmsnorm requires an explicit axis, preserves the input shape, supports
  negative axes, and computes x / sqrt(mean(x*x, axis, keepdims=True) + eps)
  without affine weights.
- layernorm requires an explicit axis, preserves the input shape, supports
  negative axes, and computes
  (x - mean(x, axis, keepdims=True)) /
  sqrt(mean((x - mean)^2, axis, keepdims=True) + eps) without affine weights.
- sum over an empty axis returns zeros; mean over an empty axis returns NaNs;
  max over an empty axis is rejected.
```

Acceptance criteria:

```text
softmax, rmsnorm, and layernorm work on Metal and match CPU reference.
```

Definition of Done:

```text
- sum, max, mean, exp, softmax, layernorm, rmsnorm, gelu, silu each have a CPU
  reference and pass CPU-vs-Metal within the reduction tolerance (§12.3).
- Reductions are tested along a non-trivial axis, not just full-tensor.
- softmax is numerically stable (max-subtraction); tested on large-magnitude
  inputs without overflow.
- Each op is registered through OpDesc, not a bespoke method (§5.6).
```

---

### Phase 7 — Experimental kernel DSL

Goal:

```text
Start the Triton-like direction.
```

Current status:

```text
The experimental namespace, @cx.experimental.kernel metadata wrapper,
restricted Python AST -> backend-neutral IR parser, text MSL emitter, in-memory
metallib compile artifact, native Metal library load/function lookup validation,
and a narrow synchronous Metal launch hook exist for the first elementwise
float32 subset. A generated add kernel now binds runtime buffers, launches on
Metal through `BackendExecution`, and matches the CPU reference. Keep the DSL
under cx.experimental until the semantics are broader and stable enough for a
public top-level API.
```

Initial design:

```python
@cx.experimental.kernel
def add_kernel(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = a[i] + b[i]
```

Implementation approach:

```text
- Start with restricted Python AST parsing
- Generate a simple internal IR
- Emit MSL for the Metal backend
- Support only simple elementwise kernels at first
```

Do not attempt full Python semantics.

Supported first features:

```text
program_id
thread_id
block_size
if condition
load
store
basic arithmetic
float32
int32 indices
```

Acceptance criteria:

```text
A user-defined Python kernel can be compiled to MSL, launched on Metal, and tested against CPU.
```

Definition of Done:

```text
- A @cx.experimental.kernel elementwise add compiles Python AST -> Cortex
  Runtime IR -> MSL, launches on Metal, and matches CPU.
- Unsupported Python constructs raise a clear compile-time error naming the
  offending node — never silently miscompile.
- The IR is documented in docs/ and is backend-agnostic (no MSL assumptions baked
  into the IR itself).
- This phase is explicitly time-boxed: if AST->IR->MSL is not working within the
  agreed spike budget, it is parked, not expanded. (See §2 review note: this is a
  product, not a phase.)
```

---

### Phase 8 — Backend interface hardening

Goal:

```text
Prepare the architecture for CUDA/ROCm without implementing them yet.
```

Tasks:

```text
- Document Backend ABI
- Separate primitive ops from kernel ops
- Define runtime launch abstraction
- Define buffer abstraction cleanly
- Define compilation target abstraction
- Add docs/BACKENDS.md
```

Acceptance criteria:

```text
A future CUDA backend can be designed without rewriting the core Tensor/Tensor metadata classes.
```

Definition of Done:

```text
- docs/BACKENDS.md specifies the Backend ABI: lifecycle, OpDesc contract, buffer
  ownership, launch abstraction, and compilation-target abstraction.
- A "null/stub backend" compiles against the interface alone (no Metal), proving
  the core has no Metal dependency.
- Primitive ops and kernel ops are separated in the interface.
- Adding a backend requires zero edits to cpp/cortex/core/ (invariant 4) —
  demonstrated by the stub backend living entirely under backends/.
```

---

### Phase 9 — CUDA backend prototype

Goal:

```text
Add the second accelerator backend.
```

This should only start after the Metal backend and core abstractions are stable.

Implemented CUDA path (2026-09-28):

```text
Cortex Runtime C++ core
  ↓
CUDA Runtime API (device 0, synchronous execution)
  ↓
Build-time nvcc-compiled static CUDA kernels
```

Initial CUDA operations:

```text
cuda registration through the existing backend registry/routing layer
device discovery
buffer allocation
copy
add_f32
mul_f32
fill_f32
```

Acceptance criteria:

```text
The same Python code works on device="metal" and device="cuda" for simple elementwise operations.
```

Definition of Done:

```text
- CUDA is added through registry/string-keyed dispatch, not as another ad hoc
  CPU/Metal branch in Python or bindings.
- device discovery, buffer copy, add/mul/fill _f32 work on device="cuda" and
  match the CPU reference.
- The SAME pytest test body runs against both metal and cuda (parametrized),
  proving API parity.
- Requires CUDA hardware/cloud access — that environment is selected and
  validated using `docs/CUDA_PHASE9_ENVIRONMENT.md` before implementation starts.
```

Status: **done** (2026-09-28). Nightblade passed the environment gate before
implementation. CUDA uses the existing native/Python backend registries and
`CudaBackend::execute`, preserving the core ABI. CPU↔CUDA float32/int32 copies
and float32 fill/add/multiply are validated against CPU through the same
backend-parametric pytest bodies used for Metal. CUDA kernels are opt-in at
build time; unsupported operations and int32 arithmetic are explicit errors.
See [validation evidence](docs/CUDA_PHASE9_VALIDATION.md). Metal cases skip on
Nightblade; this completion does not claim a new Metal hardware run.

---

### Phase 10 — MLIR integration exploration

Goal:

```text
Investigate whether Cortex Runtime IR should lower to MLIR for long-term compiler scalability.
```

This should not be part of the early MVP.

Possible future path:

```text
Cortex Runtime Kernel DSL
  ↓
Cortex Runtime IR
  ↓
MLIR dialects
  ↓
backend-specific lowering
```

Acceptance criteria:

```text
A research prototype demonstrates one simple Cortex Runtime operation lowered through MLIR.
```

Definition of Done:

```text
- A written decision record (docs/) answers: does Cortex Runtime IR lower to MLIR, or not,
  and why. A documented "no" is a valid, successful outcome of this phase.
- If yes: one op (e.g. elementwise add) lowers Cortex Runtime IR -> MLIR -> backend and
  matches CPU.
- No core or backend code is committed that assumes MLIR before this decision is
  recorded.
```

Status: **done** (2026-07-05). The decision record is
[`docs/MLIR_DECISION.md`](docs/MLIR_DECISION.md): the answer is yes — the
Phase 7 elementwise add lowers Cortex IR → MLIR (func/scf/arith/memref) →
LLVM → native code and matches the CPU reference, with guard and zero-thread
semantics preserved (`experiments/mlir/`, `tests/python/test_mlir_lowering.py`).
Verification venues: the emitter tests run in CPU CI on every push; the
end-to-end lowering was validated originally on the local Mac and again on
Nightblade in the follow-up decision. It skips when the MLIR toolchain is absent,
unless required-toolchain mode is enabled. The optional CPU runtime slice is
implemented under the [decision](docs/MLIR_RUNTIME_INTEGRATION_DECISION.md);
the C++ core and backends have no LLVM/MLIR build dependency.

---

## 15. Long-Term Vision

Long-term, Cortex Runtime could become:

```text
- a portable accelerator runtime
- a compact tensor runtime for experiments
- a Triton-inspired kernel compiler
- a backend playground for Metal, CUDA, ROCm, and MLIR
- a foundation for future inference experiments
```

Possible future capabilities:

```text
- custom kernel DSL
- autotuning
- kernel cache
- graph execution
- operator fusion
- PyTorch custom op bridge
- ONNX import
- LLM inference kernels
- quantized matmul
- CUDA backend
- ROCm backend
- Vulkan/SPIR-V backend
- MLIR backend
```

---

## 16. Explicit Non-Goals

Cortex Runtime should not try to become all of these at once:

```text
- PyTorch replacement
- JAX replacement
- TensorFlow replacement
- MLX replacement
- production-grade training framework
- distributed training framework
- universal TPU compiler
- full CUDA alternative
- full Triton replacement
```

The correct early identity is:

> A small, serious, Python-first accelerator runtime and kernel compiler project, starting with Apple Metal.

---

## 17. First 10 Implementation Tasks

Use these as the initial development sequence.

### Task 1 — Create repository skeleton

Create:

```text
pyproject.toml
CMakeLists.txt
python/cortex_runtime/__init__.py
cpp/cortex/core/
bindings/python_module.cpp
tests/python/
docs/ARCHITECTURE.md
```

Make `import cortex_runtime` work.

---

### Task 2 — Add C++ extension binding

Use nanobind to expose:

```python
cortex_runtime._core.version()
```

Expected:

```python
import cortex_runtime as cx
print(cx.__version__)
```

---

### Task 3 — Implement dtype and shape

Add C++ types:

```text
DType
Shape
```

Expose minimal Python representation.

---

### Task 4 — Implement CPU tensor

Add:

```python
cx.tensor([...], device="cpu", dtype=cx.float32)
cx.zeros(...)
cx.ones(...)
Tensor.numpy()
```

---

### Task 5 — Implement CPU add and multiply

Add:

```python
z = x + y
z = x * y
```

Only support contiguous 1D float32 initially.

---

### Task 6 — Add Metal backend discovery

Add:

```python
cx.devices()
cx.is_available("metal")
cx.device("metal")
```

Metal should appear on supported MacBook Pro hardware.

---

### Task 7 — Add Metal buffer copy

Add:

```python
x_gpu = x_cpu.to("metal")
x_back = x_gpu.cpu()
```

No GPU computation yet.

---

### Task 8 — Add first MSL kernel

Add `elementwise_add_f32`.

Run:

```python
x = cx.tensor([1, 2, 3], device="metal")
y = cx.tensor([4, 5, 6], device="metal")
z = x + y
```

Validate result.

---

### Task 9 — Add benchmarks

Create:

```text
benchmarks/bench_copy.py
benchmarks/bench_elementwise.py
```

Benchmark CPU vs Metal for tensor sizes.

---

### Task 10 — Document current architecture

Update:

```text
docs/ARCHITECTURE.md
docs/METAL_BACKEND.md
docs/ROADMAP.md
```

Document what works, what does not work, and what is intentionally out of scope.

---

## 18. Success Criteria for the First Public Version

The first public version should be considered successful if it can do this:

```python
import cortex_runtime as cx

device = cx.best_device()

x = cx.ones((1_000_000,), dtype=cx.float32, device=device)
y = cx.ones((1_000_000,), dtype=cx.float32, device=device)

z = x + y

print(z.cpu().numpy()[:5])
```

Expected:

```text
[2. 2. 2. 2. 2.]
```

And the repository should provide:

```text
- clear README
- local MacBook Pro setup instructions
- CPU backend
- Metal backend
- basic tests
- basic benchmarks
- documented roadmap
```

---

## 19. Main Risk

The main risk is scope explosion.

The project must not begin with:

```text
training
autograd
distributed systems
CUDA
ROCm
TPU
full compiler
matmul optimization
LLM inference
```

The correct first milestone is much smaller:

```text
Python → C++20 → Metal → custom MSL add kernel → correct result
```

Once that works, Cortex Runtime becomes a real project instead of only an idea.

---

## 20. Final Direction

Start with:

```text
Cortex Runtime v0.1
Target: Apple Silicon MacBook Pro
Backend: Metal
Apple bridge: Metal-cpp/C++ preferred
Primitive path: MPSGraph later
Custom kernel path: MSL
Core: C++20
Python binding: nanobind
Build: CMake + scikit-build-core
```

Then evolve toward:

```text
v0.2: more Metal ops + reductions
v0.3: matmul — naive custom MSL first, then MPSGraph
v0.4: reductions and softmax / NN primitives
v0.5: experimental Python kernel DSL
v0.6: backend interface hardening
v0.7: CUDA prototype
v0.8: MLIR exploration
```

The project is feasible if it starts as a narrow Metal runtime and grows carefully. It becomes unrealistic only if it tries to become a full PyTorch/Triton/XLA replacement from day one.
