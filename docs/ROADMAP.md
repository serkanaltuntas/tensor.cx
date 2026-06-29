# Roadmap

This file is a navigation aid. The **single source of truth for the current
phase is the Project Status block in [`PROJECT.md`](../PROJECT.md)** — update that
block when a phase completes, not this file. The full per-phase rationale,
acceptance criteria, and Definitions of Done also live in `PROJECT.md` (§14).

## Phase status

```text
[x] Phase 0   Project bootstrap
[x] Phase 1   CPU backend
[x] Phase 2   Metal backend foundation
[x] Phase 3   First Metal kernels        <- v0.1 ships here
[x] Phase 4   Runtime polish
[x] Phase 5   Matmul (custom MSL + MPSGraph)
[x] Phase 6   Reductions & NN primitives
[x] Phase 7   Experimental kernel DSL
[x] Phase 8   Backend interface hardening
[ ] Phase 9   CUDA prototype
[ ] Phase 10  MLIR exploration
```

Phase 8 is complete: the backend execution ABI now
separates primitive operations from kernel launches, carries explicit launch and
compilation-target metadata, and has a null backend scaffold that compiles
without Metal. CPU and null backend now use the shared primitive contract
validator for input/output schema and fill allocation descriptors; null backend
also uses the shared kernel contract validator. CPU fill, add/multiply, unary
transforms, reductions, and matmul now route through `CpuBackend::execute`;
Metal add/multiply, unary transforms, axis/norm transforms, reductions, and fill
now route through `MetalBackend::execute`; remaining Metal matmul and
generated-kernel dispatch paths still use typed entry points. Phase 9 has not
started; it requires a CUDA hardware or cloud development environment decision first.

## What works today (through Phase 8)

- CPU reference backend: `float32`/`int32`, contiguous 1-D/2-D, add / multiply /
  fill / zeros / ones / empty, exact NumPy round-trip.
- Metal backend: device discovery, buffer host↔device copy, static MSL kernels
  (`add_f32`/`i32`, `mul_f32`/`i32`, `fill_f32`/`i32`, `exp_f32`, `gelu_f32`,
  `silu_f32`, `softmax_f32`, `rmsnorm_f32`, `layernorm_f32`) loaded from a
  build-time embedded `.metallib`.
- Matmul: a naive custom MSL `matmul_f32` (correctness-first) **and** an MPSGraph
  optimized path. Removing the MPSGraph path leaves a working slow matmul — the
  project is not an MPSGraph wrapper (PROJECT.md §9.2).
- Reductions: `sum` and `max` on CPU and Metal for `float32` and `int32`, plus
  `mean` for `float32`, with explicit `axis`, negative-axis support, and
  CPU-vs-Metal tests on non-trivial axes.
- Unary float32 activations: `exp`, `gelu` (tanh approximation), and `silu` on
  CPU and Metal.
- Stable axis-based `softmax` for `float32` on CPU and Metal.
- Axis-based `rmsnorm` for `float32` on CPU and Metal.
- Axis-based `layernorm` for `float32` on CPU and Metal.
- Experimental kernel DSL: restricted Python AST -> backend-neutral IR -> text
  MSL -> in-memory metallib -> synchronous Metal launch for the first float32
  elementwise add subset, tested against the CPU reference.
- Backend ABI: `Backend::execute`, primitive/kernel execution classes,
  backend-neutral launch and compilation target metadata, shared primitive and
  kernel contract validators, and a null backend scaffold that builds without
  Metal.
- Error taxonomy, CPU CI, benchmarks for copy / elementwise / matmul.

## Intentionally not implemented yet

Autograd, training, streams/async, broadcasting, non-contiguous execution, wide
dtypes, broad generated-kernel semantics, top-level kernel APIs, and the
CUDA/ROCm/MLIR backends are out of scope until their phases. See PROJECT.md §16
and AGENTS.md "Out Of Scope For Early Work".

## Known deferred design work

- **PyTorch portability strategy.** The current roadmap keeps Cortex Runtime
  independent from PyTorch, but a future integration can be staged through a
  custom-op bridge before considering a full PyTorch / ATen backend. See
  [`PYTORCH_PORTABILITY_ROADMAP.md`](PYTORCH_PORTABILITY_ROADMAP.md).
- **CUDA environment for Phase 9.** The next phase needs a CUDA-capable local or
  cloud development environment before implementation starts. The decision gate
  and validation checklist are documented in
  [`CUDA_PHASE9_ENVIRONMENT.md`](CUDA_PHASE9_ENVIRONMENT.md).
- **Remaining Metal dispatch migration.** The Phase 8 ABI is defined and
  exercised by CPU, the null backend, and Metal elementwise, axis/norm,
  reduction, and fill paths. Metal matmul and generated-kernel execution are
  still routed through per-op typed entry points. See
  [`ARCHITECTURE.md`](ARCHITECTURE.md) → "Dispatch after Phase 8".
- **GIL release + Metal thread-safety.** v0.1 is synchronous and holds the GIL
  across blocking Metal submissions, so multi-threaded use is serialized and the
  pipeline cache is intentionally unlocked. Releasing the GIL (and locking the
  cache) is concurrency work beyond the synchronous v0.1 scope. See
  [`METAL_BACKEND.md`](METAL_BACKEND.md) → "Threading and the GIL".
