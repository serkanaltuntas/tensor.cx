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
[x] Phase 9   CUDA prototype             <- Nightblade, 2026-09-28
[x] Phase 10  MLIR exploration           <- done, out of order
```

Phase 10 ran ahead of Phase 9 under a documented sequencing exception:
Phase 9 was blocked on CUDA hardware/cloud access at that time, and Phase 10's Definition of Done (a decision record, optionally
one CPU/Metal-validated op) doesn't need CUDA. See
[`PHASE_SEQUENCING_DECISION.md`](PHASE_SEQUENCING_DECISION.md). Phase 9's
acceptance criteria are unaffected.

Phase 10 is complete: [`MLIR_DECISION.md`](MLIR_DECISION.md) records a "yes" —
the Phase 7 elementwise add lowers tensor.cx IR → MLIR → native code and matches
the CPU reference (`experiments/mlir/`,
`tests/python/test_mlir_lowering.py`). The optional CPU runtime slice is implemented under the
[follow-up decision](MLIR_RUNTIME_INTEGRATION_DECISION.md), with Linux x86_64
compile/launch validation. Core and backends have no LLVM/MLIR build dependency.

Phase 8 is complete: the backend execution ABI now
separates primitive operations from kernel launches, carries explicit launch and
compilation-target metadata, and has a null backend scaffold that compiles
without Metal. CPU and null backend now use the shared primitive contract
validator for input/output schema and fill allocation descriptors; null backend
also uses the shared kernel contract validator. CPU fill, add/multiply, unary
transforms, reductions, and matmul now route through `CpuBackend::execute`;
Metal add/multiply, unary transforms, axis/norm transforms, reductions, matmul,
fill, and non-empty narrow experimental generated-kernel launches now route
through `MetalBackend::execute`. Phase 9 is now complete on Nightblade, using
`CudaBackend::execute` for float32 fill/add/multiply and the existing registry
for discovery and copies. See [validation evidence](CUDA_PHASE9_VALIDATION.md).

## What works today (runtime through Phase 9, plus the Phase 10 MLIR prototype)

- Optional CUDA backend: device 0 discovery, float32/int32 CPU↔CUDA copies,
  float32 fill/add/multiply, matmul, sum/max/mean, exp/GELU/SiLU, softmax/RMSNorm/LayerNorm;
  synchronous execution through the shared ABI. [Extended primitive validation](CUDA_PRIMITIVES_VALIDATION.md).
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
  MSL -> in-memory metallib -> synchronous Metal launch for float32
  elementwise and rowwise-reduction (bounded for/accumulator) kernels, tested
  against the CPU reference (`Kernel.reference(...)` interprets the same IR
  on CPU tensors).
- Backend ABI: `Backend::execute`, primitive/kernel execution classes,
  backend-neutral launch and compilation target metadata, ordered kernel
  arguments, shared primitive and kernel contract validators, and a null backend
  scaffold that builds without Metal.
- Error taxonomy, CPU CI, benchmarks for copy / elementwise / matmul.

The user-requested [tensor API extension](TENSOR_API.md) adds ordinary
subtraction/division/negation, real scalars, shared-storage contiguous reshape,
reduction `keepdims`, explicit float32/int32 casts, and binary broadcasting.
These additions preserve the dtype/backend boundaries
and do not complete the broader CUDA product goal.

## Intentionally not implemented yet

Autograd, training, streams/async, non-contiguous execution, wide
dtypes, broad generated-kernel semantics, top-level kernel APIs, broader CUDA operations, ROCm
backends, and broader MLIR/backend integration are out of scope until their
phases. The optional CPU runtime slice follows the completed Phase 10 research;
it does not add LLVM/MLIR to the C++ core or backend build. See PROJECT.md §16 and AGENTS.md "Out Of Scope For Early Work".

## Known deferred design work

- **MLIR CUDA integration.** The [CPU-first runtime slice](MLIR_RUNTIME_INTEGRATION_DECISION.md)
  is implemented. The [CUDA ABI/toolchain decision](MLIR_CUDA_INTEGRATION_DECISION.md)
  and guarded add/subtract/multiply runtime are implemented on sm_52 with primary-context and
  native module ownership through tensor.cx buffers. The [guarded local-expression
  scope](MLIR_CUDA_EXPRESSIONS_DECISION.md) is implemented with bounded local/nested
  arithmetic. The [CUDA product completion goal](CUDA_PRODUCT_COMPLETION.md)
  tracks operator coverage, compatibility, measurements, GPU CI and distribution.

- **PyTorch portability strategy.** The current roadmap keeps tensor.cx
  independent from PyTorch, but a future integration can be staged through a
  custom-op bridge before considering a full PyTorch / ATen backend. See
  [`PYTORCH_PORTABILITY_ROADMAP.md`](PYTORCH_PORTABILITY_ROADMAP.md).
- **CUDA scope after Phase 9.** The prototype is complete on the selected
  [Nightblade environment](CUDA_PHASE9_ENVIRONMENT.md). Broader CUDA operations
  and generated kernels require separate implementation and CPU parity tests.
- **Remaining allocation routing.** Constructor-style `empty` allocation still
  uses typed backend paths because it must choose backend-specific native tensor
  objects. See [`ARCHITECTURE.md`](ARCHITECTURE.md) → "Dispatch after Phase 8".
- **Async submission and streams.** GIL release and native-layer thread-safety
  are done: the binding releases the GIL around every backend call and the
  pipeline caches are mutex-guarded, so multi-threaded use is supported. What
  remains deferred is asynchronous submission (streams, command-buffer
  batching); execution is still synchronous per op. See
  [`METAL_BACKEND.md`](METAL_BACKEND.md) → "Threading and the GIL".
