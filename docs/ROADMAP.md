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
[ ] Phase 6   Reductions & NN primitives  <- in progress
[ ] Phase 7   Experimental kernel DSL
[ ] Phase 8   Backend interface hardening
[ ] Phase 9   CUDA prototype
[ ] Phase 10  MLIR exploration
```

## What works today (through Phase 6 in progress)

- CPU reference backend: `float32`/`int32`, contiguous 1-D/2-D, add / multiply /
  fill / zeros / ones / empty, exact NumPy round-trip.
- Metal backend: device discovery, buffer host↔device copy, static MSL kernels
  (`add_f32`/`i32`, `mul_f32`/`i32`, `fill_f32`/`i32`) loaded from a build-time
  embedded `.metallib`.
- Matmul: a naive custom MSL `matmul_f32` (correctness-first) **and** an MPSGraph
  optimized path. Removing the MPSGraph path leaves a working slow matmul — the
  project is not an MPSGraph wrapper (PROJECT.md §9.2).
- Reductions: `sum` and `max` on CPU and Metal for `float32` and `int32`, with
  explicit `axis`, negative-axis support, and CPU-vs-Metal tests on non-trivial
  axes.
- Error taxonomy, CPU CI, benchmarks for copy / elementwise / matmul.

## Intentionally not implemented yet

Autograd, training, streams/async, broadcasting, non-contiguous execution, wide
dtypes, a kernel DSL, and the CUDA/ROCm/MLIR backends are out of scope until
their phases. See PROJECT.md §16 and AGENTS.md "Out Of Scope For Early Work".

## Known deferred design work

- **Unified `Backend::execute` dispatch (Phase 8).** The data-driven dispatch
  interface (`cpp/cortex/core/backend.h`) is defined but not yet wired; the live
  path is per-op typed entry points routed by the binding. See
  [`ARCHITECTURE.md`](ARCHITECTURE.md) → "Dispatch today vs. the Phase 8 target".
- **GIL release + Metal thread-safety.** v0.1 is synchronous and holds the GIL
  across blocking Metal submissions, so multi-threaded use is serialized and the
  pipeline cache is intentionally unlocked. Releasing the GIL (and locking the
  cache) is concurrency work beyond the synchronous v0.1 scope. See
  [`METAL_BACKEND.md`](METAL_BACKEND.md) → "Threading and the GIL".
