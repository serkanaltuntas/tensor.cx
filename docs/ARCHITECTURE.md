# Architecture

Cortex Runtime is organized around a Python API, a backend-neutral C++20 core,
and backend-specific implementations.

## Layers

```text
python/cortex_runtime/        Python API and user ergonomics
bindings/                     nanobind extension and Python exception boundary
cpp/cortex/core/              backend-neutral runtime types and dispatch
cpp/cortex/backends/cpu/      mandatory CPU reference backend
cpp/cortex/backends/metal/    Apple Metal backend
```

The C++ core must not expose platform-specific handles. Metal, MPSGraph, and
other Apple API types stay inside `cpp/cortex/backends/metal/`.

## Current Phase

Phase 2 provides CPU tensors, dtype and shape metadata, CPU buffer ownership,
NumPy conversion, CPU add/multiply, Metal device discovery, and exact CPU/Metal
tensor copy round-trips. Phase 3 starts the first static Metal kernels.

The public Python `Tensor` wraps backend-specific native tensor objects. CPU
operations still execute only on CPU tensors; Metal tensors can currently be
created, inspected, copied back to CPU, and used as the transfer target for
future kernels.

Phase 3 should introduce backend-neutral operation dispatch for Metal kernels
instead of adding one Python branch per operation. The Phase 2 copy helpers are
intentionally narrow bridge functions.

## Core Principles

- CPU reference behavior is mandatory for every future GPU operation.
- Operation dispatch is data-driven through `OpDesc`, not one virtual method per
  operation.
- Runtime errors should converge on `Status` / `expected<T, Status>` and be
  translated to Python exceptions at the nanobind layer. Early backend code may
  still throw direct C++ exceptions until Phase 4 error polish.
- Python and NumPy types stay outside `cpp/cortex/core/` and all backends.
- Metal-cpp handles must be RAII-wrapped and isolated inside the Metal backend.
