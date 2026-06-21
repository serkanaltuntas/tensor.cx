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

Phase 0 provides the build skeleton, importable package, native extension, and
test harness. It does not implement tensor allocation, CPU operations, Metal
buffers, or kernels yet.

## Core Principles

- CPU reference behavior is mandatory for every future GPU operation.
- Operation dispatch is data-driven through `OpDesc`, not one virtual method per
  operation.
- Runtime errors flow through `Status` / `expected<T, Status>` and are translated
  to Python exceptions at the nanobind layer.
- Python and NumPy types stay outside `cpp/cortex/core/` and all backends.
- Metal-cpp handles must be RAII-wrapped and isolated inside the Metal backend.
