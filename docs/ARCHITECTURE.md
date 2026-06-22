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

Phase 3 provides CPU tensors, dtype and shape metadata, CPU buffer ownership,
NumPy conversion, CPU add/multiply, Metal device discovery, exact CPU/Metal
tensor copy round-trips, and the first static Metal elementwise kernels.

The public Python `Tensor` wraps backend-specific native tensor objects. CPU and
Metal tensors both support add and multiply; Metal tensors also support direct
fill through `zeros` and `ones`.

Python binary operations dispatch through shared native `_core.add` and
`_core.multiply` entrypoints with CPU and Metal overloads. The early fill path is
still explicit because constructors must choose a backend-specific native tensor
type.

## Core Principles

- CPU reference behavior is mandatory for every future GPU operation.
- Operation dispatch is data-driven through `OpDesc`, not one virtual method per
  operation.
- Metal backend errors return `Status` / `Expected<T>` and are translated to
  Python exceptions at the nanobind layer. Phase 4 should extend that polish
  across the older CPU/core paths.
- Python and NumPy types stay outside `cpp/cortex/core/` and all backends.
- Metal-cpp handles must be RAII-wrapped and isolated inside the Metal backend.
