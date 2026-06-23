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

Phase 5 completed the first matmul paths: CPU reference matmul, a
correctness-first custom Metal matmul kernel, and an MPSGraph-backed Metal
matmul path. Phase 6 is complete, with `sum`, `max`, `mean`, `exp`, `gelu`,
`silu`, `softmax`, `rmsnorm`, and `layernorm` available on CPU and Metal.
Phase 7 is ready to start and should keep the experimental kernel DSL behind
the existing backend-neutral runtime boundaries.

The public Python `Tensor` wraps backend-specific native tensor objects. CPU and
Metal tensors both support add and multiply; Metal tensors also support direct
fill through `zeros` and `ones`, rank-2 float32 matmul, axis-based
`sum`/`max`/`mean` reductions, `softmax`, `rmsnorm`, `layernorm`, and
`exp`/`gelu`/`silu`.

Python binary operations dispatch through shared native `_core.add` and
`_core.multiply` entrypoints with CPU and Metal overloads. The early fill path is
still explicit because constructors must choose a backend-specific native tensor
type.

### Dispatch today vs. the Phase 8 target

`§5.6` settles a data-driven dispatch design: a single
`Backend::execute(OpDesc, inputs, outputs)` per backend, switching on the op
enum, rather than one method per operation. That interface exists in
`cpp/cortex/core/backend.h`, but through Phase 5 it is a **design placeholder —
no backend implements it and nothing calls it.** The dispatch that actually runs
is a set of per-op typed entry points in each backend (`cpu::execute_binary`,
`cpu::execute_unary`, `cpu::reduce`, `cpu::matmul`, `cpu::fill`,
`metal::execute_binary`, `metal::execute_unary`, `metal::reduce`,
`metal::matmul_custom`, `metal::matmul_mpsgraph`, `metal::fill`), selected by
the nanobind layer from the operand tensor type and device. `OpDesc` is passed
to the unary, binary, and reduction entry points (and Metal's `fill`) and tags
the op `kind`. Phase 6 adds a minimal `axis` attribute to `OpDesc` for
reduction entry points and axis-aware transforms such as softmax, rmsnorm, and
layernorm, plus an `epsilon` attribute for normalization ops. Other op
parameters, such as fill's value and matmul's backend choice, still travel as
ordinary arguments.

This is a deliberate, documented deviation kept small per the "avoid unrelated
refactors" rule: unifying the backends onto `Backend::execute` is the work of
**Phase 8 — Backend interface hardening**, whose Definition of Done already
requires a stub backend that compiles against this interface alone. Until then,
do not read `backend.h` as the live dispatch path; read it as the contract Phase
8 implements.

## Core Principles

- CPU reference behavior is mandatory for every future GPU operation.
- Operation dispatch is moving toward the data-driven `OpDesc` model (single
  `Backend::execute`, no per-op virtual method). The unified interface is
  defined but not yet wired; see "Dispatch today vs. the Phase 8 target" above.
- Metal backend errors return `Status` / `Expected<T>` and are translated to
  Python exceptions at the nanobind layer.
- Python and NumPy types stay outside `cpp/cortex/core/` and all backends.
- Metal-cpp handles must be RAII-wrapped and isolated inside the Metal backend.

## Error Taxonomy

Runtime errors must carry stable, specific messages. Python is the user-facing
boundary, so C++ errors are translated once through the nanobind extension.

```text
Condition                         Python exception  Required message fragment
Device unavailable                ValueError        device is not available
Requested Metal unavailable       ValueError        Metal is not available
Unsupported device transfer       ValueError        unsupported device transfer
Binary device mismatch            ValueError        device mismatch
Binary shape mismatch             ValueError        shape mismatch
Binary dtype mismatch             ValueError        dtype mismatch
Matmul rank mismatch              ValueError        matmul requires rank-2
Matmul shape mismatch             ValueError        matmul shape mismatch
Matmul dtype mismatch             ValueError        matmul only supports float32
Reduction axis out of range       ValueError        reduction axis is out of range
Max over empty axis               ValueError        max reduction requires a non-empty axis
Float32-only op dtype mismatch    ValueError        only supports float32
Negative shape dimension          ValueError        shape dimensions must be non-negative
Shape element-count overflow      ValueError        shape size overflow
Shape stride overflow             ValueError        shape stride overflow
Int value out of int32 range      ValueError        out of range for int32
Fill value out of int32 range     ValueError        fill value is out of range for int32
Nested data to flat factory       ValueError        flat numeric sequence
Unsupported dtype                 ValueError        unsupported dtype
Empty device type                 ValueError        device type must be a non-empty string
Invalid device index (parse)      ValueError        invalid device index
Unsupported device index          ValueError        only device index 0 is supported
Metal runtime unavailable         RuntimeError      Metal is not available
Metal internal failure            RuntimeError      failed to
```

`int32` overflow in elementwise `add`/`multiply` is **not** an error: it is
defined two's-complement wraparound, identical on the CPU and Metal paths (both
compute through `uint32`), matching NumPy. Float inputs are narrowed to
`float32`, so values may lose precision or overflow to `inf` — this is a silent,
intentional consequence of the float32-only v0.1 scope, not an error.

Guidelines:

- Invalid user input should become `ValueError`.
- User-requested unavailable devices should become `ValueError`. Backend
  unavailability discovered after native dispatch should become `RuntimeError`.
- CPU behavior is the correctness reference for Metal. A Metal operation that
  rejects an input should reject it before allocating large buffers or launching
  kernels.
- Backend-specific C++ code should prefer `Status` / `Expected<T>`. Existing
  CPU/core paths may still throw standard C++ exceptions, but those exceptions
  must cross into Python only at the binding boundary.
