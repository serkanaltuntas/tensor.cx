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
Phase 7 is complete for the first generated-kernel slice: a
`cx.experimental.kernel` metadata wrapper, restricted Python AST ->
backend-neutral IR parser, text MSL emitter, in-memory metallib compile
artifact, native Metal function validation, and a narrow synchronous Metal
launch path for float32 elementwise kernels. Keep the experimental kernel DSL
behind the existing backend-neutral runtime boundaries. See
[`KERNEL_DSL.md`](KERNEL_DSL.md). Phase 8 is complete: the backend execution ABI,
shared primitive/kernel contract validators, and null backend scaffold exist.
Phase 9 has not started; CUDA implementation is blocked until a CUDA development
environment is selected and validated.

The public Python `Tensor` wraps backend-specific native tensor objects. CPU and
Metal tensors both support add and multiply; Metal tensors also support direct
fill through `zeros` and `ones`, rank-2 float32 matmul, axis-based
`sum`/`max`/`mean` reductions, `softmax`, `rmsnorm`, `layernorm`, and
`exp`/`gelu`/`silu`.

Python binary operations dispatch through shared native `_core.add` and
`_core.multiply` entrypoints with CPU and Metal overloads. Both overloads now
route add and multiply through their backend `execute` implementations. CPU
unary transforms `exp`/`gelu`/`silu`/`softmax`/`rmsnorm`/`layernorm`, CPU
reductions `sum`/`max`/`mean`, and CPU matmul also route through that execution
contract. Metal `exp`/`gelu`/`silu` also route through `MetalBackend::execute`.
Fill dispatch routes through `Backend::execute` for both CPU and Metal, using
an output allocation descriptor. Constructor-style `empty` allocation still
uses typed paths because it must choose backend-specific native tensor types.

### Dispatch after Phase 8

`§5.6` settles a data-driven dispatch design: a single
`Backend::execute(BackendExecution)` per backend, switching on `OpDesc`, rather
than one virtual method per operation. Phase 8 completed the backend ABI
hardening work:
`cpp/cortex/core/backend.h` now defines the execution contract, separates
primitive operations from kernel launches, and carries optional launch and
compilation-target metadata. `cpp/cortex/backends/null/` compiles against that
interface alone and exists to prove the contract has no Metal dependency. The
core also exposes the primitive op input/output schema plus shared primitive
and kernel contract validators. CPU and Metal currently use the primitive
validator for migrated execution paths; the null backend uses both validators to
reject malformed primitive and kernel scaffold requests.

For primitive execution, `BackendExecution.outputs` are result slots that the
backend fills with produced tensor metadata. Allocation-style primitives such as
`fill` use an output allocation descriptor and `OpDesc.scalar_value`. The
`BackendOpClass::kKernel` form is only a contract scaffold today: it is not yet
the live ABI for the experimental generated-kernel launcher because that path
still needs ordered tensor/scalar arguments and caller-owned output tensors.

The dispatch that actually runs is still mixed after Phase 8. CPU
add/multiply, unary transforms, reductions, and matmul are routed through
`CpuBackend::execute(BackendExecution)`, which adapts existing `CpuTensor`
values to backend-neutral `Tensor` metadata and then reuses the existing CPU
operation implementations. CPU fill also routes through `CpuBackend::execute`
using an output allocation descriptor. Metal add/multiply, `exp`/`gelu`/`silu`,
and fill route through `MetalBackend::execute(BackendExecution)`, using the same
primitive validator and adapting between `MetalTensor` and core tensor metadata
at the backend boundary. Remaining Metal operations still use typed entry
points (`metal::execute_unary` for softmax/rmsnorm/layernorm, `metal::reduce`,
`metal::matmul_custom`, `metal::matmul_mpsgraph`). `OpDesc` is still passed to
those typed entry points and tags the op `kind`. Phase 6 adds a minimal `axis`
attribute to `OpDesc` for reduction entry points and axis-aware transforms such
as softmax, rmsnorm, and layernorm, plus an `epsilon` attribute for normalization
ops. Phase 8 adds `scalar_value` for fill. Constructor-style `empty` allocation
remains outside `Backend::execute` for now.

This is a deliberate, documented transition kept small per the "avoid unrelated
refactors" rule: Phase 8 hardened the ABI first; public device routing now uses
a string-keyed backend registry while preserving the existing tensor creation,
copy, and operation semantics. Device capability queries now consistently
require the target backend to be available.

### Backend Selection Status

The backend-neutral core does not name concrete backend APIs or expose platform
handles. Public Python device routing goes through `python/cortex_runtime/backend.py`,
which registers backends by string key and owns availability checks, device
names, tensor copies, fill creation, and matmul backend options. The nanobind
module also uses a small route table for native device capability helpers.

CUDA must be added by registering a new backend route, not by adding a third
ad hoc CPU/Metal branch to public Python dispatch. Backend-specific native
operations may still expose typed implementation functions while the core ABI
continues to harden around `Backend::execute` and `OpDesc`.

## Core Principles

- CPU reference behavior is mandatory for every future GPU operation.
- Operation dispatch is moving toward the data-driven `OpDesc` model (single
  `Backend::execute`, no per-op virtual method). The ABI is defined and backed
  by a null backend scaffold; CPU add/multiply, unary transforms, reductions,
  matmul, and fill have been migrated, and Metal add/multiply,
  `exp`/`gelu`/`silu`, plus fill now use the same backend execution path.
  Remaining Metal axis/norm, reduction, matmul, and generated-kernel dispatch
  still use typed entry points.
- Public backend selection uses registry/string-keyed routing; CUDA must plug
  into that route instead of adding ad hoc public dispatch branches.
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
Non-integer axis                  ValueError        axis must be an integer
Axis parse overflow               ValueError        axis is out of range
Reduction axis out of range       ValueError        reduction axis is out of range
Max over empty axis               ValueError        max reduction requires a non-empty axis
Float32-only op dtype mismatch    ValueError        only supports float32
Negative shape dimension          ValueError        shape dimensions must be non-negative
Non-integer shape dimension       ValueError        shape dimensions must be integers
Shape dimension parse overflow    ValueError        shape dimension is out of range
Shape element-count overflow      ValueError        shape size overflow
Shape stride overflow             ValueError        shape stride overflow
Int value out of int32 range      ValueError        out of range for int32
Fill value out of int32 range     ValueError        fill value is out of range for int32
Invalid float32 data value        ValueError        not convertible to float32
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
