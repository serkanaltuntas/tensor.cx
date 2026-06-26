# Backend Notes

Cortex Runtime keeps the backend-neutral runtime in `cpp/cortex/core/` and
places backend-specific implementation under `cpp/cortex/backends/`.

## Current Backends

```text
cpu     Required reference backend.
metal   Apple Silicon backend for buffer ownership, copy round-trips, and first
        elementwise kernels.
null    Phase 8 contract scaffold. Compiles against the backend ABI without
        Metal and intentionally does not execute operations.
```

The CPU backend is the correctness reference for every operation. Metal behavior
must be compared against CPU behavior before it is treated as complete.

## Current Status

The Metal backend currently supports:

```text
- device discovery
- shared Metal buffer allocation
- CPU -> Metal copy
- Metal -> CPU copy
- Python Tensor.to("metal")
- Python Tensor.cpu()
- add and multiply kernels for float32 and int32 tensors
- fill kernels for zeros/ones on float32 and int32 tensors
- exp, gelu, silu, softmax, rmsnorm, and layernorm kernels for float32 tensors
- naive custom MSL matmul for float32 rank-2 tensors
- MPSGraph matmul for float32 rank-2 tensors
- reduction kernels for sum/max on float32 and int32 tensors
- reduction kernel for mean on float32 tensors
- experimental generated-kernel validation and synchronous launch for the first
  float32 elementwise subset
```

CPU remains the correctness reference for every Metal operation.

## Backend ABI

Phase 8 defines the backend ABI in `cpp/cortex/core/backend.h`. The ABI is the
contract future backends should compile against before they are wired into live
dispatch.

### Lifecycle

Each backend owns its device/context resources and exposes a stable `name()`.
Execution uses `Status` return values; backend implementations must not rely on
exceptions crossing backend boundaries. The Python binding remains the single
place where runtime status becomes Python exceptions.

### Execution Contract

`Backend::execute(const BackendExecution&)` is the single backend entry point.
`BackendExecution` carries:

```text
op_class            BackendOpClass::kPrimitive or BackendOpClass::kKernel
op                  OpDesc operation descriptor and attributes
inputs              input tensor metadata span
outputs             mutable output tensor metadata span
launch              optional LaunchConfig, required only for kernel launches
compilation_target  optional CompilationTarget, required only for kernel launches
```

Primitive operations are library/runtime operations such as matmul or reductions
that a backend may route through platform primitives. They must not carry kernel
launch metadata. Kernel operations are project-owned static or generated kernels;
they require an explicit launch configuration and compilation target.

Primitive requests must match the core `primitive_op_schema`: the op kind
defines the expected input and output tensor metadata counts. The null backend
uses this schema to reject malformed requests before returning `kUnavailable`.
`fill` and constructor-style allocation remain outside `BackendExecution` until
the ABI has explicit backend-neutral output allocation and scalar-value
attributes.

### Launch Abstraction

`LaunchConfig` describes grid dimensions and threads-per-group dimensions. The
fields are intentionally backend-neutral so Metal, CUDA, ROCm, Vulkan/SPIR-V,
and MLIR-generated paths can map them to their native launch concepts later.
Synchronous execution remains the project default.

### Compilation Target

`CompilationTarget` identifies a backend-local artifact and entry point for
kernel execution. `KernelArtifactKind` stays backend-neutral: static library,
source text, binary module, or intermediate representation. The core does not
name Metal, MSL, PTX, SPIR-V, MLIR, or other concrete formats here; each backend
maps the neutral artifact class and opaque artifact string to its own compiled
library, generated source cache, binary blob, or IR module. Primitive operations
should leave `compilation_target` unset.

### Buffer Ownership

The core owns backend-neutral tensor and buffer abstractions. Backend-specific
buffer handles stay inside their backend directories. The core must not expose
Metal, MPSGraph, CUDA, ROCm, Vulkan, or platform API handles.

### Null Backend Scaffold

`cpp/cortex/backends/null/` validates the ABI shape without executing work. It
returns `kUnavailable` for valid execution requests and `kInvalidArgument` for
contract violations such as primitive input/output count mismatches or kernel
execution without launch metadata or a compilation target. The Python test hook
`_backend_contract_smoke_test` proves the scaffold builds and links with the
extension.

## Current Direction

Phase 8 is in progress. The backend ABI and null backend scaffold exist, and
CPU add/multiply plus unary transforms now route through `CpuBackend::execute`
without changing public Python behavior. CPU reductions, matmul, fill, and
Metal dispatch still use the existing typed entry points while the interface is
hardened.
