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
outputs             mutable output tensor metadata/result-slot span
launch              optional LaunchConfig, required only for kernel launches
compilation_target  optional CompilationTarget, required only for kernel launches
```

Primitive operations are library/runtime operations such as matmul or reductions
that a backend may route through platform primitives. They must not carry kernel
launch metadata. For ordinary primitive execution, `outputs` are result slots:
callers may pass default-constructed `Tensor` metadata, and the backend writes
the output metadata after allocating or producing the result buffer.
Allocation-style primitives such as `fill` use `outputs[0]` as an allocation
descriptor: dtype, shape, device, contiguous strides, no buffer, and offset 0.

`BackendOpClass::kKernel` is the Phase 8 scaffold for future project-owned
static or generated kernel execution. It validates launch and compilation-target
metadata, but it is not yet the live ABI for `cx.experimental` generated
kernels. The generated DSL still uses the typed Metal launch path with ordered
`KernelArgument` tensor/scalar values and caller-owned output tensors. Before
generated kernels migrate to `BackendExecution`, the ABI must gain an ordered
kernel argument channel and explicit output ownership semantics for
caller-provided outputs.

Primitive requests must match the core `primitive_op_schema`: the op kind
defines the expected input and output tensor metadata counts. The shared core
primitive contract validator rejects malformed primitive requests before a
backend executes them; CPU and null backend both use that validator. CPU `fill`
now routes through `BackendExecution` using `OpDesc.scalar_value` and an output
allocation descriptor. Constructor-style `empty` allocation and Metal fill
remain on typed paths until their ownership semantics are migrated.

Kernel requests must pass the shared core kernel contract validator before a
backend executes them: they require output metadata, launch metadata, a
launch whose grid and thread-group dimensions are all non-zero, a compilation
target, a non-empty entry point, a concrete artifact kind, and a non-empty
artifact identifier. This is still a contract scaffold, not the live
generated-kernel ABI.

### Launch Abstraction

`LaunchConfig` describes grid dimensions and threads-per-group dimensions. The
fields are intentionally backend-neutral so Metal, CUDA, ROCm, Vulkan/SPIR-V,
and MLIR-generated paths can map them to their native launch concepts later.
All six launch dimensions must be non-zero. Synchronous execution remains the
project default.

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
contract violations such as primitive input/output count mismatches, invalid
fill allocation descriptors, or kernel execution without launch metadata or a
compilation target. Primitive and kernel validation use the same core helpers
that future real backends should use. The Python test hook
`_backend_contract_smoke_test` proves the scaffold builds and links with the
extension.

## Current Direction

Phase 8 is in progress. The backend ABI and null backend scaffold exist, and
CPU fill, add/multiply, unary transforms, reductions, and matmul now route
through `CpuBackend::execute` without changing public Python behavior. Metal
dispatch still uses the existing typed entry points while the interface is
hardened.
