# Backend Notes

tensor.cx keeps the backend-neutral runtime in `cpp/tensorcx/core/` and
places backend-specific implementation under `cpp/tensorcx/backends/`.

## Current Backends

```text
cpu     Required reference backend.
metal   Apple Silicon backend for buffer ownership, copy round-trips, and first
        elementwise kernels.
cuda    Optional Phase 9 prototype: device 0, copies, float32 fill/add/multiply.
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
- experimental generated-kernel validation and synchronous launch for float32
  elementwise and rowwise-reduction kernels
```

CPU remains the correctness reference for every Metal operation.

## Backend ABI

Phase 8 defines the backend ABI in `cpp/tensorcx/core/backend.h`. The ABI is the
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
kernel_arguments    ordered tensor/uint32 argument span for kernel launches
```

Primitive operations are library/runtime operations such as matmul or reductions
that a backend may route through platform primitives. They must not carry kernel
launch metadata. For ordinary primitive execution, `outputs` are result slots:
callers may pass default-constructed `Tensor` metadata, and the backend writes
the output metadata after allocating or producing the result buffer.
Allocation-style primitives such as `fill` use `outputs[0]` as an allocation
descriptor: dtype, shape, device, contiguous strides, no buffer, and offset 0.

`BackendOpClass::kKernel` is the backend execution form for project-owned
static or generated kernel execution. Kernel requests use caller-owned concrete
output tensor metadata and an ordered `kernel_arguments` span for tensor buffers
and uint32 scalar values. The current live use is intentionally narrow:
`cx.experimental` generated kernels route non-empty launches through this ABI
for synchronous Metal-only, one-output float32 execution (elementwise and
rowwise-reduction kernels).
Zero-thread Python launches validate the Metal function and return as no-ops
before building `BackendExecution`, because core launch dimensions are non-zero
by contract.

Primitive requests must match the core `primitive_op_schema`: the op kind
defines the expected input and output tensor metadata counts. The shared core
primitive contract validator rejects malformed primitive requests before a
backend executes them; CPU, Metal, and null backends all use that validator for
the migrated execution paths. CPU and Metal `fill` route through
`BackendExecution` using `OpDesc.scalar_value` and an output allocation
descriptor. Constructor-style `empty` allocation remains on typed paths until
its ownership semantics are migrated. Matmul algorithm selection uses the
backend-neutral `OpDesc.matmul_preference` field so public auto/custom/optimized
selection can route through `BackendExecution` without naming Metal primitives
in the core.

Arithmetic extensions add subtract/divide/negate and one-input scalar variants.
Scalar variants reuse `scalar_value` and `scalar_left` (for operand order), with
the same shared schema validation. Reshape constructs typed metadata sharing
the existing buffer; reduction `keepdims` reshapes the result without a new
execution request. Dtype and ownership contracts are in [TENSOR_API.md](TENSOR_API.md).

Kernel requests must pass the shared core kernel contract validator before a
backend executes them: they require output metadata, launch metadata, a
launch whose global work and thread-group dimensions are all non-zero, a compilation
target, a non-empty entry point, a concrete artifact kind, and a non-empty
artifact data or identifier. Tensor kernel arguments and kernel outputs must be
concrete contiguous tensor metadata with buffers. Primitive `inputs` are not
used for kernel execution; ordered kernel arguments define the backend binding
ABI. `BackendExecution` spans and `KernelArgument.tensor` pointers are borrowed
for the duration of `Backend::execute`; backends must not retain them.

### Launch Abstraction

`LaunchConfig` describes backend-neutral global work-item dimensions and
threads-per-group dimensions. The `grid_*` names mean logical global work size,
not native CUDA block count or Metal threadgroup count. Metal, CUDA, ROCm,
Vulkan/SPIR-V, and MLIR-generated paths map these values to their native launch
concepts locally. All six launch dimensions must be non-zero. For the current
narrow `cx.experimental` Metal generated-kernel path, `grid_x` is the logical
1-D thread count; the Metal backend derives native threadgroup count from
`grid_x` and `threads_per_group_x`. Synchronous execution remains the project
default.

### Compilation Target

`CompilationTarget` identifies a backend-local artifact and entry point for
kernel execution. The `artifact` string is opaque backend-local artifact data or
an artifact identifier; for the current in-memory Metal path it carries metallib
bytes. `KernelArtifactKind` stays backend-neutral: static library, source text,
binary module, or intermediate representation. The core does not name Metal,
MSL, PTX, SPIR-V, MLIR, or other concrete formats here; each backend maps the
neutral artifact class and opaque artifact string to its own compiled library,
generated source cache, binary blob, or IR module. Primitive operations should
leave `compilation_target` unset.

### Buffer Ownership

The core owns backend-neutral tensor and buffer abstractions. Backend-specific
buffer handles stay inside their backend directories. The core must not expose
Metal, MPSGraph, CUDA, ROCm, Vulkan, or platform API handles.

### Null Backend Scaffold

`cpp/tensorcx/backends/null/` validates the ABI shape without executing work. It
returns `kUnavailable` for valid execution requests and `kInvalidArgument` for
contract violations such as primitive input/output count mismatches, invalid
fill allocation descriptors, or kernel execution without launch metadata or a
compilation target. Primitive and kernel validation use the same core helpers
that future real backends should use. The Python test hook
`_backend_contract_smoke_test` proves the scaffold builds and links with the
extension.

## Current Direction

Phase 8 is complete. The backend ABI and null backend scaffold exist, and CPU
fill, add/multiply, unary transforms, reductions, and matmul route through
`CpuBackend::execute` without changing public Python behavior. Metal
add/multiply, unary transforms, axis/norm transforms, reductions, matmul, fill,
and non-empty narrow experimental generated-kernel launches route through
`MetalBackend::execute`. Phase 9 is complete: CUDA uses the same execution ABI
and primitive validator, plus backend-local ownership, dtype, size, offset,
and contiguous-metadata validation. CUDA Runtime API resources stay inside
`cpp/tensorcx/backends/cuda/`; nvcc compiles static kernels at build time.
Device selection is scoped per calling thread and restored afterward. Each
operation waits for completion before returning and publishes output metadata
only after success. Copies and explicit casts support float32/int32; arithmetic
is float32-only and accepts broadcast-compatible shapes. Checked float-to-int
casts validate on the device and read back only a status flag. See the
[tensor API contract](TENSOR_API.md).
See [`CUDA_PHASE9_VALIDATION.md`](CUDA_PHASE9_VALIDATION.md).
Phase 10's MLIR
exploration is complete as a decision/prototype only; it did not add MLIR to
the backend ABI, runtime core, or backend build.
