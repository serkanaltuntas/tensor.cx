# Metal Backend

The Metal backend lives in `cpp/tensorcx/backends/metal/` and is intentionally
isolated from the backend-neutral core.

## Host API

The backend uses Metal-cpp and links the Apple Foundation, QuartzCore, and Metal
frameworks. `CMakeLists.txt` fetches a pinned Metal-cpp source revision because
the local Xcode SDK does not provide the C++ headers in the project include
path.

## Ownership Rules

Metal objects are stored with `NS::SharedPtr`. Owned Apple objects are adopted
with `NS::TransferPtr`. Do not store raw owning `MTL::` or `NS::` pointers.

`MTL::` and `NS::` types must remain inside `cpp/tensorcx/backends/metal/`.
Public Python bindings and backend-neutral core headers should talk to Metal via
plain C++ tensor/backend APIs.

## Current Behavior

Native comparisons, boolean logic, where, any/all and boolean mask selection
follow the [tensor API contract](TENSOR_API.md#comparisons-and-boolean-masks).
Mask selection uses a device prefix scan and gather, reading back only its
selected count; bool storage occupies one byte per element.


Current Metal support includes exact CPU/Metal/CPU tensor copy round-trips for
contiguous `float32`, `int32` and `bool` tensors, static Metal kernels for add,
multiply, fill, `exp`, `gelu`, `silu`, `softmax`, `rmsnorm`, `layernorm`,
`sum`, `max`, and `mean`, a naive custom MSL matmul kernel for `float32`, and an
optional MPSGraph-backed matmul path for the Apple optimized primitive route.
Reductions accept all axes (`None` by default), one integer or an axis sequence,
including negative indices. Selected zero-size axes yield zeros for `sum`, NaNs
for `mean`, and an error for `max`; `()` copies under the same dtype rules.
Multi-axis kernels index the original buffer using shared extent/stride metadata;
mean divides once after accumulation. Softmax, rmsnorm, and layernorm require an
explicit axis and preserve input shape; softmax uses max-subtraction for
numerical stability. Zero-element tensors are represented without allocating a
zero-length Metal buffer, and their copy and kernel paths are no-ops.

The MPSGraph integration is isolated in `metal_mpsgraph.mm`. The backend-neutral
core and Python binding surface do not expose MPSGraph or Objective-C types.
Building with `-DTENSORCX_ENABLE_MPSGRAPH=OFF` removes the MPSGraph path while
leaving the custom MSL matmul path available.

Phase 7 also has a narrow experimental generated-kernel hook for
`cx.experimental`: an in-memory metallib can be loaded through the Metal backend,
checked for a named function, converted into a compute pipeline through
`MetalBackend::execute` for non-empty launches, bound to runtime buffers plus
uint32 scalar arguments, and launched synchronously for float32 elementwise
and rowwise-reduction (bounded for/accumulator) kernels. Zero-thread launches
validate the Metal function and return as no-ops.

Metal buffers use shared storage for the initial copy and kernel path. This is
simple and correct for the first local runtime; future performance work may
introduce private buffers, command encoders, and explicit synchronization.

Each kernel launch wraps its command buffer/encoder in an `NS::AutoreleasePool`,
since a Python C-extension call (and especially a worker thread) has no implicit
pool to drain the autoreleased Metal objects.

## Threading and the GIL

Execution is still synchronous — every Metal op encodes, commits, and blocks on
`waitUntilCompleted` — but the binding layer releases the Python GIL around
every native backend call (execute, device copies, and experimental kernel
validation/launch). Ops issued from multiple Python threads therefore overlap
inside the native layer instead of serializing on the GIL.

The native state that concurrent calls share is guarded:

- The lazy `KernelRuntime` pipeline cache for embedded kernels is protected by a
  mutex; the returned pipeline pointers stay valid because slots are never
  cleared and the singleton lives for the whole process.
- Experimental DSL launches use a mutex-guarded pipeline cache in
  `metal_library.cpp`, keyed by the exact metallib bytes plus function name, so
  repeated launches do not recompile per call and concurrent launches cannot
  corrupt the cache. The cache is bounded (64 entries); clearing it only drops
  references, and in-flight launches keep their pipelines alive via
  `NS::SharedPtr`.
- `MTLCommandQueue` is thread-safe per Apple's documentation; `MetalContext` and
  backend singletons rely on C++11 thread-safe static initialization.
- The CPU backend is stateless per call.

Code that runs while the GIL is released must stay pure C++: no Python object
may be created, copied, or destroyed inside `without_gil` regions in the
binding. Async submission and streams remain out of scope.

## Current Limitations

```text
- Only device index 0 is supported.
- Only contiguous tensors are supported.
- Metal kernels currently support at most 2^32 - 1 elements per launch.
- Matmul currently supports float32 rank-2 tensors only.
- `mean`, `exp`, `gelu`, `silu`, `softmax`, `rmsnorm`, and `layernorm`
  currently support float32 tensors only.
- Reduction kernels are correctness-first and use one thread per output element;
  they are not optimized for large reduction axes yet.
- Execution is synchronous (no async/stream API); native calls release the GIL
  and are safe for concurrent multi-threaded use (see "Threading and the GIL"
  above).
```
