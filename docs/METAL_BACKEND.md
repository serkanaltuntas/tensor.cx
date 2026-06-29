# Metal Backend

The Metal backend lives in `cpp/cortex/backends/metal/` and is intentionally
isolated from the backend-neutral core.

## Host API

The backend uses Metal-cpp and links the Apple Foundation, QuartzCore, and Metal
frameworks. `CMakeLists.txt` fetches a pinned Metal-cpp source revision because
the local Xcode SDK does not provide the C++ headers in the project include
path.

## Ownership Rules

Metal objects are stored with `NS::SharedPtr`. Owned Apple objects are adopted
with `NS::TransferPtr`. Do not store raw owning `MTL::` or `NS::` pointers.

`MTL::` and `NS::` types must remain inside `cpp/cortex/backends/metal/`.
Public Python bindings and backend-neutral core headers should talk to Metal via
plain C++ tensor/backend APIs.

## Current Behavior

Current Metal support includes exact CPU/Metal/CPU tensor copy round-trips for
contiguous `float32` and `int32` tensors, static Metal kernels for add,
multiply, fill, `exp`, `gelu`, `silu`, `softmax`, `rmsnorm`, `layernorm`,
`sum`, `max`, and `mean`, a naive custom MSL matmul kernel for `float32`, and an
optional MPSGraph-backed matmul path for the Apple optimized primitive route.
Reductions require an explicit axis and support negative axes; `sum` over an
empty axis returns zeros, `mean` over an empty axis returns NaNs, and `max` over
an empty axis is rejected. Softmax, rmsnorm, and layernorm also require an
explicit axis and preserve input shape; softmax uses max-subtraction for
numerical stability. Zero-element tensors are represented without allocating a
zero-length Metal buffer, and their copy and kernel paths are no-ops.

The MPSGraph integration is isolated in `metal_mpsgraph.mm`. The backend-neutral
core and Python binding surface do not expose MPSGraph or Objective-C types.
Building with `-DCORTEX_ENABLE_MPSGRAPH=OFF` removes the MPSGraph path while
leaving the custom MSL matmul path available.

Phase 7 also has a narrow experimental generated-kernel hook for
`cx.experimental`: an in-memory metallib can be loaded through the Metal backend,
checked for a named function, converted into a compute pipeline through
`MetalBackend::execute` for non-empty launches, bound to runtime buffers plus
uint32 scalar arguments, and launched synchronously for the first float32
elementwise subset. Zero-thread launches validate the Metal function and return
as no-ops.

Metal buffers use shared storage for the initial copy and kernel path. This is
simple and correct for the first local runtime; future performance work may
introduce private buffers, command encoders, and explicit synchronization.

Each kernel launch wraps its command buffer/encoder in an `NS::AutoreleasePool`,
since a Python C-extension call (and especially a worker thread) has no implicit
pool to drain the autoreleased Metal objects.

## Threading and the GIL (deferred)

v0.1 execution is synchronous and effectively single-threaded: every Metal op
encodes, commits, and **blocks** on `waitUntilCompleted` while holding the Python
GIL. As a result:

- Concurrent calls from multiple Python threads are serialized, not parallel.
- The lazy `KernelRuntime` pipeline cache is therefore not yet guarded by a mutex
  (the GIL is the de-facto lock).

Releasing the GIL across the blocking submit — and the pipeline-cache locking it
would then require — is intentionally deferred. It is concurrency/performance
work beyond the synchronous v0.1 scope (AGENTS.md "Start with synchronous
execution"; async/streams are out of scope). Do not release the GIL in the
binding without first making the pipeline cache thread-safe.

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
- Execution is synchronous and holds the GIL; not safe for concurrent
  multi-threaded use yet (see "Threading and the GIL" above).
```
