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

Phase 5 supports exact CPU/Metal/CPU tensor copy round-trips for contiguous
`float32` and `int32` tensors, static Metal kernels for add, multiply, and fill,
a naive custom MSL matmul kernel for `float32`, and an optional MPSGraph-backed
matmul path for the Apple optimized primitive route. Zero-element tensors are
represented without allocating a zero-length Metal buffer, and their copy and
kernel paths are no-ops.

The MPSGraph integration is isolated in `metal_mpsgraph.mm`. The backend-neutral
core and Python binding surface do not expose MPSGraph or Objective-C types.
Building with `-DCORTEX_ENABLE_MPSGRAPH=OFF` removes the MPSGraph path while
leaving the custom MSL matmul path available.

Metal buffers use shared storage for the initial copy and kernel path. This is
simple and correct for the first local runtime; future performance work may
introduce private buffers, command encoders, and explicit synchronization.

## Current Limitations

```text
- Only device index 0 is supported.
- Only contiguous tensors are supported.
- Metal kernels currently support at most 2^32 - 1 elements per launch.
- Matmul currently supports float32 rank-2 tensors only.
```
