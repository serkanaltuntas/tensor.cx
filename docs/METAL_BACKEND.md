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

Phase 3 supports exact CPU/Metal/CPU tensor copy round-trips for contiguous
`float32` and `int32` tensors, plus static Metal kernels for add, multiply, and
fill. Zero-element tensors are represented without allocating a zero-length Metal
buffer, and their copy and kernel paths are no-ops.

Metal buffers use shared storage for the initial copy and kernel path. This is
simple and correct for the first local runtime; future performance work may
introduce private buffers, command encoders, and explicit synchronization.

## Current Limitations

```text
- Runtime errors may still throw direct C++ exceptions until Phase 4 error polish.
- Only device index 0 is supported.
- Only contiguous tensors are supported.
```
