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

Phase 2 supports exact CPU/Metal/CPU tensor copy round-trips for contiguous
`float32` and `int32` tensors. Zero-element tensors are represented without
allocating a zero-length Metal buffer, and their copy path is a no-op.

Metal buffers use shared storage for the initial copy path. This is simple and
correct for Phase 2; future performance work may introduce private buffers,
command encoders, and explicit synchronization.

## Current Limitations

```text
- No Metal kernels are implemented yet.
- CPU add/multiply do not dispatch to Metal tensors.
- Runtime errors may still throw direct C++ exceptions until Phase 4 error polish.
- Only device index 0 is supported.
```
