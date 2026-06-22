# Backend Notes

Cortex Runtime keeps the backend-neutral runtime in `cpp/cortex/core/` and
places backend-specific implementation under `cpp/cortex/backends/`.

## Current Backends

```text
cpu     Required reference backend.
metal   Apple Silicon backend for buffer ownership, copy round-trips, and first
        elementwise kernels.
```

The CPU backend is the correctness reference for every operation. Metal behavior
must be compared against CPU behavior before it is treated as complete.

## Phase 3 Status

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
```

CPU remains the correctness reference for every Metal operation.

## Phase 4 Direction

Phase 4 should polish runtime errors, shape and dtype validation, local Metal
test instructions, and richer benchmark scripts.
