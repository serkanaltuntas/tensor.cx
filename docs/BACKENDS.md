# Backend Notes

Cortex Runtime keeps the backend-neutral runtime in `cpp/cortex/core/` and
places backend-specific implementation under `cpp/cortex/backends/`.

## Current Backends

```text
cpu     Required reference backend.
metal   Apple Silicon backend for buffer ownership and copy round-trips.
```

The CPU backend is the correctness reference for every operation. Metal behavior
must be compared against CPU behavior before it is treated as complete.

## Phase 2 Status

The Metal backend currently supports:

```text
- device discovery
- shared Metal buffer allocation
- CPU -> Metal copy
- Metal -> CPU copy
- Python Tensor.to("metal")
- Python Tensor.cpu()
```

It does not execute Metal kernels yet. CPU add/multiply remain the only tensor
operations.

## Phase 3 Direction

Phase 3 should add the first Metal kernels through backend-neutral operation
dispatch. Avoid extending the Python wrapper with one branch per operation, such
as `add_metal` or `multiply_metal`, unless it is a temporary private bridge with
a documented removal path.
