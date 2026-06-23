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
- exp, gelu, and silu kernels for float32 tensors
- naive custom MSL matmul for float32 rank-2 tensors
- MPSGraph matmul for float32 rank-2 tensors
- reduction kernels for sum/max on float32 and int32 tensors
- reduction kernel for mean on float32 tensors
```

CPU remains the correctness reference for every Metal operation.

## Phase 6 Direction

Phase 6 is in progress. The current subset supports `sum`, `max`, `mean`,
`exp`, `gelu`, and `silu`. Remaining Phase 6 work should continue with stable
`softmax`, `rmsnorm`, and `layernorm`, each with CPU references before or
alongside Metal kernels.
