---
title: Backends & support
description: What CPU, Metal, CUDA, and experimental compilation support today, with validation limits.
---

This matrix describes the implemented pre-alpha runtime, not a promise of
production support. All operations are synchronous; tensors are contiguous and
only device index 0 is exposed.

| Capability | CPU | Metal | CUDA |
| --- | --- | --- | --- |
| Float32/int32/bool tensor copy | Yes | Yes | Yes |
| Float32 fill, add, multiply | Yes | Yes | Yes |
| Int32 fill, add, multiply | Yes | Yes | No |
| Float32 subtract, divide, negate and scalar arithmetic | Yes | Yes | Yes |
| Int32 subtract, negate and scalar add/subtract/multiply | Yes | Yes | No |
| Contiguous reshape (float32/int32/bool) | Yes | Yes | Yes |
| Transpose to contiguous copy (float32/int32/bool) | Yes | Yes | Yes |
| Squeeze / expand dims views (float32/int32/bool) | Yes | Yes | Yes |
| Basic indexing / slicing copies (float32/int32/bool) | Yes | Yes | Yes |
| Concat / stack / split copies (float32/int32/bool) | Yes | Yes | Yes |
| Explicit float32/int32/bool conversion | Yes | Yes | Yes |
| Comparisons and `where` (float32/int32/bool) | Yes | Yes | Yes |
| Bool fill, logic, any/all and mask selection | Yes | Yes | Yes |
| Binary arithmetic broadcasting | Yes | Yes | Yes, float32 |
| Float32 2D matmul | Yes | Custom + optional optimized path | Custom path |
| Float32 sum, max, mean | Yes | Yes | Yes |
| Int32 sum, max | Yes | Yes | No |
| Numeric reduction `keepdims` | Yes | Yes | Yes, float32 |
| All-axis / multi-axis sum, max, mean | Yes | Yes | Yes, float32 |
| Float32 exp, GELU, SiLU | Yes | Yes | Yes |
| Float32 softmax, RMSNorm, LayerNorm | Yes | Yes | Yes |
| Generated kernels | Optional MLIR subset | Experimental MSL subset | Optional MLIR subset |

## CPU

The mandatory reference backend. It runs without a GPU and is the baseline for
device comparison tests. Start here for the simplest installation and debugging
workflow. CPU-only CI covers Python 3.11, 3.12, and 3.13.

## Metal

Apple Silicon is the primary Metal target. Building requires Xcode's `metal`
and `metallib` tools. Static kernels implement tensor operations; matmul can use
an optional Apple optimized path or the custom correctness-first kernel.

`cx.matmul(a, b, backend="custom")` selects the custom implementation.
`backend="optimized"` requires the enabled Apple optimized path. macOS CI has
configurations with that path enabled and disabled. A job that skips device
tests does not establish hardware execution.

## CUDA

CUDA is an opt-in backend. Its recorded real-device validation is limited to a
GTX 980 Ti (`sm_52`) on Linux x86_64 with CUDA 12.4 and GCC 13. Tests compare
outputs with CPU references and include native contracts and GPU sanitizer runs.

Additional NVIDIA architectures and toolchain combinations still need their
own validation. The hosted CUDA build job tests compilation and no-device
behavior; it does not run on an NVIDIA GPU. A local push gate exercises the
validated GPU. Repository-wide GPU acceptance still needs an isolated runner.

The CUDA backend copies and explicitly casts `int32` tensors but rejects int32 factories and
arithmetic. Float32 matmul supports `auto` and `custom`, not `optimized`.

## Compiler boundary

MLIR is an optional compiler path, not a fourth hardware backend. Its current
CPU/CUDA runtime integration handles a guarded float32 elementwise subset.
See [experimental kernels](/docs/experimental/) for requirements and non-goals.

ROCm, Vulkan/SPIR-V, float16/bfloat16, asynchronous streams, and multi-device
execution are not implemented.

## Engineering evidence

Detailed, versioned records stay with the source:

- [Backend execution contract](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/BACKENDS.md)
- [CUDA primitive validation](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/CUDA_PRIMITIVES_VALIDATION.md)
- [Distribution and host limits](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/DISTRIBUTION.md)
