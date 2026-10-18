---
title: Backends & support
description: What CPU, Metal, CUDA, Vulkan and experimental compilation support today, with validation limits.
---

This matrix describes the implemented pre-alpha runtime, not a promise of
production support. All operations are synchronous; tensors are contiguous and
CPU/Metal/CUDA expose index 0; Vulkan has indexed routes with one physical GPU validated.

| Capability | CPU | Metal | CUDA | Vulkan (initial slice) |
| --- | --- | --- | --- | --- |
| Float32/int32/bool tensor copy | Yes | Yes | Yes | Yes |
| Float32 fill, add, multiply | Yes | Yes | Yes | Yes |
| Int32 fill, add, multiply | Yes | Yes | Yes | Yes |
| Float32 subtract, divide, negate and scalar arithmetic | Yes | Yes | Yes | Yes |
| Int32 subtract, negate and scalar add/subtract/multiply | Yes | Yes | Yes | Yes |
| Contiguous reshape (float32/int32/bool) | Yes | Yes | Yes | Yes |
| Transpose to contiguous copy (float32/int32/bool) | Yes | Yes | Yes | Yes |
| Squeeze / expand dims views (float32/int32/bool) | Yes | Yes | Yes | Yes |
| Basic indexing / slicing copies (float32/int32/bool) | Yes | Yes | Yes | Yes |
| Concat / stack / split copies (float32/int32/bool) | Yes | Yes | Yes | Yes |
| Explicit float32/int32/bool conversion | Yes | Yes | Yes | Yes |
| Comparisons and `where` (float32/int32/bool) | Yes | Yes | Yes | Yes |
| Bool fill, logic, any/all and mask selection | Yes | Yes | Yes | Yes |
| Binary arithmetic broadcasting | Yes | Yes | Yes | Yes |
| Float32 batched/vector matmul | Yes | Custom + optional optimized path | Custom path | Custom path |
| Float32 sum, max, mean | Yes | Yes | Yes | Yes |
| Int32 sum, max | Yes | Yes | Yes | Yes |
| Sum/max/mean `keepdims` | Yes | Yes | Yes | Yes |
| All-axis / multi-axis sum, max, mean | Yes | Yes | Yes | Yes |
| Float32 log, sqrt | Yes | Yes | Yes | Yes |
| Float32/int32 min, argmax | Yes | Yes | Yes | Yes |
| Float32/int32 abs, clip, topk | Yes | Yes | Yes | Yes |
| Float32 linear, affine norms, embedding, attention | Yes | Yes | Yes | Not implemented |
| Float32 exp, GELU, SiLU | Yes | Yes | Yes | Not implemented |
| Float32 softmax, RMSNorm, LayerNorm | Yes | Yes | Yes | Not implemented |
| Generated kernels | Optional MLIR subset | Experimental MSL subset | Optional MLIR subset | Not implemented |
| `empty` allocation | CPU storage | Existing CPU staging/copy | Direct device allocation | Direct device allocation |
| DLPack exchange (contiguous float32/int32/bool) | Yes | Not implemented | Device 0 | Not implemented |
| Installation and operation/dtype diagnostics | Yes | Availability/capabilities | Yes | Yes |
| Memory snapshots | System physical memory | Not implemented | Device-wide free/total | Device-local heap total |
| Native host profiling | Bounded spans | Not instrumented | Bounded spans | Not implemented |
| Persistent compiled-code cache | Optional MLIR | Experimental MSL | Optional MLIR | Not implemented |
| Bounded buffer pool | Not implemented | Not implemented | Explicit opt-in scope | Not implemented |

`empty` has unspecified contents on every backend. CUDA skips CPU payload
allocation/upload and can reuse idle storage in an explicit pool. See
[memory reuse and profiling](/docs/performance/) and the
[feature overview](/docs/status/) for the remaining scope.

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

The CUDA backend supports `int32` copies/casts, empty/zeros/ones, wrapping
add/subtract/multiply/negate, integer scalars and sum/max reductions. Division
and mean require float32. The math extension also supports int32
abs, min, argmax, clip and topk. Float32 matmul supports `auto` and `custom`, not `optimized`.

## Vulkan / SPIR-V

The optional Linux backend supports GPU copies/factories/views, casts,
transpose, basic indexing/joins, comparisons/masks, boolean/numeric reductions and
broadcast float32/int32 arithmetic and float32 vector/batched matmul.
It has real GTX 980 Ti validation, with no
claim of other-GPU acceptance. Basic float32 arithmetic uses integer GPU
significands for ties-to-even rounding and gradual underflow; arithmetic NaNs
are canonical quiet NaNs. Software Vulkan CPU devices are excluded from
GPU discovery. Numeric sum/max/mean/min/argmax preserve CPU order and execute
serially per output, with mean restricted to float32. Matmul supports
`auto`/`custom`, separate multiply/add rounding and broadcast batches.
Log/sqrt/abs/clip/topk also execute on-device; sqrt uses integer root rounding,
log uses a bounded-error series, and clip/topk preserve selection bits.
Inference, activations, normalization, DLPack and MLIR are outside this
first slice. See [Vulkan setup and limits](/docs/vulkan/).

## Compiler boundary

MLIR is an optional compiler path, not a hardware backend. Its current
CPU/CUDA runtime integration handles a guarded float32 elementwise subset.
See [experimental kernels](/docs/experimental/) for requirements and non-goals.

ROCm, float16/bfloat16, asynchronous streams and multi-device execution are not
implemented. Broader Vulkan operations and GPU coverage remain open.

## Engineering evidence

Detailed, versioned records stay with the source:

- [Backend execution contract](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/BACKENDS.md)
- [CUDA primitive validation](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/CUDA_PRIMITIVES_VALIDATION.md)
- [Distribution and host limits](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/DISTRIBUTION.md)
