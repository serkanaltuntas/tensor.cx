---
title: Features & roadmap
description: Implemented tensor.cx features, backend boundaries and remaining product work.
---

tensor.cx is a **pre-alpha runtime and experimental kernel compiler**, licensed
under Apache-2.0. The original phase milestones are complete, while the broader
product roadmap remains open. The tables below summarize the current scope;
the [support matrix](/docs/backends/) defines backend availability.

## Implemented features

| Area | Available today | Guide |
| --- | --- | --- |
| Tensor storage | Contiguous float32/int32/bool, scalars and empty shapes; explicit device copies and casts; shared reshape/squeeze/expand-dims views | [Operations](/docs/operations/) |
| Data preparation | Basic indexing/slicing, concat, stack and split as independent contiguous copies | [Index and join](/docs/operations/#api-overview) |
| Masks | Six comparisons, boolean logic, `where`, `any`/`all` and boolean selection | [Masks](/docs/operations/#boolean-masks) |
| Math | Tensor/scalar broadcasting, batched/vector matmul, all/multi-axis reductions with `keepdims`, log/sqrt/abs, min/argmax/clip/topk | [Math and selection](/docs/operations/#math-and-selection) |
| Inference | Float32 linear, affine RMSNorm/LayerNorm, int32-indexed embedding and scaled dot-product attention with boolean/additive/causal masks | [Inference](/docs/operations/#inference) |
| Storage exchange | CPU/CUDA device-0 DLPack sharing or explicit copy; optional inference-only PyTorch `linear` custom op | [DLPack & PyTorch](/docs/interoperability/) |
| Diagnostics | Standalone `tensorcx-doctor`, installation checks, operation/dtype capabilities and CPU/CUDA memory snapshots | [Diagnostics](/docs/diagnostics/) |
| Explicit kernels | Experimental Metal MSL and guarded float32 CPU/CUDA MLIR compilation; division, sqrt/abs/exp/log/rsqrt/tanh/sigmoid/SiLU, minimum/maximum, comparisons and value selection in the MLIR subset | [Experimental kernels](/docs/experimental/) |
| Compilation tools | Persistent CPU/CUDA/Metal code cache; compile/launch latency measurements and separate compilation profiles | [Performance](/docs/performance/#compilation-cache-measurements) |
| Runtime profiling | CPU/CUDA host spans and Chrome trace export; optional separate CUDA kernel event intervals | [Runtime profiling](/docs/performance/#native-runtime-profiling) |
| Manual CUDA fusion experiment | Native/separate-MLIR/fused expression comparisons with balanced unprofiled calls and separate CUDA diagnostics | [Fusion measurements](/docs/performance/#manual-cuda-fusion-experiment) |
| Generated CUDA outputs | Proved nonzero full writes omit the template copy; private output, immutable inputs and conservative legacy/partial behavior retained | [Output copies](/docs/performance/#generated-cuda-output-copies) |
| CUDA allocation | Bounded opt-in exact-size reuse, pool stats/trim, and direct `empty` allocation without CPU staging | [Memory reuse](/docs/performance/#bounded-cuda-memory-reuse) |
| Vulkan / SPIR-V | Optional Linux GPU copies/factories/views, casts, transpose, indexing/joins, comparisons/masks, boolean and numeric reductions, basic arithmetic; float32 uses ties-to-even with gradual underflow; one GTX 980 Ti validated | [Setup and limits](/docs/vulkan/) |
| Distribution | Reviewed Linux x86_64 CPU manylinux wheel pipeline for CPython 3.11–3.13; source builds for supported Metal/CUDA environments | [Installation](/docs/installation/) |

Operation/dtype combinations have explicit limits. For example, matmul,
division, mean and inference arithmetic require float32; integer indices are
used by embedding and selection. There is no implicit dtype promotion or
device transfer. `empty` contents are unspecified, including on a pool hit;
use initialized factories or write every element before reading it.

## Remaining work

The first six packages in the
[ordered product backlog](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/ROADMAP.md#product-feature-backlog)
are implemented within their documented scope. Packages 7–11 have the following
remaining parts; completed pieces above are not reset by this list.

| Track | Remaining scope |
| --- | --- |
| Diagnostics and profiling | Native Metal memory reporting and profiling; CUDA transfer timing and broader profiling coverage |
| Compiler and performance | More generated MLIR operations and reductions, automatic fusion and wider fusion validation, and lower first-process toolchain fingerprinting cost; ordinary tensor expressions are not automatically fused |
| Dtypes | FP16/BF16 implementation and numerical/performance acceptance on suitable hardware |
| Memory and execution | CPU/Metal pools, broader allocator policies, asynchronous submission/streams and command batching |
| Distribution and releases | Portable macOS/CUDA wheels, versioned releases and package-index publication; Linux ARM64 and musl/Alpine wheels are outside the current matrix |
| CUDA validation and devices | Execution on additional real NVIDIA architectures/toolchains, an isolated automatic GPU CI runner, and device-index selection beyond 0 |

Float32 `sqrt`, `abs`, minimum/maximum and bounded-error `exp`/`log`/`rsqrt`/`tanh`/`sigmoid`/`silu`
are implemented inside guarded experimental kernels, with their target-specific
numerical contracts. Ordinary tensor operations remain separate. The manual CUDA
fusion experiment compares explicit grouping on the validated local GPU; it does
not add automatic fusion or establish performance on other GPUs.

## Validation boundaries

CPU tests, package checks, macOS jobs with MPSGraph enabled/disabled and a
GPU-free CUDA build job cover different parts of the runtime. Recorded CUDA
execution is on Linux x86_64, GTX 980 Ti sm_52, CUDA 12.4 and GCC 13. A local
exact-commit push gate runs real CUDA tests, native contracts and GPU sanitizers.
It does not cover pushes from every computer or replace an isolated GPU runner;
the separate remote CUDA acceptance workflow remains disabled until that runner
is provisioned and validated. A skipped GPU job is not execution evidence.

Measurements retain their workload and shared-machine limits; no overall speedup
or cross-architecture support is promised. See [measurement guidance](/docs/performance/).

## Separate longer-term directions

Autograd, model training, distributed/multi-GPU execution, a full PyTorch/ATen
backend, general strided execution and ROCm are not implemented. Wider Vulkan operation
and hardware coverage remains open beyond the [initial slice](/docs/vulkan/).
They are separate directions rather than missing parts of the narrow PyTorch
bridge or the completed original phases.

[PROJECT.md](https://github.com/serkanaltuntas/tensor.cx/blob/main/PROJECT.md) is
the phase ledger, and the source
[roadmap](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/ROADMAP.md)
remains the canonical engineering backlog. This page is a curated user summary.
