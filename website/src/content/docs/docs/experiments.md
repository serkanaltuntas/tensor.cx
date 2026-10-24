---
title: Execution experiments
description: Validated NVIDIA OpenCL, PoCL CPU and Mesa lavapipe experiments, separate from runtime backend support.
---

The [runtime support matrix](/docs/backends/) covers CPU, Metal, CUDA and optional
Vulkan. Separate experiments investigate other execution paths on the same
physical hardware. They are not additional devices or new runtime backends.

The four experiments in the
[source device backlog](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/DEVICE_BACKLOG.md#nightblade-validation-experiments-planned)
are complete within their documented scope as of **24 October 2026**.

## Validated paths

| Hardware and path | Result | Runtime availability |
| --- | --- | --- |
| GTX 980 Ti / Vulkan-SPIR-V | Tensor operations, inference, guarded generated kernels and local GPU acceptance | Optional [Linux Vulkan backend](/docs/vulkan/), validated on this GPU |
| i7-6700K / Mesa lavapipe | 192 guarded float32 cases, two lifecycles, 1,742/1,742 owned resources released, zero validation errors | Standalone CPU software experiment; excluded from GPU discovery |
| GTX 980 Ti / NVIDIA OpenCL | 192 guarded float32 cases, two contexts, 490/490 owned resources released | Standalone GPU experiment |
| i7-6700K / PoCL OpenCL CPU | 192 guarded float32 cases, two contexts, 490/490 owned resources released | Standalone CPU experiment |

The standalone probes exercise real allocation, upload, guarded add/multiply,
readback and cleanup. Their cases include empty and tail sizes, zero/partial/full
active prefixes, separate output and first-input aliasing. Readbacks, untouched
suffixes, padding and preserved inputs match tensor.cx CPU operations, guarded
`Kernel.reference` execution and NumPy.

The NVIDIA OpenCL run used driver 580.178.04 and OpenCL C 1.2. The PoCL CPU run
used PoCL 6.0 with LLVM 18.1.8 and its pthread device. Wrong-device selection,
compiler failures and incorrect unguarded results were rejected. Clean CPU
wheel installs also ran both OpenCL probes from the source distribution.

## Scope and reproduction

**OpenCL is not a registered tensor.cx backend.** These probes do not add
`device="opencl"`, OpenCL DLPack exchange, full operation coverage, multi-device
execution, or a performance claim. PoCL and lavapipe run on the CPU; they are
not evidence of another GPU architecture. The small finite test fixtures do not
establish general float32 special-value or subnormal conformance.

Use the versioned source instructions for the external drivers/tools, exact
device selection, commands and acceptance limits:

- [NVIDIA OpenCL and PoCL CPU probes](https://github.com/serkanaltuntas/tensor.cx/tree/main/experiments/opencl)
- [Mesa lavapipe software probe](https://github.com/serkanaltuntas/tensor.cx/tree/main/experiments/vulkan)
- [Vulkan runtime setup and limits](/docs/vulkan/)

Other NVIDIA/AMD/Intel GPUs, real two-GPU execution and other operating systems
still need their own acceptance. Wider device work and the isolated GPU runner
remain in the [roadmap](/docs/status/#remaining-work).
