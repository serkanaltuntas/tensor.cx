---
title: Vulkan / SPIR-V
description: Build and use the optional Linux Vulkan backend for GPU copies, factories and basic tensor arithmetic.
---

Vulkan is an optional Linux backend with a deliberately smaller operation set
than CPU, Metal or CUDA. Its first real-device validation is on a GTX 980 Ti
with NVIDIA 580.178.04 and Vulkan device API 1.4.312. Support on other NVIDIA,
AMD and Intel GPUs still requires hardware acceptance.

## Build and run

Install a C++20 compiler, Vulkan 1.1+ headers, `glslangValidator` and `spirv-val`
for the build. Ubuntu provides the latter dependencies in `libvulkan-dev`,
`glslang-tools` and `spirv-tools`. Execution requires a matching Vulkan loader
and compute GPU driver/ICD. Vulkan is off by default.

```bash
CMAKE_ARGS="-DTENSORCX_ENABLE_METAL=OFF -DTENSORCX_ENABLE_CUDA=OFF -DTENSORCX_ENABLE_VULKAN=ON" \
  uv pip install -e ".[dev]"
```

The checked-in GLSL shader is compiled, validated and embedded as SPIR-V at
build time. Installed use needs no shader compiler, external shader file or
LLVM. The extension still imports and runs CPU operations without a usable
Vulkan loader or GPU; the Vulkan route then reports unavailable.

```python
import tensorcx as cx

print(cx.devices())
x = cx.tensor([[1., 2., 3.], [4., 5., 6.]], device="vulkan")
y = cx.tensor([10., 20., 30.], device="vulkan")
print(((x + y) * 2).numpy())
# [[22. 44. 66.]
#  [28. 50. 72.]]
print(cx.capabilities("vulkan"))
```

## Supported subset

Float32, int32 and bool support CPU copies, tensor factories, direct `empty`
and shared reshape/squeeze/expand-dims views. Float32 and int32 support
broadcast add/subtract/multiply, negation and scalar arithmetic; division is
float32 only. Int32 arithmetic wraps exactly as on CPU. `randn` uses host
random generation followed by upload. `empty` contents are unspecified.

Matmul, reductions, inference, activations, masks, indexing/joins, casts,
transpose, DLPack and generated kernels/MLIR are not implemented on Vulkan.
Unsupported operations raise errors. Inspect capabilities before choosing a
backend for a larger workload.

Operations are synchronous and use checked staging, barriers and fence waits.
Validation covers CPU parity, empty/tail sizes, int32 wraparound, a 2D dispatch
grid, threaded calls, native failure paths and fresh source-archive/wheel
installation. It establishes correctness for the recorded environment, not a
performance or cross-vendor guarantee.

## Device selection and diagnostics

`vulkan` and `vulkan:0` select the first eligible GPU. Additional eligible GPUs
use indexed routes such as `vulkan:1`; ordering is process-local, not a permanent
hardware identity. Only one physical Vulkan GPU has been tested. Mocked indexed
routing is not real two-GPU acceptance. Software CPU implementations such as
lavapipe are excluded from the GPU list.

CPU, Metal and CUDA retain their device-0 limits. CUDA/Metal keep higher
`best_device()` priority. A GPU exposed through both CUDA and Vulkan has separate
storage for each backend; use explicit CPU staging between them.

`cx.diagnostics()` reports API version, vendor-encoded raw driver version,
storage-buffer limit and total device-local heap bytes. Free/available bytes,
profiling and memory pools are not collected or implemented. For validation,
install `VK_LAYER_KHRONOS_validation` and set `TENSORCX_VULKAN_VALIDATION=1`
before import; missing requested layer support leaves Vulkan unavailable.

The [source backend guide](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/VULKAN_BACKEND.md)
contains native ownership/failure rules, validation commands and the remaining
hardware limits. See the [full support matrix](/docs/backends/) for comparisons.
