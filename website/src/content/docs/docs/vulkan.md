---
title: Vulkan / SPIR-V
description: Build and use the optional Linux Vulkan backend for tensor operations, activations, normalization and inference.
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

Basic float32 add/subtract/multiply/divide use integer significands on the GPU
to provide round-to-nearest, ties-to-even and gradual underflow even when the
driver flushes native float arithmetic. Non-NaN results match CPU float32 bits,
including signed zeros and overflow. Arithmetic NaNs are canonical quiet NaNs;
their source payload/sign is not preserved. This uses no CPU fallback or
optional 64-bit shader feature. Integer arithmetic adds work, especially for
division; no optimized-throughput claim is made.

All three dtypes also support `astype`, contiguous-copy `transpose`, basic
indexing (including negative slices), `concat`, `stack` and `split`. Reordering
preserves stored bits. Float-to-int casts truncate toward zero and reject NaN,
infinity and out-of-range values without mutating input storage. Float-to-bool
treats nonzero subnormals as true and both signed zeros as false.

Comparisons, boolean logic, `where`, leading-dimension mask selection and
boolean `any`/`all` are supported, including broadcasting, selected axes and
`keepdims`. Float32 comparison preserves NaN/signed-zero/subnormal rules through
bit ordering. `where` and masked selection preserve selected data bits. Mask
selection uses a GPU prefix scan; only the result row count is read back.
Empty `any`/`all` reductions return false/true. Boolean reductions currently
iterate serially per output; `where` uses two synchronized GPU passes.

Numeric `sum/max/mean/min` support all, single, multiple or no axes and
`keepdims`; `argmax` supports one axis or flattened search, returning int32
indices. Inputs may be float32 or int32, except mean requires float32. Int32
sum wraps; float32 sum/mean preserve the sequential CPU order and gradual
underflow using integer GPU arithmetic. Mean divides once after the sum.
Min/max preserve the first NaN's bits and first signed zero on ties; argmax
selects the first maximum/NaN. Empty sum/mean return zero/NaN, while
max/min/argmax reject empty reduction spaces. An empty axis tuple copies bits
to independent output storage. Numeric reductions currently iterate serially
per output, without tensor-value readback; throughput optimization remains open.

Float32 `matmul` / `@` supports vectors, matrices and right-aligned batch
broadcasting. `auto` and `custom` use the same kernel; `optimized` is rejected.
One GPU invocation owns each output and accumulates along K with separately
rounded multiply/add, preserving gradual underflow. Results have independent
contiguous storage. Empty outputs submit no work; zero K produces positive
zeros without reading the empty inputs. There is no CPU fallback or implicit
dtype/device conversion. This is a correctness baseline; tiled matmul and
optimized throughput remain open.

```python
a = cx.ones((2, 1, 3, 4), device="vulkan")
b = cx.ones((1, 5, 4, 6), device="vulkan")
c = cx.matmul(a, b, backend="custom")
print(c.shape)  # (2, 5, 3, 6)
```

```python
parts = cx.split(x[:, ::-1].T, [1], axis=-1)
joined = cx.concat(parts, axis=-1)
print(joined.astype(cx.int32).numpy())
# [[3 6]
#  [2 5]
#  [1 4]]
```

Float32 `log`/`sqrt` and float32/int32 `abs`/`clip`/`topk` execute on the GPU.
Sqrt uses an integer restoring root with ties-to-even rounding; log uses
significand normalization and a series near one, accepted at relative error
1e-6. Both handle subnormals, domains and special values without native float
flushing. Abs, clip and topk preserve selected stored bits. Clip supports
three-way broadcasting through two GPU passes; topk returns separate values
and int32 indices, with stable ties and original-index order for `sorted=False`.
These are correctness baselines, without optimized-throughput claims.

Float32 activations and normalization (`exp`, `gelu`, `silu`, `softmax`,
`rmsnorm`, `layernorm`) execute on-device. Linear, affine norms, embedding and
scaled dot-product attention use the same native execution path. Attention
supports broadcast bool/additive masks and causal masking; fully masked rows
have zero probabilities. Embedding validates indices on the GPU and reads
back only a four-byte error flag, including when the output width is zero.

Exp uses split-ln(2) range reduction and a bounded-error polynomial with gradual
underflow (relative tolerance 1e-6 plus one minimum subnormal). GELU/SiLU retain
the ordinary CPU formulas and special-value behavior; GELU is the tanh
approximation. Norms preserve sequential binary32 accumulation and the CPU
constant-row/epsilon rules. Attention allocates the full score/probability
matrices and makes no FlashAttention or optimized-throughput claim.

DLPack and generated kernels/MLIR are not implemented on Vulkan.
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
