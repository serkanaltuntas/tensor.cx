# Optional Vulkan/SPIR-V backend

The first Linux Vulkan runtime slice was validated on 2026-10-18 using the
public `tensorcx` API on a real NVIDIA GeForce GTX 980 Ti. This is an integrated
backend, not a standalone shader probe. It uses the existing registry,
`BackendExecution` contract and mandatory CPU reference.

## Build and use

Vulkan is **off by default**. CPU/Metal/CUDA installations do not need a Vulkan
SDK. A Vulkan source build needs Linux, a C++20 compiler, Vulkan 1.1+ headers,
`glslangValidator` and `spirv-val` on the build machine. For example, Ubuntu's
`libvulkan-dev`, `glslang-tools` and `spirv-tools` packages provide these tools.
A matching Vulkan loader, GPU driver/ICD and compute-capable GPU are needed for
execution. The initial slice does not target Windows, Android or MoltenVK.

```bash
CMAKE_ARGS="-DTENSORCX_ENABLE_METAL=OFF -DTENSORCX_ENABLE_CUDA=OFF -DTENSORCX_ENABLE_VULKAN=ON" \
  uv pip install -e ".[dev]"
uv run --no-sync python -c 'import tensorcx as cx; print(cx.devices())'
```

Nonstandard tool installations can set CMake's
`TENSORCX_VULKAN_INCLUDE_DIR`, `TENSORCX_GLSLANG_VALIDATOR` and
`TENSORCX_SPIRV_VAL`. The build compiles the checked-in GLSL compute shader to
SPIR-V, validates it and embeds its words in the extension. Runtime use needs
no shader compiler, external `.spv` file or LLVM installation.

```python
import tensorcx as cx

x = cx.tensor([[1., 2., 3.], [4., 5., 6.]], device="vulkan")
y = cx.tensor([10., 20., 30.], device="vulkan")
print(((x + y) * 2).cpu().numpy())
# [[22. 44. 66.]
#  [28. 50. 72.]]
print(cx.capabilities("vulkan"))
```

## Implemented scope

- `float32`, `int32`, `bool`: CPU upload/download, `tensor`, direct device
  `empty`, zeros/ones, and shared-storage reshape/squeeze/expand-dims views.
- `float32`, `int32`: broadcasted add/subtract/multiply, negate, integer or
  float32 scalars as appropriate, including scalar-left expressions. Int32
  arithmetic wraps modulo 2^32, matching the CPU reference.
- `float32` only: tensor/scalar division. `randn` generates data on the host and
  explicitly uploads it, as with other backends.
- All three dtypes: explicit `astype`, contiguous-copy `transpose`, basic
  indexing (including negative-step slices, integer axes, ellipsis and new
  axes), `concat`, `stack` and `split`. These execute on the GPU without
  exporting tensor data to the CPU. Same-dtype copies and reordering preserve
  stored bits, including float32 NaNs, signed zeros and subnormals.
- All three dtypes: broadcast comparisons, `where` and leading-dimension
  boolean mask selection. Boolean tensors also support logical and/or/xor/not
  and `any`/`all` over scalar, single, multiple or all axes, with `keepdims`.
- `float32`, `int32`: `sum`, `max`, `min` and `argmax`; `mean` is float32-only.
  Sum/max/mean/min support all, single, multiple or no axes and `keepdims`.
  Argmax supports one axis or a flattened search and returns int32 indices.
- `float32`: vector, matrix and broadcast-batched `matmul` / `@`, with
  `auto` and `custom` selecting the same kernel.
- `float32`: `log` and `sqrt`; `float32`/`int32`: `abs`, broadcasted `clip`
  and deterministic `topk` values/int32 indices.
- `float32`: exp/GELU/SiLU, softmax/RMSNorm/LayerNorm, linear, affine
  normalization, int32-indexed embedding and masked/causal scaled dot-product attention.
- Per-device capabilities and basic diagnostics: Vulkan API version, raw
  vendor-encoded driver version, storage-buffer limit and total device-local
  heap bytes. Available/free bytes are **not collected**. Heaps are device-wide,
  not a count of tensor-owned allocations.
- Native host profiling, separate GPU copy/compute timestamp intervals,
  owned-allocation diagnostics and bounded opt-in staging/buffer/command reuse;
  see [measurement, limits and benchmark evidence](VULKAN_PERFORMANCE.md).

Explicit guarded float32 [MLIR/SPIR-V custom kernels](VULKAN_COMPILER.md) use
private outputs and the shared compilation cache. DLPack, command batching and
asynchronous execution are not implemented on Vulkan.
Unsupported operations raise explicit errors; no CPU fallback is performed.
Other backends retain their existing capabilities.

The remaining math APIs reuse `MathPlan`. Sqrt performs a restoring base-four
integer root over two uint32 words, retaining an exact remainder for one
binary32 ties-to-even rounding. Positive subnormal inputs are normalized by
bits; signed zero is retained. Log normalizes by significand/exponent, reduces
the mantissa near one, and evaluates the odd degree-13 atanh series with the
existing RNE helpers and split ln(2). Its accepted relative tolerance is 1e-6
with no absolute slack, including values near one and positive subnormals;
correctly-rounded log is not claimed. Both handle domains/zeros/infinities
explicitly and quiet input NaNs. Abs preserves magnitude bits (including NaNs)
or wraps int32 magnitude. No shaderInt64 or optional float controls are used.

Clip applies the broadcast lower then upper bound in two checked GPU passes,
retaining a private expanded intermediate. NaN input/bound ordering and
signed-zero ties match the CPU's stored-bit selections. Topk assigns one
invocation per group, repeatedly selects in a stable total order, and writes
separate private values and int32 indices. Unsorted mode returns the selected
entries in original axis-index order. This is O(n*k) selection and O(k²)
unsorted index insertion, without an optimized throughput claim. Zero work
skips dispatch and unused metadata narrowing. Both output descriptors are
published only after success; allocation/second-pass failure preserves caller
slots and immutable inputs. See [the math contract](TENSOR_API.md#math-and-selection).

Matmul reuses the shared `MatmulPlan` for rank-one promotion, right-aligned
batch broadcasting, output shape and overflow checks. One invocation owns an
output and accumulates in row-major K order using the integer binary32 helpers
for separately rounded multiply/add, including gradual underflow. No optional
shaderInt64/float controls, host-value fallback or expanded operand copies are
used. The four-buffer layout and bounded 2D dispatch are retained. Empty
outputs submit no work; zero K writes positive zeros without reading empty
input buffers. Unused huge empty dimensions remain valid. Unsupported
preferences, malformed shapes/dtypes/owners and traversed uint32 limits reject
without replacing the caller's output slot or mutating either input. This is
a correctness baseline; tiling, library acceleration and measured optimization
remain open. See [the shared matmul contract](TENSOR_API.md#batched-matrix-multiplication).

Basic float32 add/subtract/multiply/divide now implement binary32
round-to-nearest, ties-to-even, including gradual underflow, signed zeros,
overflow and invalid operations. Their tensor and scalar paths use uint32
significand arithmetic on the GPU, without host fallback, shaderInt64, float64
or optional float-control features. Add/subtract align guard/round/sticky bits;
multiply uses a two-word exact significand product; division uses binary long
division. Rounding happens once, after any subnormal shift. These helpers also
avoid false NaNs when dividing two nonzero subnormal inputs.

Non-NaN results match CPU float32 bits under round-to-nearest, ties-to-even.
Arithmetic NaNs are canonical quiet NaNs; their input payload/sign is not
preserved. Negate, copies, shape operations and selection retain their separate
bit-preserving contracts. Int32 and bool parity are exact. This arithmetic
contract does not establish other GPUs' hardware acceptance or a numerical
contract for other operations. Empty contents are unspecified.

The integer path adds work, especially for division. A local shared-desktop
snapshot (seven warmed synchronous host calls per size, allocation/metadata/
fence included, upload/readback excluded) measured 1M-element medians of
1.04–1.19 ms before and 1.05–1.54 ms after this change across the four operations.
These are host intervals, not device times, controlled throughput comparisons
or guarantees. Profiling and measured optimization remain in the checklist.

`astype` supports every pair among float32/int32/bool. Float-to-int truncates
toward zero and rejects NaN, infinity and values outside `[-2^31, 2^31)`.
A private GPU error flag is read back before publishing the output; rejection
does not mutate the input or poison the context. Float-to-bool uses the stored
bits so even subnormal nonzero values are true, while both signed zeros are
false. Boolean transformations assign one invocation to each complete packed
word, preventing races at unaligned byte boundaries. Concat currently makes
one synchronous output pass per input; correctness is validated, throughput
optimization remains in the checklist below.

Float32 comparisons use IEEE-754 bit ordering, with NaN unordered, `!=` true
for NaN, both signs of zero equal and subnormal ordering preserved independently
of device float arithmetic controls. Bool predicates normalize any nonzero
stored byte to true. `where` and mask selection preserve selected branch/data
bits, including noncanonical bool bytes. `where` uses two synchronized passes
to retain Vulkan 1.1's four-storage-buffer minimum. Mask selection uses a GPU
inclusive prefix scan followed by stable gather; only the final four-byte
selected-row count is read back for output allocation. `any`/`all` currently
reduce serially per output element; empty reductions return false/true.
These initial algorithms establish correctness, not optimized throughput.

Numeric reductions currently assign one invocation to each output and traverse
reduced axes in input-axis order, independent of the order of an axis tuple.
Float32 sum uses the integer-significand add helper; mean performs one sum then
one division, with the CPU's float32 extent conversion. Non-NaN result bits
match that sequential CPU order, including subnormals and overflow. Sum/mean
arithmetic NaNs are canonical quiet NaNs. Int32 sum wraps modulo 2^32.
Min/max preserve the first NaN's stored bits and the first zero's sign on ties;
argmax returns the first maximum/NaN index. Empty sums are zero and empty means
are NaN; max/min/argmax reject an empty reduction even if no output elements
would be produced. An explicit empty axis tuple copies stored bits to new
storage; argmax's internal empty-axis plan instead returns zero indices.
Shape/axis/index limits are validated before publishing any output; unused
metadata on empty outputs need not fit shader indices. No tensor values are
read back for numeric reductions. Long serial reductions are a correctness
baseline; parallel algorithms and measured optimization remain open.

## Activations, normalization and inference

This increment was validated on the recorded GTX 980 Ti on 2026-10-22.

Float32 `exp`, `gelu`, `silu`, `softmax`, `rmsnorm` and `layernorm` execute
entirely on the GPU. Exp reduces by a split ln(2), evaluates a degree-ten
polynomial and scales the exponent with gradual underflow. Its acceptance is
relative error 1e-6 plus one minimum subnormal (2^-149), including underflow
and overflow boundaries. It is not claimed to be correctly rounded. GELU/SiLU
retain the ordinary CPU formulas, with relative/absolute tolerance 1e-6:
GELU uses the tanh approximation; negative infinity produces NaN for both
operations under those formulas. This is separate from the stable experimental
kernel DSL's activation contracts. Input signed zeros and subnormal arithmetic
are retained by the uint32 helpers.

One invocation owns a whole normalized axis, including non-last axes. Norms
retain sequential RNE sum/product order, correctly rounded sqrt and the CPU's
constant-row/zero-epsilon behavior. Ordinary softmax propagates nonfinite rows
under max subtraction; only attention has the all-masked-row exception.

Linear, affine normalization and attention use the shared
[inference composites](TENSOR_API.md#inference-operations) through the Vulkan
execution ABI. Native owners of every input, mask and affine parameter are
validated first. Embedding validates int32 indices on the GPU and reads back
only a private four-byte error flag; even zero-width outputs reject invalid
indices. Gathered values preserve stored bits. Attention's masked softmax
handles broadcast bool/additive masks, causal masking and subnormal scales.
Fully excluded/all-negative-infinity rows have zero probabilities; unmasked
NaNs propagate. Intermediate outputs remain private until every stage succeeds.
These synchronous algorithms retain the four-SSBO minimum and allocate full
attention score/probability matrices; optimized throughput and FlashAttention
are not claimed.

## Device identity and ownership

`vulkan` and `vulkan:0` select the first eligible GPU. Further eligible GPUs use
`vulkan:1`, etc.; `cx.device()` retains backend and index, while its display name
does not participate in equality. Enumeration order is cached for the process
and is not a permanent hardware identifier. Discrete, integrated and virtual
compute GPUs with the required limits are eligible. CPU/software devices such
as lavapipe are excluded, including when they are the only Vulkan ICD.

CPU, Metal and CUDA still accept index 0 only. Vulkan has lower automatic
selection priority than CUDA/Metal, preserving existing accelerator choices.
The same physical NVIDIA GPU may appear as both CUDA and Vulkan; their buffers
and contexts remain separate. Explicit [`.to()` placement](DEVICE_TRANSFERS.md)
now stages transfers between accelerator backends or Vulkan indices through
CPU; `copy=True` forces independent same-device storage. Separately,
[`shared_empty` / `shared_tensor`](SHARED_STORAGE.md) explicitly allocate aliases
on matching CUDA/Vulkan hardware when both backends and native sharing features
are available. Ordinary allocations remain separate; there is no automatic
topology/deduplication. Native operations reject mixed device owners.

Only one physical Vulkan GPU was available for acceptance. Mocked zero/one/two
registry routes validate parsing and callback routing; they do not establish
real two-GPU concurrency, independent failures or nonzero-device execution.

## Execution and failure contract

Device-local storage uses host-visible staging for copies. Logical bool storage
is byte-addressed; Vulkan buffers/copies are padded to four bytes. Empty tensors
carry metadata/ownership without creating zero-sized native buffers. Host byte
sizes, broadcast metadata, storage ranges and dispatch dimensions are checked
before narrowing to shader uint32 indices. Dispatch X is capped at 65,535
groups (or the device limit, if lower); larger workloads use the second grid
dimension. Device memory allocations reserve
against `maxMemoryAllocationCount` and the selected heap size.

A native mutex serializes queue, command-pool and descriptor use with the Python
GIL released. Memory barriers cover transfer/compute/readback visibility; mapped
noncoherent staging is flushed or invalidated over its mapped allocation.
Every operation waits on a fence before returning. A failed submission/wait
poisons the context and publishes no result. Cleanup requires successful idle or
the Vulkan device-lost completion rule. If idle also fails without that proof,
the one failed submission's resources and context are retained until process
exit; they are never freed while potentially in flight. Restart the process to
recover; CPU and other backend contexts are not silently substituted.

The loader is opened dynamically: a Vulkan-enabled wheel still imports and runs
CPU operations without `libvulkan.so.1` or a usable ICD. An unusable Vulkan route
is reported unavailable. `VK_DRIVER_FILES`/`VK_ICD_FILENAMES` visibility is
resolved on first discovery; change it before starting the process.

## Validation and limits

Recorded environment: Linux x86_64, GCC 13, GTX 980 Ti, NVIDIA 580.178.04,
Vulkan loader 1.4.341, device API 1.4.312, glslang 16.2.0 and SPIR-V Tools
2026.1. CUDA's `sm_52` designation is not a Vulkan portability guarantee.
Other NVIDIA cards, AMD/Intel GPUs, driver versions and hosts remain unvalidated.

Set `TENSORCX_VULKAN_VALIDATION=1` before import to request
`VK_LAYER_KHRONOS_validation` and debug utilities. The layer must be installed
and discoverable (a nonstandard installation can set `VK_LAYER_PATH`). Missing
requested validation support leaves Vulkan unavailable; it is not ignored.
The private acceptance counter `_core._vulkan_validation_errors()` must stay
zero. Raw validation logs, device UUIDs and local paths are not publication data.

```bash
TENSORCX_VULKAN_VALIDATION=1 TENSORCX_REQUIRE_BACKENDS=vulkan \
  uv run --no-sync pytest -q tests/python/test_vulkan.py \
  tests/python/test_vulkan_transform.py tests/python/test_vulkan_predicate.py \
  tests/python/test_vulkan_float32.py \
  tests/python/test_vulkan_reduction.py tests/python/test_multi_axis_reductions.py \
  tests/python/test_vulkan_matmul.py tests/python/test_batched_matmul.py \
  tests/python/test_vulkan_math.py tests/python/test_math_ops.py \
  tests/python/test_vulkan_nn.py tests/python/test_inference_ops.py \
  tests/python/test_boolean_masks.py tests/python/test_backend_parity.py
# Build native tests with TENSORCX_BUILD_TESTS=ON and TENSORCX_ENABLE_VULKAN=ON:
uv run --no-sync ctest --test-dir build/vulkan-native --output-on-failure
# Fresh host-specific Vulkan wheel built from the source archive; requires GPU:
uv run --no-sync python tools/validate_distribution.py \
  --variants vulkan --output build/distribution-vulkan
```

Real-device acceptance covers copies/fills (all three dtypes), empty/scalar/tail
shapes, broadcasts/scalars, int32 wraparound, shared views, a 2D dispatch grid,
concurrent threads and immutable inputs. Native contracts cover malformed
metadata/owners, allocation limits, unchanged outputs on rejection, fence/idle
failure retention and missing/incomplete loaders. Transformation and predicate
acceptance also covers all nine cast pairs, noncanonical bool bytes, IEEE
comparison ordering, bit-preserving conditional/leading-mask selection, empty
and multi-axis boolean reductions, packed bool 2D dispatch and million-element
mask scans. Native offset readback checks preserve destinations on rejection.
Arithmetic acceptance additionally compares non-NaN result bits against CPU
and NumPy across exponent boundaries, rounding ties, random bit patterns,
scalar-left/right paths, empty shapes, threaded calls and 2D dispatch. NaN
classification and canonical quiet bits are checked separately from CPU NaN
payloads.
Numeric acceptance covers all/selected/empty axes and keepdims, first NaN/tie
ordering, signed zeros, gradual underflow, wrapping int32, million-element
reduction spaces, 2D output dispatch, high-rank/unused empty metadata, immutable
inputs and composition with host export blocked. Native rejections retain the
caller's output slot for malformed input ownership, dtype, shape and axes.
Matmul acceptance includes vector promotion, asymmetric batch broadcasting,
zero K and empty/high-rank/unused huge shapes, separate-rounding NumPy oracles,
special/subnormal values, a >65535-workgroup output, threaded composition,
private outputs and unsupported preferences. The shared native batched contract
also runs for Vulkan, with both input owners and failure output retention checked.
High-rank tensors remain valid internally. NumPy export follows NumPy's own
rank limit and safely rejects larger ranks; reshape explicitly before export
(see [the API contract](TENSOR_API.md)).
Fresh source-archive/wheel
installation checks CPU and GPU arithmetic and hidden-driver CPU execution.
The optional CI job builds Vulkan and tests CPU/no-loader paths; it is **not**
GPU execution evidence. Existing CUDA push acceptance remains required.

## Remaining Vulkan work

The user's 2026-10-18 goal is to finish the Vulkan work that can be developed
and verified on the current Linux host. This checklist retains that full scope;
finishing one increment does not complete the backend. The CPU API is the
operation reference, and each advertised feature needs real-device acceptance.

- [x] Copies, factories, shared views and basic numeric arithmetic.
- [x] Float32 basic arithmetic gradual underflow and fixed ties-to-even rounding
  through integer GPU significands; arithmetic NaN payloads are not preserved.
- [x] All dtype conversion pairs, transpose, basic indexing and
  concat/stack/split, including packed bool and float-to-int validation.
- [x] Comparisons, boolean logic, `where`, masked selection and boolean
  reductions, selected axes and `keepdims`.
- [x] Numeric sum/max/mean/min/argmax reductions, selected axes and `keepdims`.
- [x] Vector, matrix and broadcast-batched matmul.
- [x] Remaining math: log/sqrt/abs/clip/topk, with CPU parity.
- [x] Activations, normalization and inference operations
  (`linear`, affine norms, embedding and attention), with CPU parity.
- [x] Repeatable benchmarks, host/device timing, allocation diagnostics and
  measured staging/command/descriptor/buffer reuse; [contract](VULKAN_PERFORMANCE.md).
- [x] Experimental custom kernels and MLIR-to-SPIR-V, with the existing
  guarded-kernel validation and compilation-cache contracts;
  [supported subset and acceptance](VULKAN_COMPILER.md).
- [x] Explicit accelerator/device transfer API and supported storage exchange
  where the API and driver can actually provide it; do not label a host copy
  zero-copy or publish an unsupported DLPack device type.
  The [host-staged placement baseline](DEVICE_TRANSFERS.md) is implemented;
  [explicit CUDA/Vulkan shared allocation](SHARED_STORAGE.md) now has native and
  Python real-device acceptance for both origins, including a fresh combined
  wheel, foreign CUDA mutations, DLPack lifetime/range checks and native
  failure/quarantine acceptance. Other device pairs remain outside this route.
- [ ] Portable Linux Vulkan wheels and repeatable local real-GPU acceptance
  alongside no-loader/hidden-ICD CI. Broader packaging must not require a Vulkan
  driver merely to import and use CPU.
- [ ] The separately planned lavapipe software execution probe, clearly
  distinguished from GPU acceptance (see [device backlog](DEVICE_BACKLOG.md)).

External acceptance still needs other NVIDIA/AMD/Intel GPUs and real two-GPU
hardware. Windows/Android/MoltenVK, FP16/BF16 and asynchronous public execution
remain separate extensions with their own platform/dtype/API prerequisites;
this host does not establish support for them. No new device-wide compatibility
claim follows from implementing the checklist on one GTX 980 Ti.

References: [Khronos compute tutorial](https://docs.vulkan.org/tutorial/latest/11_Compute_Shader.html),
[synchronization examples](https://docs.vulkan.org/guide/latest/synchronization_examples.html),
[Vulkan memory rules](https://docs.vulkan.org/spec/latest/chapters/memory.html),
[device-lost lifetime rules](https://docs.vulkan.org/spec/latest/chapters/devsandqueues.html#devsandqueues-lost-device).
