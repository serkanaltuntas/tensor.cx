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
- Per-device capabilities and basic diagnostics: Vulkan API version, raw
  vendor-encoded driver version, storage-buffer limit and total device-local
  heap bytes. Available/free bytes are **not collected**. Heaps are device-wide,
  not a count of tensor-owned allocations.

Matmul, reductions, activations, norms, comparisons/masks, slicing/joins,
transpose/casts, inference, DLPack, generated kernels/MLIR, profiling, memory
pools, batching and asynchronous execution are not implemented on Vulkan.
Unsupported operations raise explicit errors; no CPU fallback is performed.
Other backends retain their existing capabilities.

Float32 parity uses `rtol=1e-6, atol=1e-6` on the recorded inputs. Vulkan's basic
shader float controls do not establish a portable subnormal, NaN-payload or
bit-exact arithmetic contract across vendors; those stronger guarantees are
not claimed. Int32 and bool parity are exact. Empty contents are unspecified.

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
and contexts remain separate. Use explicit CPU staging for transfers between
accelerator backends or Vulkan indices. There is no zero-copy interoperability
or automatic topology/deduplication. Native operations reject mixed owners.

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
  uv run --no-sync pytest -q tests/python/test_vulkan.py tests/python/test_backend_parity.py
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
failure retention and missing/incomplete loaders. Fresh source-archive/wheel
installation checks CPU and GPU arithmetic and hidden-driver CPU execution.
The optional CI job builds Vulkan and tests CPU/no-loader paths; it is **not**
GPU execution evidence. Existing CUDA push acceptance remains required.

References: [Khronos compute tutorial](https://docs.vulkan.org/tutorial/latest/11_Compute_Shader.html),
[synchronization examples](https://docs.vulkan.org/guide/latest/synchronization_examples.html),
[Vulkan memory rules](https://docs.vulkan.org/spec/latest/chapters/memory.html),
[device-lost lifetime rules](https://docs.vulkan.org/spec/latest/chapters/devsandqueues.html#devsandqueues-lost-device).
