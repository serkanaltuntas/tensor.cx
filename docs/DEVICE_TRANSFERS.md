# Explicit device transfers

`Tensor.to(device, *, copy=False)` places a contiguous tensor on an explicitly
selected available device. It preserves dtype, shape and stored element bits
for float32, int32 and bool, including scalars, empty tensors and odd bool tails.
It does not cast or broadcast. String and `cx.Device` selection follow the same
canonical registry identity, including `vulkan` / `vulkan:0` aliases.

```python
import tensorcx as cx

source = cx.ones((257,), device="cuda")
placed = source.to("vulkan")             # independent host-staged copy
back = placed.to("cuda")                 # independent host-staged copy
same = placed.to("vulkan:0")             # same Python Tensor, no work
independent = placed.to("vulkan", copy=True)  # independent GPU storage
```

Only request device pairs available in the current build/runtime. CPU/Metal/CUDA
retain their index-zero restrictions; Vulkan routes have concrete indexed
ownership. Different indices are supported by the transfer routing contract,
but actual second-device execution needs separate hardware acceptance.

## Copy and readiness rules

| Request | Behavior |
| --- | --- |
| Same device, default `copy=False` | Return the original Tensor without allocation or synchronization |
| Same device, `copy=True` | Return independent contiguous storage on that device |
| Different devices, either copy value | Return independent destination storage |
| Accelerator to accelerator | Download to a temporary CPU tensor, then upload to the destination |
| CPU to/from accelerator | Use the registered native copy callback |

`copy` is strictly a Python bool. It means force a copy even on the same device;
it does not forbid copies when false. DLPack's `copy=False` has its separately
documented no-copy meaning. Same-device CPU/CUDA/Metal copies reuse the existing
identity-transpose copy implementation. Vulkan copies directly between its own
device buffers, including padded four-byte bool transfers; padding is not tensor
data. No shader launch, host upload or download is needed for that Vulkan copy.

Transfers are synchronous: producer work completes before download, the CPU
staging owner stays alive until upload completes, and the destination is ready
when `to()` returns. Source tensors/views remain valid and unchanged. Results
can outlive sources and allocation-pool scopes; staging is not shared storage.
Both endpoint routes and required callbacks are checked before download.
Allocation/copy failures propagate without publishing a partial destination,
relocating data to a different target or changing the source. A failed Vulkan
submission retains its existing poison/quarantine behavior; a plain allocation
failure does not poison a healthy context.

This changes only explicit placement. Mixed-device arithmetic still raises an
error and never invokes automatic transfers. This baseline uses no peer-copy
or external-memory route, even when two backends expose one physical GPU.

## Storage exchange boundary

CPU/CUDA [DLPack](DLPACK.md) exchange can be composed with explicit placement:

```python
import numpy as np
import tensorcx as cx

array = np.arange(6, dtype=np.float32)
vulkan = cx.from_dlpack(array, copy=False).to("vulkan")
host_view = np.from_dlpack(vulkan.to("cpu"))
cuda_tensor = vulkan.to("cuda")
# A supported CUDA consumer can now import cuda_tensor through DLPack.
```

The placement step copies data; DLPack may then share the destination's supported
storage. Native Vulkan storage is not exported under a CPU/CUDA device type or
an unimplemented Vulkan capsule convention. `vulkan.__dlpack_device__()` and
direct Vulkan DLPack exchange remain explicit errors. The
[remaining Vulkan interoperability group](VULKAN_BACKEND.md#remaining-vulkan-work)
also retains actual external-memory sharing, ownership/synchronization and
lifetime acceptance. Host staging does not complete that work.

## Measurement and acceptance

```bash
uv run --no-sync python benchmarks/bench_device_transfer.py \
  --source cuda --target vulkan --dtype float32 --profile \
  --output build/transfer-cuda-vulkan.json
```

The benchmark checks CPU parity and source preservation outside timed regions.
It measures public copy latency separately from copy → target arithmetic →
copy back, alternates sample order, and records separate optional profiles.
Times include synchronous allocation/staging/API costs. Final result deletion
is outside the clock; intermediate cleanup inside the public call is included.
They are host wall times, not device-only bandwidth or a universal speedup.
Default element counts are 1K, 16K, 256K, 1M and 16M. Unavailable endpoints fail
instead of skipping or falling back; an existing output report is not overwritten.
Reports include source/build hashes and optional Git identity without raw paths
or hardware identifiers; source archives work without Git metadata.

Tests compare every available directed pair and both copy modes across the
supported dtypes, special float bits, scalar/empty/tail shapes, views, ownership,
threads/pools, rejected routes and allocation failure. Native Vulkan tests
independently verify padded copies and recovery after a failed clone allocation.
Current real heterogeneous acceptance is CPU/CUDA/Vulkan on one GTX 980 Ti.
Mock indexed routes prove callback routing only. Other GPUs and actual two-GPU
execution remain open; no shared-storage or peer-copy claim follows from these
host-staged results.
