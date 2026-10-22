# Explicit shared storage

`cx.shared_empty(shape, dtype="float32", *, devices=(first, second))` and
`cx.shared_tensor(data, dtype=None, *, devices=(first, second))` return two
contiguous Tensor aliases over one allocation. The pair is required and its
order selects the physical allocator. Shape/dtype inference and input conversion
follow the ordinary factories. Empty contents remain unspecified.

```python
import tensorcx as cx

vk, cu = cx.shared_tensor([1.0, 2.0, 3.0], devices=("vulkan", "cuda"))
print((vk + 2).numpy())  # [3. 4. 5.]
print((cu + 2).numpy())  # [3. 4. 5.]

# CUDA VMM allocation imported into Vulkan; the aliases follow the requested order.
cu, vk = cx.shared_empty((257,), dtype=cx.float32, devices=("cuda", "vulkan"))

# Mandatory same-device reference: distinct wrappers, one ordinary CPU allocation.
a, b = cx.shared_tensor([1, 2, 3], devices=("cpu", "cpu"))
```

## Supported pairs

Repeated devices return two wrappers over one ordinary backend allocation.
Different devices require a registered native exchange route. The current
cross-backend implementation is Linux CUDA device 0 with a Vulkan index on the
**same physical GPU**, in a build enabling both backends. Native UUID matching
checks identity without publishing the identifier. Float32, int32 and bool are
supported, including scalars, empty tensors, odd bool tails and contiguous
reshape/squeeze/expand-dims views.

CUDA/Vulkan support requires compatible opaque-FD external memory and binary
semaphores. CUDA-origin storage additionally requires CUDA VMM and POSIX-FD
export; Vulkan dedicated-only import devices are rejected for that origin.
Unsupported pairs fail before materializing input data. There is no host-copy,
different-device or different-origin fallback. Two Vulkan indices, CPU/GPU,
Metal, AMD/Intel CUDA interoperability and other operating systems are not
implemented exchange routes.

CUDA VMM allocation granularity can exceed the logical tensor size substantially
(2 MiB on the validated GTX 980 Ti). Only the logical bytes are tensor data;
padding and the remaining physical allocation cannot be imported as valid
subviews. Sharing is opt-in and does not replace the normal CUDA allocator.
Shared allocations bypass allocation pools.

[`Tensor.to()`](DEVICE_TRANSFERS.md) continues to make independent cross-device
copies. Ordinary separately created CUDA/Vulkan tensors do not become aliases.
Mixed-device arithmetic remains an error, even for these shared aliases.

## Ownership, synchronization and lifetime

Initial data is uploaded once. Later CUDA/Vulkan handoffs share native memory
without a CPU staging copy. The runtime uses two imported binary semaphores,
external queue-family ownership barriers, CUDA completion and Vulkan fences.
Public execution remains synchronous; sharing does not introduce an async API.
CUDA completion currently covers the device, which can wait for unrelated work.
There is no claim that sharing is faster for every workload.

Each operation holds a stable, deduplicated ownership lease until native access
completes. This includes composite operations, custom kernels, CPU readback and
copies, with a common lock order across both backends. Nested operations reuse
the same lease. Separate private outputs are published only after completion.

Both aliases retain one physical owner and both native contexts. Either alias,
its views or an exported CUDA consumer may outlive the other. A failed handoff
or unproven native completion poisons the shared allocation; later access fails.
If completion cannot be proven during final release, the complete physical owner
and required contexts/handles are retained until process exit. An unproven
Vulkan wait retains its submission resources as well. Restart the process to
recover. Forked children reject inherited native access before taking owner
locks; start a fresh process for GPU work.

## DLPack boundary

The **CUDA alias** supports the existing [CPU/CUDA DLPack contract](DLPACK.md),
including contiguous in-range offset subviews. Imported native shared ranges
retain typed provenance, health checks and ownership leases. A malformed
descriptor cannot escape into ordinary CUDA import by exceeding logical bytes
or pointing into bool padding. Vulkan aliases do not expose a fabricated CUDA
or Vulkan DLPack device type.

Foreign CUDA consumers must keep their exported owner alive and complete their
writes before tensor.cx accesses either alias. The runtime does not intercept
external consumers or coordinate unsynchronized concurrent foreign writes.
`__dlpack__(stream=-1)` retains its no-synchronization export contract. Final
physical release proves CUDA completion even for work on foreign nonblocking
streams.

## Acceptance

The native and Python tests exercise both allocation origins against CPU values,
empty/scalar/tail cases, actual foreign CUDA writes visible through Vulkan,
independent `.to()` copies, DLPack range/subview lifetime, reversed multi-owner
concurrency and nested composites. Native fault injection separately exercises
partial allocation/import cleanup, exported FD cleanup, command cancellation,
semaphore failures, failed CUDA/Vulkan completion, final-owner quarantine and
fork rejection with an inherited locked owner mutex.

Local acceptance on the GTX 980 Ti: full CPU/CUDA/Vulkan/LLVM suite **12,539
passed / 226 expected platform skips**; the subsequent expanded shared-storage
and distribution-validator set **132 passed**; native contracts **41/41**,
including strengthened last-caller release/retention checks. CUDA memcheck found
zero errors and racecheck zero hazards in the native shared-storage contract.
Fresh combined-wheel probes passed **69** normal, **58** hidden-device CPU and
**73** LLVM-enabled checks, with CUDA MLP CPU parity. Separate CPU and Vulkan-only
wheel validation also passed. These are local acceptance results, not isolated
remote GPU CI or wider-device validation.

Use a current official Vulkan validation layer. SDK 1.4.341 records opaque FD
metadata by integer descriptor without tracking external close/reuse; repeated
CUDA-origin imports can therefore receive stale memory/semaphore metadata and
false `allocationSize-01742` reports. The real-device tests use SDK 1.4.363,
whose upstream implementation removes that FD tracking, without message filters
or disabled validation. See the [upstream state tracker](https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/vulkan-sdk-1.4.341.0/layers/state_tracker/state_tracker.h)
and [current implementation](https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/vulkan-sdk-1.4.363.0/layers/state_tracker/state_tracker.h).

For a combined source build and strict local acceptance:

```bash
CMAKE_ARGS="-DTENSORCX_ENABLE_CUDA=ON -DTENSORCX_ENABLE_VULKAN=ON" uv pip install --reinstall -e ".[dev]"
TENSORCX_REQUIRE_BACKENDS=cuda,vulkan TENSORCX_REQUIRE_SHARED_CUDA_VULKAN=1 \
  TENSORCX_VULKAN_VALIDATION=1 uv run --no-sync pytest -q tests/python/test_shared_storage.py
```

CUDA and Vulkan build prerequisites and the validation-layer search path must
already be configured; see [Vulkan setup](VULKAN_BACKEND.md) and
[CUDA environment](CUDA_PHASE9_ENVIRONMENT.md). Fresh combined-wheel acceptance
uses `tools/validate_distribution.py --variants cuda-vulkan`; it requires real
matching hardware and does not substitute CPU execution for a missing GPU.
Evidence from this single NVIDIA GPU does not establish other vendors/models,
actual two-GPU concurrency or other platforms.
