# Vulkan measurement and bounded reuse

The optional Linux Vulkan backend supports native host profiling, separate GPU
timestamp intervals, owned-allocation diagnostics and an explicit bounded reuse
scope. Execution remains synchronous. These features do not imply portable
performance, asynchronous execution or acceptance on other GPUs.

```python
import tensorcx as cx

x = cx.ones((1024,), device="vulkan")
with cx.profile(device_timing=True) as profile:
    with cx.memory_pool("vulkan", max_cached_bytes=64 * 1024 * 1024,
                        max_cached_blocks=64) as pool:
        for _ in range(5):
            y = (x + 1).softmax(-1)
        print(pool.stats())
print(profile.summary())          # host nanoseconds
print(profile.device_summary())   # separate GPU milliseconds
print(cx.diagnostics()["backends"]["vulkan"]["allocations"])
```

## Memory and lifetime contract

`cx.memory_pool(device)` uses the existing backend registry for `vulkan` and
indexed Vulkan routes. It is opt-in, current-thread, single-use and cannot nest
with another Vulkan pool on that thread. Independent threads can own independent
scopes. Unscoped execution retains direct allocation/free behavior.

The pool caches owned buffers and their bound memory, including tensor outputs,
internal metadata and host-visible staging buffers. It matches exact padded
buffer capacity and the host-visible class. Dtype is not a cache key: every
nonempty operation writes its defined result, while `empty` remains unspecified.
Staging memory stays mapped while cached; noncoherent flush/invalidate operations
remain required on each use. No implicit device or backend transfer is added.

`max_cached_bytes` bounds **idle VkMemoryRequirements.size bytes**, including
driver alignment; `max_cached_blocks` bounds idle buffer/memory pairs. Neither
limit bounds live tensors, total VRAM or driver object memory. Additionally, one
completed command frame can be retained per scope: one command buffer, fence,
descriptor pool/set and optional two-slot timestamp query pool. This fixed
object cache is outside the byte/block limits, including with a zero-byte limit.
`pool.trim()` releases both idle buffers and that frame, preserving live tensors.
Exit drains both caches; statistics remain readable afterward.

Only last-reference release can return a buffer to its **originating** active
pool. Views and escaped tensors retain their storage. A last release on another
thread can return to that origin under its lock; it never adopts the releasing
thread's pool. Releases after scope exit free normally. One device-OOM retry can
trim cached buffers and retry; other errors are not hidden or retried.

A submitted frame holds every referenced allocation. Cache return requires
successful fence completion and command-buffer/fence reset. Reset failure
discards the completed frame, increments `frame_reset_failures` and preserves
the successful tensor result. Submission/completion failure retains the existing
poisoned-context contract: if device idle cannot prove safe release, the frame
and referenced storage are quarantined rather than freed or recached. A profile
or memory pool cannot make that device usable again.

## Diagnostics

`cx.memory_info("vulkan")` keeps its existing device-local **heap total** and
unknown available/free bytes. It is not an ownership counter.
`cx.diagnostics()["backends"][device]` adds separate fields:

- `allocations`: physical owned blocks/bytes/peak, idle cached blocks/bytes,
  allocation/free/failure calls, and live block/requested-byte counts by
  `tensor`, `internal` and `staging` role. Physical bytes use
  `VkMemoryRequirements.size`; requested role bytes omit padding.
- `execution`: upload/download/dispatch/submission counts and command, fence,
  descriptor, query creation and frame reuse counts.
- `timestamps`: selected queue valid bits, nanoseconds per tick and whether
  the required timestamp functions are available.

Counters are process-lifetime snapshots for tensor.cx-owned resources on that
Vulkan context. They do not include CUDA allocations on the same physical GPU,
other programs or all driver overhead. Pool stats describe only the originating
scope; `in_use_bytes` is physical leased memory, not logical tensor size.

## Profiling clocks and errors

The [shared profiling contract](RUNTIME_PROFILING.md) applies to bounded records,
host self time, current-thread scope and explicit local export. Vulkan host
spans cover execution/validation, allocation/free, staging memcpy, command
recording, queue submission, fence wait and pool bookkeeping.

With `device_timing=True`, nonempty Vulkan compute/upload/download submissions
reserve a separate device record. Two timestamps bracket the encoded GPU work
at top/bottom pipeline stages. Results are read with availability only **after
the existing fence wait**; no extra wait is introduced. These are command
intervals, not isolated shader instructions or end-to-end Python latency.
Scheduling, contention and query recording can affect them. Staging host copies,
allocation and trailing host cleanup are outside the GPU interval.

The clock label is `vulkan_timestamps_elapsed_ms`; `device_clocks` identifies
each supported backend. A report with both CUDA and Vulkan device rows uses
`backend_device_intervals`. An empty device list does not establish timing
coverage. GPU milliseconds have no calibrated host epoch: never sum them with
host nanoseconds or place them on an invented common timeline. Chrome trace
events remain host-only, with device rows retained in metadata.

Unsupported queues/functions, failed query creation/results, unavailable
timestamps and ambiguous counter wraps produce `elapsed_ms=None` with a static
error label. They make device measurement incomplete without replacing a valid
tensor result. Counter subtraction respects `timestampValidBits`; intervals
whose host completion span permits a whole counter wrap are rejected. Disabled
timing, zero GPU work and exhausted record capacity do not create query pools.
Cached query pools may remain until trim/exit, but are not used when timing is
disabled. Ordinary operations do not require timestamp support.

## Reproduce the measurements

```bash
mkdir -p build/vulkan-performance
uv run --no-sync python benchmarks/vulkan_runtime.py \
  --device vulkan --elements 4096 --iterations 50 --rounds 6 --warmup 5 \
  --mode paired --output build/vulkan-performance/paired.json
```

The JSON records alternating direct/pooled samples, cold call, warmed per-call
means, median/range/IQR, scope entry/exit, allocation/object counter deltas and
pool statistics. Cold means first call in that scope after device initialization,
not process startup. Warmed timing includes result destruction, excluding scope
entry/exit and CPU validation. Counters cover the **whole sample**, including
cold validation download and warmup; they are not per-iteration kernel counts.
A separate three-call profiled capture reports host/device intervals; query
instrumentation is excluded from latency samples. Allocation-only work has no
GPU interval. `empty` validates metadata only, never arbitrary uninitialized data.

2026-10-22 local paired run: Linux x86_64, GTX 980 Ti, NVIDIA 580.178.04,
Vulkan device API 1.4.312, 64-bit timestamps at 1 ns/tick; validation/profiling off
for latency samples. Six rounds, 50 calls after five warmups, 4096 elements;
inference uses fixed 8×16 and 16×8 inputs. All defined results passed CPU parity.

| Workload | Direct median µs | Pooled median µs | Driver allocations, direct/pooled per sample |
| --- | ---: | ---: | ---: |
| Empty allocation | 179.97 | 1.83 | 56 / 1 |
| Fill | 835.17 | 65.49 | 169 / 4 |
| Upload | 525.33 | 34.21 | 113 / 2 |
| Download | 338.03 | 71.59 | 56 / 1 |
| Two arithmetic operations | 1597.31 | 147.80 | 337 / 5 |
| Norm → GELU → matmul → softmax | 3675.38 | 540.42 | 673 / 7 |

The first sample created one pooled command frame for every submitted workload,
compared with 56–449 direct command buffers per sample; allocation-only work
created none. Reuse reduces measured driver calls, but a shared desktop, small
workloads and this single GPU do not establish a general speedup, throughput,
bandwidth or other-architecture result. Re-run on the intended workload/driver.

References: [Khronos profiling guide](https://docs.vulkan.org/guide/latest/profiling.html),
[query results](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetQueryPoolResults.html),
[command reset](https://docs.vulkan.org/refpages/latest/refpages/source/vkResetCommandBuffer.html).
