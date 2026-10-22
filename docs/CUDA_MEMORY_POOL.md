# Bounded synchronous CUDA allocation reuse

For Vulkan's distinct physical-byte and command-frame limits, see
[Vulkan measurement and reuse](VULKAN_PERFORMANCE.md).

`cx.memory_pool("cuda")` explicitly reuses idle, exact-byte-size CUDA device-0
buffers on the current thread. Ordinary execution keeps direct allocation/free.
CPU and Metal pools are not implemented. Operations retain their existing
synchronous completion, numerical, alias and generated-output contracts.

```python
import tensorcx as cx

x = cx.ones((1024,), device="cuda")
with cx.memory_pool("cuda", max_cached_bytes=64 * 1024 * 1024,
                    max_cached_blocks=64) as pool:
    for _ in range(10):
        result = (x + x) * x
        del result
    print(pool.stats())
    pool.trim()  # Release idle blocks; keep live tensors valid.

assert pool.stats()["cached_bytes"] == 0
```

## Scope and ownership

- A pool is a single-use context. Enter/exit/trim on its owning thread; same-thread
  nesting is rejected. Separate threads may each create a pool. Calls from other
  threads do not automatically join it. Stats may be read from any thread.
- Only owned `CudaBuffer` allocations made during the context participate.
  Existing buffers and raw auxiliary allocations retain direct deletion. A
  participating buffer returns to **its original pool** when its final view/owner
  dies, including on another thread. It is not adopted into that thread's pool.
- Exit first disables reuse, then frees cached blocks. Live tensors remain valid
  and hold the primary-context/pool ownership they need. Their later destruction
  frees storage directly. Abandoned contexts are cleaned up on owning-thread
  destruction, but normal code must use `with` for deterministic release.
- A buffer exposed through DLPack is permanently excluded from reuse, even when
  the capsule is discarded; this also covers copied exports. Foreign consumers
  can use untracked streams, so the existing direct `cudaFree` boundary is kept.
  Borrowed DLPack storage remains owned by its producer and never enters the pool.
  Existing caller synchronization obligations for foreign writes still apply.
- Exact-size matching ignores dtype only at the raw byte-storage level. New
  tensor metadata and normal fill/copy/kernel work are unchanged. Simultaneously
  live tensors never reuse each other's storage. No capacity rounding, splitting,
  cross-device reuse, stream API or `cudaMallocAsync` is added.

`max_cached_bytes` is a nonnegative integer fitting the platform's `size_t`
(default 64 MiB); zero retains no idle storage. `max_cached_blocks` is an integer
in `[1,4096]` (default 64). Excess/oversized returns are freed rather than evicting
other blocks. Limits cover **cached requested payload bytes**, not live tensors,
allocator rounding, total device VRAM, or a process-wide memory quota. Multiple
thread pools have separate limits. `cx.memory_info("cuda")` reports device-wide
free memory and therefore reflects retained allocations and other GPU users.

## Completion and failures

Pool returns use a nonblocking Driver legacy-stream readiness query. A block
is cached only on success; pending work or query errors take the existing direct
free path. This protects incomplete internal copy/launch failure paths. Internal
work currently uses that stream and completes before successful public return;
external DLPack work is excluded as described above. No synchronization is added
to turn an unsafe return into a cache hit. Foreign current contexts are restored
through the existing primary-context scope. See NVIDIA's
[Driver stream query contract](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__STREAM.html)
and [CUDA allocation/free API](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-runtime-api/group__CUDART__MEMORY.html).

On a cache miss, `cudaMalloc` is used. Only an allocation OOM with cached blocks
may drain those blocks and retry **once**. The recoverable OOM state is cleared;
an unrelated pending CUDA error, drain failure or failed retry is reported.
Other allocation errors do not trigger draining/retry. Memory used by another
pool/process or by live tensors is not reclaimed. This is not an OOM guarantee.

Explicit trim/exit report cleanup errors. A free attempt relinquishes ownership:
its error can refer to earlier asynchronous work, so ambiguous frees are never
retried as if their pointer were certainly still valid. Destruction is best
effort and cannot raise; counters record failed frees/context failures when
possible. GPU/context loss is not claimed recoverable. Exceptional exit still
detaches the pool; a cleanup exception may chain with a body exception.

## Stats and profiling

`pool.stats()` returns an independent snapshot after entry, including after
exit. It contains no pointers, tensor values, process/thread IDs or user paths.

| Fields | Meaning |
| --- | --- |
| `active`, `max_cached_bytes`, `max_cached_blocks` | Scope state and configured limits |
| `cached_bytes`, `cached_blocks`, `peak_cached_bytes` | Idle payload storage currently retained / peak |
| `in_use_bytes`, `peak_in_use_bytes` | Participating checked-out payload, including buffers outliving exit |
| `hits`, `misses` | Nonzero-byte logical requests served from cache / sent to allocation |
| `allocation_calls`, `free_calls` | Runtime calls attempted by this pool, including retries/trim/late deletion |
| `allocation_failures`, `oom_retries` | Failed logical requests / one-shot recoveries attempted |
| `uncached_returns`, `readiness_failures` | Directly freed returns / unsuccessful readiness queries |
| `free_failures`, `failed_free_bytes` | Failed cleanup attempts and their requested bytes; not a measured leak size |

[Host runtime profiling](RUNTIME_PROFILING.md) adds `memory_pool` spans named
`pool_acquire` / `pool_release`. Actual `cudaMalloc` / `cudaFree` calls retain
their separate allocation/deallocation spans. A hit does not pretend to be a
runtime allocation. Pool management self time excludes nested runtime calls;
only self times may be summed. This is still host timing, not device kernel time.

## Reproduce measurements

From the CUDA-enabled source checkout, with working `nvidia-smi` telemetry:

```bash
mkdir -p build/cuda-memory-pool
uv run --no-sync python benchmarks/bench_memory_pool.py --pool off --output build/cuda-memory-pool/off.json
uv run --no-sync python benchmarks/bench_memory_pool.py --pool on --output build/cuda-memory-pool/on.json
```

Defaults cover 1K, 16K, 256K, 1M and 16M elements, seven samples after one warmup,
plus a resident 32×64 → 128 → 10 MLP. Missing CUDA fails without fallback. Cases:

- `allocation_cycle`: public `cx.empty` and local tensor destruction. Current
  CUDA builds [allocate directly](CUDA_EMPTY.md); the dated runs below still
  include the former CPU staging/upload path. Neither is bare cudaMalloc timing.
  Empty data has no numerical promise, so validation checks shape/dtype/device.
- `expression_resident`: `(x + y) * y`, pre-existing resident inputs.
- `expression_end_to_end`: two uploads, that expression and one CPU return,
  including temporary cleanup inside the call.
- `mlp_resident`: matmul → LayerNorm → GELU → matmul → softmax, resident inputs.

Each case has its own pool; warmup populates it. Pool setup/trim/exit, validation
and returned-result destruction are outside the public-call timer; intermediate
allocations and cleanup inside each call remain included. All numerical outputs
are checked against CPU outside timing. A **separate** diagnostic profile runs
after the timed samples, never as a latency sample. Stats include warmups,
validation-related activity, that diagnostic call and returned-result cleanup;
they are not counters for timed calls alone. The 64 MiB default cannot retain
two simultaneous 16M-float32 outputs, so those cases can still miss heavily.

Reports retain samples/median/p10/p90, native-region counts, before/after-close
stats, source/script/profiler/extension identities and sharing snapshots without
process identifiers. Publication is atomic and only follows successful checks;
failed runs preserve existing output. Review compact exports under the
[publication policy](../CONTRIBUTING.md#publish-only-reviewed-evidence).
Repeated on/off runs and varied shapes are required before claiming a gain;
warm steady-state reuse trades retained device memory for fewer runtime calls.

### Recorded local runs — 2026-10-17

Two stock `a6cf458` baseline runs ([1](validation/cuda-memory-pool-before-off1.json),
[2](validation/cuda-memory-pool-before-off2.json)) used the unmodified installed
runtime before rebuilding. The new worktree then ran off/on
([off 1](validation/cuda-memory-pool-after-off1.json),
[on 1](validation/cuda-memory-pool-after-on1.json)) and reversed on/off
([on 2](validation/cuda-memory-pool-after-on2.json),
[off 2](validation/cuda-memory-pool-after-off2.json)). All six reports use the
same benchmark/helper/profiler hashes; the baseline and new extension hashes
differ as expected. They identify pre-commit worktrees based on `a6cf458`, not
an already published implementation commit.

Each run passed all 16 cases with seven measured samples, one warmup and a
separate profile. All pooled cases respected their caps and ended with zero
cached and checked-out bytes after cleanup. This Linux x86_64 i7-6700K / GTX
980 Ti sm_52 / CUDA Runtime 12.4 / driver 580.178.04 system was shared with the
desktop and **another compute process in every run**. No process was stopped.
These runs demonstrate the workflow and runtime-call reduction, not isolated
or general performance. Baseline and off-control variation is substantial.

Representative public-call medians in milliseconds, run 1 / run 2:

| Case | Stock off | New off | New on |
| --- | --- | --- | --- |
| 1K public empty/destruction cycle | 0.1280 / 0.1321 | 0.1168 / 0.1333 | 0.0181 / 0.0187 |
| 1K resident expression | 0.0396 / 0.0332 | 0.1079 / 0.0315 | 0.0256 / 0.0328 |
| 16M resident expression | 6.1069 / 6.2235 | 5.7020 / 6.1827 | 2.9572 / 4.0446 |
| Resident MLP | 0.1829 / 0.1906 | 0.1449 / 0.1860 | 0.1289 / 0.1623 |

The pooled 16K public factory/destruction cycle was slower in both replicates
(off 0.0273 / 0.0281 ms, on 0.0339 / 0.0324 ms). The second pooled 1K expression
was also slightly slower than its off control; lower
medians elsewhere do not establish a universal speedup, unchanged default
latency, bare-allocation time or behavior on another GPU. Full samples and
p10/p90 retain variation and other regressions rather than selecting only gains.

Pool counters are consistent across the two on runs: nine 1K factory calls
(warmup, samples and diagnostic call) required **one allocation and eight hits**,
retaining at most 4096 bytes. Nine resident MLP calls required **four allocations
and 41 hits**, with 35,328 peak idle bytes. The 16M resident expression retained
the full 64 MiB cap, with **10 allocations and eight hits**: its two-output
working set exceeds that idle limit. This memory/call tradeoff is distinct from
the noisy end-to-end latency observations.

## Acceptance

Python tests cover limits/trim, exact sizes and cross-dtype reuse, independent
live outputs/views, escaping tensors, DLPack exclusions, cross-thread final
release, nesting/exceptions, generated private output/suffix preservation,
host-profile call counts and benchmark failure behavior. Native contracts inject
OOM/non-OOM/free errors without exhausting the desktop GPU, exercise a foreign
context, pending legacy work and late TLS deletion, and rerun every existing
primitive operation family under pooling. The strict CUDA push gate includes
these tests. CPU/Metal retain their normal paths; asynchronous execution,
broader allocator policies and other GPU architectures remain separate work.
