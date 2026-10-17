# Direct CUDA empty storage

`cx.empty(shape, dtype="float32", device="cuda")` allocates an independent
contiguous CUDA device-0 buffer directly. A registered optional Python backend
factory calls the native CUDA factory with the GIL released. That factory
validates element count and contiguous strides before using the existing
`CudaBuffer::create` allocator; byte-size/dtype checks precede allocation there.
There is no CPU payload buffer, H2D copy, fill launch or added synchronization.
CPU and Metal keep the previous factory/copy routes. No core ABI, new primitive
operation, stream API, dtype or device index is introduced.

## Contents and lifetime

**Initial values are unspecified.** Empty storage is not zeroed, scrubbed or
guaranteed finite; a pool hit may contain previous bytes. This was already the
public `empty` contract, even though the former CPU staging implementation
happened to supply zeroed buffers. Use `zeros`/`ones` for initialized data and
write every element before reading or computing with empty storage. float32,
int32 and bool use the same storage rules.

Scalars and zero-size tensors retain their shape/stride metadata. A zero-size
buffer retains context ownership but calls no `cudaMalloc`. Invalid devices are
rejected before consuming shape iterables; invalid dimensions, dtype and
size/stride/byte overflow still fail. Ordinary allocations use the existing
primary-context scope and direct deletion. With an explicit
[memory pool](CUDA_MEMORY_POOL.md), exact-size blocks reuse its existing limits,
OOM policy and readiness checks. Views retain the buffer beyond pool exit;
DLPack-exposed storage is permanently excluded from reuse. Synchronous operation
completion and foreign-write synchronization obligations remain unchanged.

## Verification

`tests/python/test_empty.py` covers CPU/device metadata, scalar/zero-size/large
empty shapes, invalid input before allocation, unavailable-device rejection,
registered direct/fallback factories, absence of host/copy/fill work, pool hits,
simultaneously live tensors and escaping views. The DLPack tests initialize every
byte through the exported CUDA pointer before comparing against CPU values;
they do not compare uninitialized values. The native CUDA contract also tests
direct allocation, initialized CPU parity and invalid metadata/dtypes, both with
and without the pool. The exact-commit CUDA push gate includes the new Python
file and anchors for the direct, pool and DLPack cases.

## Measurements

The unchanged `benchmarks/bench_memory_pool.py` compares public `empty` creation
**and destruction**, including validation/routing/metadata overhead. It is not
bare `cudaMalloc` latency. Its five sizes are 1K, 16K, 256K, 1M and 16M float32
elements. Seven unprofiled samples follow one warmup, then a separate diagnostic
native profile. Empty cases validate metadata only; the other eleven expression
and MLP controls compare initialized numerical outputs against CPU outside timing.
Returned-result validation/cleanup and pool setup/close are untimed; destruction
of the local empty tensor occurs inside the timed factory cycle.

Run from a CUDA-enabled checkout with `nvidia-smi` telemetry:

```bash
mkdir -p build/cuda-empty
uv run --no-sync python benchmarks/bench_memory_pool.py --pool off --output build/cuda-empty/off.json
uv run --no-sync python benchmarks/bench_memory_pool.py --pool on --output build/cuda-empty/on.json
```

Repeated before/after results and their limits are recorded below. The historical
[pool reports](CUDA_MEMORY_POOL.md#recorded-local-runs--2026-10-17) retain their
former CPU-staging measurement boundary.

### Recorded local runs — 2026-10-17

The baseline is clean `6720740`, measured off 1/2 then on 1/2 before rebuilding.
The new implementation was measured on 1/2 then off 1/2. All eight runs share
the same benchmark/helper/profiler hashes, with one extension hash per build.
After reports identify a dirty worktree based on `6720740`, not an already
committed release. The eight compact exports retain every factory size and
sample, pool stats and native breakdown. They explicitly omit the eleven
numerical control rows, whose CPU comparisons passed in each full local run;
no inference/workload speedup is claimed.

- Before: off [1](validation/cuda-empty-before-off1.json) /
  [2](validation/cuda-empty-before-off2.json), on
  [1](validation/cuda-empty-before-on1.json) /
  [2](validation/cuda-empty-before-on2.json).
- After: off [1](validation/cuda-empty-after-off1.json) /
  [2](validation/cuda-empty-after-off2.json), on
  [1](validation/cuda-empty-after-on1.json) /
  [2](validation/cuda-empty-after-on2.json).

Public factory/destruction medians in **milliseconds**, run 1 / run 2:

| Elements | Before off | After off | Before on | After on |
| --- | --- | --- | --- | --- |
| 1K | 0.126595 / 0.124820 | 0.128173 / 0.117344 | 0.012305 / 0.012111 | 0.003444 / 0.003343 |
| 16K | 0.029456 / 0.028457 | 0.005916 / 0.005837 | 0.025275 / 0.025552 | 0.002866 / 0.002938 |
| 256K | 0.251661 / 0.252303 | 0.113963 / 0.114601 | 0.145532 / 0.152228 | 0.002968 / 0.003014 |
| 1M | 0.848488 / 0.760149 | 0.327176 / 0.328454 | 0.500677 / 0.507881 | 0.003042 / 0.002926 |
| 16M | 46.740262 / 45.492628 | 4.587381 / 4.589906 | 44.844401 / 44.818815 | 0.002983 / 0.002977 |

Separate diagnostic profiles show one CPU allocation, one H2D copy and its
device wait in each old factory call, and none in the new calls. Unpooled calls
still allocate/free CUDA storage; warmed pooled calls use acquire/release spans.
Pool allocation/hit counts and caps remain consistent, including at 16M where
a single 64 MiB factory buffer fits the default cap. All pooled cases closed
with zero idle/live payload and no allocation/free failures.

This Linux x86_64 i7-6700K / GTX 980 Ti sm_52 / CUDA Runtime 12.4 / driver
580.178.04 machine shared its desktop and **another compute process in all eight
runs**. No process was stopped. Results support removal of staging work on this
factory path; they do not establish isolated performance, a universal speedup
or behavior on another GPU. The first unpooled 1K median was slightly slower
after the change; small-call variation remains visible in the full samples and
p10/p90. Pooled empty storage has no initialization work to time and does not
represent a numerical operation or device-only allocation latency.

Local acceptance: **87 new Python cases**, **6382 passed / 171 expected skips**
with LLVM enabled; the unavailable-LLVM run also passed with 4724 / 1829.
Native and ASan/UBSan contracts passed 4/4 (leak detection disabled). CUDA
memcheck/racecheck passed 86 Python cases and the updated native contract with
zero errors/hazards. The remaining hidden-device subprocess case passed normally;
under sanitizer injection its child still observed CUDA despite an empty
`CUDA_VISIBLE_DEVICES`, so it is excluded only from those instrumented commands,
not from the normal suite or exact-commit push gate. This is one validated GPU,
not multi-architecture or remote-GPU acceptance.
