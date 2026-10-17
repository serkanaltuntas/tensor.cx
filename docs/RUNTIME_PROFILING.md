# Native runtime host profiling

`cx.profile()` records bounded, current-thread host spans for CPU and CUDA.
It separates native allocation, copies, execution and existing waits without
adding device synchronization. Metal execution is **not instrumented**; every
report declares that limitation. This is a narrow runtime profiling increment,
not a device-event profiler, graph optimizer or new execution mode.

```python
import tensorcx as cx

x = cx.ones((1024,), device="cpu")  # or "cuda" on a supported CUDA build
with cx.profile(max_events=10000) as p:
    result = (x + x) * x

for row in p.summary():
    print(row["backend"], row["category"], row["operation"], row["self_ns"])
report = p.report()
assert report["complete"]
p.export_chrome_trace("runtime-trace.json")  # explicit local file export
```

## Timing and coverage contract

All offsets/durations use the host's monotonic C++ `steady_clock`, in integer
nanoseconds. A span measures an **attempted** native region; it is not evidence
that the operation succeeded. Public exceptions and tensor results are unchanged.

| Category | Measured host region |
| --- | --- |
| `dispatch` | Backend execution validation, planning and routing outside nested work |
| `allocation` | CPU owned-vector allocation/initialization or `cudaMalloc` |
| `deallocation` | `cudaFree` calls occurring inside the context |
| `copy_api` | CUDA copy API call or CPU concat copy/planning, excluding nested allocation |
| `copy_allocate` | CPU sequence import, same-dtype clone, NumPy export or generated-output clone; allocation and copy combined |
| `compute_sync` | CPU executors (including validation/indexing) or CUDA launch wrappers including their existing nested wait |
| `memset_api` | `cudaMemset` API call; device completion is not implied |
| `synchronize` | Existing `cudaDeviceSynchronize` / generated-kernel `cuCtxSynchronize` calls |

`duration_ns` is inclusive; `self_ns` subtracts directly nested spans. Only
**self times** may be added across events. In particular, CUDA `compute_sync`
self time excludes nested synchronization and primarily measures host launch
and wrapper overhead. Neither its self nor inclusive duration is device-only
kernel time. Copy API timing may end before device transfer completion; a later
wait can include earlier work. See NVIDIA's
[CUDA 12.4.1 synchronization behavior](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-runtime-api/api-sync-behavior.html).

`elapsed_ns` covers the active native session; `unattributed_ns` is that duration
minus recorded self times. It includes Python work, binding work outside spans,
profiler overhead, unsupported paths and, on truncation, dropped spans. CPU
vector constructors that adopt an already-filled vector do not record another
allocation. Imports/exports and clones combine their storage work. CPU frees,
arbitrary internal temporary/metadata allocations, CUDA context/module startup,
external library work and other threads are not independently attributed.
Compilation has a [separate profiler](COMPILATION_CACHE.md#compile-and-launch-profiling).

`complete` means **no instrumented spans were dropped**, not complete backend
coverage. Inspect `coverage` even when it is true. A Metal-only context can have
no events and still be complete; incidental CPU imports/exports remain CPU spans.
No Metal kernel, allocation or wait duration is claimed.

## Lifetime, limits and exports

- A context is single-use. Enter and exit on its owning thread. Nested contexts
  on that thread are rejected; separate threads may each run their own profiler.
  Work submitted by other threads is not captured automatically.
- `max_events` is an integer in `[1, 100000]` (default 10000). Storage is reserved
  before recording. Once full, operations continue unchanged, `dropped_events`
  increases and `complete` becomes false. Summary times are partial in that case.
- Normal context exit, including an exception in the body, stops recording.
  Reports become available afterwards; report snapshots are independent copies.
  Use `with`, rather than relying on destructor cleanup of manual entry.
- When disabled, native scopes only check thread-local state: they do not read
  a clock or allocate profiling storage. This is not a zero-overhead claim.
- Event fields include stable local IDs/parent IDs, static region/backend labels,
  optional operation and optional byte counts. Bytes are known requested payload
  sizes, not allocator capacity, live-memory usage or achieved bandwidth.
- No tensor values, pointers, user paths, real process/thread IDs or device UUIDs
  are collected. Chrome trace export uses synthetic IDs, microsecond offsets,
  inclusive spans and `self_ns` arguments. It writes atomically only when asked;
  the parent directory must exist. No upload occurs. Local output stays out of Git.

## Reproduce paired measurements

From the installed source checkout; this benchmark needs no LLVM toolchain:

```bash
mkdir -p build/runtime-profile
uv run --no-sync python benchmarks/bench_runtime_profile.py --device cpu --output build/runtime-profile/cpu.json
uv run --no-sync python benchmarks/bench_runtime_profile.py --device cuda --output build/runtime-profile/cuda.json
```

Unavailable requested backends fail without fallback. CUDA benchmark telemetry
also requires a working `nvidia-smi` command; `cx.profile()` itself does not.
Defaults are one warmup
and seven measured pairs per case, alternating profiled/unprofiled order, for
1K, 16K, 256K, 1M and 16M float32 elements, followed by a resident
32×64 → 128 → 10 MLP (matmul, LayerNorm, GELU, matmul, softmax).
Resident inputs/weights exist before timing. CUDA end-to-end expressions include
two transfers, two arithmetic operations, a CPU return transfer and temporary
cleanup within the call. CPU copy expressions clone inputs/output explicitly.
Both modes include intermediate allocation and cleanup inside the workload.
Returned-result destruction, profile reserve/export and CPU parity checking are
outside the call timer. Validation exports between samples may affect caches.

Each output is checked against CPU outside timing; any mismatch or truncated
profile fails before atomic report publication, preserving an existing file.
JSON includes individual public-wall and category-self samples, median/p10/p90,
profile elapsed/unattributed durations, event counts, source/build hashes,
runtime versions and CUDA sharing snapshots with process identifiers removed.
The default report omits per-span timelines; `--include-events` adds them for
local inspection. Review reports under the [publication policy](../CONTRIBUTING.md#publish-only-reviewed-evidence).

Instrumented samples include profiler overhead. Their category breakdown cannot
be substituted for unprofiled performance or device timings. Run repeated paired
measurements before making an overhead/optimization claim. A shared desktop GPU
and endpoint process snapshots cannot establish isolation. No other workload is
stopped by this benchmark. Device-event timing, Metal spans, broader allocation
accounting, memory pooling, async execution and measured fusion remain separate
work.

## Verification

### Recorded local paired runs — 2026-10-17

Two CPU runs ([1](validation/runtime-profile-cpu-run1.json),
[2](validation/runtime-profile-cpu-run2.json)) and two CUDA runs
([1](validation/runtime-profile-cuda-run1.json),
[2](validation/runtime-profile-cuda-run2.json)) each passed all 11 cases,
with seven measured pairs and one warmup per case. All profiles were untruncated;
per-sample category self times plus unattributed time equal session elapsed time.
These reviewed compact reports contain no per-span trace dumps.

Environment: Linux x86_64, i7-6700K, GTX 980 Ti sm_52, CUDA Runtime 12.4,
driver 580.178.04. Reports share identical benchmark, Python profiler and loaded
extension hashes. They were taken from the pre-commit worktree based on
`90e2230`; their dirty flag and hashes identify that state, not an already
published commit. Both CUDA runs observed another compute process and a shared
desktop; no process was stopped and these are not isolated performance results.

Representative public-call medians, milliseconds (run 1 / run 2):

| Backend/case | Unprofiled | Profiled |
| --- | --- | --- |
| CPU 16M resident expression | 123.8284 / 123.7882 | 124.0906 / 124.1810 |
| CPU resident MLP | 0.3743 / 0.3686 | 0.3702 / 0.3698 |
| CUDA 16M resident expression | 6.0453 / 6.0472 | 6.0311 / 6.0209 |
| CUDA resident MLP | 0.1587 / 0.1604 | 0.1607 / 0.1619 |

Small differences, including apparently faster instrumented medians, reflect
measurement variation; they do not establish zero overhead, a speedup, device
kernel durations or behavior on another host. Individual samples and p10/p90
are retained. This increment changes observability, not the algorithms.

### Acceptance coverage

`test_runtime_profile.py` covers numerical parity, CPU work attribution, CUDA
copies/frees/memset/waits, generated CPU/CUDA launches, hierarchical accounting,
limits/truncation, exceptions, nesting, thread isolation, safe atomic exports,
paired benchmark timing boundaries and failure-preserving output. Native
backend-contract tests also check span accounting and late TLS destructor safety.
The strict CUDA push gate includes this suite with real LLVM/CUDA requirements;
Metal compatibility is exercised by the existing macOS suite without claiming
Metal instrumentation.
