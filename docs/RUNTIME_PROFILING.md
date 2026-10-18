# Native runtime profiling

`cx.profile()` records bounded, current-thread host spans for CPU and CUDA.
It separates native allocation, copies, execution and existing waits without
adding device synchronization. `cx.profile(device_timing=True)` additionally
records CUDA kernel event intervals on the validated local sm_52 host. Metal
execution is **not instrumented**; every report declares that limitation.
This does not add graph optimization, streams or an asynchronous execution API.

Profiling reports the operations actually performed. The
[generated CUDA full-write optimization](CUDA_OUTPUT_COPY_ELISION.md) omits a
template copy and its wait for eligible v2 artifacts. Its final kernel wait is
retained; partial, zero-work and legacy launches keep their conservative path.
Historical wait counts below describe their original runtime revision.

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

## Optional CUDA device intervals

```python
x = cx.ones((1024,), device="cuda")
with cx.profile(device_timing=True) as p:
    result = (x + x) * x
print(p.device_summary())  # elapsed_ms, measured_calls, failed_calls
assert p.report()["device_complete"]
```

`device_timing` accepts a Python bool and defaults to false. CPU-only builds and
CPU/Metal work remain usable with the flag enabled, but produce no device
intervals for those backends. An empty device list is not evidence of device
execution: inspect `device_coverage` and the actual interval list.

Each nonempty static CUDA launch wrapper and generated MLIR CUDA launch records
a pair of timing-enabled Driver events on the backend's existing default stream,
inside its primary-context guard. The stop event is submitted after a successful
launch **before the existing completion wait**; elapsed time is read only after
that wait succeeds. Event handles are destroyed within the call, including early
returns. No events, GPU clock reads or extra waits occur for disabled timing,
zero-launch work or an exhausted device-record cap. Only two event handles are
live per instrumented launch, rather than retaining handles until session exit.
The Driver API preserves the Runtime launch-error slot and original launch/wait
return values. See NVIDIA's [event contract](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__EVENT.html).

`device_events` is separate from host `events`. Each row has a session-local
`id`, `host_event_id` correlation, static backend/name/optional operation labels,
`elapsed_ms` and `error`. A correlated host event may be absent after host
truncation (`host_event_id=0`). Durations retain the CUDA API's float millisecond
precision; zero is a valid measured duration at timer resolution. Device rows
have **no host timestamp**. Never add them to host self/elapsed/unattributed
time or treat them as calibrated positions on the host timeline.

`device_summary()` groups these intervals independently, retaining measured and
failed counts; its totals are partial if records failed or were dropped. Host
`summary()` is unchanged. Chrome trace `X` events remain host-only; separate
device intervals and coverage stay in trace metadata. No artificial GPU lane
or common host/GPU epoch is invented. The host report schema stays at version 1
with additive optional device fields and unchanged host clock semantics.

These are **device event intervals around submission**, not isolated instruction
times. Host submission gaps, GPU scheduling/contention and instrumentation can
affect them. Copies, memset, allocation, compilation, module/context setup and
external library/Metal work do not create device intervals. Existing copy/API
and wait spans remain host measurements. There is no transfer bandwidth,
overlap, asynchronous pipeline, clean-GPU or additional-architecture claim.

`max_events` independently bounds each of the host and device record lists.
Device capacity is reserved only when requested. `dropped_device_events` counts
omitted device records. Event create/record/read/destroy failures or failed
operations yield a static error label and `elapsed_ms=None`, never a fabricated
zero. Instrumentation errors do not replace the tensor operation's status.
Cleanup is attempted for both acquired handles even after an earlier failure;
a Driver cleanup error is reported rather than hidden. `device_errors` counts
failed recorded intervals; `device_complete` is false on errors/drops and is
`None` when timing was not requested. `host_complete` only describes host drops;
overall `complete` requires both requested recording domains to be untruncated
and all recorded device measurements to succeed. None implies full coverage.

## Host timing and coverage contract

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
| `memory_pool` | Opt-in CUDA acquire/return bookkeeping; actual runtime allocations/frees remain nested spans |
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

`host_complete` means **no instrumented host spans were dropped**, not complete
backend coverage. Inspect `coverage` even when it is true. A Metal-only context can have
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
stopped by this benchmark. Bounded opt-in [CUDA memory pooling](CUDA_MEMORY_POOL.md) is implemented.
Optional CUDA kernel event intervals are implemented as described above.
Metal/device-copy timing, broader accounting and async execution remain separate
work. The [manual fusion experiment](CUDA_FUSION_EXPERIMENT.md) uses separate
diagnostics and unprofiled wall samples. Add `--device-timing` with `--device cuda` to the paired
benchmark to retain a separate `device_time_breakdown` and device-record counts.
This measures instrumentation overhead too; the paired unprofiled samples remain
the public-call latency baseline. `--include-events` additionally retains the
separate per-interval results for local inspection. Incomplete device measurements
fail the benchmark before report publication.

## Verification

### CUDA event intervals — 2026-10-18

Local Linux x86_64/Toolkit 12.4/GTX 980 Ti sm_52 verification covers all 22
static launch wrappers and the generated MLIR launch, float32 and exact int32
parity, zero work, separate clock/cap accounting, caller preservation, repeated
launches, thread isolation, context restoration, exceptions and safe exports.
Native tests inject every event API failure, invalid/zero durations and failed
launch/completion paths, require both cleanup attempts, and verify that disabled
or capacity-exhausted timing makes no event API calls. A separate native test
also preserves a deliberately seeded Runtime error; it runs on a real GPU,
outside the memory/race-contract executable. Sanitizer error flags are unchanged.

Focused profiler/gate verification passed **97 tests**; the complete Linux
CPU/CUDA/LLVM regression passed **9,227 tests with 226 expected platform/capability
skips**. An initial full run was interrupted by SIGTERM before completion and was
excluded; the separate-session rerun completed. Ten option/report/benchmark
policy tests passed with CUDA and LLVM hidden. Native contracts passed **5/5**;
native memcheck/racecheck reported zero errors/hazards. Python sanitizer selection
passed **38 tests** per tool with zero errors/hazards, including the generated
launch. Those Python sanitizer commands disable API-error logging for the
loader's deliberate out-of-range parameter-count probe; memory/race checking
remains enabled. The mandatory native sanitizer commands keep their original
error flags.

Independent GPU QA also passed pooled float32 and exact int32 work, generated
prefix/output-alias/private-copy/suffix preservation, unchanged wait lists,
zero-work/copy/memset exclusions, event-to-host correlation and two-worker
isolation. After test tensors were released and the pool exited, its cached and
in-use counters were both zero.

Four reviewed paired reports compare host-only profiling
([1](validation/device-profile-host-1.json), [2](validation/device-profile-host-2.json))
and host plus CUDA events
([1](validation/device-profile-events-1.json), [2](validation/device-profile-events-2.json)).
Each uses three element counts (1K, 256K, 1M), resident/end-to-end expressions and
the resident MLP: seven cases, seven measured pairs and one warmup pair per
case. All outputs passed CPU parity and all profiles were complete. These runs
share benchmark/profiler/native-extension hashes and come from the pre-commit
worktree based on `c0bf17c`; they are not measurements of an already-published
commit. Other compute processes and the desktop shared the GPU in every run.

Public-call medians in milliseconds, run 1 / run 2:

| Case | Host-only unprofiled | Host-only profiled | Event-mode unprofiled | Event-mode profiled |
| --- | --- | --- | --- | --- |
| 1K resident expression | 0.033955 / 0.031540 | 0.032841 / 0.032427 | 0.032306 / 0.032988 | 0.042820 / 0.042533 |
| 1M resident expression | 0.530590 / 0.527915 | 0.514859 / 0.510768 | 0.532716 / 0.532937 | 0.525527 / 0.522480 |
| Resident MLP | 0.148914 / 0.146656 | 0.150604 / 0.148389 | 0.146608 / 0.148348 | 0.171532 / 0.171866 |

Per-case/per-operation event intervals are retained in the separate device
breakdown. Event creation/record/read/destruction has visible overhead for small
workloads; the apparently lower instrumented 1M medians illustrate measurement
variation rather than a speedup. These samples do not establish zero overhead,
isolated kernel timing, calibrated host/GPU clocks or another GPU's behavior.

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
