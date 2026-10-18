---
title: Performance
description: Reproduce measurements and interpret results within their hardware, workload, and timing boundaries.
---

tensor.cx prioritizes correctness and visible execution boundaries. There
is no universal speedup claim. Small GPU operations can be dominated by launch
and transfer costs, and results depend on shape, backend, and hardware.

## Start with a functional measurement

From an installed source checkout:

```bash
uv run --no-sync python benchmarks/bench_elementwise.py --sizes 1024 --repeats 2
uv run --no-sync python benchmarks/bench_copy.py --sizes 1024 --repeats 2
uv run --no-sync python benchmarks/bench_matmul.py --sizes 16x16x16 --repeats 2
```

These short runs check the measurement workflow. They are not a basis for a
performance comparison.

## CUDA and MLIR measurements

`benchmarks/bench_cuda.py` separately measures compilation, copies, primitives,
explicitly authored expression kernels, and an MLP pipeline. It requires real
CUDA execution and LLVM 21.1.8 and validates each result against CPU outside the
timed region. Read the complete
[method and environment setup](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/CUDA_PERFORMANCE.md)
before running it.

Recorded optimization results concern specific long contiguous rows on one
sm_52 GPU shared with a desktop. They do not establish a speedup for other GPUs
or an overall workload improvement. The engineering report keeps the raw
measurement boundaries, medians, variation, and regressions together.

## Manual CUDA fusion experiment

`benchmarks/bench_cuda_fusion.py` compares two- and four-operation expressions
through native tensor calls, separate MLIR kernels and one manually authored
MLIR kernel. It balances all six provider orders and checks every output against
CPU. Resident and transfer-inclusive cases use unprofiled public-call wall
times; separate host/CUDA-event diagnostics verify the expected kernel counts.

Generated results remain private and copy an output template per launch.
Grouping can reduce calls, allocations, copies and waits together, so this is
not an isolated kernel-time comparison or automatic tensor-expression fusion.
The GPU is shared with the desktop. Read the
[complete experiment and retained results](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/CUDA_FUSION_EXPERIMENT.md)
for measured limits, variation and regressions.

In two shared-GPU runs, the 1M four-operation resident case had a native/fused
median ratio of about 3.08, falling to about 1.28–1.29 with transfers included.
The fused 1K two-operation resident call was about 2.19–2.22 times slower than
native. These observations keep ordinary tensor dispatch unchanged and do not
establish performance on another GPU or model.

## Compilation cache measurements

Experimental CPU/CUDA/Metal compilation reuses private local code artifacts,
including across process restarts. Measure public compile and launch latency,
with optional separate compilation profiles:

```bash
mkdir -p build/compilation-profile
uv run --no-sync python benchmarks/bench_compilation_cache.py --target cpu --profile --output build/compilation-profile/cpu.json
```

Select `cuda` or `metal` only on the corresponding supported host. The report
separates the initial miss, alternating uncached/warm calls, first hits in fresh
processes, and resident public launches from 1K to 16M elements. Every artifact
and launch is checked against CPU outside timing. The profiler runs separately;
its nested cumulative times overlap and cannot be added or used as benchmark
samples. These are synchronous public API timings, not device-only kernel time.

A cache hit still checks toolchain identity and loads CPU/CUDA modules. The
initial Linux measurements found small-kernel CPU warm calls slower than
uncached calls. Bounded parallel execution of the same fresh linker queries
subsequently reduced that CPU compile latency; no identity check was removed.
First hits in new processes still pay for dependency hashing again. CUDA warm
calls were faster than uncached calls in those local runs. These observations do not establish a speedup
on another host, workload or GPU. Metal was not measured in that Linux run.

See [cache controls](/docs/experimental/) and the
[detailed method and results](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/COMPILATION_CACHE.md#compile-and-launch-profiling).
The existing CUDA benchmark keeps compilation cold. Automated kernel fusion
remains separate work.

## Native runtime profiling

Use opt-in host profiling around a CPU or CUDA workload:

```python
import tensorcx as cx

x = cx.ones((1024,), device="cpu")
with cx.profile() as p:
    result = (x + x) * x
print(p.summary())
p.export_chrome_trace("runtime-trace.json")
```

Spans distinguish allocation, copy API calls, native execution and existing
waits. Add `self_ns` only: inclusive times overlap. CUDA launch self time excludes
nested waits and is host overhead; neither value is device-only kernel timing.
Copies and memset can return before device work completes. No synchronization
is added. CPU imports, clones and NumPy exports combine allocation and copy.
Metal is not instrumented, and reports explicitly declare that limitation.

Contexts are current-thread, single-use and cannot nest on the same thread.
The default cap is 10,000 spans; `report()["complete"]` means no spans were
dropped and no requested device measurement failed, not full backend coverage.
Export is explicit and local, with synthetic
process/thread IDs and no tensor values. See the
[complete contract](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/RUNTIME_PROFILING.md).

For alternating profiled/unprofiled calls and CPU parity outside timing:

```bash
mkdir -p build/runtime-profile
uv run --no-sync python benchmarks/bench_runtime_profile.py --device cpu --output build/runtime-profile/cpu.json
```

Choose `cuda` on a supported host with `nvidia-smi` for benchmark telemetry;
no LLVM toolchain is needed. The profiler itself does not require `nvidia-smi`.
Defaults cover
five sizes from 1K to 16M and a resident MLP. Profile setup, validation and
returned-result cleanup are outside the call timer; intermediate allocation,
existing waits and temporary cleanup inside the workload remain included.
Instrumented runs include profiler overhead; use the paired unprofiled samples
for ordinary latency. Metal profiling and CUDA transfer timing remain open work.

### CUDA kernel event intervals

On the validated CUDA host, request device timing explicitly:

```python
x = cx.ones((1024,), device="cuda")
with cx.profile(device_timing=True) as p:
    result = (x + x) * x
print(p.device_summary())
```

Static CUDA and generated MLIR kernel launches produce separate `device_events`
with `elapsed_ms`, operation labels and a `host_event_id` correlation. CUDA
events bracket the launch before its existing wait; no extra synchronization is
added. These intervals can include submission gaps and GPU contention, so they
are not isolated instruction times. Copies, memset, allocations and Metal are
outside device timing. Host `summary()` stays unchanged; never add GPU durations
to host self time. Chrome trace events remain on the host clock, with device
intervals retained separately in metadata.

The host and device lists each have the `max_events` limit. Failed device
measurements have `elapsed_ms=None` and an error label; `device_complete` becomes
false on errors or device drops, while `host_complete` only tracks host drops.
Inspect coverage even when a report is complete. CPU/Metal-only work produces
no CUDA intervals. Timing defaults to disabled, and event handles are cleaned
up within each synchronous launch.

Use `--device-timing` with `--device cuda` in the benchmark above to collect a
separate device-time breakdown. Instrumentation overhead remains in profiled
wall samples; compare paired unprofiled samples when evaluating latency.
See the [numerical, timing and resource boundaries](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/RUNTIME_PROFILING.md#optional-cuda-device-intervals).

## Bounded CUDA memory reuse

```python
x = cx.ones((1024,), device="cuda")
with cx.memory_pool("cuda", max_cached_bytes=64 * 1024 * 1024,
                    max_cached_blocks=64) as pool:
    for _ in range(10):
        result = (x + x) * x
        del result
    print(pool.stats())
```

Here `x` is an existing CUDA tensor. This opt-in current-thread scope caches
idle exact-size owned buffers. Operations remain synchronous. The limits cover
cached payload, not live tensors or total VRAM. `pool.trim()` frees idle blocks;
exit stops reuse and releases cached storage while live tensors remain valid.
CPU/Metal pools and asynchronous execution are not implemented. DLPack-exposed
storage is permanently excluded to preserve foreign-stream ownership boundaries.

Use `benchmarks/bench_memory_pool.py --pool off` and `--pool on` to compare
public-call timings with CPU parity outside timing. Defaults cover five sizes
and an MLP; native profiles run separately from latency samples. At 16M float32
elements, the default 64 MiB cache cannot hold two simultaneous outputs. Gains
are workload-dependent and cost retained memory. See the
[full contract, method and evidence](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/CUDA_MEMORY_POOL.md).

CUDA `empty` now skips the CPU buffer and host upload, both with and without
the pool. This removes those calls; it does not guarantee an overall speedup.
The older pool measurements include CPU staging, while the
[direct-allocation comparison](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/CUDA_EMPTY.md)
records the changed factory path separately. `empty` contents remain unspecified.

## A useful report includes

- Exact source revision and the build actually loaded.
- GPU model and architecture, OS, driver, toolkit, and Python versions.
- Shapes, dtype, axis, random seed, warmups, and sample count.
- Whether allocation, transfers, compilation, and synchronization are timed.
- CPU parity, median latency, spread, and comparable repeated runs.
- Other workloads sharing the device and any measurement limitations.

Keep raw local reports private until reviewed. Use the repository's
[publication guidance](https://github.com/serkanaltuntas/tensor.cx/blob/main/CONTRIBUTING.md#publish-only-reviewed-evidence)
to remove personal paths and persistent device identifiers before sharing.
