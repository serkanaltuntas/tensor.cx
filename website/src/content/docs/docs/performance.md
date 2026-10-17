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

A cache hit still checks toolchain identity and loads CPU/CUDA modules. On the
measured Linux host, small-kernel CPU warm calls were slower than uncached calls;
first hits in new processes paid for dependency hashing again. CUDA warm calls
were faster in those local runs. These observations do not establish a speedup
on another host, workload or GPU. Metal was not measured in that Linux run.

See [cache controls](/docs/experimental/) and the
[detailed method and results](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/COMPILATION_CACHE.md#compile-and-launch-profiling).
The existing CUDA benchmark keeps compilation cold. Broader runtime tracing
and automated kernel fusion remain separate work.

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
