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
[method and environment setup](https://github.com/serkanaltuntas/cortex-runtime/blob/main/docs/CUDA_PERFORMANCE.md)
before running it.

Recorded optimization results concern specific long contiguous rows on one
sm_52 GPU shared with a desktop. They do not establish a speedup for other GPUs
or an overall workload improvement. The engineering report keeps the raw
measurement boundaries, medians, variation, and regressions together.

## A useful report includes

- Exact source revision and the build actually loaded.
- GPU model and architecture, OS, driver, toolkit, and Python versions.
- Shapes, dtype, axis, random seed, warmups, and sample count.
- Whether allocation, transfers, compilation, and synchronization are timed.
- CPU parity, median latency, spread, and comparable repeated runs.
- Other workloads sharing the device and any measurement limitations.

Keep raw local reports private until reviewed. Use the repository's
[publication guidance](https://github.com/serkanaltuntas/cortex-runtime/blob/main/CONTRIBUTING.md#publish-only-reviewed-evidence)
to remove personal paths and persistent device identifiers before sharing.
