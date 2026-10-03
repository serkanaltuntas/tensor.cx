# CUDA performance measurement

`benchmarks/bench_cuda.py` measures the synchronous public Python API on the
validated [Nightblade CUDA/MLIR environment](MLIR_CUDA_INTEGRATION_DECISION.md).
It requires real CUDA execution and LLVM 21.1.8; missing prerequisites fail
instead of skipping or selecting CPU. CPU providers are explicit comparisons.

Run from the source checkout after installing the CUDA-enabled package:

```bash
export CORTEX_LLVM_BIN="$PWD/build/mlir-toolchain/root/usr/lib/llvm-21/bin"
uv run python benchmarks/bench_cuda.py --output /tmp/cortex-cuda-performance.json
```

Defaults are 31 measured iterations after 5 warmups for each execution case,
and 5 fresh compilation measurements after one compilation warmup. The size
sweep is 1K, 16K, 256K, 1M and 16M float32 elements. A short functional check is:

```bash
uv run python benchmarks/bench_cuda.py --sizes 17 --repeats 3 --warmup 1 \
  --compile-repeats 1 --output /tmp/cortex-cuda-smoke.json
```

Every timed result is compared with its CPU result outside the timed region.
No report is written until every comparison passes. Existing output files are
left untouched on a failed run; use a new filename for each run and check the
exit status before consuming a report. Raw samples, median, p10 and p90 are
stored in milliseconds. The script alternates provider order each iteration.
This reduces ordering bias but does not make the experiment immune to thermal,
clock, memory allocator, cache or concurrent workload effects.

## Measurement boundaries

| Case | Included |
| --- | --- |
| Compile | Public `compile`, toolchain checks, external MLIR lowering, PTX extraction and fresh native module load; Python IR already parsed |
| One-element launch | Public compiled launch validation, private output allocation/template copy, synchronous execution and returned tensor construction |
| Host/device copy | Public tensor transfer, destination allocation and synchronous copy; host NumPy validation excluded |
| Primitive | Public dispatch, output allocation, native operation and synchronization; resident inputs |
| Expression resident | `(a + b) * b` with resident inputs, either two primitive calls or one explicitly authored MLIR kernel |
| Expression end to end | Two CPU-to-CUDA transfers, expression, output transfer to CPU, temporary tensor cleanup within the call |
| MLP resident | Matmul → LayerNorm → GELU → matmul → softmax, resident inputs/weights, intermediate allocations and synchronization |
| MLP end to end | Input and both weight transfers, complete MLP, output transfer to CPU, temporary tensor cleanup within the call |

Returned-result destruction and parity checking happen after each timer stops.
Temporary destruction *inside* an API/workload call remains included. These
are public API latencies, not CUDA-event kernel-only timings. One-element
launch is an overhead probe, not a measurement of bare `cuLaunchKernel`.
Compilation keeps OS and driver caches intact; it is not a cold-start benchmark.
Validation copies results to host between samples and may affect cache state.

The axis cases include `(32,128)` and `(32,4096)` contiguous rows, axis 0 on
`(32,4096)`, and a long `(1,65536)` slice for every reduction/normalization.
Matmul includes workload-sized, tile-tail and `256×256×256` cases. All current
activation primitives are included. These inputs validate the measurement
workflow; numerical edge coverage remains in the primitive acceptance suite.

## Reproducibility and interpretation

Reports contain seed/configuration, source revision and worktree status,
tracked-diff hash, benchmark source hash, loaded extension hash, Python/NumPy,
OS, nvcc and LLVM versions, GPU UUID/driver/architecture, memory use,
temperature, clocks, power, utilization and compute-process snapshots.
Only PIDs, memory and GPU UUIDs are collected for other compute processes;
executable paths and command lines are not collected. CUDA visibility/order
environment values are recorded explicitly. Reports from different source or
extension hashes must not be treated as the same build.

GPU snapshots are taken at the beginning and end. They do not prove exclusive
use throughout the interval. The desktop shares this GPU; establish an idle
compute window separately before claiming an optimization result. Never stop
another project's job merely to improve benchmark numbers. A warning records
other compute processes observed at either snapshot. Its absence alone does
not prove isolation, and different GPUs listed by `nvidia-smi` may not all be
used by this benchmark.

For a measured improvement, retain at least two complete runs under comparable
load, temperature and clocks. Compare the same shape, resident/end-to-end
boundary, build and output correctness. Report medians and spread, including
regressions; never extrapolate a short smoke test to a general speedup claim.
The expression comparison demonstrates **manual operation fusion** available
through the existing DSL. The compiler does not discover or fuse ordinary
tensor expressions automatically. The fused API still allocates and copies
an output template, so it is not guaranteed faster at every size.

## Acceptance status

Measurement infrastructure and CPU parity checks are implemented. Performance
acceptance remains open until reproducible measurements support actual
improvements. The first full sweep encountered another project's active GPU
compute job; it cannot establish an uncontended baseline or a speedup claim.
The full product goal, including measured optimization, remains in
[CUDA_PRODUCT_COMPLETION.md](CUDA_PRODUCT_COMPLETION.md).

2026-10-03 verification: **11 benchmark tests passed**, including real CUDA
report generation, required CUDA/LLVM failures, rejection of incorrect output,
provider ordering and exclusion of validation/result destruction from timers.
The final full-size functional sweep passed CPU parity in **71 cases**, with
7 samples after 2 warmups (compilation: 3 samples after 1 warmup). Its
[retained shared-load report](../benchmarks/results/cuda-nightblade-shared-load.json)
records other compute processes at both endpoints. This report demonstrates
measurement coverage and successful execution only; it is deliberately not
used to claim a speedup. It used the `76397f4` native runtime and the benchmark
source identified by its embedded SHA-256, before this benchmark was committed.

```bash
CORTEX_REQUIRE_MLIR_CUDA=1 uv run pytest -q tests/python/test_cuda_benchmark.py
uv run python benchmarks/bench_cuda.py --repeats 7 --warmup 2 \
  --compile-repeats 3 --output /tmp/cortex-cuda-shared-load.json
```
