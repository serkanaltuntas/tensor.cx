# CUDA performance measurement

`benchmarks/bench_cuda.py` measures the synchronous public Python API on the
validated [Nightblade CUDA/MLIR environment](MLIR_CUDA_INTEGRATION_DECISION.md).
It requires real CUDA execution and LLVM 21.1.8; missing prerequisites fail
instead of skipping or selecting CPU. CPU providers are explicit comparisons.

Run from the source checkout after installing the CUDA-enabled package:

```bash
export TENSORCX_LLVM_BIN="$PWD/build/mlir-toolchain/root/usr/lib/llvm-21/bin"
uv run --no-sync python benchmarks/bench_cuda.py --output /tmp/tensorcx-cuda-performance.json
```

Defaults are 31 measured iterations after 5 warmups for each execution case,
and 5 fresh compilation measurements after one compilation warmup. The size
sweep is 1K, 16K, 256K, 1M and 16M float32 elements. A short functional check is:

```bash
uv run --no-sync python benchmarks/bench_cuda.py --sizes 17 --repeats 3 --warmup 1 \
  --compile-repeats 1 --output /tmp/tensorcx-cuda-smoke.json
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

Historical commands below retain their original spelling and target the recorded
revisions. For a current checkout, apply the [naming migration](NAMING.md).

Measurement infrastructure, CPU parity checks and a measured long-row
optimization are implemented. The comparison below establishes a local
Nightblade improvement within its stated shared-desktop limits. It does not
establish performance on other GPUs or an overall workload speedup. The full
product goal remains in [CUDA_PRODUCT_COMPLETION.md](CUDA_PRODUCT_COMPLETION.md).

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
uv run --no-sync python benchmarks/bench_cuda.py --repeats 7 --warmup 2 \
  --compile-repeats 3 --output /tmp/cortex-cuda-shared-load.json
```

## Contiguous axis optimization — 2026-10-03

Rows of at least 256 contiguous elements now use coalesced shared-memory tiles,
parallel transforms and parallel output writes. Lane zero retains the original
left-to-right float32 accumulation order; softmax caches each exponential.
Short rows and strided axes keep the original kernel. Numerical behavior,
including cancellation, NaNs, infinities and constant-row normalization, remains
covered by CPU comparisons. This is an optimization of the static CUDA
primitives, not a change to the MLIR compiler or automatic fusion.

Four full 71-case sweeps used the same script, input seed and configuration:
31 timed samples after 5 warmups, plus 5 compilation samples. All results in
all four runs passed CPU parity. Original JSON samples and provenance are kept
in [before 1](../benchmarks/results/cuda-axis-staging/before-1.json),
[before 2](../benchmarks/results/cuda-axis-staging/before-2.json),
[after 1](../benchmarks/results/cuda-axis-staging/after-1.json) and
[after 2](../benchmarks/results/cuda-axis-staging/after-2.json).
The [complete comparison](../benchmarks/results/cuda-axis-staging/comparison.csv)
includes every provider/case, each run's median/p10/p90 and the mean of the two
medians per build. Speedup below is before mean median divided by after mean
median; it is not an aggregate throughput score.

| Operation | Shape, axis 1 | Before medians, ms (runs 1 / 2) | After medians, ms (runs 1 / 2) | Ratio |
| --- | --- | --- | --- | --- |
| sum | 32×4096 | 0.2060 / 0.2058 | 0.0956 / 0.0852 | 2.28× |
| max | 32×4096 | 1.3480 / 1.3482 | 0.3783 / 0.3649 | 3.63× |
| mean | 32×4096 | 0.1828 / 0.2058 | 0.0862 / 0.0852 | 2.27× |
| softmax | 32×4096 | 3.0387 / 3.1501 | 0.5791 / 0.5715 | 5.38× |
| rmsnorm | 32×4096 | 1.3277 / 1.3513 | 0.2312 / 0.2178 | 5.97× |
| layernorm | 32×4096 | 1.5155 / 1.5512 | 0.2974 / 0.2680 | 5.42× |
| sum | 1×65536 | 1.7491 / 1.7597 | 1.0934 / 1.0684 | 1.62× |
| max | 1×65536 | 17.2910 / 17.3036 | 5.0728 / 5.0321 | 3.42× |
| mean | 1×65536 | 1.7490 / 1.7510 | 0.9713 / 0.9709 | 1.80× |
| softmax | 1×65536 | 38.3963 / 38.4022 | 5.5344 / 5.5064 | 6.96× |
| rmsnorm | 1×65536 | 15.2046 / 15.2072 | 1.0019 / 0.9912 | 15.26× |
| layernorm | 1×65536 | 16.8659 / 16.8736 | 1.6299 / 1.6282 | 10.36× |

The loaded baseline extension SHA-256 starts `dc399436e583`, and the optimized
extension starts `96ae63b48420`; full hashes are in every report. Both pairs
match internally. The first baseline used the clean `40a3940` checkout. The
second baseline captured source edits that had not been rebuilt: its identical
loaded-extension hash establishes that it still measured the old code.
Both optimized runs measured the implementation committed with this report,
while HEAD still named `40a3940`. All four benchmark-script hashes match.
Use `uv run --no-sync` to avoid implicitly changing the installed CUDA build.

There was no concurrent project training process observed in these runs.
An additional GPU process using 282 MiB was present at every
endpoint, so the reports correctly retain the shared-GPU warning. No other
build/test/GPU task from this work ran during measurement. Endpoint GPU
conditions were comparable: start 54–59°C at 1139/3505 MHz (SM/memory), finish
63–64°C at 1328/3304 MHz. These snapshots do not prove exclusive use or fixed
clocks throughout each run. The repeated, large long-row improvements support
this bounded comparison; they must not be generalized to other hardware.

Unchanged paths show noise and some worse observations. Mean medians increased
about 15% for sum on `(32,128)`, 17%/16% for strided softmax/LayerNorm, and 19%
for `256×256×256` matmul. For example, that matmul measured 0.1126/0.1451 ms
before and 0.1495/0.1561 ms after. We have not established the cause of those
changes, and do not claim that all cases are regression-free. The raw samples
and comparison preserve them. MLP resident/end-to-end ratios are only 1.02×
and 1.01×; its short normalization rows do not use the new path, so no MLP
speedup is claimed. Dispatch at width 256 is correctness-tested; these timing
results specifically cover widths 4096 and 65536, not every enabled shape.

Verification of this implementation: **286 CUDA primitive tests passed**,
including 151 new boundary/numerical tests; **4/4 native contracts passed**.
Native contracts and all 151 new Python tests each passed device memcheck and
racecheck with zero errors or hazards. The CUDA/LLVM-required full Python suite
passed **1131 tests**, with **138 expected skips** for unavailable/unsupported
paths. CUDA acceptance is also re-run without skips by the exact-commit
[push gate](CUDA_CONTINUOUS_VALIDATION.md).

```bash
CORTEX_REQUIRE_BACKENDS=cuda uv run --no-sync pytest -q tests/python/test_cuda_primitives.py
CORTEX_REQUIRE_CUDA=1 CORTEX_REQUIRE_MLIR_CUDA=1 uv run --no-sync ctest --test-dir build/cpp-cuda --output-on-failure
CORTEX_REQUIRE_BACKENDS=cuda uv run --no-sync compute-sanitizer --tool memcheck --error-exitcode 1 .venv/bin/python -m pytest -q tests/python/test_cuda_primitives.py -k staged
CORTEX_REQUIRE_BACKENDS=cuda uv run --no-sync compute-sanitizer --tool racecheck --error-exitcode 1 .venv/bin/python -m pytest -q tests/python/test_cuda_primitives.py -k staged
CORTEX_REQUIRE_BACKENDS=cuda CORTEX_REQUIRE_MLIR=1 CORTEX_REQUIRE_MLIR_CUDA=1 uv run --no-sync pytest -q
```

The full-suite and benchmark commands use the `CORTEX_LLVM_BIN` setting above.
