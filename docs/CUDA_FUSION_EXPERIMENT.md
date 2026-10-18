# Manual CUDA elementwise fusion experiment

This experiment compares explicitly authored kernels using the existing
experimental DSL. It adds no automatic tensor-expression fusion, graph capture,
runtime API, compiler pass, allocation policy or asynchronous execution.

## Reproduce

Use the validated CUDA/LLVM environment from
[CUDA performance](CUDA_PERFORMANCE.md). Run from a CUDA-enabled source checkout:

```bash
mkdir -p build/cuda-fusion
export TENSORCX_LLVM_BIN="$PWD/build/mlir-toolchain/root/usr/lib/llvm-21/bin"
uv run --no-sync python benchmarks/bench_cuda_fusion.py --output build/cuda-fusion/run-1.json
uv run --no-sync python benchmarks/bench_cuda_fusion.py --output build/cuda-fusion/run-2.json
```

The LLVM path is an example checkout-local installation. The tools must be the
supported LLVM 21.1.8 toolchain. CUDA, the guarded generated-kernel environment,
LLVM and `nvidia-smi` are required; missing prerequisites fail without fallback.
A functional smoke run uses `--sizes 17 --repeats 6 --warmup 1
--diagnostic-repeats 1`. Smoke results are not optimization evidence.

Defaults: 1K, 16K, 256K, 1M and 16M float32 elements; six warmup rounds, 30
measured rounds and three separate diagnostic calls per provider/case. Measured
rounds must be a positive multiple of six: all six provider permutations execute
equally often, balancing every provider position. Inputs use fixed-seed bounded
random values in `[-1, 1]`; these are workloads, not a complete numerical domain.

## Compared work

| Workload | Expression | Separate kernels | Manually fused kernels |
| --- | --- | --- | --- |
| Two operations | `(a + b) * b` | 2 | 1 |
| Four operations | `((a + b) * b - a) * b` | 4 | 1 |

Each expression has three providers: ordinary native tensor operations,
separately compiled MLIR add/multiply/subtract calls, and one manually authored
MLIR kernel. The MLIR separate provider helps distinguish operation grouping
from switching between native and generated launch paths. It does not isolate
every factor: grouping also removes Python validation, private allocations,
template copies and synchronous waits between operations.

Both boundaries return a new result:

- **Resident:** inputs are already on the GPU; output allocation, dispatch,
  existing synchronization and intermediate cleanup inside the call are timed.
- **End to end:** transfer both CPU inputs, execute, copy the result back to CPU
  and release temporary device tensors inside the call. Provider wrapper setup
  is included equally in all three end-to-end providers.

Every generated launch preserves its left-input output template and allocates a
private result. The original reports below include a full template copy per
launch. Current v2 artifacts [omit that copy for proved full writes](CUDA_OUTPUT_COPY_ELISION.md);
legacy, partial and zero-work behavior remains conservative. No in-place or
suffix-preservation contract is weakened. This CLI does not enable a memory pool.

## Measurement and safety boundaries

Public-call wall samples are **unprofiled**. Compilation/module loading,
validation and returned-result destruction are outside the timers. All five
generated kernels compile before measurement with `cache=False`; OS/driver
caches are retained. CPU parity is checked after every warmup, measured and
diagnostic result (`rtol=1e-6`, `atol=1e-6`, NaNs rejected), and resident inputs
are checked for mutation after each case. Validation transfers may affect caches.

After each wall measurement, separate `cx.profile(device_timing=True)` calls
record host category self times, call/byte records and CUDA event intervals.
Diagnostics must be complete and contain exactly the expected 2/4/1 kernel
intervals. Event errors, truncation, unexpected counts or wrong outputs fail
before atomic report publication, preserving an existing report.

Diagnostic intervals are not added to wall samples or used for speed ratios.
They include submission gaps, contention and event instrumentation; they are
not isolated instruction time. Copies have host records but no GPU event
timings. Nested host byte records must not be summed into physical traffic.
Host category self times share one host clock; GPU intervals use a separate
clock. See [the profiling contract](RUNTIME_PROFILING.md).

Reports retain individual wall samples, medians/p10/p90, per-round provider
orders, separate diagnostics, source/extension/helper hashes, tool versions and
GPU snapshots. Ratios are **separate-provider median / fused median** within
the same case; values below one mean the fused public call was slower. Keep
both complete runs and all regressions. Known identifiers are removed by the
[publication exporter](../CONTRIBUTING.md#publish-only-reviewed-evidence);
manual inspection and secret scans remain required.

GPU snapshots at the endpoints cannot establish exclusive execution. Nightblade
shares the GPU with its desktop and other compute workloads. Do not terminate
other workloads, extrapolate to another GPU or claim an overall model speedup.

## Local evidence and decision

Two [complete reports](validation/cuda-fusion-1.json) / [second run](validation/cuda-fusion-2.json)
each passed all 20 cases, all three providers, 30 measured rounds (five of each
permutation) and three diagnostics per provider/case. They share benchmark,
helper, Python-profiler and loaded-extension hashes, and use the pre-commit dirty
worktree based on `0152936`. Editorial changes progressed between runs; the
tracked-diff hashes are retained individually. The runtime was unchanged.

Hardware: Linux x86_64, GTX 980 Ti sm_52, CUDA 12.4, driver 580.178.04,
LLVM 21.1.8. **Both runs observed other compute processes.** Endpoint snapshots
and repeatability do not establish an isolated GPU or causal attribution to one
removed cost. The reports retain every sample and p10/p90 spread.

Medians in milliseconds, **run 1 / run 2**; every case is included:

| Elements | Ops | Boundary | Native separate | MLIR separate | MLIR fused |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1,024 | 2 | resident | 0.033317 / 0.033280 | 0.136351 / 0.134577 | 0.074045 / 0.072999 |
| 1,024 | 2 | end_to_end | 0.080156 / 0.080574 | 0.186035 / 0.183827 | 0.123473 / 0.122269 |
| 1,024 | 4 | resident | 0.059701 / 0.059848 | 0.256668 / 0.253319 | 0.082499 / 0.081893 |
| 1,024 | 4 | end_to_end | 0.108669 / 0.106964 | 0.334973 / 0.302258 | 0.134759 / 0.131537 |
| 16,384 | 2 | resident | 0.034641 / 0.034653 | 0.137921 / 0.136748 | 0.075804 / 0.075123 |
| 16,384 | 2 | end_to_end | 0.113905 / 0.127595 | 0.218995 / 0.287515 | 0.157526 / 0.165372 |
| 16,384 | 4 | resident | 0.061205 / 0.062117 | 0.260062 / 0.263697 | 0.084975 / 0.085104 |
| 16,384 | 4 | end_to_end | 0.141526 / 0.142178 | 0.343541 / 0.348272 | 0.167297 / 0.167894 |
| 262,144 | 2 | resident | 0.261900 / 0.265620 | 0.432457 / 0.434538 | 0.231589 / 0.235878 |
| 262,144 | 2 | end_to_end | 1.369169 / 1.361556 | 1.513158 / 1.529405 | 1.325331 / 1.348075 |
| 262,144 | 4 | resident | 0.587174 / 0.628358 | 0.783614 / 0.817931 | 0.242291 / 0.251611 |
| 262,144 | 4 | end_to_end | 1.656332 / 1.631276 | 1.892905 / 1.879639 | 1.337638 / 1.329844 |
| 1,048,576 | 2 | resident | 0.610069 / 0.584055 | 0.829630 / 0.725138 | 0.458429 / 0.456878 |
| 1,048,576 | 2 | end_to_end | 3.052122 / 3.066904 | 3.326573 / 3.276255 | 2.999290 / 3.023683 |
| 1,048,576 | 4 | resident | 1.410624 / 1.405701 | 1.600661 / 1.608197 | 0.457356 / 0.456199 |
| 1,048,576 | 4 | end_to_end | 3.937425 / 3.952060 | 4.299819 / 4.175502 | 3.072462 / 3.063816 |
| 16,777,216 | 2 | resident | 6.053418 / 6.063448 | 7.216961 / 7.213969 | 5.625119 / 5.631423 |
| 16,777,216 | 2 | end_to_end | 66.178630 / 65.698734 | 67.186554 / 67.332443 | 65.832570 / 65.495807 |
| 16,777,216 | 4 | resident | 13.654308 / 13.621080 | 16.443376 / 16.418933 | 5.626981 / 5.626030 |
| 16,777,216 | 4 | end_to_end | 74.015349 / 73.093037 | 76.945055 / 76.127259 | 65.829641 / 65.950817 |

In these runs, the 1M four-op resident expression had a native/fused median
ratio of **3.084 / 3.081**. Its transfer-inclusive ratios were only
**1.282 / 1.290**. At 16M, four-op ratios were **2.427 / 2.421** resident and
**1.124 / 1.108** end to end; two-op end-to-end times were nearly equal.
Conversely, the fused 1K two-op resident call was **2.222 / 2.194 times slower**
than native, and the four-op 1K call was **1.382 / 1.368 times slower**.
The same grouping is faster than separate MLIR calls here, but this does not
justify choosing MLIR fusion over native execution for every expression/size.

All three diagnostic samples in both runs agreed on these **1M four-op resident**
host records. These are instrumented call/byte counts, not timing speed ratios:

| Provider | Kernel intervals | Allocations | Template copies | Template bytes | Wait calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native separate | 4 | 4 | 0 | 0 | 4 |
| MLIR separate | 4 | 4 | 4 | 16,777,216 | 8 |
| MLIR fused | 1 | 1 | 1 | 4,194,304 | 2 |

**Decision:** retain explicit manual fusion for suitable workloads; keep normal
tensor dispatch unchanged. There is no new automatic fusion policy. A next
local candidate was eliminating the generated output-template copy only when a
full overwrite can be proved. That follow-up is now implemented with its own
[ownership tests and before/after measurements](CUDA_OUTPUT_COPY_ELISION.md).
The original results above remain pre-optimization evidence. Automatic graph
fusion, broader workload acceptance and another NVIDIA architecture remain open.

## Verification

`tests/python/test_fusion_benchmark.py` checks CPU/CUDA workload parity, empty
and tail sizes, input/template immutability, timing boundaries and balanced
provider order. Real CUDA CLI tests check report generation, missing tools,
diagnostic truncation and preservation of an existing report. Pure policy tests
also reject incorrect output, incomplete diagnostics and unexpected launch
counts. The exact-commit CUDA gate includes this file; the LLVM CPU CI selection
executes its CPU provider cases rather than silently deselecting the file.

An independent NumPy float32 oracle passed 28 CPU/CUDA checks, including both
expressions, all providers and zero/full/partial generated launches with an
aliased left-input template. Caller bits, private output and suffix bits were
preserved. This functional evidence and the existing arithmetic acceptance
suites do not establish a new numerical domain or another GPU architecture.
