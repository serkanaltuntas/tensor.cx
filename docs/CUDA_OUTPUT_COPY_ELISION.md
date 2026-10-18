# Generated CUDA output-template copy elision

Date: 2026-10-18. Follow-up to the
[manual fusion experiment](CUDA_FUSION_EXPERIMENT.md). This removes redundant
work from guarded generated CUDA kernels without changing the public API or
normal tensor dispatch. It does not implement automatic fusion or async work.

## Eligibility and ownership

The native loader accepts an exact versioned producer contract:

```text
tensorcx.cuda.v2|linux-x86_64|llvm-21.1.8|sm52|ptx78|<operation>-f32-v1|<kinds>|<output>|<guard>|write-per-thread-no-output-read-v1
```

The Python CUDA validator already requires a canonical guarded global index,
same-index input loads, one unconditional final output store per active thread,
and no output reads. General branches, offset accesses and extra stores are
rejected before toolchain/device access. Only validated IR receives this tag.
The native loader checks the exact manifest, target, entry and Driver parameter
metadata. This tag is a **trusted producer promise**, not a native PTX-body
proof or sandbox for arbitrary PTX. The private artifact trust boundary is
unchanged; never load downloaded or untrusted cache entries.

The fast path requires all three conditions: the v2 contract, nonzero logical
work and `thread_count == numel(output.shape)`. It always allocates a new private
result. It omits the full device-to-device output-template copy and that copy's
completion wait. **Only the output argument slot is rebound to the result**;
input slots that alias the template retain the original initialized storage.
Rebinding those inputs to uninitialized result memory would be incorrect.

Partial and zero-work launches, and every legacy v1 artifact, retain full
template copying and the existing same-range input-alias remapping. Untouched
suffix bits, including NaN payloads and signed zeros, are preserved. Zero work
still validates arguments and returns a distinct value without launching zero
blocks; empty buffers require no byte copy. Legacy trusted PTX may contain
multiple stores/read-after-store and therefore cannot use the v2 promise.

Kernel completion still waits before publishing the result. Context ownership,
foreign-current-context restoration, overlap/shape/dtype checks, module lifetime
and failure cleanup are unchanged. There is no new stream or delayed error
delivery. The existing default stream orders input work before the generated
launch; see the [CUDA 12.4 default-stream contract](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/stream-sync-behavior.html).
Private output allocation and final synchronization remain real costs.

## Compatibility

Entry names, four pointer/u32 argument slots, `KernelSignature`, bindings and
backend-neutral request ABI are unchanged. CPU and Metal code paths are
unchanged. The compiler emits v2 headers; generated PTX bodies remain identical
to their legacy fixtures. Those fixtures stay v1 to test compatibility.

The manifest participates in cache identity, so old/new keys differ. Compilation
does not relabel or remove old entries. Old artifacts can still be loaded on
their conservative path; a fresh v2 cache hit retains the fast path. Unknown
contract tags are rejected. Default target/compiler selection is unchanged.

## Reproduce the paired experiment

Use the validated Linux x86_64 / sm_52 / CUDA 12.4 / Driver API 13.0 / LLVM
21.1.8 environment. The generated-kernel support gate remains narrow.

```bash
mkdir -p build/cuda-output-copy
export TENSORCX_LLVM_BIN="$PWD/build/mlir-toolchain/root/usr/lib/llvm-21/bin"
uv run --no-sync python benchmarks/bench_cuda_fusion.py --output build/cuda-output-copy/run-1.json
uv run --no-sync python benchmarks/bench_cuda_fusion.py --output build/cuda-output-copy/run-2.json
```

For a before/after experiment, install the relevant CUDA-enabled runtime and
compiler before each pair; retain the same measurement harness. The four
reports retain source/compiler/loaded-extension hashes and dirty-tree identity.
The harness changed only metadata before collecting the baselines. Measurement
code, provider order, inputs and validation are identical across all four runs.
The baseline uses the loaded runtime/compiler from `9988008`; its extension and
producer hashes agree across both baseline runs. Source/document edits progressed
while the old extension remained loaded; dirty-diff hashes are retained rather
than presented as clean builds. After rebuilding, both accepted after-runs use
the new extension/producer. An initial post-change run overlapped site checks
and was excluded before selecting the two consecutive accepted after-runs.

Each run covers 20 two/four-op, resident/end-to-end cases across 1K, 16K, 256K,
1M and 16M elements. Thirty unprofiled measured rounds balance all six provider
orders; six warmups and three separate instrumented diagnostics are retained.
Every result passes CPU parity outside timing and resident inputs are checked
for mutation. The memory pool is not enabled. Compile/module loading and returned
result cleanup are excluded; intermediate cleanup inside public calls remains
included. Copy records have no device event timing. Nested host bytes are not
physical traffic totals. See the [full method](CUDA_FUSION_EXPERIMENT.md).

Nightblade shares its GTX 980 Ti with a desktop and other compute workloads.
Endpoint GPU snapshots cannot prove exclusivity. Before/after wall changes do
not isolate one removed cost; unchanged native controls and every regression
must remain visible. No cross-GPU or model-wide speedup is established.

## Verification

`tests/python/test_output_copy.py` covers exact manifests and pre-tool invalid
IR, CPU/GPU parity, reordered parameters, left/right/both/borrowed input aliases,
scalar/empty/multidimensional outputs, zero/partial/full work and non-block-sized
tails. It checks caller bits, private results, bit-exact suffixes, copy/wait
counts, dirty pool reuse and an actual seeded legacy cache followed by a v2
miss/hit. Its nine CPU-only policy cases run in LLVM CI; its 54 real-device cases
are included in the exact-commit CUDA push gate.

Native tests additionally check exact-tag rejection, foreign-context restoration,
full/partial/zero-work counts and a legacy multiple-store alias probe at both
partial and full lengths. Existing arithmetic, numerical and profiling suites
remain mandatory; the optimization changes no math or PTX instructions.

The paired measurements and final verification evidence are recorded below.

Local focused runtime/emitter/profiler/gate checks passed **560 tests**; native
contracts passed **5/5**. Each of memcheck, racecheck and initcheck passed all
**54 new GPU cases** and the native generated-kernel executable with zero
errors/hazards. These targeted loader commands disable API-error reporting only
for the deliberate out-of-range parameter-count probe; memory/race/initialization
checks remain enabled. Mandatory push-gate sanitizer flags are unchanged.
CPU host ASan/UBSan contracts passed **2/2**. The attempted CUDA ASan run skipped
two GPU tests after Driver initialization reported out of memory; those skips
are excluded from GPU evidence. Real CUDA execution is covered by the normal
native tests and Compute Sanitizer runs above.
Independent NumPy-oracle QA passed **32 additional GPU cases**, including both
parameter orders, five alias arrangements, zero/partial/full work, caller/suffix
bits, copy/wait counts and poisoned pool reuse with zero counters after close.
The complete Linux CPU/CUDA/LLVM regression passed **9,319 tests with 226
expected platform/capability skips** (exit 0). Both code reviews and separate
test/acceptance QA completed without remaining findings. Site check/build
produced 14 pages; five preview and six local Workers browser tests, strict
deployment dry run, actionlint and publication/secret checks passed. Remote CI
and production deployment are subsequent release checks, not local GPU evidence.

## Local measurements and decision

Four reviewed reports retain every sample and p10/p90 spread:
[before 1](validation/cuda-output-copy-before-1.json),
[before 2](validation/cuda-output-copy-before-2.json),
[after 1](validation/cuda-output-copy-after-1.json),
[after 2](validation/cuda-output-copy-after-2.json).
All 20 cases per run passed all three providers, 30 balanced measured rounds
and three diagnostics each. Benchmark, helper and profiler hashes match across
all four reports; extension/producer hashes match within each pair and differ
between before and after. **All four observed other compute processes.**

Medians in milliseconds, **run 1 / run 2** within every cell:

| Elements | Ops | Boundary | Native before | Native after | MLIR separate before | MLIR separate after | MLIR fused before | MLIR fused after |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 2 | resident | 0.033455 / 0.033937 | 0.033163 / 0.032535 | 0.136080 / 0.139823 | 0.115562 / 0.115548 | 0.073547 / 0.075604 | 0.064641 / 0.063215 |
| 1,024 | 2 | end_to_end | 0.081554 / 0.080523 | 0.079394 / 0.079040 | 0.195749 / 0.183493 | 0.165427 / 0.164438 | 0.125136 / 0.123779 | 0.113390 / 0.113073 |
| 1,024 | 4 | resident | 0.059133 / 0.059248 | 0.059362 / 0.059670 | 0.254850 / 0.250613 | 0.214075 / 0.211751 | 0.082203 / 0.081054 | 0.072718 / 0.071685 |
| 1,024 | 4 | end_to_end | 0.107451 / 0.107112 | 0.105902 / 0.111626 | 0.308495 / 0.301471 | 0.264925 / 0.298411 | 0.132998 / 0.131800 | 0.121769 / 0.126753 |
| 16,384 | 2 | resident | 0.036997 / 0.034367 | 0.034458 / 0.034582 | 0.145451 / 0.137899 | 0.118501 / 0.120516 | 0.083934 / 0.075465 | 0.066434 / 0.066080 |
| 16,384 | 2 | end_to_end | 0.125127 / 0.114358 | 0.116742 / 0.114459 | 0.243629 / 0.217203 | 0.207607 / 0.199816 | 0.171551 / 0.157523 | 0.153186 / 0.149733 |
| 16,384 | 4 | resident | 0.070518 / 0.060993 | 0.061552 / 0.061283 | 0.282663 / 0.256594 | 0.219278 / 0.219653 | 0.093252 / 0.083570 | 0.075369 / 0.075406 |
| 16,384 | 4 | end_to_end | 0.163105 / 0.141877 | 0.143465 / 0.142539 | 0.411834 / 0.342380 | 0.307280 / 0.301523 | 0.189990 / 0.165801 | 0.162468 / 0.162124 |
| 262,144 | 2 | resident | 0.292752 / 0.314129 | 0.272929 / 0.271925 | 0.480348 / 0.455057 | 0.428756 / 0.405598 | 0.271208 / 0.240796 | 0.229198 / 0.216477 |
| 262,144 | 2 | end_to_end | 1.527564 / 1.381804 | 1.363161 / 1.369090 | 1.697238 / 1.533373 | 1.468943 / 1.504098 | 1.475766 / 1.361611 | 1.304503 / 1.317123 |
| 262,144 | 4 | resident | 0.602877 / 0.580542 | 0.590245 / 0.589371 | 0.827255 / 0.774787 | 0.724789 / 0.733658 | 0.265072 / 0.238828 | 0.274500 / 0.225693 |
| 262,144 | 4 | end_to_end | 1.779559 / 1.678036 | 1.641935 / 1.659304 | 2.063026 / 1.843298 | 1.793621 / 1.830938 | 1.436705 / 1.317595 | 1.281527 / 1.297241 |
| 1,048,576 | 2 | resident | 0.588553 / 0.582904 | 0.583789 / 0.560474 | 0.746020 / 0.723847 | 0.630375 / 0.638904 | 0.454595 / 0.413667 | 0.408904 / 0.409199 |
| 1,048,576 | 2 | end_to_end | 3.789872 / 3.207628 | 3.063891 / 3.157293 | 3.859308 / 3.321079 | 3.192781 / 3.326789 | 3.878959 / 3.075830 | 2.972771 / 3.121930 |
| 1,048,576 | 4 | resident | 1.446249 / 1.409502 | 1.405474 / 1.413075 | 1.648328 / 1.603542 | 1.433104 / 1.436006 | 0.456730 / 0.457856 | 0.409393 / 0.409389 |
| 1,048,576 | 4 | end_to_end | 4.422541 / 4.405265 | 4.306140 / 4.073877 | 4.732840 / 4.647493 | 4.375158 / 4.221915 | 3.486365 / 3.513291 | 3.384961 / 3.130389 |
| 16,777,216 | 2 | resident | 6.146494 / 6.107468 | 6.050248 / 6.123855 | 7.292331 / 7.293453 | 6.193370 / 6.229297 | 5.646728 / 5.639909 | 5.087722 / 5.091845 |
| 16,777,216 | 2 | end_to_end | 67.242292 / 66.782656 | 67.901241 / 68.090933 | 68.281988 / 68.141698 | 67.426385 / 68.085578 | 67.550697 / 66.749834 | 66.991726 / 66.654448 |
| 16,777,216 | 4 | resident | 13.777876 / 13.709945 | 13.699886 / 13.742704 | 16.563303 / 16.450750 | 14.646708 / 14.748015 | 5.636020 / 5.635551 | 5.090129 / 5.105315 |
| 16,777,216 | 4 | end_to_end | 74.412850 / 74.357607 | 73.329117 / 75.776971 | 80.094875 / 77.304733 | 74.023485 / 76.915121 | 69.287719 / 67.008424 | 64.670776 / 67.246273 |

Across the 40 paired case medians, before/after ratios range **0.998–1.340**
for separate MLIR and **0.966–1.305** for fused MLIR. Ratios below one are
regressions. The unchanged native provider ranges **0.960–1.237**, showing
material environmental variation; these ranges are not isolated causal speedups.
Generated regressions are retained: 256K four-op resident fused run 1 (0.966),
1M two-op end-to-end separate run 2 (0.998) and fused run 2 (0.985), and 16M
four-op end-to-end fused run 2 (0.996).

The 1M four-op fused resident median moved from 0.456730 / 0.457856 ms to
0.409393 / 0.409389 ms. Conversely, the after-run 1K two-op fused call remains
about **1.949 / 1.943 times slower than native**. Transfers dilute savings, and
these cases do not justify replacing native dispatch with generated kernels.

All three diagnostics in all four reports agree on these **1M four-op resident**
counts. Allocation and kernel counts stay unchanged:

| Provider | Kernels | Allocations | Template copies before → after | Template bytes before → after | Wait calls before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native separate | 4 | 4 | 0 → 0 | 0 → 0 | 4 → 4 |
| MLIR separate | 4 | 4 | 4 → 0 | 16,777,216 → 0 | 8 → 4 |
| MLIR fused | 1 | 1 | 1 → 0 | 4,194,304 → 0 | 2 → 1 |

Across every size/provider, resident copies/waits match the same rule.
End-to-end cases retain three transfer-copy API records and two explicit
transfer waits; a synchronous copy API call is not itself a separate recorded
wait span. Kernel intervals remain 2/4/1. Counts establish the omitted work,
not its isolated elapsed cost or total physical memory traffic.

**Decision:** retain this internal full-write optimization with strict v2
eligibility and conservative fallback. Keep ordinary dispatch, private output
allocation and final waits. A next local candidate is measuring Python launch
validation/argument preparation overhead before proposing a bounded optimization;
this is a proposal, not completed work. Automatic fusion, async execution,
other GPU/toolchain acceptance and isolated GPU CI remain open.
