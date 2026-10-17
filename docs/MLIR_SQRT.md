# Guarded float32 kernel square root

Implemented 2026-10-17. `cx.experimental.sqrt(value)` accepts exactly one
float32 value inside experimental kernels. Integer/index/boolean operands,
implicit promotion and arbitrary function calls remain unsupported. Ordinary
tensor `cx.sqrt` is a separate, already implemented operation.

## Numerical and lowering contract

CPU MLIR emits `math.sqrt` without fast-math flags; LLVM 21.1.8 lowers it to
`llvm.sqrt.f32`. CUDA emits the explicit `llvm.nvvm.sqrt.rn.f` intrinsic through
MLIR's `llvm.call_intrinsic`, producing `sqrt.rn.f32` without FTZ. LLVM 21's
generic GPU `math.sqrt` conversion instead calls external `__nv_sqrtf`, which
this device-only loader does not link. The explicit intrinsic avoids that
external dependency and keeps existing lowering flags and native ABI intact.
See the pinned [LLVM CPU conversion](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Conversion/MathToLLVM/MathToLLVM.cpp),
[GPU conversion](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Conversion/GPUToNVVM/LowerGpuOpsToNVVMOps.cpp)
and [PTX square-root specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-sqrt).

Under the validated default rounding environment, CPU/CUDA tests compare all
non-NaN result bits against NumPy float32, including positive/negative zero,
subnormal inputs, largest finite values and infinity. Negative nonzero inputs
produce NaN. NaN masks match; NaN payload/sign are not a promise. The independent
CPU reference ignores NumPy floating-point error settings for sqrt, so negative
inputs return NaN rather than raising. Other composed operations retain their
existing numerical contracts; this extension is not fast math.

## Example

Save this as inspectable Python source after the external LLVM setup in the
[CPU](MLIR_RUNTIME_INTEGRATION_DECISION.md) or
[CUDA](MLIR_CUDA_INTEGRATION_DECISION.md) integration guide:

```python
import tensorcx as cx

@cx.experimental.kernel
def magnitude(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        squared = a[i] * a[i] + b[i] * b[i]
        out[i] = cx.experimental.sqrt(squared)

device = "cpu"  # "cuda" on the validated experimental CUDA host
compiled = magnitude.compile(target=device, compiler="mlir")
a = cx.tensor([3.0, 5.0], device=device)
b = cx.tensor([4.0, 12.0], device=device)
out = cx.zeros((2,), device=device)
assert list(compiled.launch(a, b, out, 2).numpy()) == [5.0, 13.0]
assert list(out.numpy()) == [0.0, 0.0]
```

Sqrt can consume earlier float32 locals, arithmetic, another sqrt, or selected
values. Its result can feed arithmetic, comparisons and selection. Value
selection does not protect unsafe loads or make evaluation lazy.

## Compatibility and limits

The CPU and CUDA host/toolchain gates remain unchanged: Linux x86_64 and
external LLVM 21.1.8; CUDA execution is validated on sm_52 with Toolkit 12.4 and
the documented driver. CUDA still requires exactly two input parameters, one
output and one uint32 bound, all buffer accesses at the canonical guarded
index. For a single-input root, `sqrt(a[i] * b[i])` with `b` filled with ones
fits that existing CUDA signature.

The CUDA budget remains 32 locals, depth 16 and 64 operation nodes; each sqrt
costs one node. No native ABI, entry/manifest, device selection or stream API
changes. Sqrt bodies use the existing `tensorcx_expr_v1`/`expr-f32-v1` contract.
The emitted source separates sqrt from other expressions in cache keys.
Existing cache validation and native module loading remain mandatory.

CPU/CUDA launches return a private output copy, preserve unwritten suffixes,
redirect input/output aliases to that copy and leave caller tensors unchanged.
Zero work still validates inputs and artifacts; there is no NumPy fallback.

Direct MSL emits qualified `::metal::sqrt` and keeps the in-place output
contract. Metal tests use finite normal values at `rtol=atol=1e-6`; CPU/CUDA bit
checks are not a Metal bit-parity claim. General branches, generated reductions
and other mathematical intrinsics require separate increments.

## Verification

`tests/python/test_mlir_sqrt.py` checks composed/nested roots, float literals,
selection/comparisons, names shadowing the builtin, standard scalar/empty/tail
shapes, block sizes, prefixes, aliases, reordered/repeated launches, invalid
arguments and no NumPy fallback. Numerical acceptance includes 18 special bit
patterns and 4,096 seeded random bit patterns against an independent NumPy
oracle. Parser/type/unsafe-load rejection, exact/one-over CUDA complexity
bounds, PTX rounding and cold/warm/same-name cache separation are covered.
Five Metal shape cases exercise the four kernel forms separately.

The strict CUDA push gate requires the new parity/numerical/PTX anchors;
GPU-free LLVM CI executes CPU kernels and serializes PTX. Linux skips do not
validate Metal; real execution is checked by the macOS jobs. No performance,
additional GPU architecture or portable release-package claim is made.

Local Linux x86_64/LLVM 21.1.8/Toolkit 12.4/sm_52 evidence:

| Check | Result |
| --- | --- |
| New CPU and real CUDA tests | 241 passed, no skips |
| Full runtime regression | 6,623 passed, 176 expected platform/capability skips |
| CUDA sqrt launch cases under memcheck | 111 passed, zero memory errors |
| Same cases under racecheck | 111 passed, zero hazards |
| Parser/type rejection with LLVM unavailable | 14 passed |

The sanitizer selection is `cuda and not cache`; cache and complexity-bound
compilations run normally. `--report-api-errors no` excludes the native loader's
intentional out-of-range parameter probe; memory/race error exit checks stay
enabled. The five new Metal cases are skipped on Linux and require macOS CI.
