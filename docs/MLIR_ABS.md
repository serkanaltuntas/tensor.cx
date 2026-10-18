# Guarded float32 kernel absolute value

Implemented 2026-10-17. `cx.experimental.abs(value)` accepts exactly one
float32 value inside experimental kernels. Integer, boolean and index operands,
implicit promotion and arbitrary function calls remain unsupported. Ordinary
tensor `cx.abs` is a separate, already implemented operation. The kernel parser
requires the qualified experimental name; Python's builtin `abs` is not an
alias in this subset.

## Numerical and lowering contract

CPU MLIR emits `math.absf` without fast-math flags, lowered by LLVM 21.1.8 to
`llvm.fabs.f32`. CUDA explicitly emits that same LLVM intrinsic through
`llvm.call_intrinsic`, producing `abs.f32` without FTZ or an external math
library. LLVM 21's generic GPU `math.absf` conversion instead calls external
`__nv_fabsf`; the device-only loader does not link libdevice. Existing lowering
options and native ABI are unchanged. See the
[MLIR operation](https://mlir.llvm.org/docs/Dialects/MathOps/#mathabsf-mathabsfop),
pinned [GPU conversion](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Conversion/GPUToNVVM/LowerGpuOpsToNVVMOps.cpp)
and [PTX absolute-value specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-abs).

For non-NaN float32 operands the result clears the sign bit: negative zero
becomes positive zero, negative infinity becomes positive infinity, and
subnormals retain their magnitude. CPU/CUDA acceptance compares every non-NaN
result bit against an independent sign-bit oracle and compares NaN masks.
NaN payload/sign are not a promise. The CPU reference uses NumPy float32
absolute value and does not raise for NaN when the caller enables NumPy
floating-point errors. Composed arithmetic/sqrt/selection keeps its existing
numerical and eager-value rules; abs does not make other operations exact or
protect an unsafe load.

## Example

Save this as inspectable Python source after the external LLVM setup in the
[CPU](MLIR_RUNTIME_INTEGRATION_DECISION.md) or
[CUDA](MLIR_CUDA_INTEGRATION_DECISION.md) guide:

```python
import tensorcx as cx

@cx.experimental.kernel
def absolute_difference(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.abs(a[i] - b[i])

device = "cpu"  # "cuda" on the validated experimental CUDA host
compiled = absolute_difference.compile(target=device, compiler="mlir")
a = cx.tensor([-3.0, 5.0], device=device)
b = cx.tensor([1.0, 8.0], device=device)
out = cx.zeros((2,), device=device)
assert list(compiled.launch(a, b, out, 2).numpy()) == [4.0, 3.0]
assert list(out.numpy()) == [0.0, 0.0]
```

Abs can consume float literals, earlier float32 locals, arithmetic and selected
values; its result can feed another abs, sqrt, comparisons and selection.

## Compatibility and limits

Host/toolchain gates remain Linux x86_64 and external LLVM 21.1.8. CUDA execution
is validated on sm_52 with Toolkit 12.4 and the documented driver. The CUDA
signature remains exactly two input parameters, one output and one uint32
bound, with every buffer access at the guarded canonical index. CPU permits a
single input; a single-input CUDA kernel is still rejected before tools.

CUDA retains 32 locals, expression depth 16 and 64 operation nodes, counting
abs as one node alongside arithmetic, sqrt, comparisons and selection. There
are no native core, stream or device-selection changes. Abs bodies use the
existing `tensorcx_expr_v1`/`expr-f32-v1` entry/manifest. Emitted source separates
abs from sqrt in cache keys; cache validation and native loading remain
mandatory.

CPU/CUDA launch returns a private result, preserves unwritten suffixes and
leaves caller tensors unchanged. CUDA full writes use
[conditional template-copy elision](CUDA_OUTPUT_COPY_ELISION.md).
Zero work still validates inputs and artifacts. There is no NumPy fallback.

Direct MSL emits qualified `::metal::fabs`, including when a parameter is named
`fabs`, and keeps the in-place output contract. Metal acceptance uses finite
normal inputs at `rtol=atol=1e-6`; CPU/CUDA special-value bit checks are not a
Metal bit-parity claim. Integer kernel abs, general branches and generated
reductions require separate increments.

## Verification

`tests/python/test_mlir_abs.py` covers direct/composed abs, float locals named
`abs`, a parameter named `fabs`, division/sqrt/selection composition, literals,
scalar/empty/tail shapes, blocks, prefixes, aliases, reordered/repeated launches
and no NumPy fallback. It checks 18 special patterns and 4,096 seeded random bit
patterns with an independent sign-bit oracle, plus a direct single-input CPU
kernel under NumPy's error-raising policy. Types/parser arity and keywords,
unsafe loads, invalid launches, exact/one-over CUDA complexity limits and
cold/warm/same-name abs-vs-sqrt cache separation are tested. Five Metal shape
cases exercise the four composed kernels and the single-input kernel.

The strict CUDA push gate requires abs parity/numerical/PTX anchors. GPU-free
LLVM CI executes CPU kernels and serializes PTX; Metal needs real macOS device
execution. No performance, additional CUDA architecture or portable release
package claim is made.

Local Linux x86_64/LLVM 21.1.8/Toolkit 12.4/sm_52 evidence:

| Check | Result |
| --- | --- |
| New CPU and real CUDA tests | 246 passed, no skips |
| Full runtime regression | 6,869 passed, 181 expected platform/capability skips |
| CUDA test selection under memcheck | 113 passed, zero memory errors |
| Same selection under racecheck | 113 passed, zero hazards |
| Parser/type/signature rejection with LLVM unavailable | 16 passed |
| GPU-free PTX emission and validation | 4 passed |

Sanitizers select `cuda and not cache`; cache and complexity-bound compilations
run normally. `--report-api-errors no` excludes the native loader's intentional
out-of-range parameter probe; memory/race error exit checks remain enabled.
The five new Metal cases skip on Linux and require macOS CI.
