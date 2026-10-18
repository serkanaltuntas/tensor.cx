# Guarded float32 kernel reciprocal square root

Implemented 2026-10-18. `cx.experimental.rsqrt(value)` computes a bounded-error
`1 / sqrt(value)` inside experimental kernels. It accepts exactly one float32
operand: guarded loads, finite float literals, locals or nested expressions.
Integer/bool/index values, implicit promotion, keyword arguments and Python
math aliases remain unsupported. No ordinary tensor `cx.rsqrt` API is added.

## Domain and numerical contract

| Input | Result |
| --- | --- |
| Positive finite value | Positive finite reciprocal square root |
| `1` | `1` |
| `+0` | Positive infinity |
| `-0` | Negative infinity |
| Negative nonzero value, including `-inf` | NaN |
| `+inf` | Positive zero |
| NaN | NaN |

NaN payload/sign and signaling exceptions are unspecified. The CPU interpreter
uses local NumPy error handling, so domain and divide-by-zero cases return
these values even when ambient NumPy error settings request exceptions.
Positive subnormal arguments are included in CPU/CUDA numerical acceptance.
Direct Metal may flush subnormal arguments: nonzero subnormal arguments of
either sign are outside its numerical contract.

Under the validated default rounding environment, standalone finite results
are checked against a float64 `1 / sqrt(x)` rounded once to float32, using
**`rtol=1e-6, atol=0`** on CPU/MLIR, CUDA/MLIR and direct Metal. Error comparisons
use float64 to retain the relative bound at the smallest positive results.
Every positive finite float32 input has a normal reciprocal-root result;
neither underflow to zero nor overflow to infinity is allowed in that domain.
NaN classifications, infinity signs and the positive zero from `+inf` are checked
separately. This is sampled numerical acceptance, not a correctly-rounded or
bit-identical cross-target promise. Composed operations retain their own error
bounds and conditioning; approximate values can affect comparisons near a
threshold. Value selection is eager: both arms may evaluate, so it does not
protect unsafe loads. An unselected NaN does not contaminate the selected result.

## Lowering and cache policy

CPU emits `math.rsqrt` without fast-math flags. Pinned LLVM 21.1.8 converts it
to `llvm.sqrt.f32` followed by float32 `1 / root`, so the two operations round
separately. The NumPy reference also rounds the root before float32 division;
the independent numerical oracle instead rounds only the final float64 result.
No additional libm linkage or cache dependency is introduced; the existing
exp/log/[tanh](MLIR_TANH.md) `-lm` policy and CPU reserved names remain unchanged. See the
[pinned CPU conversion](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Conversion/MathToLLVM/MathToLLVM.cpp).

CUDA emits `llvm.nvvm.rsqrt.approx.f`, producing non-FTZ
`rsqrt.approx.f32` on the validated sm_52 hardware. Positive subnormal arguments
remain supported; no libdevice, CUDA SDK lookup or new external linker is
needed. The relative tolerance accommodates this approximation. See the
[pinned intrinsic](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/include/llvm/IR/IntrinsicsNVVM.td)
and [PTX reciprocal-root specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-rsqrt).

MSL emits qualified `::metal::rsqrt`, including when parameters or locals are
named `rsqrt`. Kernels containing it use the existing `-fno-fast-math` compiler
and cache policy; exp/log/min/max policies remain unchanged. Acceptance covers
normal arguments, zeros, infinities and NaNs, without a subnormal-input promise.
See the [Metal language reference](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf),
sections 1.6.3, 8.1 and 8.4.

## Example and retained boundaries

After the external LLVM setup in the [CPU](MLIR_RUNTIME_INTEGRATION_DECISION.md)
or [CUDA](MLIR_CUDA_INTEGRATION_DECISION.md) guide, save this in an inspectable
Python source file:

```python
import tensorcx as cx

@cx.experimental.kernel
def inverse_root_product(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.rsqrt(a[i] * b[i])

device = "cpu"  # "cuda" on the validated experimental CUDA host
compiled = inverse_root_product.compile(target=device, compiler="mlir")
a = cx.tensor([1.0, 4.0, 16.0], device=device)
b = cx.ones((3,), device=device)
out = cx.zeros((3,), device=device)
result = compiled.launch(a, b, out, 3)
cx.testing.assert_allclose(result, cx.tensor([1.0, 0.5, 0.25], device=device), rtol=1e-6, atol=0)
assert list(out.numpy()) == [0.0, 0.0, 0.0]
```

CPU/CUDA remain Linux x86_64/external LLVM 21.1.8 paths; CUDA execution is
validated with Toolkit 12.4 on sm_52. CUDA still needs two inputs, one output
and one uint32 bound; CPU can accept a single input. Each rsqrt costs one node
in the existing 64-operation/32-local/depth-16 CUDA budget. Native entry/ABI and
`tensorcx_expr_v1`/`expr-f32-v1` manifests are unchanged. Emitted source separates
rsqrt from other operations in artifact cache keys; validation still runs on hits.

CPU/CUDA launches return a private output copy, redirect its input aliases,
preserve unwritten suffixes and leave callers unchanged. Direct Metal keeps
in-place output. Zero work still validates inputs/artifacts. No NumPy fallback,
general broadcasting, reductions, streams or device-index extension is added.

## Verification

`test_mlir_rsqrt.py` exercises six kernel forms: product, nested/local roots,
sqrt/division/select composition, builtin name shadowing, minimum/maximum
clamping and normalization-style multiplication with epsilon. Tests cover
scalar/empty/tail shapes, blocks, prefix aliases, repeated/reordered launches,
single-input CPU, no NumPy fallback, invalid launch preservation, parser/type/
arity/unsafe-load rejection, exact/one-over CUDA complexity limits and cold/warm
rsqrt-vs-abs cache separation. Compiler checks inspect strict CPU sqrt/division
without added `-lm`, non-FTZ PTX without external functions and Metal strict mode.

Numerical inputs include special bit patterns, neighbors of zero/one/minimum
normal/maximum finite values, all representable powers of two and their
neighbors, 4,097 values near one, 4,096 logarithmically spaced values across
positive magnitudes and 4,096 seeded random bit patterns. Six direct Metal cases
exercise five finite shapes and the documented domain/special-value scope.
Input immutability uses a post-import snapshot because the existing bridge may
quiet signaling NaNs. The strict CUDA push gate requires rsqrt parity/numerical/
PTX anchors. GPU-free LLVM CI runs CPU kernels and emits PTX; Linux skips do not
validate Metal. No speedup, additional NVIDIA architecture, portable release
package or broad compiler phase completion is claimed.

Local Linux x86_64/LLVM 21.1.8/Toolkit 12.4/sm_52 evidence:

| Check | Result |
| --- | --- |
| New CPU and real CUDA tests | 334 passed (included in the focused run) |
| Focused rsqrt, log, sqrt and cache regression | 911 passed, four expected Metal skips |
| Full runtime regression | 8,149 passed, 208 expected platform/capability skips |
| CUDA selection under memcheck/racecheck | 156 passed per tool, zero errors/hazards |
| Invalid/parser/policy checks with LLVM unavailable | 17 passed |
| GPU-free PTX emission and complexity validation | 3 passed |

Sanitizers select `cuda and not cache`; cache and complexity-bound compilations
run normally. `--report-api-errors no` excludes the native loader's intentional
out-of-range parameter probe; memory/race error exit checks remain enabled.
Independent QA also checked 344,072 random-bit, near-one, positive-subnormal and
special inputs on CPU and real CUDA against a separate float64 oracle, including
caller immutability with output aliasing. This adds sampled acceptance, not an
exhaustive guarantee for every float32 input or device.
