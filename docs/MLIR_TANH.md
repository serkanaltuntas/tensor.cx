# Guarded float32 kernel hyperbolic tangent

Implemented 2026-10-18. `cx.experimental.tanh(value)` accepts exactly one
float32 operand inside experimental kernels: guarded loads, finite float
literals, locals and nested expressions. Integer/bool/index values, implicit
promotion, keyword arguments and Python math aliases remain unsupported.
No ordinary tensor `cx.tanh` API is added.

## Numerical and special-value contract

| Input | Result |
| --- | --- |
| Finite value | Bounded-error hyperbolic tangent in `[-1, 1]` |
| `+0` or `-0` | The same signed zero |
| `+inf` or `-inf` | `+1` or `-1`, respectively |
| NaN | NaN |

NaN payload/sign and signaling exceptions are unspecified. CPU references use
NumPy float32 tanh with local error handling; ambient NumPy exception settings
do not change this contract. CPU/CUDA acceptance includes positive and negative
subnormal arguments/results. Direct Metal may flush them: nonzero subnormal
arguments are outside its numerical promise.

Under the validated default rounding environment, standalone finite results
use **`rtol=1e-6, atol=0`** on CPU/MLIR, CUDA/MLIR and direct Metal. The independent
oracle computes float64 tanh and rounds once to float32; error comparisons use
float64 so a small result is not hidden by an absolute tolerance. Signed zeros,
NaN classifications, exact infinity-to-one results and the output range are
checked separately. CPU/CUDA tests also check exact signed-zero and subnormal
output bits. CUDA additionally checks input-bit preservation for
`abs(x) <= 2^-12`; CPU libm results for small normal inputs may differ by an ULP
while meeting the relative-error bound.

This is sampled numerical acceptance, not a correctly-rounded or bit-identical
cross-target promise. Composed operations retain their own error bounds and
conditioning; approximate results can affect comparisons near a threshold.
Selection is eager: both arms may evaluate, so it does not protect unsafe loads.
An unselected NaN does not contaminate the selected result.

## Lowering and cache policy

CPU emits `math.tanh` without fast-math flags. LLVM 21.1.8 lowers it through
`llvm.tanh.f32` to system `tanhf`; see the [pinned conversion](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Conversion/MathToLLVM/MathToLLVM.cpp).
Tanh uses the existing explicit `-lm` linkage after object inputs and libm
linker/runtime fingerprint policy introduced for exp/log. A shared internal
operation list controls both linkage and reserved CPU names `expf`, `logf` and
`tanhf`; collisions fail before tools/cache/native loading. This restriction is
CPU-only. [Sigmoid](MLIR_SIGMOID.md) and [SiLU](MLIR_SILU.md) emit exp and share its existing libm policy;
existing exp/log flags/cache identities and non-libm defaults remain.

CUDA uses the following symmetric approximation on the validated sm_52 target.
Let `a = abs(x)` and `z = min(a, 1)^2`:

| Region | Positive-magnitude calculation |
| --- | --- |
| `a <= 1` | `min(a,1) * (135135 + 17325*z + 378*z*z + z*z*z) / (135135 + 62370*z + 3150*z*z + 28*z*z*z)` |
| `1 < a < 10` | `(1 - exp(-2*a)) / (1 + exp(-2*a))` |
| `a >= 10` | Exactly `1` |

The first row is the `[7/6]` Padé approximant to tanh; its rational coefficients
match the Taylor series through degree 13. Horner evaluation uses explicit
float32 fused multiply-adds, divides the two polynomials, then multiplies by the
bounded argument. Near zero the ratio is one, avoiding subtractive cancellation
and preserving tiny inputs. The exp row uses non-FTZ `ex2.approx.f32` with a
float32 `-2*log2(e)` scale. Both eager branches clamp their arguments before
evaluation, avoiding overflow even for largest-finite/infinite inputs. IEEE
minimum propagates NaNs; `copysign` restores the original sign after selection.
At magnitude 10 the correctly rounded float32 result has already reached one.

No libdevice, CUDA SDK lookup, new external linker or `tanh.approx.f32` is used.
That native tanh instruction requires sm_75 and its documented error bound does
not establish this increment's tighter acceptance. See the
[PTX tanh specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-tanh)
and [base-2 exponential specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-ex2).

MSL emits qualified `::metal::tanh`, including when parameters/locals are named
`tanh`. Containing kernels use the existing `-fno-fast-math` compiler/cache
policy; other strict-operation policies are unchanged. Acceptance includes
normal arguments, zeros, infinities and NaNs. See the [Metal language reference](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf),
sections 1.6.3, 8.1 and 8.4.

## Example and retained boundaries

After external LLVM setup in the [CPU](MLIR_RUNTIME_INTEGRATION_DECISION.md)
or [CUDA](MLIR_CUDA_INTEGRATION_DECISION.md) guide, save this in an inspectable
Python source file:

```python
import tensorcx as cx

@cx.experimental.kernel
def activation_product(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.tanh(a[i] * b[i])

device = "cpu"  # "cuda" on the validated experimental CUDA host
compiled = activation_product.compile(target=device, compiler="mlir")
a = cx.tensor([-1.0, 0.0, 1.0], device=device)
b = cx.ones((3,), device=device)
out = cx.zeros((3,), device=device)
result = compiled.launch(a, b, out, 3)
cx.testing.assert_allclose(result, cx.tensor([-0.7615942, 0.0, 0.7615942], device=device), rtol=1e-6, atol=0)
assert list(out.numpy()) == [0.0, 0.0, 0.0]
```

CPU/CUDA remain Linux x86_64/external LLVM 21.1.8 paths; CUDA execution is
validated with Toolkit 12.4 on sm_52. CUDA requires two inputs, one output and
one uint32 bound; CPU can accept a single input. Tanh counts as one DSL operation
in the existing 64-node/32-local/depth-16 budget despite its internal instruction
expansion. Native entry/ABI and `tensorcx_expr_v1`/`expr-f32-v1` are unchanged.
Emitted source separates tanh in cache keys; validation still runs on hits.

CPU/CUDA launches return a private output copy, redirect its input aliases,
preserve unwritten suffixes and leave callers unchanged. Direct Metal keeps
in-place output. Zero work still validates inputs/artifacts. No NumPy fallback,
general broadcasting, reductions, streams or device-index extension is added.

## Verification

`test_mlir_tanh.py` exercises six forms: product, nested/local tanh, sqrt/division/
select composition, name shadowing, min/max clamping and multiplication with
epsilon. Tests include scalar/empty/tail shapes, blocks, prefix aliases,
repeated/reordered launches, CPU single-input compilation, no NumPy fallback,
invalid launch preservation, parser/type/arity/unsafe-load rejection, exact and
one-over CUDA budgets, cold/warm tanh-vs-abs caches and CPU/Metal compiler policy.
CPU policy coverage includes tanh/exp/log/abs/rsqrt, plus all three reserved
libm names with caching enabled and disabled.

Numerical samples cover special bits, branch and small-input boundaries,
positive/negative powers of two with neighbors, 32,769 values in `[-12,12]`,
16,385 in `[-1,1]`, 4,096 log-spaced magnitudes with both signs, and 8,192 seeded
random bit patterns. Six direct Metal cases exercise five finite shapes plus
the normal-input/special-value numerical scope. Input immutability uses a
post-import snapshot because the existing bridge can quiet signaling NaNs.
The strict CUDA gate requires tanh parity/numerical/PTX anchors. GPU-free LLVM
CI runs CPU kernels and emits PTX; Linux skips do not validate Metal. No speedup,
additional NVIDIA architecture, portable package or full compiler claim is made.

Local Linux x86_64/LLVM 21.1.8/Toolkit 12.4/sm_52 evidence:

| Check | Result |
| --- | --- |
| New CPU and real CUDA tests | 346 passed (included in the focused run) |
| Focused tanh, exp, log, rsqrt and cache regression | 1,307 passed, four expected Metal skips |
| Full runtime regression | 8,495 passed, 214 expected platform/capability skips |
| CUDA selection under memcheck/racecheck | 157 passed per tool, zero errors/hazards |
| Invalid/parser/policy checks with LLVM unavailable, including updated NumPy alias | 23 passed |
| GPU-free PTX emission and complexity validation | 3 passed |

Sanitizers select `cuda and not cache`; cache and complexity-bound compilations
run normally. `--report-api-errors no` excludes the native loader's intentional
out-of-range parameter probe; memory/race error exit checks remain enabled.
Independent QA checked 753,676 inputs on CPU and real CUDA against a separate
float64 oracle, including random bits, uniform `[-12,12]`, signed subnormals and
8,192 neighbors in each direction around `+/-1`, `+/-10` and `+/-2^-12`.
Output range, tiny-input bits, prefix aliases, unwritten suffixes and caller
immutability passed. This adds sampled acceptance, not exhaustive proof for
every float32 input or device.
