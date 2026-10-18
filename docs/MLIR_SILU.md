# Guarded float32 kernel SiLU

Implemented 2026-10-18. `cx.experimental.silu(value)` computes SiLU,
`x / (1 + exp(-x))`, for exactly one float32 argument inside experimental
kernels. Guarded loads, float literals, locals and nested expressions are
supported. Integer, boolean and index arguments, implicit promotion, keyword
arguments and Python math aliases are rejected. Ordinary tensor `cx.silu`
already exists and retains its separate implementation and contract.

## Numerical contract

| Input | Result |
| --- | --- |
| Either signed zero | Zero with the original sign |
| `+inf` | `+inf` |
| `-inf` | Negative zero, the limiting value |
| NaN | NaN |

Finite results have the input's sign, including rounded zeros; they lie between
the input and zero and are greater than `-0.3`. NaN payload/sign and signaling
exceptions are unspecified. In the default rounding environment, a separate
oracle evaluates sign-separated direct exponentials in float64 and rounds once
to float32. It does not reuse the half-exponential lowering below. Finite error
is measured in float64 against `rtol * abs(reference) + atol`:

| Target | rtol | atol | Numerical scope |
| --- | --- | --- | --- |
| CPU / MLIR and CPU reference | `1e-6` | `2^-149` | All float32 inputs, including subnormal inputs/results and underflow boundaries |
| CUDA / MLIR | `1e-5` | `2 * 2^-149` | Same scope, on the validated sm_52 target |
| Direct Metal / MSL | `1e-6` | `2^-149` | `[-80, -2n]`, `[2n, +inf]`, `x <= -128`, signed zeros and NaNs; `n` is the smallest positive normal float32 |

Direct Metal excludes nonzero subnormal inputs/results and the negative
interval `(-128, -80)` from numerical acceptance. Saturation at `x <= -128`
returns literal negative zero. CPU/CUDA gradual underflow is tested: only
one/two smallest-subnormal units of absolute allowance are used, so flushing a
normal tail result to zero cannot pass. Selected negative-tail samples must
remain nonzero, including values where float32 sigmoid would already be zero.
Representative subnormal inputs also require the exact round-to-nearest result
of `x / 2`, including signed-zero underflow.

Zero signs, NaN/infinity classifications and exact special results are checked
separately. This is sampled bounded-error acceptance, not correctly rounded,
bit-identical cross-target or exhaustive float32 validation. Composed operations
retain their own error and conditioning limits; approximate values can affect
comparisons near a threshold. Selection is eager and does not protect unsafe
loads.

## Stable lowering and cache policy

For `a = abs(x)`, each target computes:

```text
h = exp(-a / 2)
d = 1 + h*h
c = -minimum(a, 256)
positive = x / d
negative = ((c*h)*h) / d
result = positive if x >= 0 else negative
```

The two negative-tail multiplications keep their stated order. Computing
`x * sigmoid(x)` or multiplying by a separately rounded `exp(x)` would lose
representable SiLU results around `-104` to `-108`. Half exponentials stay
normal there until the final product rounds to a subnormal. The bounded
negative scale equals `x` for finite negative magnitudes below 256; larger
negative values already round to `-0`. It prevents `inf * 0` in either eager
branch. Positive inputs retain their original numerator, including `+inf`.
Exponential arguments are never positive, so this also avoids exponential
overflow for negative inputs.

CPU emits strict `math.absf`, `math.exp`, IEEE `arith.minimumf`, arithmetic and
ordered value selection. Pinned LLVM 21.1.8 lowers exp through `llvm.exp.f32`
to system `expf`; see the [pinned conversion](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Conversion/MathToLLVM/MathToLLVM.cpp).
It reuses explicit `-lm` linkage after object inputs and the existing libm linker
and runtime cache fingerprints. There is no new `math.silu` or `siluf` external
symbol. CPU kernel names `expf`, `logf` and `tanhf` remain reserved; `siluf` is
not reserved. Existing sigmoid/other-operation sources, flags and cache identities
retain their defaults. The reference uses float32 NumPy operations with local
error handling, independently of ambient NumPy exception settings.

CUDA uses non-FTZ `llvm.nvvm.ex2.approx.f` with float32 scaling by
`-log2(e)/2` to produce `h`. The multiplier's magnitude is below one, so even
the largest finite input cannot overflow that scaling step. IEEE minimum
preserves NaNs in the bounded tail scale. Explicit multiplication order,
float32 divisions and ordered selection retain the tail and special-value
rules. No libdevice, CUDA SDK lookup, new external linker or newer-architecture
activation instruction is required. The CUDA tolerance includes base conversion
and exponential approximation; see the [PTX ex2 specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-ex2).

MSL emits one private inline helper with qualified `::metal::fabs`,
`::metal::exp` and `::metal::min`, and literal `-0` saturation for `x <= -128`.
NaNs propagate through the half exponential even if the min operation selects
the numeric operand. Helper names avoid kernel/parameter/local collisions;
global qualification and one helper keep nested operand source growth linear.
Containing kernels use the existing `-fno-fast-math` compiler/cache policy,
preserving multiplication order and zero signs. See the
[Metal language reference](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).

## Example and retained boundaries

After external LLVM setup in the [CPU](MLIR_RUNTIME_INTEGRATION_DECISION.md) or
[CUDA](MLIR_CUDA_INTEGRATION_DECISION.md) guide, save this inspectable source:

```python
import tensorcx as cx

@cx.experimental.kernel
def silu_product(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.silu(a[i] * b[i])

device = "cpu"  # "cuda" on the validated experimental CUDA host
compiled = silu_product.compile(target=device, compiler="mlir")
a = cx.tensor([-20.0, 0.0, 20.0], device=device)
b = cx.ones((3,), device=device)
out = cx.zeros((3,), device=device)
result = compiled.launch(a, b, out, 3)
cx.testing.assert_allclose(result, cx.tensor([-4.1223075e-8, 0.0, 20.0], device=device), rtol=1e-5, atol=0)
assert list(out.numpy()) == [0.0, 0.0, 0.0]
```

Direct Metal uses `compile(target="metal")` and returns the in-place output.
CPU/CUDA launches return a private result, preserve unwritten suffixes and leave
callers unchanged. CUDA full writes use
[conditional template-copy elision](CUDA_OUTPUT_COPY_ELISION.md). Zero work still validates
inputs/artifacts. CUDA retains two float32 inputs, one output and one uint32
bound; CPU can accept one input. Each SiLU costs one DSL operation inside
unchanged 64-operation, 32-local and depth-16 CUDA budgets. Expanded instructions
do not enlarge the budget or alter manifests/native ABI. Source/options separate
SiLU from sigmoid in cache keys; validation still runs on hits. No compiled
NumPy fallback, broadcasting, streams, general graph compiler or device-index
extension is added.

## Verification

`test_mlir_silu.py` covers six forms: product, local/nested, sqrt/divide/select,
shadowed names, min/max and multiply-with-epsilon. It exercises scalar/empty/tail
shapes and block sizes, prefix aliases, repeated/reordered launches, caller
preservation, unsafe loads/types/launches, exact/one-over CUDA budgets,
cold/warm SiLU-vs-sigmoid caches, CPU libm policy and single-input compilation.
Metal tests cover helper/kernel/parameter/local name collisions and linear
source growth for nested calls. Six direct Metal cases cover five shapes with
the helper-collision kernel and the accepted numerical domain. Immutability
uses a post-import snapshot because the bridge can quiet signaling NaNs.

The numerical dataset includes special bits, signed powers with neighbors,
origin/saturation/underflow boundaries, 32,769 values in `[-110,110]`, 16,385
in `[-20,20]`, an extra 16,385 in `[-112,-96]`, signed log-spaced magnitudes and
seeded random bit patterns. Dedicated tests require nonzero negative tails at
representable points from `-104` to `-108`, and exact subnormal-input halves.
The strict CUDA push gate requires parity/numerical/non-FTZ, stable-tail and
subnormal-half anchors. GPU-free LLVM CI runs real CPU kernels and emits PTX;
Linux skips do not validate Metal. No speedup, additional NVIDIA architecture,
portable package or full-compiler completion claim follows from these tests.

Local Linux x86_64/LLVM 21.1.8/Toolkit 12.4/sm_52 focused verification:
**354 tests passed**. Full regression: **9,200 passed, 226 expected
platform/capability skips**. Without LLVM, 28 policy/reference tests passed;
GPU-free PTX/budget checks passed three tests. CUDA memcheck and racecheck each
passed 158 tests with zero errors/hazards. The runnable example passed on CPU
and CUDA. Independent QA checked 745,532 additional inputs on both targets
against direct float64 exponentials, including dense underflow ties, subnormal
halves, stable tails, special values and prefix alias/caller preservation.
Two code reviews and separate test/acceptance QA found no blocking issues.
Direct Metal acceptance requires fresh real-device macOS CI; it is not inferred
from these Linux results.
