# Guarded float32 kernel sigmoid

Implemented 2026-10-18. `cx.experimental.sigmoid(value)` computes the logistic
function for exactly one float32 argument inside experimental kernels. Guarded
loads, float literals, locals and nested expressions are supported. Integer,
boolean and index arguments, implicit promotion, keyword arguments and Python
math aliases are rejected. No ordinary tensor `cx.sigmoid` API is added.

## Numerical contract

| Input | Result |
| --- | --- |
| Either signed zero | Exactly `0.5` |
| `+inf` | Exactly `1` |
| `-inf` | Positive zero |
| NaN | NaN |

Results stay in `[0, 1]`. NaN payload/sign and signaling exceptions are
unspecified. This is sampled bounded-error acceptance, not correctly rounded
or bit-identical cross-target math. In the default rounding environment, the
independent oracle computes sign-separated float64 logistic values and rounds
once to float32. Finite errors are measured in float64 against
`rtol * abs(reference) + atol`:

| Target | rtol | atol | Numerical scope |
| --- | --- | --- | --- |
| CPU / MLIR and CPU reference | `1e-6` | `2^-149` | All float32 inputs, including subnormal inputs/results and underflow boundaries |
| CUDA / MLIR | `1e-5` | `2 * 2^-149` | Same scope, on the validated sm_52 target |
| Direct Metal / MSL | `1e-6` | `2^-149` | `x >= -80`, `x <= -128` and NaNs, including infinities and subnormal inputs |

The Metal interval `-128 < x < -80` is outside numerical acceptance: output
gradual underflow and its rounding boundaries are not promised there. Tiny
inputs near zero are accepted because their sigmoid rounds to `0.5`; this
differs from operations whose output preserves the tiny input itself.
CPU/CUDA gradual output underflow is tested, with only one/two smallest-subnormal
units of absolute allowance. This does not hide flushing normal results to
zero. Classification, zero sign, exact special results and range are checked
separately. Composed operations retain their own error and conditioning limits;
approximate values can change a later comparison near a threshold. Selection
is eager and does not protect unsafe loads.

## Stable lowering and cache policy

Each target computes `e = exp(-abs(x))`, `d = 1 + e`, then selects `1 / d` for
`x >= 0` and `e / d` otherwise. The exponential never receives a positive
argument. Unlike `1 / (1 + exp(-x))`, this avoids positive exponential overflow
for large negative inputs. Unlike `(1 + tanh(x / 2)) / 2`, it avoids cancellation
of small negative-tail results.

CPU emits `math.absf`, `arith.negf`, `math.exp`, addition, division and ordered
value selection without fast-math flags. Pinned LLVM 21.1.8 lowers the exp to
system `expf`; see the [pinned conversion](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Conversion/MathToLLVM/MathToLLVM.cpp).
There is no new `math.sigmoid` or `sigmoidf` external symbol. The emitted
`math.exp` reuses explicit `-lm` linkage after objects, existing libm linker and
runtime fingerprints, and cache validation. CPU kernel names `expf`, `logf` and
`tanhf` remain reserved; `sigmoidf` is not reserved. [SiLU](MLIR_SILU.md) also emits exp and reuses this policy. Other kernels retain
their existing flags, dependencies and cache keys.

CUDA clamps the absolute magnitude to `128` using IEEE minimum before
multiplication by float32 `-log2(e)`. This prevents the scaling multiplication
from overflowing for large finite inputs; at that clamp, sigmoid's negative
tail already rounds to positive zero. IEEE minimum propagates NaNs. Non-FTZ
`llvm.nvvm.ex2.approx.f` then computes `e`; strict float32 divisions and ordered
selection produce the result. No libdevice, CUDA SDK lookup, new external linker
or newer-architecture activation instruction is needed. The CUDA tolerance
includes base conversion and exponential approximation, not just the
[PTX ex2 instruction error](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-ex2).

MSL emits one private inline float32 helper with qualified `::metal::fabs` and
`::metal::exp`. Helper names avoid kernel, parameter and local collisions; calls
use global qualification, and nested calls do not duplicate the operand source.
Kernels containing sigmoid use the existing `-fno-fast-math` compiler/cache
policy. See the [Metal language reference](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).
The CPU reference uses float32 NumPy operations under local error handling, so
ambient NumPy exception settings do not change the result contract.

## Example and retained boundaries

After the external LLVM setup in the [CPU](MLIR_RUNTIME_INTEGRATION_DECISION.md)
or [CUDA](MLIR_CUDA_INTEGRATION_DECISION.md) guide, save this inspectable source:

```python
import tensorcx as cx

@cx.experimental.kernel
def logistic_product(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.sigmoid(a[i] * b[i])

device = "cpu"  # "cuda" on the validated experimental CUDA host
compiled = logistic_product.compile(target=device, compiler="mlir")
a = cx.tensor([-20.0, 0.0, 20.0], device=device)
b = cx.ones((3,), device=device)
out = cx.zeros((3,), device=device)
result = compiled.launch(a, b, out, 3)
cx.testing.assert_allclose(result, cx.tensor([2.0611537e-9, 0.5, 1.0], device=device), rtol=1e-5, atol=0)
assert list(out.numpy()) == [0.0, 0.0, 0.0]
```

Direct Metal compiles with `compile(target="metal")` and returns the in-place
output. CPU/CUDA launches return a private output copy, redirect input aliases,
preserve unwritten suffixes and leave callers unchanged. Zero work still
validates inputs/artifacts. CUDA retains two float32 inputs, one output and one
uint32 bound; CPU can accept one input. Each sigmoid costs one DSL operation
inside unchanged 64-operation, 32-local and depth-16 CUDA budgets. Expanded
instructions do not enlarge that budget or alter manifests/native ABI.
No broadcasting, streams, general graph compiler or device-index extension is
added. Source and existing options distinguish sigmoid in cache keys, with
validation on hits and no NumPy fallback during compiled launches.

## Verification

`test_mlir_sigmoid.py` covers product, local/nested, sqrt/divide/select,
shadowed-name, min/max and multiply-with-epsilon forms. Shapes include scalar,
empty, block boundaries and multidimensional tails. Tests cover prefix aliases,
repeated/reordered launches, caller preservation, invalid loads/types/launches,
exact/one-over CUDA budgets, cold/warm sigmoid-vs-abs caches, CPU libm policy and
single-input compilation. Metal tests cover helper/kernel/parameter/local name
collisions and linear source growth for nested calls.

The 68,197-value numerical dataset includes special bit patterns, neighbors
of origin/saturation/underflow boundaries, signed powers of two with neighbors,
32,769 values in `[-110, 110]`, 16,385 in `[-20, 20]`, signed log-spaced values
and seeded random bit patterns. Explicit negative-tail checks include normal
and subnormal outputs and require nonzero results where representable.
Six direct Metal cases exercise five shapes with the helper-collision kernel
and the accepted numerical domain. Input immutability uses a post-import
snapshot because the existing bridge can quiet signaling NaNs.

The strict CUDA push gate includes sigmoid parity/numerical/non-FTZ anchors.
GPU-free LLVM CI runs actual CPU kernels and PTX emission; local Linux skips
do not validate Metal. No speedup, additional NVIDIA architecture, portable
package or full-compiler completion claim follows from these tests.

Local Linux x86_64/LLVM 21.1.8/Toolkit 12.4/sm_52 evidence:

| Check | Result |
| --- | --- |
| New CPU and real CUDA tests | 351 passed |
| Full runtime regression | 8,846 passed, 220 expected platform/capability skips |
| CUDA selection under memcheck/racecheck | 157 passed per tool, zero errors/hazards |
| Invalid/parser/policy checks with LLVM unavailable | 28 passed |
| GPU-free PTX emission and complexity validation | Three passed |

Sanitizers select `cuda and not cache`; cache and complexity-bound compilations
run normally. `--report-api-errors no` excludes the native loader's intentional
out-of-range parameter probe; memory/race error exit checks stay enabled.
Independent QA checked 770,061 inputs on CPU and real CUDA against a separate
float64 oracle: random bits, uniform `[-110,110]`, signed subnormals and 8,192
neighbors on each side of negative underflow and positive saturation boundaries.
Numerical bounds, output range, tiny-input half results, prefix aliases,
unwritten suffixes and caller immutability passed. This is sampled acceptance,
not exhaustive validation of all float32 inputs or other devices.
