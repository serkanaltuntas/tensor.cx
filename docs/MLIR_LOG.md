# Guarded float32 kernel natural logarithm

Implemented 2026-10-18. `cx.experimental.log(value)` accepts exactly one
float32 value inside experimental kernels: guarded loads, finite float literals,
locals and nested expressions. Integer/bool/index values, implicit promotion,
keyword arguments and Python math aliases are unsupported. Ordinary tensor
`cx.log` is a separate operation.

## Domain and numerical contract

| Input | Result |
| --- | --- |
| Positive finite value | Bounded-error natural logarithm |
| `1` | Positive zero |
| `+0` or `-0` | Negative infinity |
| Negative nonzero value, including `-inf` | NaN |
| `+inf` | Positive infinity |
| NaN | NaN |

NaN payload/sign and signaling exceptions are unspecified. Domain errors do
not raise Python exceptions; the CPU interpreter handles NumPy error settings
locally. Positive subnormal inputs receive finite logarithms on the validated
CPU/CUDA paths. Metal may flush subnormal inputs and has no promise for nonzero
subnormal arguments; its numerical acceptance excludes them.

Acceptance uses a float64 logarithm rounded once to float32 as an independent
oracle. Under the validated default rounding environment, standalone finite
results are checked in float64 against `rtol * abs(reference) + atol`:

| Path | rtol | atol | Tested scope |
| --- | --- | --- | --- |
| CPU / MLIR | `1e-6` | `0` | Full positive float32 range, including subnormal inputs; domain/special inputs separately |
| CUDA / MLIR | `1e-6` | `2e-7` | Same scope on validated sm_52 hardware |
| Metal / MSL | `1e-6` | `0` | Positive normal inputs, zero, negative normal values, infinities and NaNs |

NaN/infinity classifications and `log(1)=+0` are checked separately. The CUDA
absolute allowance covers base-2 approximation error close to one, where the
true logarithm approaches zero. This is not a correctly rounded or bit-identical
transcendental function. Numerical acceptance is sampled, not exhaustive proof
for every input/hardware. Composed arithmetic retains its own error bounds and
can amplify errors; approximate values can affect comparisons near a threshold.
Value selection is eager: both arms may evaluate, so it does not protect unsafe
loads. An unselected NaN does not contaminate the selected value.

## Lowering and cache policy

CPU emits `math.log` without fast-math flags. LLVM 21.1.8 lowers it through
`llvm.log.f32` to system `logf`; the compiler uses the explicit libm linkage
and cache fingerprint policy introduced for [kernel exp](MLIR_EXP.md).
Exp/[sigmoid](MLIR_SIGMOID.md)/[SiLU](MLIR_SILU.md)/log/[tanh](MLIR_TANH.md) kernels link `-lm` after object inputs and fingerprint both libm linker
and runtime files. Existing exp kernels retain their flags/cache identity;
kernels with none of these operations retain their defaults.
CPU compilation rejects kernel names `logf`, `expf` and `tanhf` before tools/cache/native
loading to prevent collision with those external libm symbols. This name
restriction does not apply to CUDA or direct Metal kernels.

CUDA emits `llvm.nvvm.lg2.approx.f` and multiplies its result by float32 `ln(2)`
(`0x3f317218`). PTX contains non-FTZ `lg2.approx.f32`/`mul.f32`; positive
subnormal inputs are not silently treated as zero. No libdevice or CUDA SDK
path lookup is added. The tolerance includes the base-2 approximation and
rounded conversion to a natural logarithm. See the
[pinned intrinsic definitions](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/include/llvm/IR/IntrinsicsNVVM.td)
and [PTX logarithm specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-lg2).

MSL emits qualified `::metal::log`, including with parameters/locals named
`log`. Kernels containing log use the existing `-fno-fast-math` Metal cache
policy; exp/min/max policies remain unchanged. See the [Metal language reference](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf),
sections 1.6.3, 8.1 and 8.4, for math options, subnormal behavior and accuracy.

## Example and retained boundaries

After the external LLVM setup in the [CPU](MLIR_RUNTIME_INTEGRATION_DECISION.md)
or [CUDA](MLIR_CUDA_INTEGRATION_DECISION.md) guide, save this in an inspectable
Python source file:

```python
import tensorcx as cx

@cx.experimental.kernel
def logarithm_product(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.log(a[i] * b[i])

device = "cpu"  # "cuda" on the validated experimental CUDA host
compiled = logarithm_product.compile(target=device, compiler="mlir")
a = cx.tensor([1.0, 4.0], device=device)
b = cx.ones((2,), device=device)
out = cx.zeros((2,), device=device)
result = compiled.launch(a, b, out, 2)
cx.testing.assert_allclose(result, cx.tensor([0.0, 1.3862944], device=device), rtol=1e-6, atol=2e-7)
assert list(out.numpy()) == [0.0, 0.0]
```

CPU/CUDA remain Linux x86_64/external LLVM 21.1.8 paths; CUDA execution is
validated with Toolkit 12.4 on sm_52. CUDA needs two inputs, one output and one
uint32 bound; CPU can accept a single input. Log counts as one node in the
existing 64-operation/32-local/depth-16 CUDA budget. Native ABI and
`tensorcx_expr_v1`/`expr-f32-v1` manifest are unchanged.

CPU/CUDA launches return a private output copy, redirect its input aliases,
preserve unwritten suffixes and leave callers unchanged. Direct Metal keeps
in-place output. Zero work still validates inputs/artifacts. No NumPy fallback,
general broadcasting, reductions, streams or device-index extension is added.

## Verification

`test_mlir_log.py` checks standalone/composed log, locals/name shadowing,
exp/division/sqrt/select/min/max composition, literals, scalar/empty/tail shapes,
blocks, prefix aliases, repeated/reordered launches and no NumPy fallback.
Numerical inputs cover special values, neighbors of zero/one/minimum normal/
maximum finite values, every representable power of two and their neighbors,
4,097 values near one, 4,096 values across positive magnitudes and 4,096 seeded
random bit patterns. Parser/type/arity and unsafe-load rejection, exact/one-over
complexity bounds, non-FTZ PTX without external functions, log-vs-abs cache
separation and CPU/Metal compiler policies are covered.

Six new Metal cases exercise five finite shapes and the documented normal-input
domain/special-value acceptance. Input immutability uses a post-import snapshot,
since the existing import/export bridge can quiet signaling NaNs. The strict
CUDA push gate requires log parity/numerical/PTX anchors. GPU-free LLVM CI runs
CPU kernels and emits PTX; Linux skips are not Metal evidence. No speedup,
additional NVIDIA architecture, portable release package or broad compiler
phase completion is claimed.

Local Linux x86_64/LLVM 21.1.8/Toolkit 12.4/sm_52 evidence, under the default
rounding environment:

| Check | Result |
| --- | --- |
| New CPU and real CUDA tests, including libm name rejection | 298 passed (included in the focused/cache run) |
| Focused log, exp and existing cache regression | 627 passed, four expected Metal skips |
| Full runtime regression | 7,815 passed, 202 expected platform/capability skips |
| CUDA selection under memcheck/racecheck | 135 passed per tool, zero errors/hazards |
| Invalid/parser/policy checks with LLVM unavailable | 21 passed |
| GPU-free PTX emission and complexity validation | 3 passed |

Sanitizers select `cuda and not cache`; cache and complexity-bound compilations
run normally. `--report-api-errors no` excludes the native loader's intentional
out-of-range parameter probe; memory/race error exit checks remain enabled.
Independent QA also checked 344,072 random-bit, near-one, positive-subnormal and
special inputs on CPU and real CUDA against the same separate float64 oracle,
including caller immutability with output aliasing. This is additional sampled
acceptance, not an exhaustive guarantee for every float32 input or device.
