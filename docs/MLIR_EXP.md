# Guarded float32 kernel exponential

Implemented 2026-10-18. `cx.experimental.exp(value)` accepts exactly one
float32 value inside experimental kernels, including guarded loads, finite
float literals, locals and nested expressions. Integer/bool/index arguments,
implicit promotion, keyword arguments and other Python math aliases are
unsupported. Ordinary tensor `cx.exp` remains a separate operation.

## Numerical contract

This is a bounded-error exponential, not a correctly rounded or bit-identical
transcendental function. NaN produces NaN (payload/sign/signaling exceptions
are unspecified), either zero produces `1`, `+inf` produces `+inf` and `-inf`
produces positive zero. Results are nonnegative. Composed arithmetic retains
its existing contract; approximate values can change a later comparison close
to a decision boundary. Selection is eager and does not protect unsafe loads.

CPU/CUDA acceptance uses a float64 exponential rounded once to float32 as its
independent oracle. Finite result error is checked in float64 against
`rtol * abs(reference) + atol`:

| Path | rtol | atol | Tested numerical scope |
| --- | --- | --- | --- |
| CPU / MLIR | `1e-6` | `2^-149` | Normal/subnormal results, overflow/underflow boundaries and special inputs |
| CUDA / MLIR | `1e-5` | `2 * 2^-149` | Same scope, on validated sm_52 hardware |
| Metal / MSL | `1e-6` | `2^-149` | Normal-output inputs in `[-80, 80]`, special inputs and saturation well beyond the boundaries |

NaN and infinity classification are checked separately. The absolute allowance
is measured in the smallest float32 subnormal units, not a broad absolute
tolerance that would hide flushing a normal result to zero. CPU/CUDA gradual
underflow is tested; exact subnormal rounding is not promised. Metal has no
gradual-underflow or near-overflow-boundary accuracy claim. This is sampled
acceptance, not exhaustive validation of every float32 input or other hardware.

## Lowering and cache policy

CPU emits `math.exp` without fast-math flags. LLVM 21.1.8 lowers it to `expf`;
the compiler explicitly links system `libm` after its input objects. CPU kernel
names `expf`, `logf` and `tanhf` are rejected before compilation to prevent collisions
with external libm symbols (CUDA/Metal are unaffected). Only
kernels emitting exp (including [sigmoid](MLIR_SIGMOID.md) and [SiLU](MLIR_SILU.md)), [log](MLIR_LOG.md) or [tanh](MLIR_TANH.md) add `-lm` and fingerprint the libm linker input and
runtime library in the artifact cache. Other kernels retain their flags/keys.
The interpreter uses NumPy float32 exp with local error handling, so ambient
NumPy error settings do not turn overflow/underflow into Python exceptions.

CUDA multiplies by the float32 value of `log2(e)` (`0x3fb8aa3b`) and emits
`llvm.nvvm.ex2.approx.f`, producing non-FTZ `mul.f32`/`ex2.approx.f32` on sm_52.
No libdevice, CUDA SDK path lookup or new external linker is required. The
conversion to a rounded base-2 exponent adds error beyond the PTX instruction's
own error bound; the CUDA tolerance above includes that conversion. See the
[pinned intrinsic](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/include/llvm/IR/IntrinsicsNVVM.td)
and [PTX exponential specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-ex2).

MSL emits qualified `::metal::exp`, including with parameters/locals named
`exp`. Kernels containing exp compile with `-fno-fast-math`, using the existing
strict Metal cache policy; min/max kernels retain the same policy and other
kernels retain their defaults. See the [Metal language reference](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf),
sections 1.6.3, 8.1 and 8.4, for math options and target numerical limits.

## Example and retained boundaries

After the external LLVM setup in the [CPU](MLIR_RUNTIME_INTEGRATION_DECISION.md)
or [CUDA](MLIR_CUDA_INTEGRATION_DECISION.md) guide, save this in an inspectable
Python source file:

```python
import tensorcx as cx

@cx.experimental.kernel
def exponential_product(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.exp(a[i] * b[i])

device = "cpu"  # "cuda" on the validated experimental CUDA host
compiled = exponential_product.compile(target=device, compiler="mlir")
a = cx.tensor([0.0, 1.0], device=device)
b = cx.ones((2,), device=device)
out = cx.zeros((2,), device=device)
result = compiled.launch(a, b, out, 2)
cx.testing.assert_allclose(result, cx.tensor([1.0, 2.7182817], device=device), rtol=1e-5, atol=0)
assert list(out.numpy()) == [0.0, 0.0]
```

CPU/CUDA remain Linux x86_64/external LLVM 21.1.8 paths; CUDA execution is
validated with Toolkit 12.4 on sm_52. CUDA still needs two inputs, one output
and one uint32 bound; CPU can accept a single input. Exp counts as one node in
the existing 64-operation/32-local/depth-16 CUDA budget. The native ABI and
`tensorcx_expr_v1`/`expr-f32-v1` manifest remain unchanged.

CPU/CUDA launches return a private result, preserve unwritten suffixes and leave
callers unchanged. CUDA full writes use
[conditional template-copy elision](CUDA_OUTPUT_COPY_ELISION.md). Direct Metal keeps
in-place output. Zero work still validates inputs/artifacts. There is no NumPy
fallback, general broadcasting, reduction, stream or device-index extension.

## Verification

`test_mlir_exp.py` covers direct/composed exp, locals and name shadowing,
division/sqrt/select/min/max composition, literals, scalar/empty/tail shapes,
blocks, prefix aliases, repeated/reordered launches and no NumPy fallback.
Numerical inputs include 16 special patterns, 32 float32 neighbors on each
side of four saturation/subnormal boundaries, 4,097 evenly spaced inputs and
4,096 seeded random bit patterns. It also checks type/arity/parser rejection,
unsafe loads, exact/one-over complexity bounds, PTX without FTZ or external
functions, cold/warm exp-vs-abs cache separation and CPU/Metal compile policies.
Six new Metal cases exercise five finite shapes and the documented numerical
range/special-input acceptance. The strict CUDA push gate requires exp
parity/numerical/PTX anchors; GPU-free LLVM CI runs CPU kernels and emits PTX.
Linux skips are not Metal evidence; macOS CI must execute the new cases.
No speedup, additional CUDA architecture, portable package or broad compiler
phase completion is claimed.

Local Linux x86_64/LLVM 21.1.8/Toolkit 12.4/sm_52 evidence, under the default
rounding environment:

| Check | Result |
| --- | --- |
| New CPU and real CUDA tests | 291 passed (included in the focused/cache run) |
| Focused exp and existing cache regression | 329 passed, four expected Metal skips |
| Full runtime regression | 7,517 passed, 196 expected platform/capability skips |
| CUDA selection under memcheck/racecheck | 134 passed per tool, zero errors/hazards |
| Invalid/parser/policy checks with LLVM unavailable | 17 passed |
| GPU-free PTX emission and complexity validation | 3 passed |

Sanitizers select `cuda and not cache`; cache and complexity-bound compilations
run normally. `--report-api-errors no` excludes the native loader's intentional
out-of-range parameter probe; memory/race error exit checks remain enabled.
Metal numerical input immutability uses a snapshot after tensor import, since
signaling NaNs may be quieted by the existing import/export bridge.
