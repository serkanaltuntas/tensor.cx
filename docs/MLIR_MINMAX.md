# Guarded float32 kernel minimum and maximum

Implemented 2026-10-17. `cx.experimental.minimum(lhs, rhs)` and
`cx.experimental.maximum(lhs, rhs)` accept exactly two float32 values inside
experimental kernels. These are per-element expressions, separate from ordinary
`cx.min`/`cx.max` tensor reductions. Integer/bool/index operands, implicit
promotion, keyword arguments and Python builtin `min`/`max` aliases are not
part of this kernel subset. Finite float literals and earlier float32 locals
are accepted alongside guarded buffer loads.

## Numerical contract

Both operations propagate NaN: if either operand is NaN, the result is NaN.
NaN payload/sign and signaling exceptions are not promised. Otherwise the
result is the smaller/larger operand, ordering `-0` below `+0` independently of
argument order. Equal nonzero values retain their common float32 bits.

| Inputs (either order) | minimum | maximum |
| --- | --- | --- |
| `-0`, `+0` | `-0` | `+0` |
| `-0`, `-0` | `-0` | `-0` |
| `+0`, `+0` | `+0` | `+0` |
| number, NaN | NaN | NaN |
| `-inf`, `+inf` | `-inf` | `+inf` |

Subnormal magnitudes are preserved by the standalone operations. Acceptance
compares NaN masks and all non-NaN result bits against an independent
comparison/zero-sign oracle. NumPy's default minimum/maximum zero tie behavior
can depend on operand order; the CPU interpreter explicitly fixes those ties.
Composed arithmetic, sqrt and selection retain their respective contracts.
Both operands are values; these functions do not provide lazy branches or
replace a memory-safety guard.

## Lowering

CPU uses `arith.minimumf`/`arith.maximumf` without fast-math flags and LLVM
21.1.8's corresponding intrinsics. CUDA emits `llvm.minimum.f32`/
`llvm.maximum.f32` via `llvm.call_intrinsic`. LLVM expands them for the validated
sm_52 target into min/max plus NaN and signed-zero handling; no libdevice or
sm_80-only `min.NaN`/`max.NaN` instruction is required. See
[MLIR semantics](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithminimumf-arithminimumfop),
[LLVM semantics](https://llvm.org/docs/LangRef.html#llvm-minimum-intrinsic) and the
[pinned NVPTX lowering](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/lib/Target/NVPTX/NVPTXISelLowering.cpp).

MSL uses generated helpers with integer NaN classification and ordered float32
bit keys. Each operand is evaluated once at the helper call; qualified helper
names avoid parameter/local/kernel-name collisions. This preserves zero signs
and subnormal bits without floating-point min/max assumptions. Kernels containing
minimum/maximum also compile with `-fno-fast-math`; the option participates in
the Metal artifact cache key. Kernels without these operations keep their
existing command/options/cache identity. Direct Metal results for standalone
min/max receive special-value bit checks; finite composed expressions retain
the existing `rtol=atol=1e-6` acceptance. This is not a broader Metal arithmetic
bit-parity claim. See [Metal math options](https://developer.apple.com/documentation/metal/mtlcompileoptions/fastmathenabled).

## Example

After the external LLVM setup in the [CPU](MLIR_RUNTIME_INTEGRATION_DECISION.md)
or [CUDA](MLIR_CUDA_INTEGRATION_DECISION.md) guide, save this in an inspectable
Python source file:

```python
import tensorcx as cx

@cx.experimental.kernel
def capped_positive(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.minimum(cx.experimental.maximum(a[i], 0.0), b[i])

device = "cpu"  # "cuda" on the validated experimental CUDA host
compiled = capped_positive.compile(target=device, compiler="mlir")
a = cx.tensor([-3.0, 5.0], device=device)
b = cx.tensor([1.0, 4.0], device=device)
out = cx.zeros((2,), device=device)
assert list(compiled.launch(a, b, out, 2).numpy()) == [0.0, 4.0]
assert list(out.numpy()) == [0.0, 0.0]
```

## Retained boundaries

Host gates remain Linux x86_64/external LLVM 21.1.8; CUDA execution is validated
on sm_52 with Toolkit 12.4 and the documented driver. CUDA still requires two
input parameters, one output and one uint32 bound, with canonical guarded
indices. Each minimum/maximum costs one node in the existing 64-operation
budget; 32 locals and depth 16 remain the limits. CPU permits other supported
buffer counts. No native ABI, stream, device-index or general reduction change.

Bodies use the existing `tensorcx_expr_v1`/`expr-f32-v1` CUDA manifest. Source
and compilation options separate operations/policies in cache keys; native
loading/validation remain mandatory. CPU/CUDA launches return a private output
copy, redirect its input aliases, preserve unwritten suffixes and leave caller
tensors unchanged. Direct Metal keeps in-place output. Zero work still
validates inputs/artifacts. There is no NumPy fallback.

## Verification

`test_mlir_minmax.py` covers standalone and composed operations, float literals,
locals, selector/sqrt/abs/division composition, standard scalar/empty/tail shapes,
block sizes, prefix aliases, repeated/reordered launches, invalid arguments,
no NumPy fallback and same-name min-vs-max cache separation. Special numerical
acceptance uses all 324 pairs of 18 patterns plus 4,096 seeded random bit pairs.
Types/arity/parser rejection, unsafe loads, exact/one-over complexity limits,
sm_52 PTX, helper naming and strict/default Metal compiler/cache policies are
checked. Nine new Metal cases cover five finite shapes, two standalone IEEE
checks and two kernel/helper-name collisions.

The strict CUDA push gate includes parity/numerical/PTX anchors. GPU-free LLVM
CI runs CPU kernels and serializes PTX. Metal execution requires actual macOS
CI; Linux skips do not prove it. No performance, additional CUDA architecture,
portable package or completed broad compiler phase is claimed.

Local Linux x86_64/LLVM 21.1.8/Toolkit 12.4/sm_52 evidence:

| Check | Result |
| --- | --- |
| New CPU and real CUDA tests | 357 passed, no skips |
| Full runtime regression | 7,226 passed, 190 expected platform/capability skips |
| CUDA test selection under memcheck | 154 passed, zero memory errors |
| Same selection under racecheck | 154 passed, zero hazards |
| Rejection/helper/compiler-policy checks with LLVM unavailable | 42 passed |
| GPU-free PTX emission and complexity validation | 5 passed |

Sanitizers select `cuda and not cache`; cache and complexity-bound compilations
run normally. `--report-api-errors no` excludes the native loader's intentional
out-of-range parameter probe; memory/race error exit checks remain enabled.
The nine new Metal cases skip on Linux and require macOS CI.
