# Float32 kernel comparisons and select

Date: 2026-10-17. This extension adds float32 comparisons and explicit value
selection to guarded experimental CPU/CUDA MLIR bodies. It does not add a
general branch compiler or automatic fusion of ordinary tensor operations.

## Contract

Use `cx.experimental.select(condition, on_true, on_false)` inside a kernel.
The condition must be a comparison or an earlier boolean local; both values
must be float32 loads, floating literals, float locals or float expressions.
Write `1.0`, not `1`. All three arguments are positional. Outside a kernel the
function raises `NotImplementedError`; ordinary tensors still use `cx.where`.

The runtime slice accepts `<`, `<=`, `>`, `>=`, `==`, `!=` between two float32
expressions. Results are internal boolean values: they may feed selection or
be assigned to a local. Boolean output buffers, arithmetic on booleans,
boolean-vs-boolean comparisons, integer/mixed comparisons in the body and
truthiness conversion are outside this slice. The canonical integer launch
guard remains unchanged. `if/else`, Python conditional expressions, boolean
`and/or/not`, new scalar arguments and broadcasting are not enabled.

Selection consumes values, not lazy branches. The reference evaluates every
argument; compiler optimizations may eliminate unused work, but users must
assume both values can be evaluated. All loads in the condition and either
value must independently use the canonical guarded index. Selection cannot
hide an out-of-bounds load or unsupported expression. CUDA still rejects all
output reads; CPU retains its earlier guarded-output-read policy. IEEE NaN
produced in an unselected floating-point value does not contaminate the result.

CPU/CUDA emit `arith.cmpf` and `arith.select` without fast-math flags. The first
five comparisons are ordered: NaN makes them false. `!=` uses unordered-or-not-
equal (`une`), so NaN != x is true. Positive and negative zero compare equal;
the chosen float32 value retains its sign bit. Tests compare NaN masks and all
non-NaN bits, including subnormals and infinities; NaN payload/sign are not a
promise. Selection is not an alias for minimum/maximum with a different NaN
or tie policy. See [MLIR comparison and selection semantics](https://mlir.llvm.org/docs/Dialects/ArithOps/).

## Example

Save this in an inspectable Python source file:

```python
import tensorcx as cx

@cx.experimental.kernel
def lower_value(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        take_left = a[i] <= b[i]
        out[i] = cx.experimental.select(take_left, a[i], b[i])

device = "cpu"  # "cuda" on the validated experimental CUDA host
a = cx.tensor([2.0, 6.0], device=device)
b = cx.tensor([1.0, 8.0], device=device)
out = cx.zeros((2,), device=device)
compiled = lower_value.compile(target=device, compiler="mlir")
assert list(compiled.launch(a, b, out, 2).numpy()) == [1.0, 6.0]
assert list(out.numpy()) == [0.0, 0.0]
```

Comparisons, selected values and arithmetic may be nested. For a float mask,
write `cx.experimental.select(a[i] < b[i], 1.0, 0.0)`; inputs used only in the
condition are still validated and included in the signature. The independent
`kernel.reference(...)` runs on CPU with the same geometry/shape rules.

## Boundaries and compatibility

Installation and host gates stay in the [CPU](MLIR_RUNTIME_INTEGRATION_DECISION.md)
and [CUDA](MLIR_CUDA_INTEGRATION_DECISION.md) guides. LLVM 21.1.8 remains external
and optional for ordinary import/build/tensor use. CUDA still requires the
validated Linux x86_64/sm_52/Toolkit 12.4/Driver API 13.0 environment.

CUDA retains two inputs, one output and one uint32 bound. Its maximum remains
32 local assignments, expression depth 16 and 64 operation nodes. That last
budget now counts arithmetic, comparison and selection nodes together; a
comparison plus selection costs two. Bounds are checked before recursive
shared validation or tools. General runtime branches/reductions remain deferred.

Comparison/select bodies use the existing `tensorcx_expr_v1`/`expr-f32-v1`
entry/manifest and pointer/u32 layout. Native code, context/module ownership
and legacy single-add/subtract/multiply entries are unchanged. The emitted
source distinguishes predicates and value order in the private cache key;
cache and native-load validation still run. See the [cache trust contract](COMPILATION_CACHE.md).

CPU/CUDA MLIR launch returns a private output copy, redirects its input aliases
to that copy, preserves unwritten suffixes and leaves caller tensors unchanged.
Zero work still validates arguments/artifacts. There is no NumPy fallback.

Direct MSL kernels also accept the selector, emitted as the qualified Metal
builtin with false/true/condition argument order. They retain their in-place
output contract. Metal coverage uses finite normal inputs at the existing
`rtol=atol=1e-6`; the CPU/CUDA IEEE bit checks are not a Metal bit-parity claim.

## Verification

`test_mlir_select.py` covers all six predicates, boolean locals, nested selection,
arithmetic/division values, float masks, names shadowing the selector, all
standard shapes/blocks, prefix/zero work, input/output aliases, repeated and
reordered launches, invalid arguments, no NumPy fallback, cache reuse and
same-name predicate separation. Each CPU/CUDA kernel is checked against an
independent NumPy oracle, including 289 special-value pairs and 1,024 random-bit
pairs. It also checks PTX predicate/selection instructions, strict operand
types, parser arity/keywords, hidden unsafe loads and exact/one-over complexity
bounds. Five Metal shape cases exercise the same ten kernel forms separately.

The strict CUDA push gate includes this file plus parity/numerical/PTX anchors.
GPU-free LLVM CI executes CPU kernels and emits PTX. Metal execution requires
the existing macOS CI jobs; local Linux skips do not validate Metal. This
feature makes no performance, additional GPU architecture or release-package
claim.

Local evidence on Linux x86_64/LLVM 21.1.8/Toolkit 12.4/sm_52:

| Check | Result |
| --- | --- |
| New select tests, required CPU and real CUDA | 875 passed, no skips |
| Full regression | 5,683 passed, 176 expected platform/capability skips |
| CUDA launch cases under memcheck | 426 passed, zero memory errors |
| Same cases under racecheck | 426 passed, zero hazards |
| Type/parser rejection with LLVM unavailable | 18 passed |

Sanitizers select `cuda and not cache`; cache and boundary-compilation tests
run normally. `--report-api-errors no` excludes the loader's intentional
out-of-range parameter query; memory/race error exit checks remain enabled.
The new Metal cases are skipped on this Linux host and require real macOS CI.
