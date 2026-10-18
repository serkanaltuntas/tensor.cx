# Float32 kernel division

Date: 2026-10-17. This bounded extension adds `/` to experimental float32
kernel bodies. It does not complete a new compiler phase or add graph fusion.

## Contract

Both operands must be float32 loads, float locals, floating literals or
float32 expressions. Integer, boolean and mixed integer/float division are
rejected before compiler tools; write `2.0`, not `2`. Floor division, remainder,
power, integer/index division and broadcasting remain unsupported. CPU/CUDA
runtime bodies still require the canonical guarded index and exact shapes.
CUDA retains two input parameters, one output, one uint32 bound, at most 32
locals/64 operation nodes and expression depth 16. `/` counts as one operator;
the [selection extension](MLIR_SELECT.md) also counts comparisons and selectors.

CPU and CUDA emit `arith.divf` without fast-math flags. On the pinned LLVM
21.1.8/sm_52 path, CUDA emits `div.rn.f32`, with no approximate reciprocal or
flush-to-zero division. Each operation keeps float32 rounding; local reuse and
parentheses preserve the expression tree. Zero division produces IEEE infinity
or NaN rather than Python `ZeroDivisionError`. Tests compare non-NaN bits,
including signed zero, subnormal inputs/results, underflow and overflow; NaN
payload/sign are not promised. The NumPy interpreter remains independent of
generated code and narrows each operation to float32.

The direct MSL path also accepts float32 `/`. Its existing compilation/math
policy is unchanged. Metal validation covers finite-input parity at the
existing `rtol=atol=1e-6`; the CPU/CUDA IEEE edge contract is not a promise of
bit-identical Metal division.

See [MLIR arithmetic](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithdivf-arithdivfop)
and NVIDIA's [PTX division specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-div).

## Example

Save as a Python source file so the parser can inspect the function:

```python
import tensorcx as cx

@cx.experimental.kernel
def ratio(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        quotient = a[i] / b[i]
        out[i] = quotient * 0.5 + b[i]

device = "cpu"  # "cuda" on the validated experimental CUDA host
a = cx.tensor([2.0, 6.0], device=device)
b = cx.tensor([1.0, 2.0], device=device)
out = cx.zeros((2,), device=device)
compiled = ratio.compile(target=device, compiler="mlir")
result = compiled.launch(a, b, out, 2)
assert list(result.numpy()) == [2.0, 3.5]
assert list(out.numpy()) == [0.0, 0.0]
```

Installation/host limits stay in the [CPU](MLIR_RUNTIME_INTEGRATION_DECISION.md)
and [CUDA](MLIR_CUDA_INTEGRATION_DECISION.md) guides. LLVM remains an optional
external toolchain; ordinary tensor use/import/build does not need it.

## Runtime, ABI and cache

CPU/CUDA MLIR launch returns a private result, preserves unwritten suffixes and
leaves caller tensors unchanged. CUDA full writes use
[conditional template-copy elision](CUDA_OUTPUT_COPY_ELISION.md).
Zero work validates arguments/artifacts; errors do not publish partial output.
No Python/NumPy execution fallback is added.
Direct MSL kernels retain their existing in-place output contract; the Metal
test checks that result/output identity and unchanged input buffers.

CUDA division, including a single `a[i] / b[i]`, uses the existing
`tensorcx_expr_v1` entry and `expr-f32-v1` manifest. Pointer/u32 layout, native
module/context ownership and opaque artifact IDs are unchanged. Legacy
single-add/subtract/multiply entries and PTX fixtures stay unchanged. The
manifest describes the private ABI; it does not prove arbitrary external PTX
computes a validated expression or provide a public PTX-loading API.

The emitted source already participates in the cache key, so distinct division
expressions do not share artifacts. Existing supported kernels retain their
source/key format. Cache permission, dependency and native-load checks remain
mandatory; see the [cache contract](COMPILATION_CACHE.md).

## Verification

`tests/python/test_mlir_division.py` covers NumPy/reference/native CPU/CUDA
parity, scalar/empty/1/255/256/257/multidimensional shapes, block sizes 1/7/256,
partial/zero work, all input/output alias combinations, repeated/reordered
launches, float literals, invalid launches, no NumPy fallback, private cache
reuse/separation, before-tools type rejection and actual PTX instructions.
Numerical edges use 361 special-value pairs plus 1,024 deterministic random-bit
pairs per CPU/CUDA target. Five Metal shape cases check three division kernels
through the existing direct MSL path in `test_experimental.py`.

The exact-commit CUDA push gate includes the new file and numerical/PTX test
anchors. GPU-free LLVM CI exercises CPU execution and actual PTX emission;
normal CPU/no-toolchain and Metal jobs retain their platform-specific gates.
No performance improvement, additional NVIDIA architecture or portable binary
distribution is claimed by this feature.

Local verification on Linux x86_64, LLVM 21.1.8, CUDA Toolkit 12.4 and sm_52:

| Check | Result |
| --- | --- |
| New division tests, CPU and real CUDA required | 302 passed, no skips |
| Full regression suite | 4,808 passed, 171 expected platform/capability skips |
| CUDA division launch cases under memcheck | 142 passed, zero memory errors |
| Same cases under racecheck | 142 passed, zero hazards |
| Invalid-type/unsupported-op tests with LLVM unavailable | 15 passed |

The sanitizer runs select `cuda and not cache`; cache tests run normally.
`--report-api-errors no` excludes the loader's intentional out-of-range
`cuFuncGetParamInfo` probe from API diagnostics; memory/race error exit checks
remain enabled. Metal execution is unavailable on this Linux host, so the five
new Metal cases require the existing real macOS CI jobs. Local skips are not
Metal validation.
