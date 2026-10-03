# MLIR CUDA guarded local expressions: scope decision

Date: 2026-09-29 · Baseline: `480425f` · Status: **implemented on Nightblade, 2026-10-03; verification recorded below**.

This is the next bounded extension after [CUDA add/subtract/multiply](MLIR_CUDA_INTEGRATION_DECISION.md).
The original decision was documentation-only. The 2026-10-03 implementation
now supports this bounded subset; it does not change phase-completion status.
The broader [CUDA product completion goal](CUDA_PRODUCT_COMPLETION.md) remains
active beyond this compiler increment.

## Selected subset

Keep `kernel.compile(target="cuda", compiler="mlir").launch(...)`, exactly two
distinct input parameter names, one output parameter and one uint32 guard.
All four parameters must be used; parameter order may vary. Preserve the
canonical global index and the single outer `if i < n` guard. Inside that guard:

- Permit zero or more float32 local assignments followed by exactly one final
  `out[i] = expression` store. Locals are assigned once, cannot shadow parameters
  or the index, and may reference only earlier locals.
- Expressions contain canonical `a[i]`/`b[i]` loads, earlier float32 locals,
  finite float literals, and nested `+`, `-`, `*`. Both input parameters must
  occur in loads. Repeated loads and reuse of a local are allowed.
- Reject reads from the output parameter, multiple stores, statements after
  the store, undefined/reassigned locals, and locals outside the guard. Actual
  tensor objects may still alias, including input/output aliases.
- Reject integer/bool literals in arithmetic, runtime float scalar arguments,
  division, comparisons in the body, branches, loops, reductions, indexing
  other than `i`, extra parameters, broadcasting and non-contiguous views.
- Bound this first compiler slice to 32 local assignments, 64 binary arithmetic
  nodes in the guarded body and expression-tree depth 16. A leaf has depth 1;
  a binary node has depth `1 + max(children)`. Count each syntactic expression
  once, without expanding local references. Reject larger bodies before tools.
  These are initial compiler complexity limits, not hardware limits.

Save this example as inspectable Python source. Both explicit CPU and CUDA
compilation are available in their documented environments:

```python
import cortex_runtime as cx

@cx.experimental.kernel
def blend(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        delta = a[i] - b[i]
        scaled = delta * 0.5
        out[i] = scaled + b[i]

compiled = blend.compile(target="cuda", compiler="mlir")
a = cx.ones((257,), device="cuda")
b = cx.zeros((257,), device="cuda")
result = compiled.launch(a, b, b, 257)
assert result.cpu().numpy()[0] == 0.5
assert b.cpu().numpy()[0] == 0.0
```

The equivalent nested store `(a[i] - b[i]) * 0.5 + b[i]` is in scope. This is
explicit arithmetic within one user-authored kernel. Automatic fusion of
separate Tensor operations, graph tracing and a general optimizer are deferred.

## Numerical and ownership contract

Preserve the IR expression tree and statement dependencies. Each arithmetic
operation computes float32 with its own rounding; local reuse must not change
evaluation semantics. Reuse the finite float32 literal normalization in
[`format_f32_constant`](../python/cortex_runtime/_compiler/emitter.py).
Reject NaN/infinite/out-of-range literals before invoking tools; tensor data
may contain NaNs, infinities, signed zeros and subnormals.

Do not enable fast math, reassociation, flush-to-zero or multiply/add contraction.
Here, fusion means a single launch with register intermediates, not FMA
semantics. Source flags alone are insufficient proof: verify generated PTX and
device results for a rounding-sensitive multiply/subtract chain before enabling
this slice. If the pinned lowering/JIT cannot preserve separate rounding,
leave the feature disabled and record the blocker rather than relax parity.

Keep exact shape/dtype matching, the current synchronous context/module owners,
immutable inputs and new-result semantics. Copy the supplied output privately,
preserve its unwritten suffix, redirect its input aliases to the copy, and
publish only on success. All accesses remain at the same guarded index, and
the single final store cannot affect another thread's inputs. Zero work still
validates arguments and returns a copy without launching zero blocks.

## Integration and artifact identity

Keep the current Linux x86_64 / LLVM 21.1.8 / sm_52 / Runtime 12.4 / Driver API
13.0 gate. Keep the four existing pointer/u32 argument slots, parameter-order
metadata, private registry IDs and backend-neutral execution request.

Retain `cortex_add_v1`, `cortex_sub_v1`, `cortex_mul_v1` and their manifests for
the currently supported single-operation forms. Use a new `cortex_expr_v1`
entry and `expr-f32-v1` operation tag for extended bodies under the existing
`cortex.cuda.v1` envelope. Native loading must match entry, manifest and Driver
parameter metadata; dispatch must match the entry to the resolved module.
Different expressions may share an entry name because each module has a unique
opaque registry ID. The manifest identifies a supported ABI/subset; it does
not prove arbitrary externally supplied PTX implements the validated expression.
Loading remains a private compiler interface, not a public PTX execution API.

Implementation belongs in the CUDA Python validator/emitter and backend-private
loader. Reuse CPU guard/type validation where sound, then apply stricter CUDA
body checks; do not narrow existing CPU syntax to match this subset. Emit local
bindings as SSA values and nested arithmetic recursively, with generated names
independent of user identifiers. Validate direct IR inputs as well as parsed
Python so the private compiler cannot bypass scope checks. Any shared literal
helper extraction must preserve existing CPU behavior. No LLVM dependency,
CUDA handles or new primitive operation is added to the C++ core.

## Implementation sequence and acceptance gates

1. Add failing examples and negative tests for the selected syntax. Keep a CPU
   reference and compiled CPU result for every supported expression. Confirm
   parser/type validation and resource limits fail before subprocess execution.
2. Add the bounded GPU emitter and generate a small expression PTX fixture with
   the pinned toolchain. Check separate float32 operations, guard/index structure,
   entry/manifest and parameter widths. Inspect the contraction discriminator's
   PTX for separate rounding and absence of fused multiply-add instructions.
3. Add only the new native entry/tag pairing. Execute the fixture using real
   Cortex buffers before exposing extended public compile acceptance. Reject
   mismatched old/new entries and manifests, forged requests and stale IDs.
4. Enable the public path after CPU/reference/device parity and ownership tests
   pass. Preserve current single-op fixture lowering and the previous add/sub/mul
   behavior. Verify distinct expression modules cannot dispatch each other's
   artifacts accidentally, including concurrent execution and destruction.
5. Run focused and full Python suites, native CTest and the documented CUDA
   sanitizers. Check actual CUDA-off builds with and without LLVM, CPU-only
   emitter CI, missing tool/GPU errors, cleanup and no implicit fallback. Obtain
   two code reviews plus Test and Acceptance QA before committing implementation.

Required coverage:

| Area | Cases and expected result |
| --- | --- |
| Expressions | Local chain, equivalent nested tree, reused local, repeated load, constants on either side, noncommutative operand order, reordered parameters |
| Rejections | Every excluded construct above; exact complexity limits accepted and one-over limits rejected before tools; finite overflow/non-finite constants; direct malformed IR |
| Geometry | Scalars, empty and multidimensional shapes; 1/255/256/257 elements; blocks 1/7/256; zero and partial prefixes |
| Ownership | Distinct output, output aliases left/right/both; unchanged inputs/suffix; repeated launches; validation failure leaves caller output unchanged |
| Numerics | CPU interpreter, compiled CPU and explicit NumPy float32 steps; normal finite values at project tolerance; exact discriminator, signed zero and subnormal cases; NaN masks/infinity signs without NaN-payload promises |
| Native lifecycle | Wrong shape/dtype/device/context, argument layout and guard, invalid block/grid, wrong entry/tag/artifact, concurrent lifetime and restored foreign context |

No speedup or reduced memory-traffic claim follows from this decision. The
private output copy remains. Benchmarking is a separate task after correctness;
other GPUs/hosts, Metal parity and broader compiler features need separate gates.

## Evidence for this decision

Code inspection at `480425f` found CPU local/nested expression support in
[`_signature`](../python/cortex_runtime/_compiler/cpu.py), the shared
[emitter](../python/cortex_runtime/_compiler/emitter.py), and the float32
[reference interpreter](../python/cortex_runtime/experimental.py). The current
[CUDA signature validator](../python/cortex_runtime/_compiler/cuda.py) required
one binary store over two loads and deliberately rejected this extension.

On 2026-09-29, a temporary inspectable Python probe compiled local and nested
forms of the example plus `product = a[i] * b[i]; out[i] = product - 1.0` on
the pinned CPU toolchain. For zero, partial and full work, all three matched
the reference exactly and preserved the supplied output; all three were
rejected by the current CUDA validator. For float32 inputs `1 + 2**-23` and
`1 - 2**-23`, the CPU/reference chain produced zero, while exact multiplication
followed by subtracting one and one final float32 rounding produced `-2**-46`.
This is a required discriminator for future CUDA validation, not evidence that
CUDA expression execution or contraction control is already implemented.

The relevant existing regression selection was also run:

```bash
CORTEX_LLVM_BIN="$PWD/build/mlir-toolchain/root/usr/lib/llvm-21/bin" CORTEX_REQUIRE_MLIR=1 uv run pytest tests/python/test_mlir_runtime.py tests/python/test_mlir_cuda_runtime.py -k 'native_parity or unsupported_before_tools or gpu_emitter' -q
```

**30 passed, 265 deselected**. The probe was temporary; no new runtime, test
fixture or public capability is shipped in this documentation-only decision.


## Implementation verification — 2026-10-03

The private CUDA compiler now validates bounded direct IR before recursive
shared validation, emits local SSA bindings/nested arithmetic and retains the
old three single-operation entry/fixture paths unchanged. Native loading adds
only the `cortex_expr_v1`/`expr-f32-v1` pairing. The checked expression fixture
computes `a*b-1` with separate `mul.rn.f32` and `add.rn.f32` instructions;
no `fma`/`mad` instruction appears. Real CUDA execution gives zero for the
rounding discriminator above. Signed zero and subnormal checks also passed.
The fixture ran through native Cortex buffers before public expression tests.

Final local evidence (CUDA-enabled build restored after CUDA-off checks):

| Configuration | Result |
| --- | --- |
| Full Python suite, required LLVM and CUDA, including expression tests | 800 passed, 149 skipped |
| Actual CUDA-off build with required LLVM | 334 passed, 615 skipped |
| Actual CUDA-off build with LLVM absent | 277 passed, 672 skipped |
| Native CTest, including expression fixture | 4/4 passed |
| Native ASan/UBSan with the documented Nightblade workaround | 4/4 passed |

The CUDA-off extension's dynamic dependencies contain neither CUDA nor LLVM.
Tests exercise exact complexity bounds with real CPU/GPU compilation, as well
as one-over rejection before tools. Code review found an index/parameter
shadowing gap in direct IR; the fix and before-tools regression are included.
GPU-free MLIR CI now selects expression fixture lowering and validation tests.
Commands follow the [CUDA integration verification](MLIR_CUDA_INTEGRATION_DECISION.md#runtime-usage-and-verification--2026-09-28),
with `tests/python/test_mlir_cuda_expressions.py` for focused expression coverage.

Broader operator, GPU compatibility, performance and packaging work is tracked separately; this feature is not overall product
completion. Metal execution and other NVIDIA architectures have not been
verified by this increment.
