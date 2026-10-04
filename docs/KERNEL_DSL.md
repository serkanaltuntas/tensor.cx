# Experimental Kernel DSL

This document defines the Phase 7 starting point. The DSL is experimental and
must stay behind `cx.experimental` until the AST -> IR -> backend path is proven
with tests.

## Status

Phase 7 is complete for the first narrow slice. The current API records kernel
metadata, parses a small restricted Python AST subset into backend-neutral IR,
emits text MSL, compiles that MSL into an in-memory metallib artifact when Apple
Metal command-line tools are available, validates generated functions through
native Metal library lookup, and can launch float32 elementwise kernels and rowwise-reduction
(bounded `for`/accumulator) kernels on Metal. `Kernel.reference(...)` executes the same IR on CPU tensors as the
interpreter-based reference path under the same launch contract.

```python
import tensorcx as cx

@cx.experimental.kernel(target="metal")
def add_kernel(a, b, out, n):
    i = (
        cx.experimental.program_id(0) * cx.experimental.block_size()
        + cx.experimental.thread_id()
    )
    if i < n:
        out[i] = a[i] + b[i]

print(add_kernel.name)
print(add_kernel.parameters)
print(add_kernel.parse_ir())
print(add_kernel.emit_msl())
compiled = add_kernel.compile(target="metal")
print(len(compiled.metallib))
print(compiled.validate_metal_function())

x = cx.ones((4,), dtype=cx.float32, device="metal")
y = cx.ones((4,), dtype=cx.float32, device="metal")
out = cx.empty((4,), dtype=cx.float32, device="metal")

add_kernel(x, y, out, 4, block_size=2)
print(out.cpu().numpy())
```

Do not expose a top-level `cx.kernel` API until the experimental API has a
working compiler path and stable semantics.

## First Supported Shape

The first real compiler slice should support only:

```text
operation class: elementwise
dtype: float32
outputs: one output tensor
inputs: contiguous tensors with exact matching shape
device target: Metal first, CPU reference required
execution: synchronous
```

The first generated kernel should be an elementwise add equivalent to:

```text
out[i] = a[i] + b[i]
```

It should not replace the existing static `add_f32` kernel. Its purpose is to
prove the compiler path:

```text
Python function
  -> restricted Python AST
  -> tensor.cx kernel IR
  -> generated MSL
  -> Metal launch
  -> CPU-vs-Metal test
```

## Non-Goals For Phase 7

Do not add these while proving the first DSL path:

```text
broadcasting
autograd
async execution
multi-output kernels
dynamic shapes
runtime dtype promotion
matmul
general Python semantics
CUDA/ROCm/MLIR lowering (research prototype only, see experiments/mlir/)
```

Rowwise reductions via bounded `for`/accumulator landed after the Phase 7
acceptance slice; control flow beyond `if` and one `for range(param)` level
(nested loops, while, else branches) remains unsupported.

## IR Rules

The IR must be backend-neutral. It may describe concepts such as parameters,
indices, loads, stores, constants, arithmetic, comparisons, and simple branches.
It must not contain MSL syntax, Metal handles, Objective-C types, command
queues, threadgroup details, or backend-specific buffer objects.

Expected first IR nodes:

```text
Kernel
Name
Call
BinaryOp
Compare
If
Load
Store
Assign
Constant
```

The current parser supports this intentionally small syntax subset:

```text
assignment to a local name
single-output store through one-dimensional subscript syntax
one-dimensional tensor load syntax
cx.experimental.program_id(0)
cx.experimental.thread_id()
cx.experimental.block_size()
+, -, *
single comparisons: <, <=, >, >=, ==, !=
if blocks without else
numeric constants
for var in range(bound_name) with an assignment-only body (any local
    name parses; launching requires the bound to be a scalar parameter)
loop-carried accumulator reassignment (acc = acc + ..., only for names
    defined before the loop; type-preserving; loop-locals do not escape)
```

For-loops are the first post-elementwise construct (rowwise reductions).
Launch-safety rules for loops are structural, like the store guard:

```text
- The loop must sit inside the `index < scalar_limit` store guard.
- The range bound must be a scalar parameter (also passed at launch).
- Loads inside the body must use the row-major pattern
  buffer[row * limit + loop_var]; the launch layer enforces
  numel(buffer) == numel(output) * limit exactly.
- A buffer cannot be loaded both elementwise and loop-indexed, and the
  output buffer cannot be loaded inside a loop.
- A zero loop limit runs zero iterations (sum-over-empty stays 0); the
  zero-element input buffer is bound via a provably-unread placeholder
  because zero-element Metal tensors have no native buffer.
```

Unsupported AST nodes must fail at compile time with a clear message naming the
node type. Silent fallback or partial miscompilation is not acceptable.

## MSL Emission And Compile Rules

The current `emit_msl()` path is golden-testable text generation. The current
`compile(target="metal")` path shells out to the Apple command-line Metal tools
to produce an in-memory metallib artifact. `CompiledKernel.validate_metal_function()`
then loads that artifact through the native Metal backend and verifies that the
generated function can be found.

It currently assumes:

```text
float32 tensor buffers
one mutable output buffer
const input buffers
uint scalar parameters
one-dimensional program_id/thread_id/block_size mapping
```

The compiled metallib is not cached. The current launch path creates a pipeline,
binds runtime buffers and uint32 scalar arguments in source-parameter order,
dispatches synchronously, and returns the single output tensor object supplied by
the caller. It is intentionally limited to float32 Metal tensors, one output
buffer, exact tensor shape matches, one-dimensional dispatch, and buffer
loads/stores guarded by a scalar bound such as `if i < n:`. The guard bound must
match the launch `thread_count`, and `thread_count` must not exceed the output
tensor size.

## API Rules

The experimental API starts here:

```python
cx.experimental.kernel
cx.experimental.program_id
cx.experimental.thread_id
cx.experimental.block_size
```

`@cx.experimental.kernel` currently returns a metadata wrapper with `parse_ir()`,
text-only `emit_msl()`, `compile(target="metal")` for an in-memory metallib
artifact, and experimental non-empty launch through `BackendExecution`.
`CompiledKernel` exposes `validate_metal_function()` for library load/function
lookup and `launch(...)` for the first synchronous Metal execution path.
`Kernel.reference(...)` executes the same IR on CPU tensors under the same
launch contract, interpreting with MSL/C-matching scalar semantics (uint32
wraparound, C literal typing for signed-vs-unsigned comparisons, float32
arithmetic); because CPU tensors are immutable values it returns a new cpu
Tensor instead of mutating `out`.

Emitted MSL preserves parentheses around comparisons used inside other
expressions. Floating-point `!=` is true when either operand is NaN in both
the reference and the general MLIR emitter. This does not widen the bounded
MLIR runtime subset to accept comparison-based kernel bodies.

The public API should move slowly:

```text
1. Keep all Phase 7 work under cx.experimental.
2. Add docs and tests with each supported syntax feature.
3. Keep CPU reference behavior available before enabling Metal launch.
4. Only consider cx.kernel after generated-kernel semantics are broader and
   stable enough for a non-experimental API.
```

## Acceptance Bar

Phase 7 is complete for the first accepted slice:

```text
- A Python elementwise add kernel is parsed into backend-neutral IR.
- The IR emits MSL without backend assumptions leaking into core IR.
- The generated kernel launches on Metal.
- The result matches the CPU reference.
- Unsupported syntax produces stable compile-time errors.
```

Broader DSL semantics, caching, and top-level `cx.kernel` remain future work.
CPU execution for DSL kernels exists as `Kernel.reference(...)`, the
interpreter-based reference path for generated kernels.


## Optional MLIR CPU compilation

Use `kernel.compile(target="cpu", compiler="mlir").launch(...)` for the explicit
Linux x86_64 float32 elementwise slice. The default compiler and callable kernel
keep the existing Metal route. Setup, the narrower supported body, CPU output
ownership, toolchain errors, and validation are maintained in the
[MLIR runtime decision](MLIR_RUNTIME_INTEGRATION_DECISION.md#using-the-cpu-runtime).


## Optional MLIR CUDA elementwise compilation

`kernel.compile(target="cuda", compiler="mlir").launch(...)` executes the narrow
guarded float32 add/subtract/multiply subset and returns a new CUDA tensor without mutating the
supplied output. The canonical index/guard and exact shapes are required; the
two input parameter names must differ, but tensor arguments may alias.
Supported host/toolchain, block limits, context/module ownership and verification
are maintained in the [CUDA integration record](MLIR_CUDA_INTEGRATION_DECISION.md).
Bounded locals and nested arithmetic are supported by the
[expression extension](MLIR_CUDA_EXPRESSIONS_DECISION.md), with separate float32 rounding.
