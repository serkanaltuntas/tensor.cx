# Experimental Kernel DSL

This document defines the Phase 7 starting point. The DSL is experimental and
must stay behind `cx.experimental` until the AST -> IR -> backend path is proven
with tests.

## Status

Phase 7 has started, but no user-defined kernel is compiled or launched yet.
The current API records kernel metadata and rejects launch/compile attempts with
clear `NotImplementedError` messages.

```python
import cortex_runtime as cx

@cx.experimental.kernel(target="metal")
def add_kernel(a, b, out, n):
    # Future syntax. The function body is not compiled yet.
    pass

print(add_kernel.name)
print(add_kernel.parameters)
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
  -> Cortex Runtime kernel IR
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
reductions
matmul
Python control-flow beyond a simple if
general Python semantics
CUDA/ROCm/MLIR lowering
```

## IR Rules

The IR must be backend-neutral. It may describe concepts such as parameters,
indices, loads, stores, constants, arithmetic, comparisons, and simple branches.
It must not contain MSL syntax, Metal handles, Objective-C types, command
queues, threadgroup details, or backend-specific buffer objects.

Expected first IR nodes:

```text
Kernel
Parameter
ProgramId
ThreadId
BlockSize
BinaryOp
Compare
If
Load
Store
Constant
```

Unsupported AST nodes must fail at compile time with a clear message naming the
node type. Silent fallback or partial miscompilation is not acceptable.

## API Rules

The experimental API starts here:

```python
cx.experimental.kernel
cx.experimental.program_id
cx.experimental.thread_id
cx.experimental.block_size
```

`@cx.experimental.kernel` currently returns a metadata wrapper. Calling
`compile()` or launching the wrapper is intentionally disabled until the
compiler path exists.

The public API should move slowly:

```text
1. Keep all Phase 7 work under cx.experimental.
2. Add docs and tests with each supported syntax feature.
3. Keep CPU reference behavior available before enabling Metal launch.
4. Only consider cx.kernel after a real add kernel compiles and launches.
```

## Acceptance Bar

Phase 7 is not complete until:

```text
- A Python elementwise add kernel is parsed into backend-neutral IR.
- The IR emits MSL without backend assumptions leaking into core IR.
- The generated kernel launches on Metal.
- The result matches the CPU reference.
- Unsupported syntax produces stable compile-time errors.
```

Until then, the project status should remain Phase 7 in progress.
