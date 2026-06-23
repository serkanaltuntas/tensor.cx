# PyTorch Portability Roadmap

This document compares three possible product and engineering tracks for Cortex
Runtime. It is not the phase ledger. The current project phase and formal
acceptance criteria remain in [`PROJECT.md`](../PROJECT.md).

The central question is whether Cortex Runtime should stay focused on its
current runtime/compiler roadmap, become a narrow PyTorch extension path, or
grow into a portability layer that lets PyTorch run on more accelerator
backends.

## Executive Summary

Cortex Runtime should first become a correct, backend-neutral tensor runtime.
That is the foundation for every other path.

After that, the lowest-risk PyTorch integration is a custom operation bridge:
PyTorch remains the framework, while selected operations call into Cortex
Runtime kernels or primitives.

Only after the runtime and custom operation bridge are proven should the project
consider a full PyTorch backend / ATen bridge. That third track is the closest
match for the long-term portability vision, but it is also a much larger product
line than the current Cortex Runtime roadmap.

```text
Track 1: Cortex Runtime core
  Build the portable runtime and compiler foundation.

Track 2: PyTorch custom op bridge
  Let PyTorch call selected Cortex Runtime operations.

Track 3: PyTorch backend / ATen bridge
  Make Cortex Runtime a backend substrate for PyTorch devices.
```

## Track Comparison

| Track | Primary goal | PyTorch role | Scope | Risk | Performance target |
| --- | --- | --- | --- | --- | --- |
| Cortex Runtime core | Build a standalone tensor runtime and compiler foundation | Independent | Runtime, devices, buffers, ops, kernels, compiler experiments | Medium | Strong performance for supported Cortex ops |
| PyTorch custom op bridge | Call selected Cortex kernels from PyTorch | PyTorch stays the main framework | Narrow integration for specific ops or fused kernels | Medium-high | Match or beat PyTorch for selected operations |
| PyTorch backend / ATen bridge | Let PyTorch run on Cortex-backed accelerators | PyTorch frontend and autograd, Cortex backend substrate | Broad ATen/device/backend integration | Very high | Competitive performance on real PyTorch workloads |

## Track 1: Cortex Runtime Core

This is the current project roadmap.

The goal is to build a small but serious accelerator runtime:

```text
Python API
  -> C++20 backend-neutral core
  -> CPU reference backend
  -> Metal backend
  -> static kernels, primitive paths, and later compiler paths
```

This track should remain focused on correctness, backend boundaries, CPU
references, tests, and carefully expanding operator coverage.

### What This Track Should Deliver

- A reliable Python tensor API for Cortex-owned tensors.
- Backend-neutral C++ abstractions for devices, buffers, tensors, and operation
  dispatch.
- A mandatory CPU reference path for every accelerator operation.
- Metal support first, then later backend hardening for CUDA, ROCm, Vulkan/SPIR-V,
  or MLIR exploration.
- Static kernels first, then an experimental kernel DSL only after the runtime is
  stable.

### What This Track Does Not Deliver By Itself

- PyTorch compatibility.
- PyTorch autograd integration.
- PyTorch operator coverage.
- A drop-in PyTorch training backend.
- Competitive performance for arbitrary PyTorch training systems.

### Success Criteria

This track succeeds when Cortex Runtime can reliably run its own supported
operations across CPU and accelerator backends with clear semantics, strong
tests, and honest benchmarks.

## Track 2: PyTorch Custom Op Bridge

This is the recommended first PyTorch integration path.

The idea is to let PyTorch call Cortex Runtime for selected operations without
trying to make Cortex Runtime responsible for all of PyTorch.

```text
PyTorch model / training or inference code
  -> selected custom op
  -> torch-cortex bridge
  -> Cortex Runtime OpDesc / kernel / primitive
  -> accelerator backend
```

PyTorch remains responsible for tensors, modules, autograd, optimizers, data
loading, and the broader training system. Cortex Runtime is used only where it
has a specific kernel, primitive, or portability advantage.

### Good First Targets

- Fused elementwise operations.
- Normalization kernels.
- Softmax variants.
- Specialized matmul experiments.
- Inference-only accelerator experiments.
- Operations where PyTorch has no good path for a target device.

### Required Capabilities

- A stable C ABI or C++ API boundary that PyTorch extensions can call.
- A `torch-cortex` extension package.
- Tensor data movement between PyTorch tensors and Cortex Runtime buffers.
- Clear ownership rules for temporary buffers and device memory.
- CPU fallback or explicit unsupported-operation errors.
- Per-op benchmarks against native PyTorch.
- Optional autograd formulas for custom ops that participate in training.

### Success Criteria

This track succeeds when a PyTorch program can call a small set of Cortex-backed
operations, get correct results, and show useful performance or portability
value for those operations.

It does not need to make arbitrary PyTorch models run on Cortex devices.

## Track 3: PyTorch Backend / ATen Bridge

This is the long-term portability vision.

The goal is to make Cortex Runtime act as a substrate under PyTorch so that
PyTorch can target accelerators through Cortex Runtime.

```text
PyTorch frontend, nn.Module, autograd, optimizers
  -> PyTorch dispatcher / ATen backend integration
  -> torch-cortex device backend
  -> Cortex Runtime operation dispatch or compiler
  -> Metal / CUDA / ROCm / Vulkan / SPIR-V / other backend
  -> accelerator
```

If successful, user code could eventually look like this:

```python
import torch

model = model.to("cortex")
x = x.to("cortex")
y = model(x)
```

This is the closest path to making PyTorch more portable across accelerators.
It is also much larger than the current Cortex Runtime roadmap.

### Required Capabilities

- A PyTorch device backend, likely starting with `PrivateUse1` or the current
  PyTorch custom backend mechanism.
- ATen operator coverage for common tensor operations.
- Correct PyTorch semantics for shapes, broadcasting, dtype promotion, strides,
  views, indexing, reductions, RNG, and error behavior.
- A memory allocator compatible with PyTorch expectations.
- Stream, event, synchronization, and async execution support.
- Forward and backward operator coverage for training.
- Integration with vendor primitives where available.
- A compiler or graph lowering path for fusion and workload-level performance.
- A large compatibility and regression test suite.

### Why This Is Hard

Running one kernel is not enough to be a PyTorch backend. PyTorch programs rely
on many subtle tensor semantics and thousands of operator paths. Training adds
backward kernels, optimizer behavior, memory pressure, mixed precision,
synchronization, and graph-level performance concerns.

For older or unsupported accelerators, the backend may also need a target such
as Vulkan/SPIR-V or OpenCL instead of CUDA or ROCm. Getting full device capacity
requires architecture-specific kernels, memory tiling, occupancy tuning,
primitive libraries, or an optimizing compiler path.

### Success Criteria

This track succeeds only when real PyTorch workloads can run on a Cortex-backed
device with correct behavior and competitive performance for a defined workload
class.

The first realistic target should be narrow:

```text
one device backend
one dtype family
one model family
one small training or inference smoke test
explicit unsupported-op behavior
benchmarks against native PyTorch where native support exists
```

## Recommended Sequencing

The tracks should not run as three equal priorities from the start.

### Stage 1: Finish The Runtime Foundation

Continue the current Cortex Runtime roadmap until the core backend boundaries,
operation dispatch, CPU reference behavior, Metal path, benchmarks, and docs are
stable.

Decision gate:

```text
Cortex can add new ops without changing core architecture.
CPU-vs-device tests are routine.
Benchmarks exist for each performance claim.
Backend-specific code stays behind backend boundaries.
```

### Stage 2: Build A Narrow PyTorch Custom Op Bridge

Add a small `torch-cortex` integration package and route a few selected PyTorch
custom operations into Cortex Runtime.

Decision gate:

```text
At least one PyTorch custom op is correct.
The op has clear CPU fallback or unsupported behavior.
The op is benchmarked against native PyTorch where possible.
The bridge does not compromise Cortex Runtime's backend-neutral design.
```

### Stage 3: Evaluate A Real PyTorch Backend

Only begin the ATen backend track after the custom op bridge proves that Cortex
Runtime has practical value from PyTorch.

Decision gate:

```text
A target accelerator family is chosen.
A first workload class is chosen.
Required ATen operator coverage is listed.
Memory, stream, dtype, layout, and autograd requirements are documented.
Unsupported behavior is explicit.
```

## Strategic Recommendation

The best path is incremental:

```text
1. Build Cortex Runtime well.
2. Prove value inside PyTorch through selected custom ops.
3. Use that evidence to decide whether a full PyTorch backend is justified.
```

This keeps the current project grounded while leaving room for the larger
portability vision. Cortex Runtime should not try to become a PyTorch
replacement. The stronger long-term role is to become a clean accelerator
substrate that PyTorch or other frameworks can target when that integration is
worth the cost.

## Non-Goals For The Near Term

- A drop-in replacement for PyTorch.
- Running arbitrary PyTorch training systems on Cortex devices.
- Matching native PyTorch performance across all operators.
- Supporting every dtype, layout, view, and dispatch path.
- Supporting every legacy or unsupported accelerator at full capacity.
- Treating vendor primitive libraries as optional when performance depends on
  them.

## Open Questions

- Which accelerator family should define the first portability target after
  Metal?
- Should the first PyTorch integration be inference-only, training-capable, or
  explicitly split?
- Should Cortex Runtime expose a stable C ABI before PyTorch integration starts?
- Should the compiler path target Cortex-owned IR first, MLIR first, or a smaller
  backend-specific lowering path?
- What is the smallest PyTorch workload that would prove the backend vision is
  real without expanding into a full framework project too early?
