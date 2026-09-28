# MLIR CUDA integration: ABI and toolchain decision

Date: 2026-09-28 · Baseline: `636e3e7` · Status: **scope decided; research
device execution verified; public runtime integration not implemented**.

This answers the next-work request after the [CPU runtime slice](MLIR_RUNTIME_INTEGRATION_DECISION.md).
The delivered change is this decision and a reproducible device-only research
probe. It adds no CUDA compiler target, public pointer access, runtime module
loader, or generated-kernel capability to Cortex. Phase 9/10 status is unchanged.

## Selected first implementation slice

An explicit experimental `kernel.compile(target="cuda", compiler="mlir")`
will return a distinct `CompiledCudaKernel` with `.launch(...)`. This API is
**planned, not available**. Default compilation/call behavior remains Metal;
CPU MLIR compilation and static CUDA operations retain their existing behavior.
No automatic target/compiler fallback is allowed.

First enable only guarded float32 elementwise add with one output, two input
buffers, one uint32 guard parameter, contiguous exact-match shapes, and the
canonical global-index pattern used by the CPU slice. Support scalar, empty,
multidimensional and partial-prefix inputs through flattening. Reject loops,
reductions, offset/strided access, unused parameters, mixed arithmetic,
comparisons over floats and broader expression bodies before invoking tools.
Subtraction/multiply and more DSL constructs require their own parity coverage.

The returned CUDA tensor is a new value, matching the compiled CPU/reference
ownership contract: privately copy the original output including its unwritten
suffix, redirect all input aliases of that output to the copy, and publish only
after successful synchronization. Existing inputs remain unchanged. Zero work
still validates arguments/module and returns the copy without a zero-sized
Driver launch. This intentionally differs from the existing Metal DSL's output
mutation; document that distinction on the future artifact type.

## Device pipeline and toolchain

Select Linux x86_64, device index 0, compute capability **5.2**, LLVM/MLIR
**21.1.8**, PTX ISA **7.8**, CUDA Toolkit/Runtime **12.4**, and the validated
Nightblade driver **580.178.04** as the first acceptance environment.
Other devices/SMs, hosts and toolchain versions fail clearly until separately
validated; this is a bounded initial support policy, not a claim of universal
incompatibility. The driver's reported CUDA API level is 13.0; it is not the
installed Toolkit version.

The research fixture uses parallel `gpu.func`/`gpu.block_id`/`gpu.block_dim`/
`gpu.thread_id`, arithmetic/control flow, and explicit LLVM pointer accesses:

```text
gpu/scf/arith + llvm pointer operations
  -> convert-scf-to-cf
  -> convert-gpu-to-nvvm (index-bitwidth=64)
  -> reconcile-unrealized-casts
  -> nvvm-attach-target (chip=sm_52 features=+ptx78 O=2)
  -> gpu-module-to-binary (format=assembly)
  -> PTX -> CUDA Driver JIT -> device execution
```

This invokes the NVPTX backend inside the pinned `mlir-opt`; no external
`llc` invocation or host shared library is needed. Do not use the full
host-offloading pipeline or introduce MLIR's host launcher into Cortex.
The [MLIR GPU compilation documentation](https://mlir.llvm.org/docs/Dialects/GPU/#gpu-compilation)
distinguishes device serialization from host offloading, and requires explicit
parallel IR. The [pinned GPU-to-NVVM tests](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/test/Conversion/GPUToNVVM/gpu-to-nvvm.mlir)
provide version-specific lowering behavior.

Use no fast math, reassociation, or contraction. The current add fixture needs
no libdevice or device linker. Do not relabel the CPU emitter's serial loop.
Future runtime code must emit this GPU structure from validated Cortex IR;
the checked-in probe's hand-authored MLIR is not evidence of that frontend.

LLVM remains external and optional at build/import and ordinary execution.
Keep strict executable/version checks and invalid-explicit-directory failures,
bounded subprocess timeouts, isolated temporary files, and no downloads.
Validate the serialized assembly's target, address width, entry and parameter
schema before loading; this is an internal trusted compiler artifact, not a
sandbox for arbitrary PTX. The research extractor only handles its single,
fixed assembly object. Production extraction needs tests for malformed/multiple
objects and must fail closed. Package native linking adds `CUDA::cuda_driver`
only when CUDA is enabled; CPU/Metal builds require no Driver/LLVM libraries.

## Device argument ABI v1

For the first add entry, arguments follow the original parameter order:

| Parameter | Device representation | Host argument storage |
| --- | --- | --- |
| a, b, out | 64-bit generic device pointers to contiguous f32 elements | Stable `CUdeviceptr` slots |
| n | 32-bit unsigned logical work-item count | Stable `uint32_t` slot |

The three pointers are pointer values, not CPU memref descriptors or pointers
to such descriptors. Pass addresses of these host slots to
`cuLaunchKernel(..., kernelParams, nullptr)`; that Driver entry copies each
argument from its host address. See the [CUDA 12.4 launch contract](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__EXEC.html).
Do not pass both `kernelParams` and the packed `extra` buffer.

No hidden CPU thread-count/block-size arguments: `n == thread_count`, block
size comes from the hardware launch, and the kernel uses a **64-bit global
index** before checking `index < zero_extend(n)`. For nonzero logical work,
compute `blocks = 1 + (thread_count - 1) / block_size` in checked arithmetic.
Validate 1D geometry, positive uint32 block size, device and function block
limits, and the device grid limit before launch. Rounded-up lanes must branch
before any load/store; uint32 wraparound must never bypass that guard.

Keep core `BackendExecution`, `LaunchConfig`, `KernelArgument` and
`CompilationTarget` unchanged. Use `kKernel`, `kBinary` and an opaque
process-local weak-registry artifact ID; resolve to a strong module owner in
`CudaBackend::execute`. The private module stores PTX, entry, ABI version,
ordered argument kinds/output/guard indices, SM, PTX and LLVM versions.
Validate that manifest against compiled entry parameter metadata and repeat
the shared and CUDA-specific concrete buffer/device/dtype/size/shape/offset
checks before any device access. Manifest mismatch, unknown/expired ID,
missing symbol, wrong target or invalid launch returns Status through the
single nanobind translation boundary.

## Context, buffers and module ownership

Use device 0's retained **primary context** for all Cortex CUDA allocations,
copies, static and generated launches, and destructors. A shared backend-private
RAII owner is retained by buffers and modules. A scoped Driver context push/pop
restores the caller's thread-local context on success and error. Never call
`cudaDeviceReset`/`cuDevicePrimaryCtxReset` or create an independent context
for production modules. NVIDIA documents the [shared primary context](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__PRIMARY__CTX.html)
and [Runtime/Driver context interaction](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-runtime-api/driver-vs-runtime-api.html).

The existing `DeviceScope` only selects/restores a Runtime device; it does not
prove restoration of an already-current foreign Driver context on the same
device. Replace that policy consistently across CUDA entry points rather than
introducing a second ownership regime only for generated kernels. Keep pointer
accessors backend-private; the core and Python API must not expose CUDA handles.

Load with `cuModuleLoadDataEx` and capture bounded JIT diagnostics; look up
the entry and retain the module through every synchronous launch. Keep typed
tensor owners, argument slots and module alive while the GIL is released.
Use the existing synchronous/default-stream policy and wait before returning.
Unload under the correct context only after the last launch owner releases;
then release its primary-context owner. Keep no persistent compilation cache.
Destructor failures must not throw; device-loss handling must not free under
the wrong context or report a successful result.

## Local evidence and its limits

Nightblade, 2026-09-28: LLVM 21.1.8, nvcc/ptxas 12.4.131, Runtime API version
12040, Driver API version 13000, NVIDIA driver 580.178.04, GTX 980 Ti / sm_52.
GCC 13 is used only to compile the research host harness. No driver/system
package changes were needed.

`experiments/mlir/probe_cuda.py` lowers `cuda_add.mlir`, verifies the PTX
header and four parameter widths, and asks CUDA 12.4 ptxas to assemble it.
The independently loaded PTX (not that cubin) executes through the Driver API.
`cuda_driver_probe.cpp` allocates via Runtime API and verifies each input
allocation's context matches the retained primary context.

**48 cases passed**: sizes 0/1/255/256/257, blocks 7/256, zero/partial/full
logical work, distinct output and input/output alias. Checks cover unchanged
read-only inputs, suffix and rounded-lane sentinels. Results match both
`Kernel.reference` and CPU add. A foreign context is restored after normal
execution and an intentionally missing-symbol failure. The C++ research
harness mutates its explicitly chosen output; it does not yet implement the
planned public output-copy contract.

```bash
export CORTEX_LLVM_BIN="$PWD/build/mlir-toolchain/root/usr/lib/llvm-21/bin"
uv run python experiments/mlir/probe_cuda.py --compile-only
uv run python experiments/mlir/probe_cuda.py
CORTEX_REQUIRE_MLIR=1 CORTEX_REQUIRE_MLIR_CUDA=1 uv run pytest tests/python/test_mlir_cuda_probe.py -q
```

The focused suite passed **11 tests, no skips**. The full CUDA-enabled suite
with both require flags passed **358 tests**, with **149 skips** for Metal and
deferred CUDA capabilities. With tools unavailable and require flags unset,
the new file passed 9 tests and cleanly skipped its 2 optional integration
checks. Two code reviews and Test/Acceptance QA passed; review prompted checks
of all three allocations' contexts. No runtime native files or package APIs
changed, so the research harness compile/run supplies the new native evidence.
Remote CI and local Metal execution were not verified in this task.

The compile-only route needs
no CUDA SDK or device; the hardware route deliberately fails if the selected
host/toolchain is absent. The MLIR CI job runs the compile-only tests under
require-mode; hosted runners do not establish GPU execution evidence.

**This does not close the CUDA integration gate.** The probe uses direct
Runtime allocations, not `CudaBuffer`, has a fixed hand-authored kernel,
and does not enter `CudaBackend::execute`. It proves the selected toolchain,
device ABI and context interoperability for that fixture. It does not prove
runtime module registry safety, concurrent object destruction, public API
semantics, other GPUs, Metal behavior or performance.

## Next implementation and acceptance

1. Introduce backend-private shared primary-context ownership; cover existing
   copies/static operations as well as proposed modules. Verify caller context
   and device restoration, including foreign contexts, failure and threads.
2. Add a private native PTX module/launch path through `CudaBackend::execute`.
   **First run the checked fixture against actual Cortex CudaBuffer objects**
   with native forged-request tests before exposing the public compile pair.
3. Add the narrow Cortex IR GPU emitter and `CompiledCudaKernel`; enforce the
   structure/typing and launch contracts before tools/native execution.
4. Compare generated add against `Kernel.reference`, CPU and static CUDA add:
   scalar/empty/multidimensional shapes; 1/255/256/257 sizes; partial and zero
   work; repeated launches; aliases; multiple Python threads; destruction during
   launch; immutable inputs and unchanged output on failed operations.
5. Reject bad/mixed tools, wrong SM/PTX/manifest/entry, stale IDs, malformed PTX,
   missing GPU/driver, wrong concrete buffers/devices/dtypes/shapes/strides,
   argument-count/kind errors, guard mismatch and excessive block/grid sizes.
   Record JIT/compiler/synchronization errors and verify cleanup.
6. Require device tests on Nightblade without skips and compile/build negatives
   in CPU-only CI. Preserve CPU MLIR, no-LLVM import/build, static CUDA, Metal and
   unavailable-backend coverage. Run native sanitizer checks where applicable,
   two code reviews and Test/Acceptance QA before enabling the capability.

The next recommended task is this bounded **Cortex CUDA add compile/launch
implementation**, starting with context ownership and real Cortex buffers.
Broader GPU lowering, reductions, fusion, async APIs and other SM targets remain
separate work. The decision adds no user priority or deadline.
