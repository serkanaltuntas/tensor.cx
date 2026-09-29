# MLIR CUDA integration: ABI and toolchain decision

Date: 2026-09-28 · Baseline: `636e3e7` · Status: **bounded CUDA add/subtract/multiply runtime implemented** (extended 2026-09-29).

This answers the next-work request after the [CPU runtime slice](MLIR_RUNTIME_INTEGRATION_DECISION.md).
The original decision and device-only research probe are retained below as
historical evidence. The subsequent implementation now connects validated
Cortex IR, native modules and real Cortex buffers through CudaBackend. Phase
9/10 status is unchanged; this bounded slice does not complete a broad compiler phase.

## Selected first implementation slice

An explicit experimental `kernel.compile(target="cuda", compiler="mlir")`
returns a distinct `CompiledCudaKernel` with `.launch(...)`. This API is
**implemented for the validated environment below**. Default compilation/call behavior remains Metal;
CPU MLIR compilation and static CUDA operations retain their existing behavior.
No automatic target/compiler fallback is allowed.

The initial add slice, extended on 2026-09-29, supports guarded float32
elementwise add/subtract/multiply with one output, two input
buffers, one uint32 guard parameter, contiguous exact-match shapes, and the
canonical global-index pattern used by the CPU slice. Support scalar, empty,
multidimensional and partial-prefix inputs through flattening. Reject loops,
reductions, offset/strided access, unused parameters, mixed arithmetic,
comparisons over floats and broader expression bodies before invoking tools.
Each body contains exactly one binary operation over two input loads. Nested
expressions and additional DSL constructs remain outside this slice.

The returned CUDA tensor is a new value, matching the compiled CPU/reference
ownership contract: privately copy the original output including its unwritten
suffix, redirect all input aliases of that output to the copy, and publish only
after successful synchronization. Existing inputs remain unchanged. Zero work
still validates arguments/module and returns the copy without a zero-sized
Driver launch. This intentionally differs from the existing Metal DSL's output
mutation; this distinction is documented on `CompiledCudaKernel`.

## Device pipeline and toolchain

Select Linux x86_64, device index 0, compute capability **5.2**, LLVM/MLIR
**21.1.8**, PTX ISA **7.8**, CUDA Toolkit/Runtime **12.4**, and the validated
Nightblade driver **580.178.04** as the first acceptance environment. Runtime
validation checks Driver API level **13.0** (13000), not the vendor patch string.
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
The private CUDA compiler now emits this GPU structure from validated Cortex IR.
The checked-in probe remains historical evidence; runtime tests additionally
compare the emitted PTX against the native fixture and execute parsed kernels.

LLVM remains external and optional at build/import and ordinary execution.
Keep strict executable/version checks and invalid-explicit-directory failures,
bounded subprocess timeouts, isolated temporary files, and no downloads.
Validate the serialized assembly's target, address width, entry and parameter
schema before loading; this is an internal trusted compiler artifact, not a
sandbox for arbitrary PTX. The research extractor handles its single fixed object. The runtime extractor
rejects malformed/multiple objects, wrong headers, entries and pointer/u32
schemas; native loading also verifies the manifest and Driver parameter metadata. Package native linking adds `CUDA::cuda_driver`
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

The prior `DeviceScope` selected/restored only a Runtime device. It has been
replaced across CUDA entry points with retained primary-context ownership and
Driver push/pop. Native tests verify foreign-context restoration after load,
launch, copies, static add, validation failure and destruction. Keep pointer
accessors backend-private; the core and Python API must not expose CUDA handles.

Load with `cuModuleLoadDataEx` and capture bounded JIT diagnostics; look up
the entry and retain the module through every synchronous launch. Keep typed
tensor owners, argument slots and module alive while the GIL is released.
Use the existing synchronous/default-stream policy and wait before returning.
Unload under the correct context only after the last launch owner releases;
then release its primary-context owner. Keep no persistent compilation cache.
Destructor failures must not throw; device-loss handling must not free under
the wrong context or report a successful result.

## Research evidence and its limits

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

**The research probe alone did not close the integration gate.** It uses direct
Runtime allocations, not `CudaBuffer`, has a fixed hand-authored kernel,
and does not enter `CudaBackend::execute`. It proves the selected toolchain,
device ABI and context interoperability for that fixture. It does not prove
runtime module registry safety, concurrent object destruction, public API
semantics, other GPUs, Metal behavior or performance.

## Runtime implementation acceptance

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

These acceptance checks were applied to the runtime implementation below and
the subsequent subtraction/multiply extension. The [local-expression scope
decision](MLIR_CUDA_EXPRESSIONS_DECISION.md) now defines the next implementation
and parity gates; that CUDA capability is not yet implemented. Broader GPU
lowering, reductions, automatic graph fusion, async APIs and other SM targets
remain separate work. The decision adds no user priority or deadline.


## Runtime usage and verification — 2026-09-28

Build with `CORTEX_ENABLE_CUDA=ON`, Toolkit 12.4+ and a usable NVIDIA Driver
library `libcuda.so.1`. The compiled MLIR slice is narrower than the static
CUDA backend: Linux x86_64, sm_52, Runtime 12.4 and Driver API 13.0 are checked
before lowering/loading. The native build itself never links LLVM/MLIR.

Save inspectable DSL source in a Python file:

```python
import cortex_runtime as cx

@cx.experimental.kernel
def add(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = a[i] + b[i]

a = cx.ones((257,), device="cuda")
out = cx.zeros((257,), device="cuda")
compiled = add.compile(target="cuda", compiler="mlir")
result = compiled.launch(a, a, out, 257, block_size=256)
assert result.cpu().numpy()[0] == 2.0
assert out.cpu().numpy()[0] == 0.0
```

Exactly two distinct input parameter names, one output parameter and one guard
are supported; actual input/output tensor objects may alias. Parameter order
may vary, including a leading uint32 scalar. Only a single guarded
`out[i] = a[i] + b[i]`, `out[i] = a[i] - b[i]`, or
`out[i] = a[i] * b[i]` store is enabled. Other bodies fail before tools run.
`compile(target="cuda")` still fails unless `compiler="mlir"` is explicit.

Implementation files: `_compiler/cuda.py` validates/emits/lowers device code;
`CompiledCudaKernel` shares the existing Python launch contract; the native
CUDA context, buffer and kernel modules own resources and run through
`CudaBackend::execute`. No CUDA types were added to core interfaces and no raw
pointer API was exposed. The PTX manifest and Driver parameter offsets/sizes
are checked before registration. Weak registry IDs do not retain modules;
active native calls hold strong owners while the GIL is released.

The native fixture in `tests/cpp/fixtures/cuda_add_sm52.ptx` is generated from
the checked research add using LLVM 21.1.8, with entry renamed to
`cortex_add_v1` and the v1 manifest prepended. It is a small intentional test
asset, not a runtime cache. The runtime emitter equality test prevents it from
silently drifting. Native fixture execution against Cortex buffers passed
before the public compile API was connected.

CUDA-enabled extensions now link the Driver library directly, following this
decision. GPU-free CUDA build CI supplies the Toolkit stub under its runtime
SONAME in an isolated temporary search path solely to test imports and
unavailability. Stubs are never packaged or used for real execution.
An actual CUDA-off build remains independent of Driver/Toolkit libraries.
Review identified the required CI loader setup; it was added and exercised
locally before final validation.

Validation commands:

```bash
export CORTEX_LLVM_BIN="$PWD/build/mlir-toolchain/root/usr/lib/llvm-21/bin"
CORTEX_REQUIRE_MLIR=1 CORTEX_REQUIRE_MLIR_CUDA=1 uv run pytest tests/python/test_mlir_cuda_runtime.py -q
CORTEX_REQUIRE_MLIR=1 CORTEX_REQUIRE_MLIR_CUDA=1 uv run pytest -q
uv run cmake --build build/cpp-cuda
CORTEX_REQUIRE_MLIR_CUDA=1 uv run ctest --test-dir build/cpp-cuda --output-on-failure
uv run cmake --build build/cpp-cuda-sanitizers
ASAN_OPTIONS=halt_on_error=1:detect_leaks=1:protect_shadow_gap=0 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 CORTEX_REQUIRE_MLIR_CUDA=1 uv run ctest --test-dir build/cpp-cuda-sanitizers --output-on-failure
```

The sanitizer shadow-gap setting is the existing [Nightblade CUDA workaround](CUDA_PHASE9_VALIDATION.md).
The CUDA-off/LLVM-absent and CUDA-on build recipes remain in that record and the
[CPU integration validation](MLIR_RUNTIME_INTEGRATION_DECISION.md).
Final local validation on Nightblade:

| Configuration | Result |
| --- | --- |
| CUDA runtime integration tests, required LLVM and CUDA | 69 passed |
| Full suite, CUDA enabled and required LLVM/CUDA | 427 passed, 149 skipped |
| Actual CUDA-off build, required LLVM | 313 passed, 263 skipped |
| Actual CUDA-off build, LLVM absent | 259 passed, 317 skipped |
| Native CUDA CTest | 4/4 passed |
| Native CUDA ASan/UBSan CTest | 4/4 passed |

The CUDA-off extension's dynamic dependencies contain no CUDA or LLVM
libraries. The CUDA-enabled extension also passed import, CPU fallback and
unavailability checks with the isolated Driver stub. The CUDA-enabled build
was restored before the final full-suite run. Skips cover unavailable Metal
execution and deferred backend capabilities. Metal execution and remote CI
results are not claimed by local Nightblade validation. No performance claim
is made.


## Guarded subtraction/multiply extension — 2026-09-29

The public compile/launch API and ownership contract are unchanged. Replace
`+` in the usage example above with `-` or `*` to compile subtraction or
multiplication; change the result assertion to `0.0` for subtraction or `1.0`
for multiplication. This does not add a static CUDA subtraction primitive.

The compiler emits `arith.addf`, `arith.subf`, or `arith.mulf` without fast-math
flags. Entries are `cortex_add_v1`, `cortex_sub_v1`, and `cortex_mul_v1`;
manifest operation tags are respectively `add-f32-v1`, `sub-f32-v1`, and
`mul-f32-v1`. The native loader accepts only these entry/tag pairings, and
backend dispatch checks the requested entry against the resolved module.
The add ABI and fixture remain unchanged. The sub/mul PTX fixtures are generated
by the runtime emitter with the pinned LLVM toolchain; tests compare all three
fixtures against fresh lowering.

Parity tests cover the Python CPU reference, compiled MLIR CPU, NumPy and
static CUDA add/multiply where available. Distinct operands check subtraction
order, including reordered parameters. Scalar/empty/multidimensional shapes,
block boundaries, partial prefixes, all input/output alias combinations,
repeated and concurrent launches, signed zero, subnormals, infinities and NaNs
are covered. Native tests exercise each new entry with real Cortex buffers,
partial outputs, context restoration and mismatched operation rejection.

The validation commands above passed on Nightblade for this extension:

| Configuration | Result |
| --- | --- |
| CUDA runtime integration tests, required LLVM and CUDA | 231 passed |
| Full suite, CUDA enabled and required LLVM/CUDA | 589 passed, 149 skipped |
| Actual CUDA-off build, required LLVM | 315 passed, 423 skipped |
| Actual CUDA-off build, LLVM absent | 259 passed, 479 skipped |
| Native CUDA CTest | 4/4 passed |
| Native CUDA ASan/UBSan CTest | 4/4 passed |

The CUDA-off extension links neither CUDA nor LLVM libraries. The CUDA-enabled
build is restored after these checks. The existing sm_52/toolchain restrictions remain; Metal execution and remote CI
are not covered by local Nightblade validation. No performance claim is made.
