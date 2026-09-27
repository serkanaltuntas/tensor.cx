# MLIR runtime integration: first scope

Date: 2026-09-28 · Baseline: `8151530` · Status: **scope decided; implementation not started**.

This is the engineering decision requested after Phase 9. It resolves the
integration questions left by [Phase 10](MLIR_DECISION.md); it does not claim
that the runtime already compiles or executes MLIR kernels. Phase 9 and Phase
10 remain complete. No new implementation phase is marked done.

## Decision and rationale

Start with **explicit, experimental CPU compilation** of guarded float32
elementwise kernels. Reuse standard MLIR dialects and external LLVM 21.1.8 tools
to produce a host shared library, then execute it through the existing backend
ABI. Keep the CPU interpreter as the independent semantic reference.

| Question | Selected approach | Reason / deferred alternative |
| --- | --- | --- |
| First execution target | CPU on Linux x86_64 (Nightblade) | Existing lowering now passes locally. CUDA-first would combine new GPU lowering, module loading, context ownership, and runtime integration. |
| IR | Existing Cortex IR → `func/scf/arith/memref` → LLVM | No custom Cortex dialect until an operation needs semantics the standard dialects cannot express. |
| Compilation | External tools → native shared library before launch | Reuses the prototype and avoids linking MLIR or LLVM ExecutionEngine into the extension. In-process JIT is deferred. |
| Runtime entry | `CpuBackend::execute`, `BackendOpClass::kKernel` | The same validation/ownership boundary used by generated Metal kernels; no ctypes/NumPy execution bypass. |
| Second target | CUDA device-only lowering via `gpu`/NVVM → PTX | A later slice with its own hardware acceptance; no generated host scheduler replacing Cortex's backend. |
| Existing Metal path | Retain direct MSL/metallib compilation | No automatic switch and no claim of MLIR-to-Metal support. |

This selects a bounded next implementation task, not autograd, a graph compiler,
operation fusion, tiling, broad dtype support, or production packaging.

## First slice: user contract

Planned API (not available yet):

```python
compiled = add_kernel.compile(target="cpu", compiler="mlir")
result = compiled.launch(a_cpu, b_cpu, out_cpu, n, thread_count=n, block_size=64)
expected = add_kernel.reference(a_cpu, b_cpu, out_cpu, n, thread_count=n, block_size=64)
```

- Add a `compiler` keyword to `Kernel.compile`, defaulting to existing behavior.
  Require the explicit pair `target="cpu", compiler="mlir"` for this slice.
  Existing `compile()` / `compile(target="metal")` and `Kernel.__call__` keep
  their current Metal selection; no silent CPU fallback. Unsupported pairs fail.
- Provide a separate CPU compiled-artifact type with `.launch(...)`; preserve
  the existing Metal `CompiledKernel` fields and `validate_metal_function()`.
  Keep all new API under `cx.experimental`; no `device="mlir"` registration.
- Initial subset: one output, contiguous float32 buffers flattened internally
  with public shapes preserved, uint32 scalars, guarded elementwise indexing,
  float32 `+`, `-`, `*`, and local assignments. Reject loops, float comparisons,
  mixed int/float arithmetic, signed ordered comparisons, and unsupported index
  expressions before compilation. The existing rowsum prototype remains research
  evidence, not a promise of compiled runtime reductions in this slice.
- Reuse the shared launch contract: exact elementwise shapes; guard bound equals
  `thread_count`; `thread_count <= output_numel`; positive uint32 block size.
  Partial-prefix execution preserves the unwritten suffix. Zero work validates
  arguments and the loaded artifact, then returns without entering a non-zero
  `LaunchConfig`. Do not copy the looser research harness launch rules.
- Match CPU `Kernel.reference`: return a new tensor and leave caller tensors
  unchanged. Copy the supplied output's initial contents and redirect every
  argument aliasing that output to the same private copy. Other input aliases
  remain shared. Failures never publish a partially computed result.

## Compiler and native boundary

Move reusable emission into a private Python compiler module, retaining the
experiment as a thin consumer of that one implementation. Separate MSL checks
from shared DSL semantics carefully: the CPU reference must remain independent
of generated code, and existing Metal rejection tests must still pass.

The compiler produces a native library plus a versioned internal manifest:
backend/host target, entry point, ordered tensor/uint32 signature, output index,
launch contract, compiler version, and ABI version. A single generated wrapper
with C ABI `void cortex_launch_v1(void **args)` avoids calling arbitrary typed
functions through an incorrectly cast function pointer. Each argument slot
points to its typed storage: a rank-1 float32 memref descriptor or a uint32
scalar; the last two slots hold uint32 thread count and block size. The wrapper
adapts this packed call to the generated `_mlir_ciface_*` entry point.

For this 64-bit CPU target the private descriptor is allocated pointer, aligned
pointer, int64 offset, one int64 size, and one int64 stride. Validate its C layout
and the generated signature in native tests. Dynamic memrefs require descriptors;
do not apply the bare-pointer convention that requires static dimensions.
See the [LLVM lowering ABI](https://mlir.llvm.org/docs/TargetLLVMIR/#c-compatible-wrapper-emission).
The descriptor and loader stay inside the CPU backend, outside `core/`.

The native CPU backend owns the loaded module through RAII. Use an opaque,
process-local artifact ID in existing `CompilationTarget.artifact` with
`KernelArtifactKind::kBinary`; the backend resolves it to a live module and
manifest. Do not pass raw pointers, MLIR objects, or Python/NumPy types through
core metadata. The binding validates/registers compiler output and constructs
`BackendExecution`; the backend repeats metadata/signature checks, validates
concrete CPU buffers, and calls the wrapper with the GIL released. Loader and
execution errors return `Status`, translated once at nanobind.

The compiled Python object retains native module ownership; each launch holds a
strong reference until completion, including when another thread drops the
object. Temporary library files live until loading/ownership rules allow cleanup.
Reject unknown/expired artifact IDs and ABI mismatches. No persistent disk cache
or unbounded global module retention in this slice; compile once and reuse the
returned object. Module lifetime, cleanup, and concurrent launches need tests.

## Toolchain and distribution

MLIR stays optional at package import, ordinary CPU/Metal/CUDA use, and package
build time. Compilation explicitly discovers `CORTEX_LLVM_BIN` and verifies
executable tools and actual version output (`mlir-opt`, `mlir-translate`, `clang`
from LLVM **21.1.8**); sharing a directory alone does not prove version equality.
An invalid explicit setting fails rather than falling back. Run subprocesses
with argument arrays, bounded timeouts, isolated temporary output, and useful
stderr. No implicit downloads or compiler installation from runtime APIs.

Use the proven CPU lowering passes from `experiments/mlir/lower_add.py`, no fast
math/reassociation or floating-point contraction, and a host-compatible shared
library. Build/import tests without LLVM remain mandatory. The first native
loader is enabled only for the validated Linux x86_64 target; unsupported hosts
fail clearly for this experimental compile pair and retain existing runtime
behavior. macOS support requires separate loader/ABI verification, including
preservation of the working Metal path.

## CUDA follow-up gate

Before adding `compile(target="cuda", compiler="mlir")`, demonstrate a
**device-only** guarded add lowering through `gpu`/NVVM and LLVM NVPTX codegen,
then load and run it against Cortex CUDA buffers on Nightblade. The current
serial CPU `scf.for` launch loop cannot simply be relabeled as a GPU kernel.
GPU block/thread IDs must implement the existing logical grid, including a
rounded-up block guard that prevents writes beyond logical `thread_count`.

Keep allocation and launch ownership inside `CudaBackend`. A Driver API module
loader must use the same device-0 primary context as existing Runtime API
allocations, restore thread-local context state, hold modules alive through
synchronous completion, and route launches through the existing kernel ABI.
Specify and test the actual device argument ABI; do not assume CPU memref
packing is a CUDA kernel ABI. Explicitly pin/validate SM and PTX compatibility
on the GTX 980 Ti (`sm_52`, CUDA 12.4 environment) before enabling the capability.
Tool recognition of `sm_52` alone is not execution evidence.

MLIR's [GPU compilation pipeline](https://mlir.llvm.org/docs/Dialects/GPU/#gpu-compilation)
provides lowering and host offloading machinery; this design selects only the
device compilation portion. ROCm, SPIR-V/Metal, and wider generated kernels
remain separate decisions.

## Acceptance and next implementation task

The next task is **the CPU slice above**, in this order: optional toolchain and
emitter extraction; packed wrapper and owned native module; CPU kernel execution
through `BackendExecution`; experimental compile/launch API; regression and
required-toolchain CI evidence. No runtime integration is complete until:

1. Add/subtract/multiply compare with `Kernel.reference`; add/multiply also
   compare with existing CPU primitives (no new subtraction primitive is required).
   Scalar, empty, 1/255/256/257 elements, multidimensional contiguous tensors,
   partial work, zero work, repeated launch, and input/output alias cases pass.
2. Invalid shapes/dtypes/scalars/index patterns, unsupported IR, missing/mixed
   tools, compiler failure/timeout, bad library/symbol/ABI and expired IDs fail
   before unsafe execution; native contract tests cover forged requests.
3. Caller inputs remain unchanged; concurrent launches and module destruction
   cannot free live code/buffers; resources are released after last use.
4. Required-MLIR tests run without skips under pinned tools. Existing CPU-only,
   CUDA, Metal compilation, and unavailable-backend jobs stay valid. CPU and
   Metal compilation selection/return behavior has regression coverage.
5. Code reviewers and Test/Acceptance QA pass; docs distinguish supported hosts
   and implemented subsets. Performance claims require separate benchmarks.

## Evidence for this decision

On 2026-09-28, baseline `8151530`, Nightblade Linux x86_64, Python 3.12.14:
Ubuntu packages `mlir-21-tools`, `libmlir-21`, `clang-21`, and `llvm-21`, all
`1:21.1.8-6ubuntu1`, were downloaded with `apt-get download` and extracted with
`dpkg-deb -x` into ignored `build/mlir-toolchain/root`. Existing system LLVM/Clang
shared libraries satisfy this host's dependencies; this extracted layout is a
local validation fixture, not a standalone distribution. No system packages or
drivers were changed; binaries/debs are not committed.

```bash
export CORTEX_LLVM_BIN="$PWD/build/mlir-toolchain/root/usr/lib/llvm-21/bin"
CORTEX_REQUIRE_MLIR=1 uv run pytest tests/python/test_mlir_lowering.py -q
uv run python experiments/mlir/lower_add.py
uv run python experiments/mlir/lower_rowsum.py
"$CORTEX_LLVM_BIN/llc" -march=nvptx64 -mcpu=help
```

Results: **18 passed, no skips**; add (4096 elements), guard preservation, zero
work, and rowsum (64×128) matched CPU references. LLVM 21.1.8 lists `sm_52` as an
NVPTX target. This confirms the existing CPU prototype/toolchain; it does **not**
validate the planned packed wrapper, native module loader, or MLIR-generated
CUDA execution. Those are the implementation acceptance gates above.
