# MLIR runtime integration: first scope

Date: 2026-09-28 · Baseline: `8151530` · Status: **CPU slice implemented and locally validated**.

This is the engineering decision requested after Phase 9. It resolves the
integration questions left by [Phase 10](MLIR_DECISION.md). The following scope
was implemented on 2026-09-28 after user approval. Phase 9 and Phase 10 remain
complete; this bounded slice does not mark a new broad compiler phase done.

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

Implemented experimental API:

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

## Acceptance gate

The CPU slice follows this order: optional toolchain and emitter extraction;
packed wrapper and owned native module; CPU kernel execution through
`BackendExecution`; experimental compile/launch API; regression and
required-toolchain CI coverage. The checks below define its acceptance:

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


## Using the CPU runtime

Install the package normally; LLVM is not needed for package import, build, CPU
primitives, or the independent interpreter. To opt into compiled CPU kernels,
install **LLVM/MLIR 21.1.8** externally and point `CORTEX_LLVM_BIN` at a directory
containing executable `mlir-opt`, `mlir-translate`, and `clang`. Without that
variable, the compiler looks for those names on PATH. All three versions are
checked at each compilation. An invalid explicit directory fails without PATH
fallback. The runtime never installs or downloads a toolchain.

Save this example in a Python file (the DSL needs inspectable function source):

```python
import cortex_runtime as cx

@cx.experimental.kernel
def add_kernel(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = a[i] + b[i]

a = cx.ones((257,), device="cpu")
out = cx.zeros((257,), device="cpu")
compiled = add_kernel.compile(target="cpu", compiler="mlir")
result = compiled.launch(a, a, out, 257)
assert result.numpy()[0] == 2.0
assert out.numpy()[0] == 0.0
```

The runtime subset requires the canonical global-index assignment shown above,
followed by one `if index < bound` body. Every load/store uses that same index;
body expressions use float32 loads, floating constants, local values and
`+`/`-`/`*`. Local assignments follow existing DSL rules (no reassignment).
All parameters must be used as buffers or the uint32 bound. This structural
restriction proves bounds safety before compilation; it is narrower than the
research emitter. Public multidimensional shapes remain exact-match metadata,
while the native descriptor flattens contiguous storage.

Each compiled object owns a native library loaded by the CPU backend. Temporary
compiler files are removed after loading, and the library unloads when its last
owner is released. Launch holds a strong owner across GIL release; registry IDs
are weak, process-local, and cannot revive expired modules. There is no cache,
serialization, or externally supported artifact-loading API. Native libraries
are trusted compiler output; the internal manifest detects ABI/signature errors
and is not a sandbox for arbitrary native code.

Tool failures, version mismatches and per-command 60-second timeouts raise
`RuntimeError`; unsupported IR and invalid launch contracts raise
`KernelCompileError`/`ValueError`/`TypeError`. No interpreter fallback occurs.
The CPU runtime supports Linux x86_64 only; the existing MSL/Metal API remains
the default. Rowsum/reductions and generated CUDA/Metal via MLIR remain deferred.

## Implementation validation — 2026-09-28

Validation uses the Nightblade toolchain described above. Runtime tests compare
native add/subtract/multiply with the independent interpreter, and add/multiply
with CPU primitives. Cases include scalar and empty shapes, sizes 1/255/256/257,
multidimensional tensors, partial/zero work, repeated launches, aliases,
concurrent lifetime, compiler failure/cleanup, and explicit-toolchain errors.
Native fixture tests additionally exercise malformed metadata, missing symbols,
manifest mismatch, expired registry IDs, and unchanged outputs on failure.
The fixture needs no LLVM installation and runs in ordinary CPU and sanitizer CI.

| Check | Local result |
| --- | --- |
| Required LLVM runtime + research tests | **82 passed, 0 skipped** |
| Full CPU-only suite with LLVM | **293 passed, 203 skipped** (unavailable accelerators) |
| Full CPU-only suite without LLVM | **241 passed, 255 skipped** (accelerators and optional MLIR) |
| Full CUDA-enabled suite with LLVM | **347 passed, 149 skipped** (Metal and deferred CUDA capabilities) |
| CPU native contracts, including loader/ABI fixture | **2/2 passed** |
| CPU native ASan/UBSan contracts with standard options | **2/2 passed** |
| CPU + CUDA native ASan/UBSan contracts | **3/3 passed** with Nightblade CUDA shadow-gap workaround |

Commands (from the source repository, after installing the matching build):

```bash
export CORTEX_LLVM_BIN="$PWD/build/mlir-toolchain/root/usr/lib/llvm-21/bin"
CORTEX_REQUIRE_MLIR=1 uv run pytest tests/python/test_mlir_runtime.py tests/python/test_mlir_lowering.py -q
CORTEX_REQUIRE_MLIR=1 uv run pytest -q
# Run on the CPU-only build to verify no toolchain requirement:
CORTEX_LLVM_BIN=/nonexistent CORTEX_REQUIRE_MLIR="" uv run pytest -q
uv run cmake --build build/cpp-baseline
uv run ctest --test-dir build/cpp-baseline --output-on-failure
uv run cmake --build build/cpp-cuda-sanitizers
ASAN_OPTIONS=halt_on_error=1:detect_leaks=1:protect_shadow_gap=0 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 uv run ctest --test-dir build/cpp-cuda-sanitizers --output-on-failure
```

The Python builds used `CC=gcc-13 CXX=g++-13` and `uv pip install -e ".[dev]"`
with `CMAKE_ARGS="-DCORTEX_ENABLE_METAL=OFF -DCORTEX_ENABLE_CUDA=OFF"` for CPU;
CUDA used `-DCORTEX_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=52` plus
`CUDACXX=nvcc CUDAHOSTCXX=g++-13`. Native build configuration follows the
[existing C++/sanitizer setup](CUDA_PHASE9_VALIDATION.md); the CUDA shadow-gap
workaround is specific to this host. CPU native tests also passed with the
standard `halt_on_error=1:detect_leaks=1:strict_string_checks=1` ASan options,
without the workaround.

Correctness and architecture reviews passed after fixing the normalized IR
intrinsic names. Test QA identified a sanitizer CI filter that excluded the new
native tests; it now runs the full native suite. Acceptance QA checked the
documented workflow and validation limitations. Manual QA also confirmed that
the mapped native image disappears after the compiled object's last owner is
released.

CI now runs both MLIR files in require-mode and builds/runs the native fixture
without LLVM in ordinary CPU and sanitizer jobs. Metal execution cannot be
repeated on this Linux host; existing Metal tests and CI jobs are retained.
Remote CI status is not claimed as local evidence. No performance or production
readiness claim is made.

The follow-up [CUDA ABI/toolchain decision](MLIR_CUDA_INTEGRATION_DECISION.md)
is now recorded with sm_52 research execution evidence. The next task is bounded
CUDA add runtime integration; the Cortex-buffer/context/module gate above
remains open until that integration is tested.
