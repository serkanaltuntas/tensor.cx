# MLIR Lowering Decision (Phase 10)

```text
Question:  Does Cortex Runtime IR lower to MLIR?
Decision:  YES — validated end-to-end; adopt MLIR as the intended long-term
           compiler lowering path for the kernel DSL, but do NOT integrate it
           into the runtime yet.
Date:      2026-07-05
Phase:     10 — MLIR integration exploration (run ahead of Phase 9 under
           docs/PHASE_SEQUENCING_DECISION.md)
Evidence:  experiments/mlir/ + tests/python/test_mlir_lowering.py
```

## What was demonstrated

One operation (the Phase 7 elementwise `add` kernel) was lowered from Cortex
Runtime IR through MLIR to native code and matched against the mandatory CPU
reference (PROJECT.md §5.4), exactly as the Phase 10 Definition of Done asks:

```text
@cx.experimental.kernel add
  -> parse_ir()                          existing Phase 7 frontend, unchanged
  -> emit_mlir()                         func/scf/arith/memref dialects
  -> mlir-opt (LLVM 21.1.8)              lower to the LLVM dialect
  -> mlir-translate --mlir-to-llvmir     LLVM IR
  -> clang -O2 -shared                   native library (CPU backend target)
  -> ctypes via _mlir_ciface_*           rank-1 memref descriptor ABI
  -> compared against cx CPU add         §12.3 elementwise tolerance
```

Validated behaviors (see `experiments/mlir/lower_add.py`):

- 4096-element float32 add matches the Cortex CPU backend result.
- Guard semantics are preserved: with `thread_count > n`, threads with
  `i >= n` do not write (`if i < n:` maps to `scf.if` + unsigned `cmpi ult`).
- A zero-thread launch is a no-op.
- Float constants are narrowed to float32 and formatted as valid MLIR
  literals (Python's `repr` forms like `1e-05` are not valid MLIR); the
  emitted constant parses back to exactly the float32 value. Non-finite and
  out-of-float32-range constants are rejected loudly.
- Emitter failure modes are loud, mirroring the DSL rule: unsupported
  constructs raise `MlirEmitError` naming the construct — never a silent
  miscompile. Rejected loudly: mixed int/float arithmetic, reserved
  (`cortex_`-prefixed) parameter names, missing output store, non-`(0)`
  `program_id` axes, comparison results used in arithmetic, and ordered
  comparisons involving negative integer constants (MSL types those as
  signed `int` and compiles a signed compare; the prototype only emits
  unsigned predicates, so it refuses rather than silently diverging).

## The mapping

Cortex IR proved to be a natural subset of MLIR's standard dialects. No custom
dialect was needed for this scope:

```text
Cortex IR                      MLIR
----------------------------   ------------------------------------------
buffer parameter (float32)     memref<?xf32> function argument
scalar parameter (uint)        i32 argument, unsigned ops (divui/cmpi ult)
launch grid (Metal dispatch)   scf.for over the global thread index, with
                               thread_count/block_size as trailing i32 args
program_id(0)/thread_id()/     derived from the loop variable:
block_size()                   gi / bs, gi % bs, bs
+, -, * (int / float)          arith.addi/subi/muli / arith.addf/subf/mulf
comparisons                    arith.cmpi (unsigned; eq/ne sign-agnostic;
                               signed ordered compares rejected loudly) /
                               arith.cmpf
if without else                scf.if
load / store                   memref.load / memref.store (index_castui)
assignment                     SSA binding (parser already rejects
                               branch-local escapes, so this is 1:1)
```

## Why "yes"

1. **The IR maps 1:1 onto standard dialects.** Every Phase 7 IR node has a
   direct `func`/`scf`/`arith`/`memref` equivalent; the emitter is ~300 lines
   of straightforward tree walking with no semantic gymnastics.
2. **MLIR buys real backend leverage later.** Only the LLVM-CPU lowering was
   exercised here, but the emitted module uses standard dialects
   (func/scf/arith/memref) that are the documented entry points for MLIR's
   `gpu`/`spirv`/NVVM lowering paths — the portability direction PROJECT.md
   §15 wants for CUDA/ROCm/Vulkan, without hand-writing one emitter per
   backend the way MSL emission works today. Those accelerator paths remain
   unvalidated claims until a future phase exercises them.
3. **The toolchain is already viable locally.** Homebrew `llvm@21` ships
   `mlir-opt`/`mlir-translate` and the pipeline runs in well under a second
   for this kernel; no LLVM build was required.
4. **Guard and launch semantics survive the mapping.** The Metal-style grid
   contract (`program_id * block_size + thread_id`, bound guard) lowered
   without semantic drift — the part most likely to invalidate the idea did
   not.

## Why not integrate now (boundaries)

The "yes" is a direction, not an immediate dependency. Integration into the
runtime is deferred until after Phase 9 (CUDA prototype), because:

1. **Toolchain weight and drift.** The keg-only `llvm@21` must be pinned
   explicitly (`-DMLIR_DIR=...`); making the runtime build depend on it today
   adds a heavy, version-sensitive dependency for zero user-visible gain while
   the DSL supports one elementwise pattern.
2. **The DSL is still narrow.** Phase 7 semantics (one output, float32,
   1-D guarded elementwise) do not yet exercise anything MLIR is uniquely good
   at (fusion, tiling, multi-target lowering). Adopting it early would freeze
   ABI/dialect choices against a moving frontend.
3. **Phase order.** PROJECT.md sequences the CUDA prototype before broad
   compiler work; a CUDA backend will also generate the concrete requirements
   (NVVM vs PTX path, launch ABI) that should shape the MLIR integration.

Consequences, per the Phase 10 DoD:

- The runtime core and backends contain **no** MLIR code, includes, build
  flags, or assumptions. The prototype lives in `experiments/mlir/` and reads
  only the public `cortex_runtime.experimental` IR dataclasses.
- `tests/python/test_mlir_lowering.py` keeps the evidence executable: emitter
  tests run everywhere; the end-to-end case skips cleanly without the
  toolchain (same convention as Metal tests).

## Revisit triggers

Re-open integration (a new decision record) when any of these happens:

```text
- Phase 9 lands a CUDA backend and a second kernel-lowering target is real.
- The DSL grows past guarded elementwise (reductions, tiling, fusion needs).
- A distribution decision requires shipping generated kernels without
  invoking vendor compilers (metal/xcrun) at user machines.
```

At that point the open design questions are: custom `cortex` dialect vs staying
on standard dialects, `gpu`/`spirv` vs NVVM lowering for accelerators, JIT
(`ExecutionEngine`) vs AOT shared-library compilation, and CMake integration
against a pinned LLVM/MLIR release.
