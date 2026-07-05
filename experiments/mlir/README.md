# Phase 10 MLIR Lowering Prototype

Research prototype answering the Phase 10 question: *does Cortex Runtime IR
lower to MLIR?* The recorded answer and rationale live in
[`docs/MLIR_DECISION.md`](../../docs/MLIR_DECISION.md); this directory is the
executable evidence.

This code is intentionally **outside** the runtime: nothing under
`python/cortex_runtime/` or `cpp/cortex/` imports, links, or assumes MLIR. The
prototype only consumes the backend-neutral IR dataclasses that
`cortex_runtime.experimental` already exposes.

## Contents

```text
cortex_ir_to_mlir.py   Cortex kernel IR -> MLIR (func/scf/arith/memref) emitter
lower_add.py           end-to-end pipeline: IR -> MLIR -> LLVM -> dylib ->
                       ctypes execution -> comparison vs the Cortex CPU backend
```

## Pipeline

```text
@cx.experimental.kernel add          (Phase 7 frontend, unchanged)
  -> kernel.parse_ir()               backend-neutral Cortex IR
  -> emit_mlir()                     func/scf/arith/memref dialects
  -> mlir-opt                        --convert-scf-to-cf --convert-to-llvm
                                     --reconcile-unrealized-casts
  -> mlir-translate --mlir-to-llvmir LLVM IR
  -> clang -O2 -shared               native library
  -> ctypes (_mlir_ciface_*)         rank-1 memref descriptors
  -> compare vs cx CPU add           PROJECT.md §12.3 elementwise tolerance
```

The Metal launch grid maps to a single `scf.for` over the global thread index;
`program_id(0)`, `thread_id()`, and `block_size()` are derived from the loop
variable and a `block_size` function argument, so guard semantics (`if i < n:`)
are preserved exactly.

## Running

```bash
uv run python experiments/mlir/lower_add.py
```

Prints `PHASE10-PROTOTYPE-OK` on success; exits 77 with a `SKIP` line when the
toolchain is missing. Tests: `uv run pytest tests/python/test_mlir_lowering.py`
(the end-to-end case skips without the toolchain, like the Metal tests).

## Toolchain

Discovery order: `$CORTEX_LLVM_BIN`, the Homebrew `llvm@21` keg
(`/opt/homebrew/opt/llvm@21/bin`), then `PATH`. Requires `mlir-opt`,
`mlir-translate`, and `clang` from the same LLVM release (validated against
Homebrew LLVM 21.1.8 — see `docs/PHASE_SEQUENCING_DECISION.md` for the local
toolchain record).
