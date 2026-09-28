# Phase 10 MLIR Lowering Prototype

Research prototype answering the Phase 10 question: *does Cortex Runtime IR
lower to MLIR?* The recorded answer and rationale live in
[`docs/MLIR_DECISION.md`](../../docs/MLIR_DECISION.md); this directory is the
executable evidence.

These research harnesses stay outside the runtime. Their emitter now delegates
to the private compiler module shared with the optional CPU runtime. The C++
build does not link LLVM/MLIR, and ordinary package import needs no toolchain.

## Contents

```text
cortex_ir_to_mlir.py   thin compatibility import of the shared private emitter
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

CI exercises this for real: the `mlir-lowering` job in
[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) installs LLVM/MLIR
21 and runs the lowering tests with `CORTEX_REQUIRE_MLIR=1`, which turns a
missing toolchain into a hard failure so the evidence cannot silently
green-skip. Set the same variable locally to enforce the toolchain instead of
skipping.

## Toolchain

Discovery order: `$CORTEX_LLVM_BIN`, the Homebrew `llvm@21` keg
(`/opt/homebrew/opt/llvm@21/bin`), then `PATH`. Requires `mlir-opt`,
`mlir-translate`, and `clang` from the same LLVM release (validated against
Homebrew LLVM 21.1.8 — see `docs/PHASE_SEQUENCING_DECISION.md` for the local
toolchain record).

## Follow-up integration scope

The [runtime integration decision](../../docs/MLIR_RUNTIME_INTEGRATION_DECISION.md)
records the CPU-first plan, native execution boundary, and fresh Nightblade
LLVM 21.1.8 evidence. These scripts remain research harnesses; their ctypes
bridge and looser guard test are separate from the implemented public runtime
launch contract.


## CUDA device-only probe

The [CUDA integration decision](../../docs/MLIR_CUDA_INTEGRATION_DECISION.md)
records the selected ABI and exact host/toolchain. These research files do not
enable a Cortex CUDA compiler target:

- `cuda_add.mlir`: fixed guarded GPU add fixture; raw pointer/u32 arguments.
- `probe_cuda.py`: GPU-to-NVVM/PTX lowering, parameter checks and CPU parity.
- `cuda_driver_probe.cpp`: Runtime allocations, Driver module launch, context
  restoration and sentinel checks on the selected Nightblade GPU.

```bash
uv run python experiments/mlir/probe_cuda.py --compile-only
uv run python experiments/mlir/probe_cuda.py
```

Both require `CORTEX_LLVM_BIN` selecting LLVM 21.1.8. The second additionally
requires CUDA 12.4 headers/libraries/ptxas, `g++-13` (or `--cxx`), and sm_52.
Output artifacts live in a temporary directory and are removed after validation.
The separate runtime now supports the validated guarded CUDA add subset. These
research harnesses remain independent historical/toolchain evidence; runtime
acceptance additionally exercises parsed Cortex IR, Cortex buffers and backend execution.
