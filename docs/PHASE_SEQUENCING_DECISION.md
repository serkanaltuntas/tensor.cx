# Phase Sequencing Decision: Phase 9 Paused, Phase 10 Started Without CUDA

## Context

The original phase plan (`PROJECT.md` §14, `AGENTS.md` "Development Phases")
sequences work strictly: Phase 9 (CUDA prototype) before Phase 10 (MLIR
exploration). `AGENTS.md` states explicitly: "Only move to ROCm, Vulkan, MLIR,
or broad compiler work after the earlier phases are working and tested."

Phase 9 is blocked on a CUDA hardware/cloud environment decision
(`docs/CUDA_PHASE9_ENVIRONMENT.md`). The developer currently has no CUDA-capable
machine or cloud access. Waiting on that blocker idles the project indefinitely.

## Decision

Phase 9 stays paused, not skipped. Phase 10 (MLIR exploration) is started now,
ahead of Phase 9, as a scoped exception to the documented phase order.

This is possible without violating the spirit of the sequencing rule because
Phase 10's own Definition of Done (`PROJECT.md` §14, Phase 10) does not require
CUDA:

```text
- A written decision record answers: does Cortex Runtime IR lower to MLIR, or
  not, and why. A documented "no" is a valid, successful outcome.
- If yes: one op (e.g. elementwise add) lowers Cortex Runtime IR -> MLIR ->
  backend and matches CPU.
```

The CPU/Metal-validated single-op case, if pursued, needs no CUDA hardware.

## Guardrails

- **Phase 9 status is unchanged.** It remains "not started, blocked on CUDA
  environment decision." This decision does not mark Phase 9 done, skip its
  acceptance criteria, or substitute Phase 10 work for it.
- **Phase 10 scope stays exactly as documented.** Either:
  1. a written decision record concluding MLIR lowering is not pursued now
     (a valid, complete outcome), or
  2. a decision record plus one op (elementwise add) prototyped through
     Cortex IR -> MLIR -> backend, matched against the CPU reference.
  No broader compiler work, no additional ops, no core/backend code that
  assumes MLIR before the decision record exists (per the original Phase 10
  Definition of Done).
- **No core or backend code changes are implied by this decision alone.**
  This document only reorders *when* Phase 10 research may start; it does not
  change what Phase 10 requires.
- **Resuming Phase 9.** When a CUDA hardware or cloud environment becomes
  available, Phase 9 resumes at its documented starting point
  (`docs/CUDA_PHASE9_ENVIRONMENT.md` decision gate), independent of whatever
  Phase 10 concluded.

## Status

```text
Date recorded:      2026-07-05
Phase 9:            not started, blocked on CUDA environment (unchanged)
Phase 10:           not started -> in progress (sequencing exception, this record)
Reason for reorder: no CUDA hardware/cloud access currently available
```
