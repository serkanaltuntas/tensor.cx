# CUDA product completion work

Started: 2026-10-03. User instruction: complete all items in the CUDA completion
list. This authorizes broader operator/product work beyond the previous narrow
prototype. This ledger preserves the full objective across implementation
increments; a completed increment does not complete the overall objective.

| Requirement | Completion evidence required | Current state |
| --- | --- | --- |
| Guarded local/nested expressions | Selected scope, CPU/reference/GPU parity, native ABI/ownership and numerical edge checks | Locally verified on Nightblade; see [scope](MLIR_CUDA_EXPRESSIONS_DECISION.md) |
| Expression acceptance suite | Required GPU runs, sanitizer results, malformed input and lifecycle checks, reviewed documentation | Locally verified; full Python 800 passed/149 skipped, native and sanitizer 4/4 each |
| CUDA operator coverage | Matmul, sum/max/mean, exp/GELU/SiLU, softmax, RMSNorm and LayerNorm through shared dispatch, CPU parity and errors | Locally verified; [primitive validation](CUDA_PRIMITIVES_VALIDATION.md) |
| Other GPU/toolchain support | Explicit support matrix, runtime selection, actual execution evidence on additional NVIDIA architectures; no inferred support from compile-only checks | Pending; only Nightblade sm_52 accessible so far |
| Performance | Reproducible compile/launch/copy/end-to-end measurements and measured optimizations; no unsupported speedup claims | [Measurement harness and long-row optimization](CUDA_PERFORMANCE.md) locally verified with two before/two after sweeps; 1.62–15.26× on measured long rows, shared-desktop limits and unchanged-path regressions recorded; other GPU performance unverified |
| Continuous GPU validation | Working GPU execution runner/process with strict no-skip acceptance and retained run evidence, alongside CPU/build CI | [Nightblade push gate](CUDA_CONTINUOUS_VALIDATION.md) implemented and locally exercised; repository-wide remote/PR GPU runner coverage still pending access |
| Product workload and distribution | Defined end-to-end workload, clean wheel installation, packaged artifacts/dependencies, documented supported environments and release checks | MLP and CPU/CUDA sdist→wheel clean-install checks locally verified on Nightblade; [distribution evidence and host limits](DISTRIBUTION.md). CPU packaging CI added; remote run not yet verified |

Implementation sequence: expressions and acceptance, operator coverage, an
end-to-end float32 inference example using those operations, measurement and
optimization, compatibility and continuous GPU validation, clean distribution
and final requirement-by-requirement audit. Run available compatibility checks
alongside implementation. A small deterministic two-layer MLP is the initial
workload candidate; standalone operator tests remain mandatory even for
operations it does not exercise.

The actual additional-GPU matrix depends on accessible machines/runners; the
user has been asked for available infrastructure. Implementable local work
continues while this information is pending. Do not convert missing hardware
evidence into a pass or shrink the objective to sm_52-only completion.

The explicitly deferred separate product tracks (a full PyTorch backend,
autograd, multi-GPU training and async/stream APIs) were not items in the
completion list. They remain separate from this objective. Source changes must
still follow review, QA, signed commit and push rules in AGENTS.md.
