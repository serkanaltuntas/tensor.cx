# CUDA product completion work

Started: 2026-10-03. User instruction: complete all items in the CUDA completion
list. This authorizes broader operator/product work beyond the previous narrow
prototype. This ledger preserves the full objective across implementation
increments; a completed increment does not complete the overall objective.

| Requirement | Completion evidence required | Current state |
| --- | --- | --- |
| Guarded local/nested expressions | Selected scope, CPU/reference/GPU parity, native ABI/ownership and numerical edge checks | Locally verified on Nightblade; see [scope](MLIR_CUDA_EXPRESSIONS_DECISION.md) |
| Expression acceptance suite | Required GPU runs, sanitizer results, malformed input and lifecycle checks, reviewed documentation | Locally verified; full Python 800 passed/149 skipped, native and sanitizer 4/4 each |
| CUDA operator coverage | Matmul, sum/max/mean, exp/GELU/SiLU, softmax, RMSNorm and LayerNorm through shared dispatch, CPU parity and errors | Locally verified; [primitive validation](CUDA_PRIMITIVES_VALIDATION.md); [native int32 fill/arithmetic/sum/max increment](CUDA_INT32.md) |
| Other GPU/toolchain support | Explicit support matrix, runtime selection, actual execution evidence on additional NVIDIA architectures; no inferred support from compile-only checks | Pending; only Nightblade sm_52 accessible so far |
| Performance | Reproducible compile/launch/copy/end-to-end measurements and measured optimizations; no unsupported speedup claims | [Measurement harness and long-row optimization](CUDA_PERFORMANCE.md) locally verified with two before/two after sweeps; 1.62–15.26× on measured long rows, shared-desktop limits and unchanged-path regressions recorded. Opt-in [runtime profiling](RUNTIME_PROFILING.md) separates host allocation/copy/waits and CUDA kernel event intervals; device-copy/Metal timing and other GPU performance remain open |
| Continuous GPU validation | Working GPU execution runner/process with strict no-skip acceptance and retained run evidence, alongside CPU/build CI | [Nightblade push gate and GitHub workflow](CUDA_CONTINUOUS_VALIDATION.md) implemented; local gate verified. Automatic remote GPU runs remain disabled until an isolated runner is provisioned and push/PR runs pass; fork PR policy and additional hardware remain open |
| Product workload and distribution | Defined end-to-end workload, clean wheel installation, packaged artifacts/dependencies, documented supported environments and release checks | MLP and CPU/CUDA sdist→wheel clean-install checks locally verified on Nightblade; [distribution evidence and host limits](DISTRIBUTION.md). Remote CPU packaging job on `2281ca9` verified; this is not remote GPU acceptance |

The [manual CUDA fusion experiment](CUDA_FUSION_EXPERIMENT.md) records two
balanced full-size runs, native/separate-MLIR/fused comparisons, separate event
diagnostics and small-input regressions. It does not add automatic graph fusion
or extend the validated GPU matrix.

The subsequent [generated-output optimization](CUDA_OUTPUT_COPY_ELISION.md)
omits the template copy only for proved nonzero full writes of v2 artifacts.
Private allocation, caller immutability, partial suffix preservation, legacy
copying and final synchronization are retained. Two before/two after reports
separate removed copy/wait counts from shared-GPU latency observations.

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

Additional completed allocation increments are bounded opt-in synchronous
[CUDA memory reuse](CUDA_MEMORY_POOL.md) and
[direct CUDA empty allocation](CUDA_EMPTY.md). These preserve the existing
device-0 scope; CPU/Metal pools, asynchronous execution and broader device
validation remain open. The [product feature backlog](ROADMAP.md#product-feature-backlog)
tracks these alongside diagnostics, compiler, dtype and distribution work.
