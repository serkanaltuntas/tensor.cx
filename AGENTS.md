# AGENTS.md

Single source of agent guidance for tensor.cx. Read natively by
Codex and by Claude Code (via `CLAUDE.md`, which only points here). Keep all
agent/process guidance in this file — do not fork it per tool.

## Project Identity

tensor.cx is the product brand for this independent Python-first accelerator
runtime and kernel compiler project. The Python distribution and import name
are both `tensorcx`. The first target is Apple Silicon with Metal.

Current naming:

```text
Product brand: tensor.cx
Python package/import: tensorcx
Documentation alias: import tensorcx as cx
Python extension module: tensorcx._core
C++ source root: cpp/tensorcx/
C++ namespace: tensorcx
```
The naming and migration contract is in `docs/NAMING.md`. Use the new names in
code and live instructions; preserve historical reports and existing remote URLs.
The architecture must remain backend-neutral so CUDA, ROCm, Vulkan/SPIR-V, and
MLIR paths can be added later without rewriting the core runtime.

The first real milestone is intentionally small:

```text
Python API -> C++20 core -> Metal backend -> static MSL add kernel -> correct result
```

Do not treat the early project as a PyTorch, JAX, MLX, Triton, XLA, or training
framework replacement.

## Source Of Truth

Read `PROJECT.md` before making architectural or scope decisions. This file
summarizes the development process, but `PROJECT.md` contains the full roadmap,
rationale, and acceptance criteria.

When instructions conflict, follow this priority:

1. User request in the current task.
2. `AGENTS.md`.
3. `PROJECT.md`.
4. Existing code and tests.

If a user request would expand scope beyond the current phase, call that out and
prefer the smallest phase-compatible implementation.

## Approved Stack

Use these defaults unless the user explicitly changes direction:

```text
Python: 3.11+
Core runtime: C++20
Apple Metal host API: Metal-cpp/C++ preferred
GPU kernels: Metal Shading Language (.metal)
Python bindings: nanobind
Python package manager: uv
Build: CMake + scikit-build-core
Tests: pytest + focused C++ unit tests
Benchmarks: Python scripts, optional C++ microbenchmarks
```

Use `uv` for Python environments, dependency installation, editable installs,
and Python command execution. Do not use `pip`, `python -m venv`, Poetry, PDM, or
Conda for project workflow commands unless the user explicitly requests it.

Do not introduce Rust or Zig for the initial runtime, compiler, or backend core.
They may be considered later only for peripheral tooling.

## Project Status

`PROJECT.md` contains a Project Status block. Treat it as the current phase
ledger. Update it only when a phase acceptance criteria and Definition of Done
are actually satisfied.

The v0.1 target was achieved at the end of Phase 3: CPU backend, Metal buffer
copies, and first static MSL elementwise kernels passing CPU-vs-Metal tests.
Use the `PROJECT.md` Project Status block for the current phase.

## Codex / Claude Code Coordination

This repository may be edited by Codex and Claude Code in alternating sessions.
`AGENTS.md` is the shared guidance file, and `CLAUDE.md` points Claude Code here
so the guidance does not fork between tools.

At the start of every session, read `PROJECT.md` and this file. Inspect the
workspace for unexpected changes before editing. At the end of every session,
summarize the task, files changed, commands run, verification, open follow-ups,
and next recommended task in the final response.

Keep `PROJECT.md` for phase-level truth. Do not mark a phase done in
`PROJECT.md` unless its acceptance criteria and Definition of Done have passed.

## Git And Commit Rules

When a task is done, reviewed, QA-checked, tested, and verified, commit it
without asking again. Do not commit half-finished work, known-failing changes,
or changes that have not been verified. If verification cannot be run, report
that clearly and do not auto-commit unless the user explicitly asks.

Commit with the contributor's configured Git identity. Never impersonate the
maintainer or another contributor. Maintainer automation uses the existing
repository-local signing setup; do not disable signing or change its key.
Do not commit credentials, personal paths, device identifiers or raw local logs.
Before publishing artifacts, follow CONTRIBUTING.md and the publication checks.

## Reviewer Agent Gate

Every task that changes repository files must receive a professional review
before it is committed. Read-only Q&A, investigations, and user-requested
no-edit tasks do not require reviewer agents unless they produce architecture,
policy, or release decisions that should be recorded in the repository.

When multi-agent tooling is available, consult reviewer agents after
implementation and before the final commit.

Use at least one reviewer agent for non-trivial documentation or configuration
tasks. For trivial text-only changes with no behavioral, process, or
architecture impact, an explicit self-review is acceptable.

Use at least two reviewer agents for code changes:

```text
Correctness reviewer: bugs, edge cases, API behavior, error handling, ownership.
Architecture reviewer: phase scope, backend boundaries, test placement, docs structure, maintainability.
```

Reviewer agents should be asked for findings only, ordered by severity, with
file and line references where possible.

Severity mapping:

```text
P0: critical correctness, data loss, security, or build breakage; always blocking.
P1: major regression or architecture violation; always blocking.
P2: moderate bug, missing required test, or process violation; blocking unless
    explicitly classified as non-blocking with rationale.
P3: minor maintainability, wording, or style issue; not blocking by default.
```

Resolve blocking findings before commit. If a non-blocking finding is
intentionally deferred, document why in the final response and keep it
phase-appropriate.

If reviewer agents are unavailable, perform an explicit self-review pass and
state that the reviewer-agent tooling was unavailable. Do not skip review
silently.

## Quality Assurance Agent Gate

Every task that changes repository files must receive a professional QA pass
before it is committed. QA is separate from code review: reviewers judge code
quality, architecture, and maintainability, while QA agents judge whether the
task is testable, verified, and safe from user-visible regressions.

When multi-agent tooling is available, consult QA agents after implementation
and after local verification evidence exists. Verification evidence means the
relevant command output, import/build/test result, manual inspection note, or a
clear statement that no meaningful command applies. If a required verification
command cannot run, QA records the risk and the Git And Commit Rules control
whether the task can be committed.

For code changes, consult at least one QA agent. For non-trivial
phase-completion, release, Metal/backend, packaging, or workflow changes,
consult two QA perspectives:

```text
Test QA: coverage behavior, missing cases, regression risk, command evidence.
Acceptance QA: user workflow, phase acceptance criteria, documented behavior, release readiness.
```

For trivial text-only changes with no behavioral, process, or architecture
impact, an explicit self-QA pass is acceptable. Read-only Q&A, investigations,
and user-requested no-edit tasks do not require QA agents unless they produce
decisions that should be recorded in the repository.

QA agents should be asked for pass/fail risks only, ordered by severity, with
file and line references or exact commands where possible.

QA severity mapping uses the same P0-P3 scale as the Reviewer Agent Gate:

```text
P0/P1: always blocking.
P2: blocking unless explicitly classified as non-blocking with rationale.
P3: not blocking by default.
```

Resolve blocking QA findings before commit. Non-blocking QA findings may be
documented with rationale. If QA agents are unavailable, perform an explicit
self-QA pass and state that QA-agent tooling was unavailable.

## Project-Side Agent Roles

Use these roles when coordinating Codex, Claude Code, and sub-agents:

```text
Implementation agent: owns the task, edits files, runs commands, integrates feedback.
Correctness reviewer: finds bugs, edge cases, API issues, error-handling gaps.
Architecture reviewer: checks phase scope, backend boundaries, maintainability, docs.
Test QA: checks test coverage, missing scenarios, regression risk, verification commands.
Acceptance QA: checks user workflow, phase Definition of Done, docs, release readiness.
```

## Repository Layout

Target layout:

```text
python/tensorcx/        Python user API
cpp/tensorcx/core/               backend-neutral C++ runtime
cpp/tensorcx/backends/cpu/       CPU reference backend
cpp/tensorcx/backends/metal/     Metal backend using C++/Metal-cpp where possible
cpp/tensorcx/backends/metal/kernels/
                            static MSL kernels
bindings/                   Python extension binding
tests/python/               pytest correctness tests
tests/cpp/                  C++ unit tests
benchmarks/                 local benchmark scripts
docs/                       architecture and backend notes
experiments/                phase-scoped research prototypes outside the
                            runtime (e.g. experiments/mlir/ for Phase 10)
```

Keep these conceptual boundaries even if filenames evolve.

## Development Phases

`PROJECT.md` is authoritative for the current phase checklist and phase status.
Do not infer the current phase from this abbreviated sequence. Phase 8 —
Backend interface hardening and Phase 9 CUDA prototype are complete. Nightblade
is the selected CUDA validation host; preserve its documented environment gate
when moving CUDA development to another host.

Completed (Phase 10 ran ahead of Phase 9 under a documented exception):

```text
Phase 0   Project bootstrap
Phase 1   CPU backend
Phase 2   Metal backend foundation
Phase 3   First Metal kernels
Phase 4   Runtime polish
Phase 5   MPSGraph matmul
Phase 6   Reductions and NN primitives
Phase 7   Experimental kernel DSL
Phase 8   Backend interface hardening
Phase 9   CUDA prototype — Nightblade validated, 2026-09-28
Phase 10  MLIR exploration — done; decision recorded in docs/MLIR_DECISION.md
          (yes-path validated; optional CPU runtime slice implemented separately)
```

Current phase:

```text
Phase 9 and Phase 10 are complete. The first optional MLIR CPU runtime slice
is implemented and locally validated; see docs/MLIR_RUNTIME_INTEGRATION_DECISION.md.
Broader compiler/backend work remains outside that slice. CUDA generated-kernel
add/subtract/multiply runtime integration is implemented on the validated sm_52 environment;
see docs/MLIR_CUDA_INTEGRATION_DECISION.md for usage, tests and strict subset.
Wider generated operations or targets require their own parity/acceptance tests.
```

`docs/CUDA_PHASE9_ENVIRONMENT.md` records the selected host and entry criteria;
`docs/CUDA_PHASE9_VALIDATION.md` records acceptance evidence. CUDA remains a
small optional backend: discovery, allocation/copy and float32 primitives.
The user-authorized product extension adds matmul, reductions, activations and
normalization; see docs/CUDA_PRIMITIVES_VALIDATION.md and the full goal ledger.
Do not enable broader CUDA capabilities without implementation and CPU parity
validation. `docs/PHASE_SEQUENCING_DECISION.md` retains the historical exception
that allowed Phase 10 research before CUDA was available; it grants no general
permission to skip other phase gates. Experimental DSL work stays under
`cx.experimental` and keeps CPU references mandatory.

Public device routing uses a small backend registry/string-keyed mechanism.
CUDA is registered through that route. New backends must use it instead of
adding ad hoc device branches.

## Architecture Rules

The C++ core must be backend-neutral. Do not expose Metal, MPSGraph, CUDA, ROCm,
Vulkan, or platform-specific handles from `cpp/tensorcx/core/`.

The CPU backend is mandatory. Every GPU operation must have a CPU reference path
and tests comparing CPU and GPU results with dtype-appropriate tolerances.

The Metal backend is the first accelerator backend, not the central abstraction.
Use it through backend interfaces for devices, buffers, copies, and operations.

Operation dispatch is data-driven. Do not grow `Backend` with one virtual method
per op. Use a single execution entry point driven by an `OpDesc` enum plus
attributes, with CPU implementation and tests added before device paths.

Keep primitive operations separate from custom kernels:

```text
Primitive path: MPSGraph or platform libraries later, for matmul and similar ops.
Custom kernel path: project-owned static .metal kernels first, generated kernels later.
```

Prefer Metal-cpp and plain C++ for Apple backend host code. Keep any
platform-specific integration hidden behind the Metal backend interface.

Start with synchronous execution. Submit the command buffer, wait for completion,
and return the result. Do not add streams or async APIs for v0.1.

Start with contiguous row-major tensors. Strides may exist as metadata, but
non-contiguous execution is out of scope at first.

Initial dtype support should stay small: `float32` first, then `int32` if needed.
Do not start with `float16`, `bfloat16`, quantized types, or broad NumPy dtype
coverage.

Core functions should return `Status` or `expected<T, Status>`. Do not let C++
exceptions cross the Metal-cpp boundary. Translate runtime status into Python
exceptions exactly once at the nanobind layer.

Use RAII for every resource. Tensors should be cheap metadata values over shared
buffers; copying a tensor creates a view, not a data copy. Store Metal-cpp
objects in RAII handles: `NS::TransferPtr` for owned values, `NS::RetainPtr` for
borrowed values, and `NS::SharedPtr` as the stored handle. Do not store raw
`MTL::` or `NS::` pointers.

NumPy and Python types live only in the Python package or nanobind layer. They
must not appear in `cpp/tensorcx/core/` or any backend.

## Python API Expectations

Keep the first API compact:

```python
import tensorcx as cx

x = cx.tensor([1, 2, 3], dtype=cx.float32, device="cpu")
y = cx.ones((3,), dtype=cx.float32, device="cpu")
z = x + y

z.numpy()
z.cpu()
z.to("metal")
cx.devices()
cx.best_device()
```

Required early constructors and properties:

```text
cx.tensor(data, dtype=None, device=None)
cx.empty(shape, dtype, device)
cx.zeros(shape, dtype, device)
cx.ones(shape, dtype, device)
Tensor.shape
Tensor.dtype
Tensor.device
Tensor.numpy()
Tensor.cpu()
Tensor.to(device)
Tensor.__add__
Tensor.__mul__
```

Avoid broad compatibility features until the runtime foundation is solid. In
v0.1, elementwise operations require exact shape and dtype matches. Do not add
broadcasting, implicit casts, or silent reinterpretation in that original scope.
The user-authorized post-v0.1 API extension now includes explicit float32/int32
`astype`, ordinary binary tensor broadcasting, contiguous-copy transpose, and
shared-storage squeeze/expand_dims, all-axis/multi-axis sum/max/mean, and
basic indexing plus concat/stack/split as contiguous copies, followed by bool
tensors, comparisons, logical masks, where, any/all and masked selection, and float32 batched matmul
with batch broadcasting and rank-one vector promotion;
see `docs/TENSOR_API.md`. Normalizations still require one explicit axis.
Implicit dtype promotion, device transfer, general strides and generated-kernel
broadcasting remain outside that extension.

## Metal Backend Rules

For v0.1, prefer static `.metal` kernels checked into the repository. Compile
them to a `.metallib` at build time and load that library at runtime. Do not use
runtime MSL string compilation before the Phase 7 kernel DSL.

Expected first kernels:

```text
fill_f32
elementwise_add_f32
elementwise_mul_f32
```

The Metal backend should own:

```text
Metal device selection
MTLCommandQueue creation
MTLBuffer allocation
Metal library loading
compute pipeline creation
command buffer submission
synchronization
Metal-specific error reporting
```

Keep all Metal handles inside the Metal backend. No `MTL::` or `NS::` types
should appear outside `cpp/tensorcx/backends/metal/`.

Do not implement matmul in v0.1. In Phase 5, implement a naive custom MSL
`matmul_f32` first and compare it against the CPU reference. Only after that add
MPSGraph matmul as the optimized primitive path. Removing the MPSGraph path
should still leave a slow but working custom MSL matmul.

## Change Invariants

Check these before finishing any non-trivial change:

```text
1. Every GPU op has a CPU reference and a CPU-vs-device test.
2. No Apple/Metal type appears outside cpp/tensorcx/backends/metal/.
3. No NumPy/Python type appears in cpp/tensorcx/core/ or any backend.
4. The core never names a concrete backend; live public selection must stay on
   registry/string-keyed routing, including future CUDA dispatch.
5. Every Metal handle is RAII-wrapped; no manual retain/release calls.
6. Adding an op = OpDesc enum entry + CPU backend impl + test, in that order.
7. A public Python API change ships with docs and tests in the same change.
```

## Testing Requirements

Testing is part of the feature, not cleanup.

Every implemented operation needs tests for:

```text
valid inputs
shape or dtype mismatch errors where applicable
CPU correctness
Metal correctness when Metal is available
CPU vs Metal comparison for every Metal operation
```

Metal tests must skip cleanly on machines where Metal is unavailable.

Default tolerances for `cx.testing.assert_allclose`, unless an op overrides them:

```text
float32 elementwise:    rtol=1e-6, atol=1e-6
float32 reductions/sum: rtol=1e-5, atol=1e-5   (accumulation order differs)
float32 matmul:         rtol=1e-4, atol=1e-4   (FMA + tiling differences)
int32:                  exact equality
```

Looser tolerances for reductions/matmul are intentional — GPU and CPU accumulate
in different orders. Bit-exact equality is correct for integer ops. Boolean
tensors also use exact equality.

Run the most relevant test command before finishing a task. Typical commands:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

uv run pytest                                  # full suite
uv run pytest tests/python/test_tensor_cpu.py  # single file
uv run pytest -k backend_contract              # focused test selection
```

When C++ tests are relevant, run them as part of changes touching `cpp/`:

```bash
NANOBIND_DIR="$(uv run python -c 'import nanobind; print(nanobind.cmake_dir())')"
PYTHON_EXECUTABLE="$(uv run python -c 'import sys; print(sys.executable)')"
cmake -S . -B build/cpp-tests -DTENSORCX_ENABLE_METAL=OFF -DTENSORCX_BUILD_TESTS=ON -Dnanobind_DIR="${NANOBIND_DIR}" -DPython_EXECUTABLE="${PYTHON_EXECUTABLE}"
cmake --build build/cpp-tests
ctest --test-dir build/cpp-tests --output-on-failure
```

## Benchmarking Rules

Do not optimize blindly. Add or update benchmarks before making performance
claims.

Initial benchmark sizes:

```text
1K elements
16K elements
256K elements
1M elements
16M elements
```

Initial benchmark scripts:

```bash
uv run python benchmarks/bench_copy.py
uv run python benchmarks/bench_elementwise.py
```

It is acceptable for small tensors to be slower on GPU because launch and copy
overheads dominate.

## Documentation Rules

Keep documentation honest and current. When behavior changes, update the
relevant docs in the same task.

When a phase is completed, update the Project Status block in `PROJECT.md` in
the same change.

Important docs:

```text
README.md                 setup and basic usage
docs/ARCHITECTURE.md      core runtime design
docs/BACKENDS.md          backend abstraction and future backend notes
docs/METAL_BACKEND.md     Metal-specific implementation details
docs/ROADMAP.md           phased project status
```

Document what works, what is partial, and what is intentionally not implemented.

## Out Of Scope For Early Work

Do not add these to v0.1:

```text
autograd
model training
distributed training
CUDA backend
ROCm backend
TPU backend
full graph compiler
dynamic kernel DSL
ONNX import
PyTorch replacement features
high-performance matmul from scratch
quantized LLM inference
async streams
non-contiguous tensor execution
wide dtype support
```

Scope control is the main project risk. Prefer a small working runtime over a
large unfinished design.

## Coding Guidelines

Keep interfaces narrow and explicit. Prefer simple ownership rules over clever
abstractions in the early runtime.

Use C++20 RAII for resource management. Avoid raw owning pointers. Keep
platform-specific lifetimes contained in backend implementation classes.

Return clear errors for unsupported devices, dtype mismatches, shape mismatches,
unavailable Metal support, and failed backend operations.

Use Python for ergonomics and tests, C++ for runtime behavior and Metal host
integration, and MSL for kernels.

Use nanobind for the Python extension and NumPy bridge. Do not introduce
pybind11 in new code.

Avoid unrelated refactors. If a task reveals a design issue, make the smallest
change that keeps the current phase moving and document follow-up work.

## Agent Workflow

Before editing:

1. Read `PROJECT.md` and this file.
2. Inspect the current tree and existing tests.
3. Identify the current development phase.
4. Keep the change scoped to the requested task and phase.

While editing:

1. Preserve backend-neutral boundaries.
2. Add CPU behavior before or alongside accelerator behavior.
3. Add tests with the feature.
4. Keep Metal-specific code in the Metal backend directories.
5. Prefer static kernels over generated kernels until the runtime is stable.

Before finishing:

1. Run relevant formatting, build, and test commands when available.
2. For repository-changing tasks, follow the Reviewer Agent Gate, resolve
   blocking reviewer findings, and document any non-blocking reviewer findings.
3. For repository-changing tasks, follow the Quality Assurance Agent Gate,
   resolve blocking QA findings, and document any non-blocking QA findings.
4. Report exactly what changed.
5. Report what was tested and any commands that could not be run.
6. Note remaining phase-appropriate follow-ups and the next recommended task.

## Acceptance Bar For Early Public Version

The first public version should make this work:

```python
import tensorcx as cx

device = cx.best_device()

x = cx.ones((1_000_000,), dtype=cx.float32, device=device)
y = cx.ones((1_000_000,), dtype=cx.float32, device=device)

z = x + y

print(z.cpu().numpy()[:5])
# [2. 2. 2. 2. 2.]
```

The repository should also include clear setup instructions, CPU and Metal
backends, basic correctness tests, basic benchmarks, and accurate architecture
docs.
