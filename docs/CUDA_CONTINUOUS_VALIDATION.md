# Continuous CUDA validation

Nightblade has an opt-in local `pre-push` gate for updates to `origin/main`.
The hook validates the **outgoing commit SHA** supplied by Git, including a
topic branch pushed to remote `main`. It does not validate the current working
tree in place. A failed acceptance run rejects the push and retains its logs.
Other remotes and remote branches are outside this hook's scope.

This is a local push gate, not a hosted GitHub Actions GPU runner. Pushes from
other computers, changes made through GitHub and pull requests are not covered.
On 2026-10-03 the self-hosted runner list was empty. An isolated GPU runner
is still required for repository-wide coverage. The CPU, packaging and CUDA
compile-only Actions jobs remain separate.

## Acceptance path

`tools/cuda_push_gate.py` archives the exact commit with `git archive`, extracts
it into a disposable directory and builds/installs a CUDA wheel in a fresh uv
environment. It does not modify the developer's package/environment or include
uncommitted source changes. Tests and native fixtures come from that snapshot.
The controller script is identified separately by SHA-256 in each result.

Every run requires:

- the validated Nightblade CUDA/LLVM environment (GCC 13, CUDA 12.4, sm_52,
  LLVM 21.1.8), uv, and `compute-sanitizer`;
- 595 or more tests from the CUDA discovery/primitive and MLIR runtime/expression
  acceptance files, with no skipped, failed, errored or xfailed tests;
- four named native CTest contracts, with no skipped or failed cases;
- CUDA backend native contracts under both memcheck and racecheck;
- the CUDA MLP example reporting CPU parity and shape `(32,10)`.

The controller's pytest plugin also rejects non-strict XPASS, even if a test
overrides pytest's default `xfail_strict` policy. Required CUDA capabilities and MLIR modes are
set explicitly. JUnit reports are inspected after command success, so a
zero-exit pytest/CTest run that skipped acceptance does not pass. Test-count
floors and named anchors detect accidental selection loss; deliberate changes
to acceptance scope require review and corresponding gate updates.

Builds use two parallel jobs. CPU/GPU correctness may be checked while another
workload runs, but these timings are not performance evidence. The gate is
synchronous: a normal push waits for validation. It never stops another job.

Results live at `build/cuda-gate/<UTC>-<commit>/`: source archive hash, controller
hash, dependency/toolchain/GPU details, pytest/CTest XML, compiler output,
sanitizer logs, MLP output and `result.json`. Only `status: passed` is accepted.
Failed runs record the failure; interrupted processes may leave partial logs.
The temporary source/build/environment is removed on normal completion or
handled failure. Retained results are ignored by Git and are not automatically
deleted. Preserve evidence needed for releases before manually pruning older
run directories. Missing evidence is never treated as a pass.

## Local setup and operation

Installation is explicit and repository-local; cloning does not activate hooks.
First inspect `git config --get core.hooksPath` and the existing hooks directory.
Do not overwrite or mask existing hooks; integrate them deliberately if present.
Nightblade had no custom hooks when this gate was installed.

```bash
git config --local cortex.cudaLlvmBin "$PWD/build/mlir-toolchain/root/usr/lib/llvm-21/bin"
git config --local core.hooksPath .githooks
git push origin main
```

The versioned hook must be executable. It uses uv with Python 3.12 and executes
the local controller, while the tested product source is the outgoing commit.
The LLVM path is machine-local Git configuration; no credential is stored in
the repository. `LD_LIBRARY_PATH`/`LD_PRELOAD`, Python import overrides and
pytest argument/plugin overrides are cleared for child processes.

Manual acceptance without pushing or enabling a hook:

```bash
uv run --no-project --python 3.12 python tools/cuda_push_gate.py \
  --revision HEAD --llvm-bin build/mlir-toolchain/root/usr/lib/llvm-21/bin
```

Inspect `result.json` and referenced logs on failure, fix the environment/code,
then retry. The gate does not cache successes: subsequent pushes recheck the
actual device and tools. Like all client-side hooks, this is not a security or
branch-protection boundary and can be bypassed locally. A fresh clone, a
different host or a hook bypass must not be represented as a GPU-validated push.

## Evidence and remaining scope

Initial exact-commit execution on `40529ae` passed 595 Python tests, all four
native contracts, both device sanitizer runs and the MLP. The installed hook
then passed the same checks with Git protocol input and inherited `GIT_DIR`.
GPU hiding caused the required-device run to fail (pytest exit 4), with a
retained failed result. Fourteen regression tests passed, covering push filtering,
skip/failure/anchor rejection, non-strict XPASS/XFAIL, timeout diagnostics and hook stdin/exit handling.
The [retained evidence](validation/cuda-push-gate.json) distinguishes the hook
rehearsal from an actual remote update. Current
evidence and remaining repository-wide runner work are tracked in
[CUDA_PRODUCT_COMPLETION.md](CUDA_PRODUCT_COMPLETION.md).
