# Continuous CUDA validation

> Naming update (2026-10-04): commands, source paths, and symbols in this living
> document use the current tensorcx spelling. Dated results describe runs
> under the former names; they are not new validation runs. For historical
> revisions, use the reverse mapping in [NAMING.md](NAMING.md).

Nightblade has an opt-in local `pre-push` gate for updates to `origin/main`.
The hook validates the **outgoing commit SHA** supplied by Git, including a
topic branch pushed to remote `main`. It does not validate the current working
tree in place. A failed acceptance run rejects the push and retains its logs.
Other remotes and remote branches are outside this hook's scope.

This is a local push gate, not a hosted GitHub Actions GPU runner. Pushes from
other computers, changes made through GitHub and pull requests are not covered.
On 2026-10-03 the self-hosted runner list was empty. An isolated GPU runner
is still required for repository-wide coverage. The CPU, packaging and CUDA compile-only Actions
jobs remain separate; their success does not establish GPU execution.

## GitHub workflow and runner rollout

[`cuda-acceptance.yml`](../.github/workflows/cuda-acceptance.yml) connects the
same strict acceptance command to main pushes, same-repository pull requests,
and manual dispatch. It checks out `github.sha` explicitly and verifies HEAD
before archiving it. For a PR this is GitHub's test merge commit, not merely
the branch head. Permissions are read-only for repository contents, checkout
does not persist credentials, and shared dependency caching is disabled.
The workflow publishes only the sanitized `result.json` summary for 14 days,
including failure status. Raw logs and JUnit remain local to the runner; retain
them privately for diagnosis rather than uploading the entire build directory. Its job timeout is 30 minutes.

The workflow is **prepared, not deployed GPU coverage**. Automatic jobs require
repository variable `TENSORCX_CUDA_CI_ENABLED` to equal `true`. With the variable
unset, those jobs skip; a skip is not a passing GPU acceptance result. Manual
dispatch bypasses this rollout switch to validate a new runner. It still
requires actual hardware and fails on absent CUDA/LLVM prerequisites.
Do not use the job as an enforced GPU acceptance check until deployment has
passed a manual run, a main push and a PR merge-commit run with retained logs.

2026-10-03 preparation checks: actionlint 1.7.12 passed both workflow files.
The workflow's acceptance command ran locally against `0412144`, passing 746
Python tests, four native contracts, memcheck/racecheck and the MLP. The
[retained rehearsal](validation/cuda-workflow-rehearsal.json) binds these
results to the workflow hash and explicitly records that no GitHub GPU runner
was registered or exercised. It is command-level evidence, not a deployment.

Runner contract:

- Linux x86_64, labels `self-hosted`, `linux`, `x64`, `tensorcx-cuda-sm52`.
- The existing validated environment: sm_52, CUDA Runtime 12.4, Driver API
  13.0, LLVM 21.1.8, GCC/G++ 13, `nvcc`, `compute-sanitizer`, Git and working
  driver libraries. The native runtime enforces the documented environment
  gate; a runner label alone is not compatibility evidence.
- Repository variable `TENSORCX_CUDA_LLVM_BIN` contains the absolute LLVM `bin`
  path **inside the runner environment**. `setup-uv` supplies uv; the gate
  creates its own Python 3.12 environment and installs the snapshot.
- A dedicated, disposable runner environment with no personal home directory,
  SSH/GPG keys, workspace records or Docker socket exposed to job code.
  A developer desktop account is not an isolated CI environment.

Provision the selected isolated GPU environment first, then use GitHub's
[runner registration instructions](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners)
for this repository and assign the custom label. Keep registration credentials
out of Git and reports. GitHub supports
[ephemeral registration](https://docs.github.com/en/actions/reference/runners/self-hosted-runners)
with `--ephemeral`; that handles one job only. Continuous coverage additionally
requires replenishing and cleaning runner environments between jobs, not
leaving a single completed ephemeral registration as a claimed deployment.

After a manual run passes, verify that the uploaded `result.json` names the
event SHA, reports `status: passed`, contains at least 595 Python tests and four
native contracts without skips/errors/failures, and includes both device
sanitizers and the CUDA MLP. The current suite has 746 Python acceptance tests.
Enable the automatic-run variable only after the runner lifecycle is ready,
then verify push and PR runs before closing the continuous-validation ledger.
This workflow does not provision a machine, register a runner or change branch
protection by itself.

Fork PRs do not run on this runner. Extending that scope requires a reviewed
isolation and approval policy; `pull_request_target` must not be used to run
untrusted head code with privileged base-repository credentials. See GitHub's
[self-hosted runner security guidance](https://docs.github.com/en/actions/reference/security/secure-use).
This restriction and the still-unregistered runner remain explicit rollout
limits, not evidence that repository-wide GPU acceptance is complete.

## Acceptance path

`tools/cuda_push_gate.py` archives the exact commit with `git archive`, extracts
it into a disposable directory and builds/installs a CUDA wheel in a fresh uv
environment. It does not modify the developer's package/environment or include
uncommitted source changes. Tests and native fixtures come from that snapshot.
The controller script is identified separately by SHA-256 in each result.

Every run requires:

- the validated Nightblade CUDA/LLVM environment (GCC 13, CUDA 12.4, sm_52,
  LLVM 21.1.8), uv, and `compute-sanitizer`;
- 595 or more tests from the CUDA discovery/primitive, MLIR runtime/expression,
  and public tensor API acceptance files, with no skipped, failed, errored or
  xfailed tests; named anchors include arithmetic, reduction `keepdims`, casts,
  broadcasting, transpose, squeeze, expand_dims and multi-axis reductions;
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
git config --local tensorcx.cudaLlvmBin "$PWD/build/mlir-toolchain/root/usr/lib/llvm-21/bin"
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
