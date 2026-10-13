# Installation and device diagnostics

Product feature 7 adds local CPU/CUDA diagnostics and declared operation/dtype
contracts. It does not establish support for new GPU architectures or devices.

```bash
tensorcx-doctor
tensorcx-doctor --json
tensorcx-doctor --json --check-tools
# Equivalent when the console script is not on PATH:
python -m tensorcx_doctor --json
```

The standalone `tensorcx_doctor` module remains usable if `tensorcx._core` or
NumPy cannot be imported. It emits `status: import_error`, installation advice,
and exit code 1. It intentionally omits the raw loader exception, which can
contain personal paths. A successfully collected report exits 0 even when an
optional backend/compiler is unavailable. This is a diagnostic snapshot, not
an acceptance test or an assertion that every optional feature works.
An unrecoverable collector/native mismatch emits `diagnostics_error` and exits
1. Per-backend query failures are contained; failed availability discovery is
`available: null` rather than a claim that the device is unavailable.

```python
import tensorcx as cx

report = cx.diagnostics()
cuda_copy_dtypes = cx.capabilities("cuda")["copy"]  # requires an available CUDA backend
memory = cx.memory_info("cpu")
```

`diagnostics(check_tools=False)` distinguishes compiled native backends from
usable devices. It reports package/Python/platform versions, operation dtype
contracts for available registered backends, CPU host memory and CUDA device 0
memory, architecture and runtime/Driver API versions. The numeric CUDA Driver
API version is not the NVIDIA driver package version. Metal build/availability
and operation metadata are included; Metal memory is `not_collected` in this
increment. Unknown measurements are JSON `null`, never fabricated zeros.

`capabilities(device)` maps operation names to accepted **data input** dtypes.
For embedding these describe the weight tensor; indices still require int32.
For `where`/masked selection these describe the selected data; the mask still
requires bool. Output dtype, other operands, shapes, axis/scalar restrictions,
and matmul algorithm choices remain governed by [TENSOR_API.md](TENSOR_API.md).
Capability metadata is declared support, not a kernel execution probe. CUDA
int32 storage/copies, casts, predicates, indexing and selected math operations
do not imply int32 fill/arithmetic/sum/max support. Reports expose those gaps.

`memory_info(device)` returns `scope`, `total_bytes` and `available_bytes`:

- CPU `host_physical`: system physical memory; Linux availability comes from
  `MemAvailable`, including reclaimable caches. It is not process RSS,
  tensor-owned bytes or a container/cgroup allocation allowance.
- CUDA `cuda_device`: an instantaneous CUDA free/total memory query in the
  backend-owned device 0 primary context. Other processes and driver reservations
  can affect these values; total minus available is not tensor.cx-owned usage.
- Providers without measurements use `not_collected` and null values. Direct
  provider failures raise; the aggregate report records `query_status: failed`.

No tensors are allocated, kernels launched, configurations changed, tools
downloaded or device contexts reset. Discovery/querying can initialize an
accelerator context. CUDA queries preserve a foreign current Driver context.

LLVM executable presence is reported without running external programs by
default. `--check-tools` / `check_tools=True` runs `mlir-opt`, `mlir-translate`
and `clang` with `--version`, each limited to three seconds. A matching LLVM
21.1.8 toolchain and a compatible native runtime target are both required for
`ready_targets`. Ordinary operations need no LLVM. An explicit empty or missing
`TENSORCX_LLVM_BIN` does not silently fall back to PATH.

The report contains no environment values, paths, GPU UUIDs, hostname or raw
tool output. It is local-only and is never uploaded automatically.

## Verification

`tests/python/test_diagnostics.py` executes every declared built-in
operation/dtype pair on available backends, checks unsupported CUDA int32 gaps,
report isolation, unavailable/hidden devices, bounded opt-in tool checks,
standalone CLI import failure and CUDA foreign-context preservation. Fresh
CPU/CUDA wheel checks must also verify the packaged standalone module and CLI.
Metal execution remains subject to the existing macOS CI and hardware gate.

2026-10-13 Linux x86_64 validation: the CUDA-enabled full suite passed 4,438
tests (162 expected unavailable/optional skips); the installed CPU wheel full
suite passed 2,113 (957 unavailable/optional skips). Focused diagnostics and
distribution checks passed 257 tests, native contracts passed 4/4, and native
ASan/UBSan contracts passed 4/4 with the documented CUDA
`protect_shadow_gap=0` setting. Diagnostic device checks passed 235 tests each
under memcheck/racecheck with zero errors/hazards. The hidden-device CLI test
was run normally, separately from these instrumentation runs: Compute
Sanitizer makes the injected child process see CUDA despite its empty
`CUDA_VISIBLE_DEVICES` setting. Both CPU/CUDA sdist-to-wheel installations
passed the standalone module and console-script probes. A NumPy-free wheel
installation produced the expected redacted import-error report and exit 1.
