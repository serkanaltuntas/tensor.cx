# Local Vulkan acceptance

`tools/vulkan_push_gate.py` validates an **exact Git commit** on a real Linux
Vulkan GPU. It archives the resolved SHA, builds a wheel from that snapshot and
installs it in a fresh temporary uv environment. Runtime tests, native fixtures
and installed-wheel probes come from the snapshot, including when the working
tree has uncommitted changes. The local controller, common support and strict
pytest plugin have separate hashes in the result.

This complements the no-loader/hidden-ICD GitHub CI job and the
[portable wheel pipeline](PORTABLE_WHEELS.md). It is local acceptance, not an
isolated GitHub GPU runner or evidence for other vendors/architectures.

## Required checks

- Actual Vulkan GPU availability with the requested Khronos validation layer;
  missing layer support makes discovery fail. Every device fixture checks zero
  validation errors. There is no CPU fallback for GPU acceptance.
- Full Vulkan operation/compiler tests plus the shared int32/public tensor,
  cast/broadcast, shape, multi-axis reduction, indexing/join, mask, batched
  matmul, math, inference and explicit device-transfer contracts. CPU references
  remain mandatory. General API tests include all available compiled backends;
  transfer tests cover every ordered device pair, raw bits and forced copies.
- Real guarded MLIR/SPIR-V launches, special values, private outputs, ownership,
  threads, pooling, timing and persistent/fresh-process cache checks.
- Native CPU and Vulkan contracts, including failed/unproven waits and fresh
  no-loader/incomplete-loader fixtures; JUnit must contain every named anchor.
- Installed-wheel GPU, MLIR and hidden-device probes, checked outside the
  checkout with Python `-I`.
- A real Mesa CPU ICD discovery check and separate
  [lavapipe compute experiment](../experiments/vulkan/README.md): 192 actual
  software execution/readback cases, two complete resource lifecycles, CPU
  references, zero validation errors and production GPU exclusion. The software
  result is never counted as GPU execution.

The strict pytest controller rejects every skip, XFAIL and XPASS, including
non-strict XPASS. A successful process exit is insufficient: JUnit also rejects
skips/failures/errors, insufficient test counts and missing named anchors.
Both command output and test evidence are retained; a failed/interrupted run
must not be represented as successful acceptance.

## Two explicit modes

The default Vulkan-only mode builds CPU+Vulkan and runs 13 named native
contracts and at least 4,296 Python tests. Exactly two explicitly heterogeneous CUDA tests are deselected:
mixed CUDA/Vulkan profiling and CUDA/Vulkan threaded transfers. CUDA sharing is
not advertised by this mode. Other acceptance tests cannot skip.

`--with-cuda` builds CPU+CUDA+Vulkan. At least 6,249 Python tests, all 41 named native contracts and the full
shared-allocation Python suite are mandatory, including both allocation origins,
foreign CUDA mutations, DLPack provenance/lifetime, concurrency and generated
kernels. Neither heterogeneous test is deselected. The real native shared
contract also runs under CUDA memcheck and racecheck. Missing CUDA, matching
physical-GPU sharing or supported compiler/runtime prerequisites rejects this
mode. Its current CUDA environment is the existing Nightblade sm_52/CUDA 12.4
contract; choosing this flag does not widen CUDA hardware support.

The count floors are the measured single-GPU acceptance scope, not a substitute
for named anchors or per-test success. Additional devices/tests can increase the
counts; reducing coverage requires an explicit policy review. The selected
minimum and controller hashes are recorded in each result.

## Manual operation

Use uv, Git, a C++20 compiler, CMake/Ninja, Vulkan headers, `vulkaninfo`, the
Khronos validation layer, a healthy Mesa lavapipe ICD and a real GPU loader/ICD.
LLVM 21.1.8 (`mlir-opt`, `mlir-translate`) and `glslangValidator`, `spirv-opt`,
`spirv-link`, `spirv-val` are required for compiler acceptance. CUDA mode also
needs the validated CUDA toolchain, GCC 13 and compute-sanitizer.

For an SDK layout with `include/` and `bin/`, from the repository root:

```bash
uv run --no-project --python 3.12 python tools/vulkan_push_gate.py \
  --revision HEAD \
  --sdk /path/to/vulkan-sdk \
  --llvm-bin /path/to/llvm-21/bin \
  --validation-layer /path/to/validation-layer-manifests \
  --software-icd /usr/share/vulkan/icd.d/lvp_icd.json
```

Add `--with-cuda` for the combined contract, or `--no-with-cuda` to override an
enabled local configuration. `--shader-bin` and `--vulkaninfo` override their
SDK `bin/` defaults. `--library-dir` explicitly supplies a nonstandard tool
runtime library directory; inherited `LD_LIBRARY_PATH`/`LD_PRELOAD` are cleared.
`--gpu-icd` pins GPU discovery to a selected manifest; an invalid/missing GPU ICD
must fail real-GPU acceptance. `--cc`/`--cxx` choose the compiler; combined mode
defaults to GCC/G++ 13, Vulkan-only mode to system `cc`/`c++`.

The source compiler contract and layer setup are described in
[VULKAN_COMPILER.md](VULKAN_COMPILER.md) and
[VULKAN_BACKEND.md](VULKAN_BACKEND.md). Use a validation layer that can correctly
track the exercised external-memory operations; the recorded combined run uses
official SDK 1.4.363 validation rather than the older stale-FD metadata tracker.
No validation messages are filtered or disabled by the gate.

## Opt-in local push integration

The existing `.githooks/pre-push` still limits coverage to `origin/main` updates
and still runs the **mandatory CUDA gate first**. `pre_push_acceptance.py` reads
Git's stdin once and supplies its exact bytes to each controller. Only an
explicit repository-local `tensorcx.vulkanGateEnabled=true` adds Vulkan. A CUDA
failure prevents the second gate and rejects the push; a Vulkan failure also
rejects it. Invalid opt-in configuration fails closed. Unrelated remote refs
need no GPU setup. Cloning does not activate hooks or either GPU environment.

Configure paths locally, never commit personal paths or credentials:

```bash
git config --local tensorcx.vulkanSdk "$PWD/build/vulkan-toolchain/root/usr"
git config --local tensorcx.vulkanLlvmBin "$PWD/build/mlir-toolchain/root/usr/lib/llvm-21/bin"
git config --local tensorcx.vulkanValidationLayer "$PWD/build/vulkan-shared-runtime/validation-layer"
git config --local tensorcx.vulkanSoftwareIcd /usr/share/vulkan/icd.d/lvp_icd.json
git config --local tensorcx.vulkanLibraryDir "$PWD/build/vulkan-toolchain/root/usr/lib/x86_64-linux-gnu"
git config --local tensorcx.vulkanGpuIcd /usr/share/vulkan/icd.d/nvidia_icd.json
git config --local tensorcx.vulkanWithCuda true
git config --local tensorcx.vulkanCc gcc-13
git config --local tensorcx.vulkanCxx g++-13
git config --local tensorcx.vulkanGateEnabled true
```

Optional matching keys are `tensorcx.vulkanShaderBin`, `tensorcx.vulkanInfo`,
`tensorcx.vulkanCc` and `tensorcx.vulkanCxx`. CLI values take precedence. Retain
the existing CUDA setup and inspect `core.hooksPath` before integrating hooks;
do not overwrite another hook installation. See the
[CUDA gate's setup and security limits](CUDA_CONTINUOUS_VALIDATION.md).

Results live in `build/vulkan-gate/<UTC>-<SHA>/`: immutable source/wheel hashes,
controller hashes, dependency/tool/discovery details, raw logs, JUnit and a
sanitized `result.json`. Only `status: passed` is acceptance. Temporary source,
native builds and environments are removed after normal completion/handled
failure. Logs stay ignored and are not automatically published. Runs are not
cached; another push checks the exact outgoing SHA again. These are correctness
checks on a shared desktop, not performance measurements or a client-side
security boundary.
