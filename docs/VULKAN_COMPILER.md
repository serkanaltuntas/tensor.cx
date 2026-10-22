# Guarded MLIR / SPIR-V kernels

The optional Vulkan backend supports explicit experimental float32 kernel
compilation. The kernel body is emitted as typed SPIR-V-dialect MLIR, verified
with LLVM 21.1.8, serialized and linked with the project's numerical helpers.
The final shader is validated for Vulkan 1.1 and uses SPIR-V 1.3 with only the
core `Shader` capability. It is executed through the existing `BackendExecution`
kernel ABI; the core has no Vulkan or LLVM dependency.

```python
import tensorcx as cx

@cx.experimental.kernel
def blend(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        delta = a[i] - b[i]
        out[i] = delta * 0.5 + b[i]

a = cx.ones((257,), device="vulkan")
b = a + 2
out = cx.zeros((257,), device="vulkan")
compiled = blend.compile(target="vulkan", compiler="mlir")
result = compiled.launch(a, b, out, 257, block_size=256)
print(result.numpy())
print(compiled.cache_hit)
```

## Tools and selection

Use a Vulkan-enabled Linux build and a usable compute GPU, as described in
[the backend guide](VULKAN_BACKEND.md). Compilation additionally requires
LLVM **21.1.8** `mlir-opt`/`mlir-translate`, `glslangValidator` and SPIR-V Tools
`spirv-opt`, `spirv-link`, `spirv-val`. Clang is not needed for this device-only
target. Tools are discovered when compiling, never by ordinary tensor imports.

```bash
export TENSORCX_LLVM_BIN=/opt/llvm-21/bin
export TENSORCX_VULKAN_BIN=/opt/vulkan/bin
tensorcx-doctor --json --check-tools
```

The shader-tool directory is optional when its executables are on `PATH`.
Explicit empty/missing directories fail rather than falling back silently.
Diagnostics report `llvm.vulkan_tools` and target-specific
`llvm.ready_targets.vulkan`; that target needs two LLVM tools, while CPU/CUDA
also need Clang. The existing aggregate LLVM bundle status still describes its
three-tool CPU/CUDA bundle. Version probes are opt-in and bounded to three
seconds each. No shader is compiled or launched by diagnostics.

`target="vulkan"` and `"vulkan:0"` select the first eligible GPU; indexed
targets use the same concrete-device ownership as tensors. The artifact is bound
to that device and process on native load. Unavailable indices fail; devices
and CPU execution are never substituted. Explicit `compiler="mlir"` is required;
the existing `auto` Metal compiler behavior is unchanged.

## Complete supported guarded subset

- One canonical global index followed by one `if i < n` guard.
- Zero, one or two float32 input buffers, one float32 output template and one
  uint32 scalar bound. Tensor shapes must match exactly; all accesses use `i`.
- Up to 32 unique local assignments and one final output store. Expressions
  have depth at most 16 and at most 64 arithmetic/comparison/intrinsic nodes.
- Float32 literals, add/subtract/multiply/divide, sqrt/abs/exp/log/rsqrt/tanh/
  sigmoid/SiLU and minimum/maximum.
- All six float32 comparisons produce internal booleans. `select` eagerly
  evaluates both float32 values and chooses their bits.

The shared guarded runtime/type validator remains mandatory. Output parameter
reads, arbitrary indices/branches, loops/reductions, broadcasting, strided
buffers, additional scalar arguments and other dtypes are rejected. This matches
the bounded straight-line MLIR runtime direction; the broader Metal DSL is not
silently accepted by a less capable target.

## Numerical and ownership contracts

The compiler links the same canonical uint32 RNE/gradual-underflow helpers used
by [ordinary Vulkan operations](VULKAN_BACKEND.md). The helper library supplies
numerical functions; the user's expressions, resources and guard are generated
in MLIR, rather than replaced by a GLSL expression or a CPU interpreter.
Arithmetic uses separate float32 rounding and no FMA contraction. Basic
arithmetic, sqrt and reciprocal sqrt preserve the corresponding CPU float32
rules, including subnormal inputs/results. Arithmetic NaN payloads are not
promised; abs and value selection preserve their defined bits.

Exp/log/tanh have the backend's bounded-error implementations. Individual
intrinsics and the stable sigmoid/SiLU formulas are checked with `rtol=1e-6`
plus one minimum subnormal. Arbitrary compositions have no fixed relative-error
bound: cancellation can amplify a helper's approximation error. Each operation
still uses its separately rounded float32 helper semantics. Minimum/maximum
propagate NaNs and order -0 below +0. Comparisons are
ordered except `!=`, which is true for NaN. The experimental stable sigmoid and
half-exponential SiLU follow [their DSL contracts](MLIR_SIGMOID.md) and
[SiLU contract](MLIR_SILU.md), including negative-tail subnormal preservation;
ordinary tensor SiLU retains its separately documented formula.

`n` must equal `thread_count`, which defaults to the output size and cannot
exceed it. `block_size` is a positive uint32 within the selected GPU's invocation
and X-dimension limits. A one-dimensional logical grid can use two physical
Vulkan workgroup dimensions; every rounded-up invocation still checks the guard.
The native loader validates the manifest, entry, resource layouts, uint32 storage
and launch ABI before creating a fresh device-owned pipeline. Only the validated
LocalSize X literal varies with block size. One last-block pipeline is cached
per loaded module; active calls/quarantined frames may retain older variants.

Each launch returns private GPU storage. It copies the output template on-device
before execution, retaining untouched suffixes and zero-work values. Inputs and
the supplied output remain unchanged, including when they alias. There is no
CPU data fallback or new public asynchronous execution. Completed command-frame
reuse and failed-wait quarantine retain the [resource lifetime contract](VULKAN_PERFORMANCE.md).
Frames also retain their compiled pipeline until completion is proven.

## Cache and measurement

The [existing private compilation cache](COMPILATION_CACHE.md) stores code,
never loaded handles. Identity includes the typed MLIR, canonical helper source,
tool executables/dependencies, options and native ABI. Every hit checks tools,
SPIR-V validity and native resource/entry ABI, then performs a fresh device load/
JIT. `cache=False` and `TENSORCX_DISABLE_CACHE` retain their existing behavior.
Fresh processes can reuse code while creating their own context and module.
The existing native-artifact trust boundary applies; cached code is not a
sandbox or a format for untrusted downloaded programs.
Inherited modules and compiler load/support queries are rejected after `fork`
before touching registry or driver locks; start a fresh process with spawn/exec
for GPU compilation and execution.

```bash
mkdir -p build/vulkan-compiler
uv run --no-sync python benchmarks/bench_compilation_cache.py --target vulkan \
  --repeats 5 --restart-repeats 3 --profile \
  --output build/vulkan-compiler/measurements.json
```

The benchmark checks CPU parity and cold/warm/restart cache behavior outside the
timed compile regions, with separate profiled captures. Warm validation/JIT can
cost more than compilation for small programs; no universal speedup is implied.
Runtime host/GPU execution is measured separately with `cx.profile()`.

Tests cover every supported intrinsic with IEEE special values and subnormals,
block sizes, partial/empty work, aliases, concurrency/pooling, metadata rejection,
fresh-process cache reuse and checksum-valid invalid shader rejection. Native
tests independently exercise 2D physical dispatch and failed-completion pipeline
lifetimes without requiring LLVM during the base build. Real-device evidence is
limited to the selected GTX 980 Ti/NVIDIA driver; other GPUs are unproven.
The CPU MLIR CI job also lowers every intrinsic and validates the resulting
SPIR-V without a GPU, including a tool directory containing no Clang.
`TENSORCX_REQUIRE_MLIR_SPIRV=1` makes missing tools fail those tests;
`TENSORCX_REQUIRE_MLIR_VULKAN=1` separately requires actual GPU execution.

Local acceptance on 2026-10-22 used Linux x86_64, GTX 980 Ti, NVIDIA 580.178.04,
LLVM 21.1.8, glslang 16.2.0 and SPIR-V Tools 2026.1. The required CPU/CUDA/Vulkan
suite passed 11,994 tests with 226 expected platform/capability skips; all 144
compiler tests and all 16 native tests passed. Fresh host-specific Vulkan and
CPU wheels passed ordinary, hidden-GPU and explicit-compiler installation modes.
Independent review and test/acceptance QA found no remaining blocking issues.
Compiler acceptance alone does not establish portable packaging or acceptance
on other GPUs; see the separate [portable wheel pipeline](PORTABLE_WHEELS.md).

References: [MLIR SPIR-V dialect](https://mlir.llvm.org/docs/Dialects/SPIR-V/),
[Khronos SPIR-V Tools](https://github.com/KhronosGroup/SPIRV-Tools).
