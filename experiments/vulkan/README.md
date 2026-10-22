# Mesa lavapipe software execution probe

This standalone Linux experiment validates Vulkan execution on a **CPU software
device**. It does not register lavapipe in tensor.cx's production GPU registry,
add a supported GPU architecture, or change normal CPU/Vulkan selection.

`lavapipe_probe.cpp` uses the actual Vulkan loader and the explicitly selected
ICD. It requires one `VK_PHYSICAL_DEVICE_TYPE_CPU` device with the Mesa LLVMpipe
driver ID/name, Vulkan 1.2 compute limits and `VK_LAYER_KHRONOS_validation`.
An NVIDIA/other GPU ICD, missing requested validation support or failed execution
rejects the run; there is no device substitution.

`probe_lavapipe.py` compiles and validates the GLSL/SPIR-V fixture, builds the
C++20 executable, checks every readback against tensor.cx CPU primitives,
guarded `Kernel.reference` execution and NumPy, then checks production GPU
exclusion in a fresh Python process. The runtime used for that last check must
have Vulkan compiled in; an unrelated CPU-only wheel cannot prove exclusion.

## Run

Use the project's uv environment with NumPy and a Vulkan-enabled tensor.cx
installed, a C++20 compiler, Vulkan headers/loader, `glslangValidator`,
`spirv-val`, Khronos validation and a healthy Mesa lavapipe ICD. The SDK/tools
and driver are external dependencies; none is bundled or installed system-wide
by the probe. From the source checkout root:

```bash
uv run --no-sync python experiments/vulkan/probe_lavapipe.py \
  --sdk /path/to/vulkan-sdk \
  --validation-layer /path/to/validation-layer-manifests \
  --software-icd /usr/share/vulkan/icd.d/lvp_icd.json
```

`--cxx` defaults to `g++-13`; choose another C++20 compiler explicitly if needed.
`--shader-bin` overrides the SDK `bin/` directory. `--library-dir` supplies an
optional nonstandard library search directory for tools and the loader; without
it the system loader is used. Inherited Vulkan visibility/layer and loader
injection overrides are cleared. No implicit ICD search is used for execution.

Default output is a fresh `build/lavapipe-probe/<UTC>/` directory. `--output`
chooses another **new** directory; existing evidence is never overwritten.
The sanitized `result.json` must have `status: passed`. Failed prerequisites,
compilation, Vulkan calls, CPU parity, cleanup or registry checks leave a failed
report and a nonzero exit. Raw native readbacks, binaries and command logs stay
ignored; they are not publication data.

## Acceptance contract

- Two complete create/execute/destroy cycles in one native process.
- 192 cases: float32 add/multiply, separate output and alias-to-first-input,
  sizes 0, 1, 63, 64, 65, 255, 256, 257 and 1025, and zero/half/full active
  prefixes (deduplicated for small sizes).
- Real Vulkan host-visible buffer allocation, mapped upload/flush as needed,
  descriptor/pipeline creation, command submission, fence completion and
  readback/invalidation as needed. This deliberately tests software Vulkan;
  no GPU transfer bandwidth or interoperability claim follows.
- A 64-lane shader with an explicit bounds guard. Even zero-active cases submit
  a guarded group over padded sentinel storage; this is a probe of the guard,
  not a change to the runtime's allocation-free empty-tensor contract.
- Exact result bits, untouched inactive output and rounded-dispatch padding,
  immutable second input and immutable first input for separate-output cases.
  The inputs are bounded finite dyadic values; this minimal experiment does not
  establish the full runtime's operation/dtype or IEEE special-value coverage.
- Balanced explicitly owned Vulkan handles after each complete cycle, and zero
  validation errors including teardown. Implicit descriptor/command allocations
  are released by their owned pools. Failed/unproven completion is not success;
  unproven idle exits the short-lived process without freeing in-flight objects.
- A healthy software ICD remains absent from `cx.devices()`/Vulkan availability,
  while ordinary CPU execution still works in a fresh installed runtime.

The [exact-commit local Vulkan gate](../../docs/VULKAN_CONTINUOUS_VALIDATION.md)
runs this experiment from the archived outgoing source against its freshly
installed wheel, in both Vulkan-only and combined CUDA/Vulkan modes. Its software
result is separate from real GPU acceptance.

## Reviewed local acceptance — 2026-10-23

On Linux x86_64, i7-6700K, Mesa 26.0.8 / LLVM 21.1.8 lavapipe reported a
CPU device, `MESA_LLVMPIPE` (driver ID 13), driver name `llvmpipe` and Vulkan
1.4.335. Khronos SDK 1.4.363 validation was explicitly enabled. Two independent
fresh builds/runs each passed all 192 cases, balanced 1,742 owned objects over
two complete cycles and reported zero validation errors. A fresh Vulkan-enabled
runtime excluded this ICD from GPU discovery and still executed CPU operations.

Negative checks rejected an NVIDIA GPU ICD and an unavailable requested
validation layer. A real run with the shader bounds guard removed produced
incorrect inactive/padding values that the CPU oracle rejected, despite zero
Vulkan validation errors. Focused probe/gate regression tests passed 97 cases.

This establishes the bounded software experiment above. It does not establish
GPU execution, performance, full runtime operation coverage on lavapipe, other
Mesa versions or another GPU architecture. Raw local evidence remains ignored.

Sources: [Mesa ICD selection](https://docs.mesa3d.org/install.html),
[Mesa source-tree description](https://docs.mesa3d.org/sourcetree.html),
[Khronos device/driver properties](https://docs.vulkan.org/spec/latest/chapters/devsandqueues.html),
[mapped memory](https://docs.vulkan.org/spec/latest/chapters/memory.html),
[synchronization](https://docs.vulkan.org/guide/latest/synchronization_examples.html).
