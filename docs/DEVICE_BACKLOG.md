# Device and multi-device backlog

Created 2026-10-04 at the user's request: track remaining device families and
multi-device requirements in a dedicated checklist. This is a planning track;
unchecked items are not implemented or supported. The sequence below is a
proposal, not a change to the user-prioritized [product feature order](ROADMAP.md#product-feature-backlog).

## Starting point

CPU, Metal, CUDA and the initial Linux Vulkan slice have implemented execution
paths, with validation bounded by documented hardware and toolchains. CPU, Metal
and CUDA accept index 0 only; Vulkan has indexed registry/ownership routing:
[Python selection](../python/tensorcx/device.py),
[native metadata](../cpp/tensorcx/core/device.h).
Multiple available backend types do not establish multi-device execution.

ROCm and wider Vulkan/SPIR-V coverage remain among the directions named in
[PROJECT.md](../PROJECT.md#15-long-term-vision). MLIR is a compiler path into
execution targets, not another physical device family. TPU, NPU/Apple Neural
Engine, oneAPI/SYCL and WebGPU have no committed implementation track here.

## Nightblade validation experiments (planned)

Added 2026-10-18 at the user's request. These four experiments are planned on
Nightblade's existing GTX 980 Ti and i7-6700K. They explore execution paths on the
same physical hardware, not four additional devices or multi-GPU coverage.
The GPU Vulkan item was implemented and validated on 2026-10-18, followed by
the separate lavapipe software probe on 2026-10-23. The two OpenCL experiments
remain unexecuted.

- [x] **GTX 980 Ti via Vulkan/SPIR-V:** NVIDIA compute discovery and real GPU
  execution integrated through the runtime; copies/factories/basic arithmetic,
  CPU parity, native failure contracts and fresh wheel installation validated.
  [Scope and evidence](VULKAN_BACKEND.md). Other Vulkan GPUs remain unvalidated.
- [ ] **GTX 980 Ti via OpenCL:** establish the NVIDIA OpenCL environment,
  enumerate the selected GPU and compiler capabilities, then run the same
  minimal float32 probe against the CPU reference.
- [x] **i7-6700K via Mesa lavapipe:** explicit CPU ICD execution passed 192
  guarded float32 add/multiply cases against CPU references, including empty,
  partial/tail, alias and immutable-input checks. Two complete resource
  lifecycles balanced 1,742 owned objects with zero validation errors;
  production GPU discovery still excludes the software device.
  [Experiment and reviewed acceptance](../experiments/vulkan/README.md).
- [ ] **i7-6700K via PoCL:** establish the OpenCL CPU environment, explicitly
  select a PoCL CPU device and run the float32 probe against the existing CPU
  reference. This does not establish NVIDIA OpenCL support.

Each experiment first verifies prerequisites and reports the actual selected
implementation, device, driver/compiler versions and capabilities. An installed
ICD manifest alone is not execution evidence. Missing tools/runtime support
leave the item pending; do not silently substitute another implementation.
Acceptance requires real allocation/upload, guarded elementwise execution,
readback with CPU parity, empty/tail sizes and resource cleanup. Retain reviewed
results and limitations separately from raw local logs.

The remaining unchecked items record planned validation only. OpenCL is not
implemented; a standalone probe would not complete backend integration,
packaging or operation coverage. OpenCL is a newly recorded
exploration track, not an implemented family in the existing support matrix.
Implementation scope and ordering remain separate decisions; no delivery dates
or change to the product-feature priority order are implied.

References: [Vulkan/SPIR-V](https://registry.khronos.org/vulkan/),
[NVIDIA OpenCL](https://developer.nvidia.com/opencl),
[Mesa lavapipe](https://docs.mesa3d.org/sourcetree.html),
[PoCL device drivers](https://portablecl.org/docs/html/drivers.html).

## 1 Device identity and discovery

The [initial Vulkan slice](VULKAN_BACKEND.md#device-identity-and-ownership) now
implements canonical indexed routing, owner checks and bounded capabilities.
The cross-backend checklist stays open: legacy index-zero restrictions and
real multi-GPU/nonzero-index acceptance have not been removed or completed.

- [ ] Define canonical backend-plus-index identity through Python, bindings,
  registries and native dispatch. Preserve existing unindexed device strings;
  specify defaults, equality, invalid indices and unavailable-device errors.
- [ ] Enumerate all usable devices per backend and select a nonzero device
  explicitly. Define process-local index stability and visibility/remapping
  behavior; do not treat an ordinal as a permanent hardware identifier.
- [ ] Expose per-device capabilities: operations, dtypes, memory information,
  architecture and compiler availability. Define deterministic `best_device()`
  selection without silently relocating existing tensors.
- [ ] Handle one physical GPU exposed through multiple backends: document
  identity/topology limitations and keep backend buffers distinct unless an
  explicit, validated interoperability path exists.

Acceptance: zero-, one- and multiple-device discovery; both string and Device
selection; unavailable/hidden devices; existing index-0 API compatibility.
Complete this contract before either new backend adopts device selection.

## 2 Ownership and independent execution

- [ ] Carry the full device identity through allocation, tensors/views,
  primitive dispatch, generated kernels and destruction; remove index-0
  assumptions throughout the implementation, not just the parser.
- [ ] Scope contexts, queues, allocator state and loaded kernel artifacts to
  their owning device. Key reusable compiled code by backend, architecture,
  toolchain and compilation options; keep loaded modules device/context-aware.
- [ ] Define mixed-device operation errors and explicit placement. Keep views
  on their owning device and prevent pointer/context reuse on another device.
- [ ] Support independent work on two devices in one process, including
  concurrent host threads, device-local synchronization, cleanup and failures
  without changing another thread's device selection.

Acceptance: execute on device 0 and device 1, interleave allocation/launch/free,
check CPU parity on each, and exercise wrong-device arguments and resource
lifetime failures. Mock devices may test routing, but real hardware is required
to close execution acceptance.

## 3 Explicit transfers between devices

- [x] Specify `Tensor.to(...)` for same-device, same-backend/different-device
  and cross-backend transfers, including dtype/shape preservation, copy versus
  alias behavior, empty tensors and source/destination lifetime.
- [x] Establish a synchronous host-staged transfer baseline for supported
  device pairs, with clear errors for unsupported pairs and allocation failure.
- [ ] Add capability-gated direct peer copies where available, with a tested
  host-staged fallback when peer access is absent or cannot be enabled.
- [ ] Integrate future streams/events and memory pools only after their shared
  contracts exist. Define source completion, destination readiness, buffer
  lifetime and cross-device dependencies before exposing asynchronous copies.

Acceptance: round trips and CPU parity for supported dtypes and device pairs;
peer-enabled and peer-unavailable paths; memory-pressure errors; separate copy
and end-to-end measurements. See product backlog item 10 for async/pool work.
The [implemented placement contract](DEVICE_TRANSFERS.md) has real
CPU/CUDA/Vulkan acceptance on Nightblade. Mocked Vulkan indices establish
routing only. The [explicit shared-allocation API](SHARED_STORAGE.md) now supports
matching CUDA/Vulkan aliases in both allocation directions. Actual two-GPU
execution, peer copies and other device-pair sharing remain open; this does not
close the broader device interoperability scope.

## 4 Remaining backend families

- [ ] **ROCm:** select a supported hardware/toolchain test environment, then
  implement discovery, ownership, allocation/copy and a minimal float32
  fill/add/multiply slice through the shared backend contract. Validate CPU
  parity and multi-device behavior before expanding operation coverage.
- [ ] **Wider Vulkan/SPIR-V acceptance:** the optional Linux runtime is
  implemented on one GTX 980 Ti, including the initial slice, shape/mask/math
  operations, reductions and float32 batched matmul. Continue the
  [canonical Vulkan checklist](VULKAN_BACKEND.md#remaining-vulkan-work) and
  expand the vendor/device matrix and real multi-device execution evidence.
- [ ] For each backend, record primitive and generated-kernel capabilities
  separately; validate optional builds, missing-runtime behavior, clean package
  installation and supported-host limits before declaring support.

An earlier proposal listed ROCm before Vulkan; Linux Vulkan was implemented
first on the available GPU on 2026-10-18. Remaining work depends on available
hardware and product priorities, without committed delivery dates.

## 5 Workloads and continuous acceptance

- [ ] Demonstrate explicit partitioning of independent batches across two
  devices and reassembly with single-device/CPU parity.
- [ ] Demonstrate an explicitly placed inference pipeline across devices,
  including intermediate transfers; measure transfer cost as well as compute.
- [ ] Maintain an execution matrix covering single-device regressions,
  same-backend multi-device operation and supported heterogeneous device pairs.
  Record device count, architecture, runtime/toolchain and outgoing commit;
  missing hardware is pending evidence, not a passing skip.
- [ ] Connect required real-device runs to isolated CI with retained evidence.
  Additional NVIDIA architectures and CUDA runner rollout remain owned by the
  [CUDA completion ledger](CUDA_PRODUCT_COMPLETION.md); reference their evidence
  here instead of maintaining duplicate completion states.

Suggested first implementation: device identity/discovery, then two-device
CUDA ownership and synchronous copies on a host with two accessible GPUs.
Backend-neutral design must also accommodate Metal and the future backends.
Hardware availability remains an open prerequisite, not an assumed resource.

Collectives, automatic sharding, multi-process/multi-host execution and
distributed training require separate scope decisions. The workloads above
exercise explicit placement; they do not claim those broader capabilities.
