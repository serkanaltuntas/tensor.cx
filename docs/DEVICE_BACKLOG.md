# Device and multi-device backlog

Created 2026-10-04 at the user's request: track remaining device families and
multi-device requirements in a dedicated checklist. This is a planning track;
unchecked items are not implemented or supported. The sequence below is a
proposal, not a change to the user-prioritized [product feature order](ROADMAP.md#product-feature-backlog).

## Starting point

CPU, Metal and CUDA have implemented execution paths, with validation bounded
by their documented hardware and toolchains. Python currently accepts only
device index 0, although Python and C++ device metadata contain an index:
[Python selection](../python/tensorcx/device.py),
[native metadata](../cpp/tensorcx/core/device.h).
Multiple available backend types do not establish multi-device execution.

ROCm and Vulkan/SPIR-V are the remaining backend families named in
[PROJECT.md](../PROJECT.md#15-long-term-vision). MLIR is a compiler path into
execution targets, not another physical device family. TPU, NPU/Apple Neural
Engine, oneAPI/SYCL and WebGPU have no committed implementation track here.

## 1 Device identity and discovery

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

- [ ] Specify `Tensor.to(...)` for same-device, same-backend/different-device
  and cross-backend transfers, including dtype/shape preservation, copy versus
  alias behavior, empty tensors and source/destination lifetime.
- [ ] Establish a synchronous host-staged transfer baseline for supported
  device pairs, with clear errors for unsupported pairs and allocation failure.
- [ ] Add capability-gated direct peer copies where available, with a tested
  host-staged fallback when peer access is absent or cannot be enabled.
- [ ] Integrate future streams/events and memory pools only after their shared
  contracts exist. Define source completion, destination readiness, buffer
  lifetime and cross-device dependencies before exposing asynchronous copies.

Acceptance: round trips and CPU parity for supported dtypes and device pairs;
peer-enabled and peer-unavailable paths; memory-pressure errors; separate copy
and end-to-end measurements. See product backlog item 10 for async/pool work.

## 4 Remaining backend families

- [ ] **ROCm:** select a supported hardware/toolchain test environment, then
  implement discovery, ownership, allocation/copy and a minimal float32
  fill/add/multiply slice through the shared backend contract. Validate CPU
  parity and multi-device behavior before expanding operation coverage.
- [ ] **Vulkan/SPIR-V:** select a tested device/driver and compilation path,
  define required compute/memory capabilities, then implement the same minimal
  slice and multi-device contract. Expand the vendor/device matrix through
  real execution evidence, not API availability alone.
- [ ] For each backend, record primitive and generated-kernel capabilities
  separately; validate optional builds, missing-runtime behavior, clean package
  installation and supported-host limits before declaring support.

ROCm followed by Vulkan is a proposed order; actual order depends on available
hardware and product priorities. Neither has a committed delivery date.

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
