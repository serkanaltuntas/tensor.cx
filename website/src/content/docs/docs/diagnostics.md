---
title: Diagnostics
description: Inspect tensor.cx installation, operation support, CPU/CUDA memory and Vulkan owned allocations.
---

Inspect your installed runtime with:

```bash
tensorcx-doctor
tensorcx-doctor --json
tensorcx-doctor --json --check-tools
```

The standalone command reports installation problems even when the native
runtime cannot be imported. Use `python -m tensorcx_doctor --json` if the console
script is outside your PATH. Reports stay local; nothing is uploaded.

For a working runtime:

```python
import tensorcx as cx

report = cx.diagnostics()
operations = cx.capabilities("cpu")
memory = cx.memory_info("cpu")
```

The report separates backends compiled into your package from devices currently
available. Capability lists describe accepted data input dtypes; shapes, masks,
embedding indices and other arguments retain their operation-specific rules.
They are declared support, rather than an execution test.

CPU memory describes system physical memory, not your process or container's
allocation limit. CUDA memory describes device 0's instantaneous free/total
memory, including the effects of other users of that GPU. Missing measurements
are `null`; Metal memory reporting is not implemented yet.

Cached CUDA pool blocks still occupy device memory. `pool.stats()` reports the
payload owned by that pool, rather than process-wide or total GPU memory.
Use [host profiling and memory reuse](/docs/performance/) to examine allocation,
copy and wait costs, or try the [runtime tools example](/docs/examples/#inspect-and-reuse-storage).
Native Metal profiling and CUDA transfer timing remain open.

LLVM checks do not execute tools by default. `--check-tools` runs bounded version
checks for the optional LLVM 21.1.8 toolchain. Ordinary tensor operations do not
need LLVM, and compiler readiness does not establish support for every kernel.

For available Vulkan devices, opt-in checks also inspect glslang and SPIR-V
Tools. `llvm.ready_targets.vulkan` uses the two LLVM tools needed by that target
and its shader tools; it does not require Clang. No kernel is compiled or
launched by diagnostics. See [Vulkan kernels](/docs/experimental/#vulkan-kernels).

See the [detailed contract and limitations](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/DIAGNOSTICS.md).

## Vulkan diagnostics

The [optional Vulkan backend](/docs/vulkan/) reports its declared capabilities,
API version, vendor-encoded raw driver version, storage-buffer limit and total
device-local heap bytes. Available/free bytes are not collected. CUDA/Metal keep
higher automatic selection priority; Vulkan indices are process-local and only
one physical Vulkan GPU has been validated.

Separate `allocations` fields report tensor.cx-owned physical, live requested and
cached bytes; they do not represent free VRAM. `execution` counts allocation-related
command/fence/descriptor/query creation and reuse, while `timestamps` describes
queue clock support. Pool limits use physical memory requirements, including
alignment. See [Vulkan timing and reuse](/docs/performance/#vulkan-timing-and-reuse)
for the distinct heap, allocation and pool scopes.
