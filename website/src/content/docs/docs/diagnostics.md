---
title: Diagnostics
description: Inspect tensor.cx installation, operation support and CPU/CUDA memory.
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
Native Metal profiling and device-event timings remain open.

LLVM checks do not execute tools by default. `--check-tools` runs bounded version
checks for the optional LLVM 21.1.8 toolchain. Ordinary tensor operations do not
need LLVM, and compiler readiness does not establish support for every kernel.

See the [detailed contract and limitations](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/DIAGNOSTICS.md).
