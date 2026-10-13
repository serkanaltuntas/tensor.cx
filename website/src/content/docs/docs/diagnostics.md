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

LLVM checks do not execute tools by default. `--check-tools` runs bounded version
checks for the optional LLVM 21.1.8 toolchain. Ordinary tensor operations do not
need LLVM, and compiler readiness does not establish support for every kernel.

See the [detailed contract and limitations](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/DIAGNOSTICS.md).
