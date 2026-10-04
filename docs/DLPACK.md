# DLPack and the optional PyTorch bridge

Product feature 6 adds shared storage exchange and one inference custom op.
It does not implement the full PyTorch backend/ATen track in the
[portability roadmap](PYTORCH_PORTABILITY_ROADMAP.md).

## Exchange contract

```python
import numpy as np
import tensorcx as cx

array = np.arange(6, dtype=np.float32).reshape(2, 3)
shared = cx.from_dlpack(array, copy=False)
view = np.from_dlpack(shared)
assert view.ctypes.data == array.ctypes.data
independent = cx.from_dlpack(shared, copy=True)
```

`cx.from_dlpack(source, *, copy=None)` accepts a producer with `__dlpack__` and
`__dlpack_device__`, or an unconsumed legacy/versioned capsule. Every Tensor
exports these methods. The supported contract is:

| Property | Supported behavior |
| --- | --- |
| Devices | CPU and CUDA device 0, same-device exchange |
| Dtypes | float32, int32 and bool (one byte, one lane) |
| Layout | C-contiguous; scalar, empty, singleton axes and aligned byte offsets |
| Versions | Legacy capsules and DLPack 1.0 versioned capsules |
| `copy=None` / `False` | Share writable storage; unsupported inputs raise |
| `copy=True` | Allocate an independent contiguous same-device copy |
| Read-only input | Versioned read-only storage requires `copy=True` |
| Metal/MPS, other dtypes/devices/layouts | Explicit errors; no implicit fallback |

General strided execution and layout conversion are not supported, including
with `copy=True`. Singleton and empty-axis strides can be noncanonical because
they do not change the accessible elements. Shape, dtype, alignment, offset
overflow and device declarations are validated; a raw capsule remains a trusted
native-memory interface, not a safe parser for arbitrary pointers.

`Tensor.__dlpack__(*, stream=None, max_version=None, dl_device=None, copy=None)`
exports a legacy capsule unless `max_version >= (1, 0)`. Versioned copies set
`DLPACK_FLAG_BITMASK_IS_COPIED`; the legacy representation has no flags field.
`dl_device` can only name the tensor's existing device. Use `.to(...)` explicitly
for transfers. Consumed capsules cannot be reused.

### Ownership and mutation

The imported native buffer retains the producer's managed tensor until the last
sharing view/export is destroyed. Its deleter runs exactly once. Export retains
the tensor.cx buffer even after the source Python tensor is deleted. Mutations
through any writable sharing owner are visible to all owners; concurrent writes
and reads need caller coordination. NumPy may expose a read-only view of a
writable export. This does not make other owners immutable.

CPU uses pointer/count spans and a native shared owner. CUDA validates device
memory on device 0's primary context and retains that context with the borrowed
buffer. DLPack structs and Python objects stay in the bindings; the backend-neutral
core has no DLPack, PyTorch or concrete GPU types. The public C interchange
boundary is the vendored DLPack ABI, not a promise of a stable C++ binary ABI.
Independently imported identical memory ranges retain the experimental kernel
API's alias semantics. Partially overlapping input/output ranges are rejected
before kernel execution; the reference interpreter also rejects partial overlap
between arguments. Kernels continue to write a private output copy.

### CUDA synchronization and runtime compatibility

The runtime remains synchronous. Import requests CUDA's legacy default stream
and completes producer work with device synchronization, including imports of
bare capsules. This conservative step can wait for unrelated work; zero-copy
describes storage sharing, not asynchronous execution or zero overhead.

Export of existing storage needs no additional synchronization because tensor.cx
operations finish before returning. CUDA accepts `None`, `1` (legacy default),
`2` (per-thread default), positive stream pointers, and `-1` (no synchronization);
`0` and other negative values are rejected. `stream=-1, copy=True` is rejected
because the synchronous copy requires synchronization. Export copies support
only `stream=None` or `1`; other consumer streams require an explicit copy before
export. Sharing existing storage accepts all the stream forms above.
CPU requires `stream=None`.
After sharing, synchronize later foreign GPU writes before tensor.cx reads or
re-exports that buffer; tensor.cx cannot track another framework's future work.

Both libraries must load compatible CUDA runtime libraries in the same process.
The bridge imports PyTorch before tensor.cx. In a process importing tensor.cx
first, an older already-loaded `libcudart.so` can prevent a newer PyTorch wheel
from importing. Select a compatible build/runtime search path before starting
Python; the bridge does not replace drivers or alter the system loader.

## PyTorch custom op

The optional package lives in `integrations/torch-tensorcx`. Install a suitable
PyTorch build and tensorcx first, then install the bridge from the repository:

```bash
uv pip install ./integrations/torch-tensorcx
uv run --no-sync python examples/torch_bridge.py
uv run --no-sync pytest tests/python/test_torch_bridge.py -q
```

`torch_tensorcx.from_torch(tensor, copy=None)` and
`torch_tensorcx.to_torch(tensor, copy=False)` exchange supported storage.
`torch_tensorcx.linear(input, weight, bias=None)` calls the registered
`torch.ops.tensorcx.linear` operation. Inputs use float32 on one supported device;
shapes are `(..., in_features)`, `(out_features, in_features)` and optionally
`(out_features,)`. Output is fresh storage; inputs are not mutated.

The op registers fake metadata for graph capture and is checked with
`torch.library.opcheck` and `torch.compile(..., backend="eager", fullgraph=True)`.
This validates custom-op graph capture, not arbitrary compiler backend coverage
or fusion. Inputs requiring gradients must be explicitly detached; there is no
autograd formula. Sparse, noncontiguous and unresolved conjugate/negative views
are rejected. No automatic CPU fallback, MPS sharing, general ATen backend,
training support or transparent model replacement is provided.

Importing tensorcx does not import or require PyTorch. The bridge is a separate
pure-Python distribution; tensorcx remains the native runtime dependency.

## Measurement and acceptance

```bash
uv run --no-sync python benchmarks/bench_torch_bridge.py --device cpu
uv run --no-sync python benchmarks/bench_torch_bridge.py --device cuda
```

The benchmark checks each result against native PyTorch linear, proves sharing
by pointer equality, warms up both providers and reports synchronized wall-clock
medians/minima. Bridge timing includes storage exchange and result allocation.
CUDA timings include synchronization; one PyTorch CPU thread is used. Shared
desktop measurements are not evidence of an isolated accelerator speedup.

The first local samples (2026-10-04, Linux x86_64, CUDA sm_52, PyTorch
2.13.0+cu126) are retained as [CPU](../benchmarks/results/dlpack-bridge-cpu.json)
and [CUDA](../benchmarks/results/dlpack-bridge-cuda.json) reports. Across the three
tested shapes, bridge linear was slower than native PyTorch. For `(M,K,N) =
(32,64,32)`, CPU median was 0.250 ms versus 0.007 ms and CUDA median was
0.473 ms versus 0.027 ms. These shared-host samples prove the measurement path
and correctness, not a speedup. The increment's benefit is explicit shared
storage and the ability to call the runtime through PyTorch's dispatcher.

Local verification: 4203 core tests passed with 161 expected optional/platform
skips; the optional bridge passed 72 real CPU/CUDA tests. Native and ASan/UBSan
contracts passed 4/4. The 182 DLPack tests passed CUDA memcheck and racecheck with
zero reported errors/hazards; 70 bridge tests passed memcheck (the two compiler
capture cases were checked separately without instrumentation). CPU/CUDA
sdist-to-wheel fresh installations passed LLVM-present, LLVM-absent and GPU-hidden
modes with exchange checks. These results do not imply other GPU architectures
or portable release wheels are validated.

Acceptance covers metadata errors, legacy/versioned negotiation, copy/read-only
behavior, exact-once deletion, views surviving producers, NumPy sharing, real
PyTorch CPU/CUDA sharing, a non-default producer stream, inference parity, custom
op schema/fake behavior and compile capture. Core builds and tests remain usable
without PyTorch. Real Metal regression and explicit unsupported Metal exchange
are checked separately from CPU/CUDA interoperability.

Protocol references: [DLPack Python specification](https://dmlc.github.io/dlpack/latest/python_spec.html),
[Array API export contract](https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html),
[PyTorch custom operators](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html).
