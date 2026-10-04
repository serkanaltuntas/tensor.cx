---
title: DLPack & PyTorch
description: Share supported CPU and CUDA tensor storage and call a tensor.cx inference operation from PyTorch.
---

## Share storage with DLPack

```python
import numpy as np
import tensorcx as cx

array = np.arange(6, dtype=np.float32).reshape(2, 3)
shared = cx.from_dlpack(array, copy=False)
view = np.from_dlpack(shared)
assert view.ctypes.data == array.ctypes.data
independent = cx.from_dlpack(shared, copy=True)
```

Exchange supports contiguous **CPU and CUDA device 0** storage with float32,
int32 or bool elements. Scalars, empty shapes and aligned offsets are supported.
`copy=False` shares storage; `copy=True` explicitly isolates it. Sharing owners
keep memory alive after the source object is deleted. Writable owners see each
other's mutations. A DLPack capsule can be consumed only once.

Metal/MPS sharing, other dtypes/devices and general strided layouts are not
supported. Unsupported requests raise an error. Read-only versioned inputs
require `copy=True`; copying does not enable arbitrary layouts.

CUDA import synchronizes producer work. Later writes from another framework
must be synchronized before tensor.cx uses the shared buffer. This is a
synchronous exchange implementation, even though it avoids copying the data.

## Optional PyTorch integration

Install tensorcx and a compatible PyTorch build, then from the repository root:

```bash
uv pip install ./integrations/torch-tensorcx
uv run --no-sync python examples/torch_bridge.py
```

```python
import torch
import torch_tensorcx as bridge

x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
w = torch.ones((4, 3))
shared = bridge.from_torch(x)
assert bridge.to_torch(shared).data_ptr() == x.data_ptr()
result = bridge.linear(x, w)
torch.testing.assert_close(result, torch.nn.functional.linear(x, w))
```

`linear` is a registered float32 CPU/CUDA custom operation with fresh output,
optional bias and fake metadata for graph capture. The bridge is inference-only;
explicitly detach inputs requiring gradients. It is not a full PyTorch device
backend and has no autograd formula. Importing tensorcx itself does not require
PyTorch.

Both libraries need compatible CUDA runtime libraries in the same process.
The bridge imports PyTorch first; if your application imports tensorcx first,
ensure a newer PyTorch wheel is not forced to use an older already-loaded CUDA
runtime. No general speedup over native PyTorch is claimed.

See the [full exchange contract and benchmarks](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/DLPACK.md)
for synchronization, ownership and current limitations.
