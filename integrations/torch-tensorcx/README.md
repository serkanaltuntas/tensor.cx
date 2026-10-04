# torch-tensorcx

Optional inference-only PyTorch integration for tensor.cx. Install tensorcx and
a compatible PyTorch build first, then from this repository root:

```bash
uv pip install ./integrations/torch-tensorcx
uv run --no-sync python examples/torch_bridge.py
```

Import `torch_tensorcx` to register `torch.ops.tensorcx.linear`. The public
`linear` wrapper supports contiguous float32 CPU/CUDA-device-0 inputs, a
two-dimensional weight and an optional one-dimensional bias. It returns fresh
storage and never mutates inputs. `from_torch` and `to_torch` share float32,
int32 and bool storage; an explicit copy isolates it. Inputs requiring gradients
must be explicitly detached. There is no autograd formula, MPS bridge or ATen
device backend. `tensorcx` itself has no PyTorch dependency.

See [the exchange contract](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/DLPACK.md) for ownership, synchronization,
CUDA runtime compatibility, benchmarks and limitations.
