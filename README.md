# Cortex Runtime

Cortex Runtime is the current working name for a Python-first accelerator
runtime and future kernel compiler for tensor computation, starting with Apple
Metal.

The first milestone is deliberately small:

```text
Python API -> C++20 core -> Metal backend -> static MSL add kernel -> correct result
```

## Development Setup

Use `uv` for Python environments and commands:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
uv run pytest
```

## Current Status

Phase 2 provides CPU tensors, dtype and shape metadata, CPU buffer ownership,
NumPy conversion, `zeros`/`ones`/`empty`, CPU add/multiply for contiguous 1D
`float32` and `int32` tensors, Metal device discovery, and CPU/Metal tensor copy
round-trips. Phase 3 starts the first static Metal kernels.

```python
import cortex_runtime as cx

print(cx.devices())

x = cx.tensor([1, 2, 3], device="cpu")
y = cx.tensor([4, 5, 6], device="cpu")
z = x + y

print(z.numpy())
# [5 7 9]

if cx.is_available("metal"):
    x_gpu = x.to(cx.device("metal"))
    x_back = x_gpu.cpu()
    print(x_back.numpy())
    # [1 2 3]
```

## Naming

```text
Working product name: Cortex Runtime
Python package/import: cortex_runtime
Documentation alias: import cortex_runtime as cx
Python extension module: cortex_runtime._core
C++ source root: cpp/cortex/
C++ namespace: cortex
```
