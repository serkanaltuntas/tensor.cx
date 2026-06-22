# Cortex Runtime

Cortex Runtime is the current working name for a Python-first accelerator
runtime and future kernel compiler for tensor computation, starting with Apple
Metal.

The first milestone is deliberately small:

```text
Python API -> C++20 core -> Metal backend -> static MSL add kernel -> correct result
```

## Development Setup

Cortex Runtime is developed first on Apple Silicon macOS. The default Apple
build enables Metal and requires the Apple Metal command-line tools.

```bash
command -v uv
xcode-select -p
xcrun --find metal
xcrun --find metallib
```

Use `uv` for Python environments and commands:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
uv run pytest
```

On a machine without Metal, or when validating the CPU-only path:

```bash
CMAKE_ARGS="-DCORTEX_ENABLE_METAL=OFF" uv pip install -e ".[dev]"
uv run pytest
```

Run the basic local benchmark with:

```bash
uv run python benchmarks/bench_elementwise.py
```

## Current Status

Phase 3 provides CPU tensors, dtype and shape metadata, CPU buffer ownership,
NumPy conversion, CPU add/multiply, Metal device discovery, CPU/Metal tensor copy
round-trips, and the first static Metal elementwise kernels for add, multiply,
and fill.

```python
import cortex_runtime as cx

print(cx.devices())

x = cx.tensor([1, 2, 3], device="cpu")
y = cx.tensor([4, 5, 6], device="cpu")
z = x + y

print(z.numpy())
# [5 7 9]

device = cx.best_device()
x = cx.ones((1_000_000,), dtype=cx.float32, device=device)
y = cx.ones((1_000_000,), dtype=cx.float32, device=device)
z = x + y

print(z.cpu().numpy()[:5])
# [2. 2. 2. 2. 2.]
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
