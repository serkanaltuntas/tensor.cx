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
uv run python benchmarks/bench_copy.py
uv run python benchmarks/bench_matmul.py
```

The benchmark scripts use the Phase 4 benchmark sizes by default: 1K, 16K,
256K, 1M, and 16M float32 elements.

A local Apple Silicon sample run is committed at
`benchmarks/sample_phase4_apple_silicon.txt`. The Phase 5 matmul sample is at
`benchmarks/sample_phase5_matmul_apple_silicon.txt`.

## Verification

CPU-only CI runs on GitHub Actions with Metal disabled:

```bash
CMAKE_ARGS="-DCORTEX_ENABLE_METAL=OFF" uv pip install -e ".[dev]"
uv run pytest
```

Local Metal verification should be run on Apple Silicon macOS:

```bash
uv pip install -e ".[dev]"
uv run pytest
uv run python benchmarks/bench_elementwise.py
uv run python benchmarks/bench_copy.py
uv run python benchmarks/bench_matmul.py
```

To verify that custom Metal matmul works without the optimized primitive path:

```bash
CMAKE_ARGS="-DCORTEX_ENABLE_MPSGRAPH=OFF" uv pip install -e ".[dev]"
uv run pytest tests/python/test_matmul.py
```

Use a smaller benchmark smoke test while iterating:

```bash
uv run python benchmarks/bench_elementwise.py --sizes 1024 --repeats 2
uv run python benchmarks/bench_copy.py --sizes 1024 --repeats 2
uv run python benchmarks/bench_matmul.py --sizes 16x16x16 --repeats 2
```

## Current Status

Phase 5 provides CPU reference matmul, a correctness-first custom Metal matmul
kernel, and an optimized Metal primitive path. The custom kernel remains
available through `cx.matmul(a, b, backend="custom")`; `backend="optimized"`
uses the Apple optimized primitive path when it is enabled.

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

a = cx.randn((2, 3), device="metal", seed=1)
b = cx.randn((3, 4), device="metal", seed=2)
c = cx.matmul(a, b)

print(c.shape)
# (2, 4)
```

### Behavior notes

- Supported dtypes are `float32` and `int32`. Float inputs are narrowed to
  `float32`, so values may lose precision or overflow to `inf`.
- `int32` `add`/`multiply` overflow wraps (defined two's-complement), matching
  NumPy and identical on CPU and Metal.
- Execution is synchronous and holds the Python GIL; the runtime is not yet safe
  for concurrent multi-threaded use (see `docs/METAL_BACKEND.md`).

## Naming

```text
Working product name: Cortex Runtime
Python package/import: cortex_runtime
Documentation alias: import cortex_runtime as cx
Python extension module: cortex_runtime._core
C++ source root: cpp/cortex/
C++ namespace: cortex
```
