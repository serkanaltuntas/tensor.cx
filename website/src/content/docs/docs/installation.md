---
title: Installation
description: Build tensor.cx from source with uv, beginning with the CPU reference backend.
---

tensor.cx is installed from source. There is no published PyPI release or
general-purpose binary wheel to install. Repository access is required; the
repository is currently private pending the public launch.

## Prerequisites

- Python 3.11 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- Git and a C++20-capable compiler (for example GCC 13 on Linux or Apple Clang
  with Xcode on macOS).
- Python development headers for your interpreter when using a system Python.
- Internet access for build dependencies. The Python build backend manages CMake
  (3.20+) and nanobind; the Metal build also fetches Metal-cpp headers.

The commands below use a POSIX shell on Linux or macOS. Windows is not part of
the validated installation path.

## Build the CPU reference

```bash
git clone https://github.com/serkanaltuntas/cortex-runtime.git
cd cortex-runtime
uv venv --python 3.12
source .venv/bin/activate
CMAKE_ARGS="-DTENSORCX_ENABLE_METAL=OFF -DTENSORCX_ENABLE_CUDA=OFF" uv pip install -e ".[dev]"
uv run --no-sync python -c "import tensorcx as cx; print(cx.devices())"
```

The CPU build reports `['cpu']` and does not require GPU libraries. Continue to
[your first tensor](/docs/first-tensor/) or run the CPU correctness checks:

```bash
uv run --no-sync pytest tests/python/test_tensor_cpu.py
```

Use `--no-sync` when running an explicitly configured build: it prevents uv from
replacing that build as part of environment synchronization.

## Enable Metal

On an Apple Silicon Mac, install Xcode and its Metal command-line tools first.
Both discovery commands must succeed:

```bash
xcrun -sdk macosx --find metal
xcrun -sdk macosx --find metallib
CMAKE_ARGS="-DTENSORCX_ENABLE_METAL=ON -DTENSORCX_ENABLE_CUDA=OFF" uv pip install --reinstall -e ".[dev]"
uv run --no-sync python -c "import tensorcx as cx; print(cx.devices())"
```

The default `TENSORCX_ENABLE_METAL=AUTO` can fall back to CPU when the tools are
missing. Explicit `ON` instead fails clearly if Metal cannot be built.

## Enable CUDA

CUDA requires an NVIDIA device, installed driver libraries, the CUDA toolkit,
and a compatible host compiler. It is disabled by default.

The validated configuration is Linux x86_64, CUDA 12.4, GCC 13, and a GTX 980 Ti
(`sm_52`). This command targets **that configuration**:

```bash
CC=gcc-13 CXX=g++-13 CUDAHOSTCXX=g++-13 \
  CMAKE_ARGS="-DTENSORCX_ENABLE_METAL=OFF -DTENSORCX_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=52" \
  uv pip install --reinstall -e ".[dev]"
uv run --no-sync python -c "import tensorcx as cx; assert cx.is_available('cuda'); print(cx.devices())"
```

Other GPUs need their matching architecture and a compatible toolkit. Changing
the architecture flag does not establish support: see the
[CUDA validation boundary](/docs/backends/#cuda).
A CUDA build requires `libcuda.so.1` even at import. Use the CPU build on hosts
without those driver libraries.

LLVM/MLIR is optional and unnecessary for ordinary tensor operations. See
[experimental kernels](/docs/experimental/) for its separate requirements.
