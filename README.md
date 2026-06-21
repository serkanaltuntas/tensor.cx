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

Phase 0 bootstraps the package, CMake build, nanobind extension, pytest setup,
and architecture docs. Tensor allocation and backends start in later phases.

## Naming

```text
Working product name: Cortex Runtime
Python package/import: cortex_runtime
Documentation alias: import cortex_runtime as cx
Python extension module: cortex_runtime._core
C++ source root: cpp/cortex/
C++ namespace: cortex
```
