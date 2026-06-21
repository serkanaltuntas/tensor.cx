"""Python entrypoint for Cortex Runtime."""

from __future__ import annotations

from . import _core

__version__ = _core.version()
float32 = _core.float32
int32 = _core.int32
Tensor = _core.Tensor
tensor = _core.tensor
empty = _core.empty
zeros = _core.zeros
ones = _core.ones


def version() -> str:
    """Return the Cortex Runtime package version."""
    return _core.version()


__all__ = [
    "Tensor",
    "__version__",
    "empty",
    "float32",
    "int32",
    "ones",
    "tensor",
    "version",
    "zeros",
]
