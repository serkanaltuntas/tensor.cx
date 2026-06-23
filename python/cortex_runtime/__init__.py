"""Python entrypoint for Cortex Runtime."""

from __future__ import annotations

from . import _core, testing
from .device import Device, best_device, device, device_name, devices, is_available
from .tensor import Tensor, empty, exp, matmul, matmul_backends, max, mean, ones, randn, sum, tensor, zeros

__version__ = _core.version()
float32 = _core.float32
int32 = _core.int32


def version() -> str:
    """Return the Cortex Runtime package version."""
    return _core.version()


__all__ = [
    "Tensor",
    "__version__",
    "Device",
    "best_device",
    "device",
    "device_name",
    "devices",
    "empty",
    "exp",
    "float32",
    "int32",
    "is_available",
    "matmul",
    "matmul_backends",
    "max",
    "mean",
    "ones",
    "randn",
    "sum",
    "tensor",
    "testing",
    "version",
    "zeros",
]
