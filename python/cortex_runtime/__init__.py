"""Python entrypoint for Cortex Runtime."""

from __future__ import annotations

from . import _core
from .device import Device, best_device, device, device_name, devices, is_available
from .tensor import Tensor, empty, ones, tensor, zeros

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
    "float32",
    "int32",
    "is_available",
    "ones",
    "tensor",
    "version",
    "zeros",
]
