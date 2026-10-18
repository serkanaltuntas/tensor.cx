"""Device helpers for tensor.cx."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import backend as _backend


@dataclass(frozen=True, slots=True)
class Device:
    """A concrete runtime device."""

    type: str
    index: int = 0
    name: str | None = field(default=None, compare=False)

    def __str__(self) -> str:
        return self.type if self.index == 0 else f"{self.type}:{self.index}"


def _normalize_device(value: str | Device | None) -> str:
    if value is None:
        return "cpu"

    if isinstance(value, Device):
        device_type, index = value.type, value.index
    elif isinstance(value, str):
        device_type, separator, index_str = value.partition(":")
        if separator:
            try:
                index = int(index_str)
            except ValueError:
                raise ValueError(f"invalid device index: {index_str!r}") from None
        else:
            index = 0
    else:
        raise TypeError("device must be a string, Device, or None")

    if not device_type:
        raise ValueError("device type must be a non-empty string")
    if not isinstance(index, int) or isinstance(index, bool):
        raise ValueError("device index must be an integer")
    # Indexed routes opt in through the registry; legacy devices retain index 0.
    if index != 0 and not _backend.allows_device_index(device_type):
        raise ValueError("only device index 0 is supported")
    if index < 0:
        raise ValueError("device index must be non-negative")
    return device_type if index == 0 else f"{device_type}:{index}"


def devices() -> list[str]:
    return _backend.devices()


def is_available(device: str | Device) -> bool:
    return _backend.is_available(_normalize_device(device))


def device_name(device: str | Device) -> str:
    return _backend.device_name(_normalize_device(device))


def device(value: str | Device) -> Device:
    canonical = _normalize_device(value)
    if not is_available(canonical):
        raise ValueError(f"device is not available: {canonical}")
    device_type, _, suffix = canonical.partition(":")
    return Device(type=device_type, index=int(suffix) if suffix else 0, name=device_name(canonical))


def best_device() -> str:
    return _backend.best_device()
