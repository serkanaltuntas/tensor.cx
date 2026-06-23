"""Device helpers for Cortex Runtime."""

from __future__ import annotations

from dataclasses import dataclass

from . import _core


@dataclass(frozen=True, slots=True)
class Device:
    """A concrete runtime device."""

    type: str
    index: int = 0
    name: str | None = None

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
    # The index guard applies to both string and Device inputs.
    if index != 0:
        raise ValueError("only device index 0 is supported")
    return device_type


def devices() -> list[str]:
    return list(_core.devices())


def is_available(device: str | Device) -> bool:
    return bool(_core.is_available(_normalize_device(device)))


def device_name(device: str | Device) -> str:
    return _core.device_name(_normalize_device(device))


def device(value: str | Device) -> Device:
    device_type = _normalize_device(value)
    if not is_available(device_type):
        raise ValueError(f"device is not available: {device_type}")
    return Device(type=device_type, index=0, name=device_name(device_type))


def best_device() -> str:
    return "metal" if is_available("metal") else "cpu"
