"""Explicit bounded allocation reuse through registered backend routes."""
from __future__ import annotations

import operator
import struct
import threading
import numpy as np
from . import backend
from .device import _normalize_device


def _limit(value, name, low, high):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    try:
        number = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]") from error
    if not low <= number <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return number


class MemoryPool:
    """Single-use, owning-thread scope; stats remain available after exit."""
    def __init__(self, device, size, blocks):
        self._device = device
        self._size = size
        self._blocks = blocks
        self._native = None
        self._used = False
        self._entered = False
        self._owner = None
        self._lock = threading.Lock()

    def __enter__(self):
        with self._lock:
            if self._used:
                raise RuntimeError("memory pool contexts are single-use")
            route = backend.require_backend(self._device)
            if route.memory_pool is None:
                raise ValueError(f"memory pools are not supported by backend: {self._device}")
            native = route.memory_pool(self._size, self._blocks)
            native.enter()
            self._native = native
            self._used = self._entered = True
            self._owner = threading.get_ident()
        return self

    def __exit__(self, *args):
        with self._lock:
            if not self._entered or self._owner != threading.get_ident():
                raise RuntimeError("memory pool must exit on its owning thread")
            try:
                self._native.exit()
            finally:
                self._entered = False
        return False

    def trim(self):
        """Free cached blocks on the active owning thread; keep live tensors."""
        with self._lock:
            if not self._entered or self._owner != threading.get_ident():
                raise RuntimeError("memory pool trim requires its active owning thread")
            self._native.trim()

    def stats(self):
        if self._native is None:
            raise RuntimeError("memory pool stats are available after entry")
        return dict(self._native.stats())

    def __del__(self):
        try:
            if self._entered and self._owner == threading.get_ident():
                self._native.exit()
        except Exception:
            pass


def memory_pool(device="cuda", *, max_cached_bytes=64 * 1024 * 1024, max_cached_blocks=64):
    """Cache CUDA or Vulkan allocations in an explicit thread-local scope.

    Limits cover idle cached storage, not live tensors. External DLPack storage
    is never cached. Vulkan limits cover actual backing-memory bytes and also
    permit one completed command/fence/descriptor/query frame per scope.
    Operations remain synchronous; CPU/Metal pools are absent.
    """
    size = _limit(max_cached_bytes, "max_cached_bytes", 0, (1 << (8 * struct.calcsize("P"))) - 1)
    blocks = _limit(max_cached_blocks, "max_cached_blocks", 1, 4096)
    return MemoryPool(_normalize_device(device), size, blocks)
