"""Python entrypoint for Cortex Runtime."""

from __future__ import annotations

from . import _core

__version__ = _core.version()


def version() -> str:
    """Return the Cortex Runtime package version."""
    return _core.version()


__all__ = ["__version__", "version"]
