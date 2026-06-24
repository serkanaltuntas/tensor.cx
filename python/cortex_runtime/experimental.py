"""Experimental APIs for Cortex Runtime.

Nothing in this module is stable yet. Phase 7 uses it as a controlled place to
prototype the kernel DSL without committing the top-level ``cortex_runtime`` API.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from types import FunctionType
from typing import Callable


_SUPPORTED_TARGETS = {"auto", "cpu", "metal"}


@dataclass(frozen=True, slots=True)
class Kernel:
    """Metadata wrapper for a Phase 7 experimental kernel function."""

    fn: Callable
    target: str = "auto"

    def __post_init__(self) -> None:
        if not isinstance(self.fn, FunctionType):
            raise TypeError("cx.experimental.kernel expects a Python function")
        _validate_kernel_function(self.fn)
        object.__setattr__(self, "target", _validate_target(self.target))

    @property
    def name(self) -> str:
        return self.fn.__name__

    @property
    def parameters(self) -> tuple[str, ...]:
        return tuple(inspect.signature(self.fn).parameters)

    def compile(self, *, target: str | None = None):
        selected = self.target if target is None else _validate_target(target)
        raise NotImplementedError(
            "experimental kernel DSL compilation is not implemented yet "
            f"for target {selected!r}"
        )

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "experimental kernel DSL launch is not implemented yet; "
            "use this scaffold only for Phase 7 metadata and validation"
        )


def _validate_target(target: str) -> str:
    if not isinstance(target, str):
        raise TypeError("experimental kernel target must be a string")
    if target not in _SUPPORTED_TARGETS:
        raise ValueError("experimental kernel target must be 'auto', 'cpu', or 'metal'")
    return target


def _validate_kernel_function(fn: Callable) -> None:
    signature = inspect.signature(fn)
    for parameter in signature.parameters.values():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            raise ValueError("experimental kernels require explicit positional parameters")
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            raise ValueError("experimental kernels do not support keyword-only parameters")


def kernel(fn: Callable | None = None, *, target: str = "auto"):
    """Decorate a Python function as a Phase 7 experimental kernel.

    The decorator records stable metadata now. Compilation and launch are still
    intentionally disabled until the AST -> IR -> MSL pipeline lands.
    """

    selected_target = _validate_target(target)

    def decorate(func: Callable) -> Kernel:
        if not isinstance(func, FunctionType):
            raise TypeError("cx.experimental.kernel expects a Python function")
        return Kernel(func, target=selected_target)

    if fn is None:
        return decorate
    return decorate(fn)


def program_id(axis: int) -> int:
    raise NotImplementedError("program_id is only valid inside compiled experimental kernels")


def thread_id() -> int:
    raise NotImplementedError("thread_id is only valid inside compiled experimental kernels")


def block_size() -> int:
    raise NotImplementedError("block_size is only valid inside compiled experimental kernels")


__all__ = [
    "Kernel",
    "block_size",
    "kernel",
    "program_id",
    "thread_id",
]
