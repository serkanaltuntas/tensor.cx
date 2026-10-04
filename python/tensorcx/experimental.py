"""Experimental APIs for tensor.cx.

Nothing in this module is stable yet. Phase 7 uses it as a controlled place to
prototype the kernel DSL without committing the top-level ``tensorcx`` API.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import inspect
import operator
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
from types import FunctionType
import weakref
from typing import Callable, NoReturn, TypeAlias


_SUPPORTED_TARGETS = {"auto", "cpu", "metal", "cuda"}
_SUPPORTED_BINARY_OPS = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
}
_SUPPORTED_COMPARE_OPS = {
    ast.Lt: "lt",
    ast.LtE: "lte",
    ast.Gt: "gt",
    ast.GtE: "gte",
    ast.Eq: "eq",
    ast.NotEq: "neq",
}
_SUPPORTED_INTRINSICS = {
    "cx.experimental.program_id": 1,
    "cx.experimental.thread_id": 0,
    "cx.experimental.block_size": 0,
}
_MSL_BINARY_OPS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
}
_MSL_COMPARE_OPS = {
    "lt": "<",
    "lte": "<=",
    "gt": ">",
    "gte": ">=",
    "eq": "==",
    "neq": "!=",
}


# The IR and its MSL validation are pure functions of the (immutable) kernel
# function, cached per function object so repeated launch()/reference() calls
# do not re-read source, re-parse, or re-emit text.
_IR_CACHE: "weakref.WeakKeyDictionary[Callable, IRKernel]" = (
    weakref.WeakKeyDictionary()
)
_MSL_VALIDATED: "weakref.WeakKeyDictionary[Callable, bool]" = (
    weakref.WeakKeyDictionary()
)


def _validated_msl(fn: Callable, kernel_ir: IRKernel) -> None:
    if fn not in _MSL_VALIDATED:
        _emit_msl(kernel_ir)
        _MSL_VALIDATED[fn] = True


class KernelCompileError(ValueError):
    """Raised when an experimental kernel cannot be parsed into IR."""


@dataclass(frozen=True, slots=True)
class IRKernel:
    """Backend-neutral kernel IR parsed from a restricted Python function."""

    name: str
    parameters: tuple[str, ...]
    body: tuple[IRStatement, ...]


@dataclass(frozen=True, slots=True)
class CompiledKernel:
    """Compiled artifact for a Phase 7 experimental kernel."""

    name: str
    target: str
    ir: IRKernel
    msl_source: str
    metallib: bytes

    def validate_metal_function(self, function_name: str | None = None) -> str:
        """Load this metallib through Metal and verify a function exists."""
        if self.target != "metal":
            raise ValueError("compiled kernel validation requires target 'metal'")

        selected = self.name if function_name is None else function_name
        if not isinstance(selected, str):
            raise TypeError("Metal function name must be a string")
        if not selected:
            raise ValueError("Metal function name cannot be empty")
        if "\x00" in selected:
            raise ValueError("Metal function name cannot contain null bytes")

        from . import _core

        return _core.validate_metal_library_function(self.metallib, selected)

    def launch(
        self,
        *args,
        thread_count: int | None = None,
        block_size: int = 256,
    ):
        """Launch this compiled Phase 7 kernel on Metal.

        This is intentionally narrow: float32 Metal tensors, uint32 scalar
        arguments, exact shape matches, one output buffer, synchronous execution.
        """
        if self.target != "metal":
            raise ValueError("compiled kernel launch requires target 'metal'")

        (
            _,
            loop_limits,
            ordered,
            output_tensor,
            output_numel,
            _,
            normalized_thread_count,
            normalized_block_size,
        ) = _launch_contract(
            self.ir,
            args,
            thread_count,
            block_size,
            device="metal",
            prefix="experimental Metal kernel",
            requires="launch requires Metal tensors",
        )
        native_arguments = _bind_metal_arguments(
            ordered, loop_limits, output_tensor, output_numel
        )
        if normalized_thread_count == 0:
            self.validate_metal_function()
            return output_tensor

        from . import _core

        _core.launch_metal_library_function(
            self.metallib,
            self.name,
            native_arguments,
            output_tensor._impl,
            normalized_thread_count,
            normalized_block_size,
        )
        return output_tensor


@dataclass(frozen=True, slots=True)
class CompiledCpuKernel:
    """Optional MLIR CPU artifact; owns its loaded native module."""

    name: str
    target: str
    ir: IRKernel
    mlir_source: str
    _module: object

    def launch(self, *args, thread_count: int | None = None, block_size: int = 256):
        """Return a new CPU tensor, preserving the supplied output and its suffix."""
        _, _, ordered, _, _, _, threads, block = _launch_contract(
            self.ir, args, thread_count, block_size, device="cpu",
            prefix="experimental MLIR CPU kernel", requires="launch requires CPU tensors",
        )
        from . import _core
        from .tensor import Tensor

        values = [value._impl if kind == "buffer" else value
                  for kind, _, value in ordered]
        return Tensor(_core._launch_cpu_kernel(self._module, values, threads, block))


@dataclass(frozen=True, slots=True)
class CompiledCudaKernel:
    """Optional MLIR CUDA elementwise artifact; launch returns a new CUDA tensor."""

    name: str
    target: str
    ir: IRKernel
    mlir_source: str
    _module: object

    def launch(self, *args, thread_count: int | None = None, block_size: int = 256):
        _, _, ordered, _, _, _, threads, block = _launch_contract(
            self.ir, args, thread_count, block_size, device="cuda",
            prefix="experimental MLIR CUDA kernel", requires="launch requires CUDA tensors",
        )
        from . import _core
        from .tensor import Tensor
        values = [value._impl if kind == "buffer" else value for kind, _, value in ordered]
        return Tensor(_core._launch_cuda_kernel(self._module, values, threads, block))


@dataclass(frozen=True, slots=True)
class IRName:
    name: str


@dataclass(frozen=True, slots=True)
class IRConstant:
    value: int | float


@dataclass(frozen=True, slots=True)
class IRCall:
    name: str
    args: tuple[IRExpression, ...]


@dataclass(frozen=True, slots=True)
class IRBinaryOp:
    op: str
    lhs: IRExpression
    rhs: IRExpression


@dataclass(frozen=True, slots=True)
class IRCompare:
    op: str
    lhs: IRExpression
    rhs: IRExpression


@dataclass(frozen=True, slots=True)
class IRLoad:
    buffer: str
    index: IRExpression


@dataclass(frozen=True, slots=True)
class IRAssign:
    target: str
    value: IRExpression


@dataclass(frozen=True, slots=True)
class IRStore:
    buffer: str
    index: IRExpression
    value: IRExpression


@dataclass(frozen=True, slots=True)
class IRIf:
    condition: IRExpression
    body: tuple[IRStatement, ...]


@dataclass(frozen=True, slots=True)
class IRFor:
    """Bounded sequential loop: ``for var in range(limit)``.

    ``limit`` is the name of a scalar parameter. Bodies are restricted to
    local assignments (the accumulator pattern); loads inside the body must
    follow the row-major pattern validated at launch.
    """

    var: str
    limit: str
    body: tuple[IRStatement, ...]


IRExpression: TypeAlias = (
    IRName | IRConstant | IRCall | IRBinaryOp | IRCompare | IRLoad
)
IRStatement: TypeAlias = IRAssign | IRStore | IRIf | IRFor


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

    def compile(
        self, *, target: str | None = None, compiler: str = "auto",
    ) -> CompiledKernel | CompiledCpuKernel | CompiledCudaKernel:
        selected = self.target if target is None else _validate_target(target)
        if not isinstance(compiler, str):
            raise TypeError("experimental kernel compiler must be a string")
        if compiler not in {"auto", "mlir"}:
            raise ValueError("experimental kernel compiler must be 'auto' or 'mlir'")
        if compiler == "mlir":
            if selected == "cpu":
                from ._compiler.cpu import compile_kernel
            elif selected == "cuda":
                from ._compiler.cuda import compile_kernel
            else:
                raise ValueError("MLIR compilation requires target 'cpu' or 'cuda'")
            return compile_kernel(self.parse_ir())
        resolved = "metal" if selected == "auto" else selected
        if resolved != "metal":
            raise NotImplementedError(
                "experimental kernel DSL compilation is only implemented "
                "for target 'metal'"
            )

        ir = self.parse_ir()
        msl_source = _emit_msl(ir)
        return CompiledKernel(
            name=ir.name,
            target=resolved,
            ir=ir,
            msl_source=msl_source,
            metallib=_compile_msl_to_metallib(ir.name, msl_source),
        )

    def parse_ir(self) -> IRKernel:
        """Parse (and cache) the kernel function into backend-neutral IR."""
        cached = _IR_CACHE.get(self.fn)
        if cached is None:
            cached = _parse_kernel_function(self.fn)
            _IR_CACHE[self.fn] = cached
        return cached

    def emit_msl(self) -> str:
        """Emit text MSL for the Phase 7 experimental kernel subset."""
        return _emit_msl(self.parse_ir())

    def __call__(
        self,
        *args,
        thread_count: int | None = None,
        block_size: int = 256,
        target: str | None = None,
    ):
        return self.compile(target=target).launch(
            *args,
            thread_count=thread_count,
            block_size=block_size,
        )

    def reference(
        self,
        *args,
        thread_count: int | None = None,
        block_size: int = 256,
    ):
        """Execute this kernel's IR on CPU tensors as the reference semantics.

        Interprets the backend-neutral IR with MSL-matching semantics (uint32
        wraparound arithmetic, C literal/declaration typing for signed vs
        unsigned comparisons, float32 arithmetic) under the same launch
        contract as the Metal path (guard bound == thread_count, exact size
        checks, zero-thread no-op). Unlike the Metal launch, CPU tensors are
        immutable values, so the result is returned as a NEW cpu Tensor; the
        ``out`` argument supplies the shape, dtype, and initial contents of
        unwritten elements and is not mutated.
        """
        kernel_ir = self.parse_ir()
        # Parity with the Metal path: kernels rejected by MSL emission (unused
        # parameters, type-changing reassignment) must not execute here either.
        _validated_msl(self.fn, kernel_ir)
        (
            output_param,
            _,
            ordered,
            output_tensor,
            _,
            scalars,
            normalized_thread_count,
            normalized_block_size,
        ) = _launch_contract(
            kernel_ir,
            args,
            thread_count,
            block_size,
            device="cpu",
            prefix="experimental kernel reference",
            requires="execution requires CPU tensors",
        )
        arrays = _bind_reference_arrays(ordered)
        out_array = arrays[output_param]
        if normalized_thread_count > 0:
            _reference_execute(
                kernel_ir,
                arrays,
                scalars,
                normalized_thread_count,
                normalized_block_size,
            )

        from .tensor import tensor as _make_tensor

        return _make_tensor(
            out_array.reshape(output_tensor.shape), dtype="float32", device="cpu"
        )


def _validate_target(target: str) -> str:
    if not isinstance(target, str):
        raise TypeError("experimental kernel target must be a string")
    if target not in _SUPPORTED_TARGETS:
        raise ValueError("experimental kernel target must be 'auto', 'cpu', 'metal', or 'cuda'")
    return target


def _normalize_uint32(value, message: str, *, allow_zero: bool) -> int:
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        normalized = operator.index(value)
    except TypeError:
        raise ValueError(message) from None
    if normalized < 0 or normalized > 2**32 - 1:
        raise ValueError(message)
    if not allow_zero and normalized == 0:
        raise ValueError(message)
    return int(normalized)


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _launch_contract(
    kernel_ir: IRKernel,
    args: tuple[object, ...],
    thread_count,
    block_size,
    *,
    device: str,
    prefix: str,
    requires: str,
):
    """Normalize the launch contract shared by compiled kernels and CPU reference.

    Runs the guard analysis exactly once and returns everything both callers
    need; keeping this single stops the launch contracts from drifting.
    """
    normalized_block_size = _normalize_uint32(
        block_size, "block_size must be a positive uint32", allow_zero=False
    )
    if thread_count is None:
        normalized_thread_count = None
    else:
        normalized_thread_count = _normalize_uint32(
            thread_count, "thread_count must be a uint32", allow_zero=True
        )
    output_param = _single_output_parameter(kernel_ir)
    guard_param, loop_limits = _launch_guard_analysis(kernel_ir, output_param)
    ordered, output_tensor, output_numel, scalar_arguments = (
        _classify_launch_arguments(
            kernel_ir,
            output_param,
            args,
            loop_limits=loop_limits,
            device=device,
            prefix=prefix,
            requires=requires,
        )
    )
    if normalized_thread_count is None:
        normalized_thread_count = output_numel
    if normalized_thread_count > output_numel:
        raise ValueError("thread_count cannot exceed output tensor size")
    if scalar_arguments[guard_param] != normalized_thread_count:
        raise ValueError("kernel guard bound must match thread_count")
    return (
        output_param,
        loop_limits,
        ordered,
        output_tensor,
        output_numel,
        scalar_arguments,
        normalized_thread_count,
        normalized_block_size,
    )


def _single_output_parameter(kernel_ir: IRKernel) -> str:
    output_names = tuple(dict.fromkeys(_iter_store_buffers(kernel_ir.body)))
    if len(output_names) != 1:
        raise KernelCompileError("kernel launch requires exactly one output buffer")
    return output_names[0]


def _launch_guard_analysis(
    kernel_ir: IRKernel,
    output_param: str,
) -> tuple[str, dict[str, str]]:
    buffer_names = set(_iter_buffer_references(kernel_ir.body))
    parameter_names = set(kernel_ir.parameters)
    guard_params: set[str] = set()
    loop_limits: dict[str, str] = {}
    # Names whose values the structural bounds proofs depend on: every store/
    # load index name and every name referenced in an if condition. Reassigning
    # any of them inside a loop body would invalidate a proof made against the
    # pre-loop value (the guard may be an OUTER if, so checking only the
    # enclosing guard's index is not enough).
    index_names: set[str] = set()

    def collect_index_names(statements) -> None:
        for statement in statements:
            if isinstance(statement, IRStore):
                index_names.update(_iter_expression_names_of(statement.index))
            elif isinstance(statement, IRAssign):
                for load in _iter_expression_loads(statement.value):
                    index_names.update(_iter_expression_names_of(load.index))
            elif isinstance(statement, IRIf):
                index_names.update(
                    _iter_expression_names_of(statement.condition)
                )
                collect_index_names(statement.body)
            elif isinstance(statement, IRFor):
                collect_index_names(statement.body)

    collect_index_names(kernel_ir.body)

    def visit_loop(loop: IRFor, guard: IRExpression | None) -> None:
        # Loads inside a for-loop body are only provably in bounds when the
        # loop sits inside the store guard `row < n` and every load uses the
        # row-major pattern `buffer[row * limit + var]`; the launch layer then
        # enforces numel(buffer) == n * limit exactly.
        if (
            not isinstance(guard, IRCompare)
            or guard.op != "lt"
            or not isinstance(guard.lhs, IRName)
            or not isinstance(guard.rhs, IRName)
        ):
            raise KernelCompileError(
                "kernel launch requires for-loops to be inside the "
                "index < scalar_limit store guard"
            )
        if loop.limit in buffer_names or loop.limit not in parameter_names:
            raise KernelCompileError(
                "kernel launch requires the for-loop range bound to be a "
                "scalar parameter"
            )
        row_name = guard.lhs.name
        guard_params.add(guard.rhs.name)
        for inner in loop.body:
            if not isinstance(inner, IRAssign):
                raise KernelCompileError(
                    "kernel launch requires for-loop bodies to contain only "
                    "local assignments"
                )
            if inner.target == row_name or inner.target in index_names:
                # Reassigning a guarded/index name inside the loop would
                # invalidate the structural bounds proofs made against its
                # pre-loop value (including proofs under OUTER guards).
                raise KernelCompileError(
                    "kernel launch does not support reassigning the guarded "
                    "index inside a for-loop"
                )
        expected_index = IRBinaryOp(
            op="add",
            lhs=IRBinaryOp(op="mul", lhs=IRName(row_name), rhs=IRName(loop.limit)),
            rhs=IRName(loop.var),
        )
        for inner in loop.body:
            for load in _iter_expression_loads(inner.value):
                if load.buffer == output_param:
                    raise KernelCompileError(
                        "kernel launch does not support loading the output "
                        "buffer inside a for-loop"
                    )
                if load.index != expected_index:
                    raise KernelCompileError(
                        "kernel launch requires for-loop loads to use the "
                        "row-major pattern buffer[row * limit + loop_var]"
                    )
                existing = loop_limits.get(load.buffer)
                if existing is not None and existing != loop.limit:
                    raise KernelCompileError(
                        "kernel launch requires a loop-indexed buffer to use "
                        "one loop limit"
                    )
                loop_limits[load.buffer] = loop.limit

    def visit(statements: tuple[IRStatement, ...], guard: IRExpression | None) -> None:
        for statement in statements:
            if isinstance(statement, IRFor):
                visit_loop(statement, guard)
            elif isinstance(statement, IRAssign):
                guard_params.update(
                    _expression_guard_parameters(
                        statement.value,
                        guard,
                        buffer_names=buffer_names,
                        parameter_names=parameter_names,
                    )
                )
            elif isinstance(statement, IRStore):
                guard_param = _store_guard_parameter(
                    statement,
                    guard,
                    output_param=output_param,
                    buffer_names=buffer_names,
                    parameter_names=parameter_names,
                )
                guard_params.add(guard_param)
                guard_params.update(
                    _expression_guard_parameters(
                        statement.index,
                        guard,
                        buffer_names=buffer_names,
                        parameter_names=parameter_names,
                    )
                )
                guard_params.update(
                    _expression_guard_parameters(
                        statement.value,
                        guard,
                        buffer_names=buffer_names,
                        parameter_names=parameter_names,
                    )
                )
            elif isinstance(statement, IRIf):
                guard_params.update(
                    _expression_guard_parameters(
                        statement.condition,
                        guard,
                        buffer_names=buffer_names,
                        parameter_names=parameter_names,
                    )
                )
                visit(statement.body, statement.condition)
            if isinstance(statement, (IRAssign, IRStore, IRIf)):
                if isinstance(statement, IRAssign):
                    expressions = (statement.value,)
                elif isinstance(statement, IRStore):
                    expressions = (statement.index, statement.value)
                else:
                    expressions = (statement.condition,)
                for expression in expressions:
                    for load in _iter_expression_loads(expression):
                        elementwise_loaded.add(load.buffer)

    elementwise_loaded: set[str] = set()
    visit(kernel_ir.body, guard=None)
    if len(guard_params) != 1:
        raise KernelCompileError(
            "kernel launch requires all stores to share one scalar guard"
        )
    conflicting = sorted(elementwise_loaded & set(loop_limits))
    if conflicting:
        raise KernelCompileError(
            "kernel launch does not support loading a buffer both elementwise "
            f"and loop-indexed: {', '.join(conflicting)}"
        )
    return next(iter(guard_params)), loop_limits


def _iter_expression_names_of(expression: IRExpression):
    if isinstance(expression, IRName):
        yield expression.name
    elif isinstance(expression, IRLoad):
        yield from _iter_expression_names_of(expression.index)
    elif isinstance(expression, (IRBinaryOp, IRCompare)):
        yield from _iter_expression_names_of(expression.lhs)
        yield from _iter_expression_names_of(expression.rhs)
    elif isinstance(expression, IRCall):
        for argument in expression.args:
            yield from _iter_expression_names_of(argument)


def _iter_expression_loads(expression: IRExpression):
    if isinstance(expression, IRLoad):
        yield expression
        yield from _iter_expression_loads(expression.index)
    elif isinstance(expression, (IRBinaryOp, IRCompare)):
        yield from _iter_expression_loads(expression.lhs)
        yield from _iter_expression_loads(expression.rhs)
    elif isinstance(expression, IRCall):
        for argument in expression.args:
            yield from _iter_expression_loads(argument)


def _expression_guard_parameters(
    expression: IRExpression,
    guard: IRExpression | None,
    *,
    buffer_names: set[str],
    parameter_names: set[str],
) -> set[str]:
    if isinstance(expression, IRLoad):
        guard_param = _load_guard_parameter(
            expression,
            guard,
            buffer_names=buffer_names,
            parameter_names=parameter_names,
        )
        nested = _expression_guard_parameters(
            expression.index,
            guard,
            buffer_names=buffer_names,
            parameter_names=parameter_names,
        )
        return {guard_param, *nested}
    if isinstance(expression, IRCall):
        result: set[str] = set()
        for argument in expression.args:
            result.update(
                _expression_guard_parameters(
                    argument,
                    guard,
                    buffer_names=buffer_names,
                    parameter_names=parameter_names,
                )
            )
        return result
    if isinstance(expression, IRBinaryOp):
        return {
            *_expression_guard_parameters(
                expression.lhs,
                guard,
                buffer_names=buffer_names,
                parameter_names=parameter_names,
            ),
            *_expression_guard_parameters(
                expression.rhs,
                guard,
                buffer_names=buffer_names,
                parameter_names=parameter_names,
            ),
        }
    if isinstance(expression, IRCompare):
        return {
            *_expression_guard_parameters(
                expression.lhs,
                guard,
                buffer_names=buffer_names,
                parameter_names=parameter_names,
            ),
            *_expression_guard_parameters(
                expression.rhs,
                guard,
                buffer_names=buffer_names,
                parameter_names=parameter_names,
            ),
        }
    return set()


def _load_guard_parameter(
    load: IRLoad,
    guard: IRExpression | None,
    *,
    buffer_names: set[str],
    parameter_names: set[str],
) -> str:
    if not isinstance(load.index, IRName):
        raise KernelCompileError(
            "kernel launch requires buffer loads indexed by a guarded local name"
        )
    if not isinstance(guard, IRCompare) or guard.op != "lt":
        raise KernelCompileError(
            "kernel launch requires buffer loads to be guarded by index < scalar_limit"
        )
    if guard.lhs != load.index or not isinstance(guard.rhs, IRName):
        raise KernelCompileError(
            "kernel launch requires buffer loads to be guarded by index < scalar_limit"
        )
    if guard.rhs.name in buffer_names or guard.rhs.name not in parameter_names:
        raise KernelCompileError(
            "kernel launch guard limit must be a scalar parameter"
        )
    return guard.rhs.name


def _store_guard_parameter(
    store: IRStore,
    guard: IRExpression | None,
    *,
    output_param: str,
    buffer_names: set[str],
    parameter_names: set[str],
) -> str:
    if store.buffer != output_param:
        raise KernelCompileError("kernel launch requires exactly one output buffer")
    if not isinstance(store.index, IRName):
        raise KernelCompileError(
            "kernel launch requires stores indexed by a guarded local name"
        )
    if not isinstance(guard, IRCompare) or guard.op != "lt":
        raise KernelCompileError(
            "kernel launch requires stores to be guarded by index < scalar_limit"
        )
    if guard.lhs != store.index or not isinstance(guard.rhs, IRName):
        raise KernelCompileError(
            "kernel launch requires stores to be guarded by index < scalar_limit"
        )
    if guard.rhs.name in buffer_names or guard.rhs.name not in parameter_names:
        raise KernelCompileError(
            "kernel launch guard limit must be a scalar parameter"
        )
    return guard.rhs.name


def _classify_launch_arguments(
    kernel_ir: IRKernel,
    output_param: str,
    args: tuple[object, ...],
    *,
    loop_limits: dict[str, str],
    device: str,
    prefix: str,
    requires: str,
):
    """Shared launch-contract validation for the Metal and reference paths.

    Classifies arguments into buffers and uint32 scalars, enforces float32
    dtype and the target device, checks elementwise shapes against the output
    shape and loop-indexed buffer sizes against output_numel * loop_limit, and
    returns the ordered entries plus the output tensor and scalar map. Keeping
    this single ensures the reference contract can never drift from Metal's.
    """
    if len(args) != len(kernel_ir.parameters):
        raise TypeError(
            f"{kernel_ir.name} expects {len(kernel_ir.parameters)} argument(s), "
            f"got {len(args)}"
        )

    from .tensor import Tensor

    buffer_names = set(_iter_buffer_references(kernel_ir.body))
    ordered: list[tuple[str, str, object]] = []
    scalar_arguments: dict[str, int] = {}
    elementwise_shapes: list[tuple[int, ...]] = []
    loop_tensors: list[tuple[str, Tensor]] = []
    output_tensor = None

    for parameter, argument in zip(kernel_ir.parameters, args, strict=True):
        if parameter in buffer_names:
            if not isinstance(argument, Tensor):
                raise TypeError(f"{prefix} buffer arguments must be Tensor objects")
            if argument.device != device:
                raise ValueError(f"{prefix} {requires}")
            if argument.dtype != "float32":
                raise ValueError(
                    "experimental kernels currently support float32 tensor buffers"
                )
            if parameter in loop_limits:
                loop_tensors.append((parameter, argument))
            else:
                elementwise_shapes.append(argument.shape)
            ordered.append(("buffer", parameter, argument))
            if parameter == output_param:
                output_tensor = argument
        else:
            scalar_value = _normalize_uint32(
                argument, "kernel scalar arguments must be uint32", allow_zero=True
            )
            scalar_arguments[parameter] = scalar_value
            ordered.append(("scalar", parameter, scalar_value))

    if output_tensor is None:
        raise KernelCompileError("kernel launch requires exactly one output buffer")
    if any(shape != output_tensor.shape for shape in elementwise_shapes):
        raise ValueError(f"{prefix} tensor shapes must match")

    output_numel = _numel(output_tensor.shape)
    for parameter, tensor in loop_tensors:
        limit_value = scalar_arguments[loop_limits[parameter]]
        expected = output_numel * limit_value
        if _numel(tensor.shape) != expected:
            raise ValueError(
                f"{prefix} loop-indexed buffer size must equal "
                "output size times the loop limit "
                f"({parameter}: {_numel(tensor.shape)} != {expected})"
            )

    return ordered, output_tensor, output_numel, scalar_arguments


def _bind_metal_arguments(ordered, loop_limits, output_tensor, output_numel):
    """Bind classified arguments to native Metal launch arguments."""
    native_arguments = []
    placeholder_indices: list[int] = []
    for kind, parameter, value in ordered:
        if kind == "scalar":
            native_arguments.append(value)
            continue
        if parameter in loop_limits and _numel(value.shape) == 0:
            # Zero-element Metal tensors have no native buffer to bind. This is
            # only reachable when the loop limit is 0 (the size equation in
            # _classify_launch_arguments enforces numel == out_numel * limit),
            # so the loop body never executes and the buffer is provably never
            # read; bind the output buffer as a placeholder.
            placeholder_indices.append(len(native_arguments))
        native_arguments.append(value._impl)
    if output_numel > 0:
        for index in placeholder_indices:
            native_arguments[index] = output_tensor._impl
    return native_arguments


def _bind_reference_arrays(ordered):
    """Bind classified buffer arguments to numpy arrays for interpretation.

    The same Tensor bound to two buffer parameters must share one array, as
    the Metal path binds one native buffer twice. Tensor.numpy() copies, so
    the interpreter's stores never mutate caller-visible tensors.
    """
    shared: dict[int, object] = {}
    return {
        parameter: shared.setdefault(id(value._impl), value.numpy().reshape(-1))
        for kind, parameter, value in ordered
        if kind == "buffer"
    }


_UINT32_MASK = 2**32 - 1


def _reference_execute(kernel_ir, arrays, scalars, thread_count, block_size):
    """Interpret kernel IR per thread with MSL/C-matching scalar semantics.

    Integers are 32-bit bit patterns with a C-style type tag: literals that
    fit int32 are "int", locals get their MSL declared type (uint for
    intrinsics and non-negative constants), arithmetic follows C conversions
    (uint wins), and ordered comparisons are signed only when BOTH operands
    are int-typed — matching what the emitted MSL compiles to. Floats use
    numpy float32 so per-element results match the Metal path.
    """
    import numpy as np

    def as_signed(bits):
        return bits - 2**32 if bits >= 2**31 else bits

    def to_float(tag, value):
        if tag == "float":
            return value
        return np.float32(as_signed(value) if tag == "int" else value)

    def eval_expression(expression, env, intrinsics):
        if isinstance(expression, IRName):
            return env[expression.name]
        if isinstance(expression, IRConstant):
            if isinstance(expression.value, float):
                return ("float", np.float32(expression.value))
            # The parser rejects literals outside int32 range, so a bare
            # integer literal is always C-typed int.
            return ("int", expression.value & _UINT32_MASK)
        if isinstance(expression, IRCall):
            return ("uint", intrinsics[expression.name])
        if isinstance(expression, IRLoad):
            index_tag, index_bits = eval_expression(
                expression.index, env, intrinsics
            )
            if index_tag == "float":
                raise KernelCompileError("buffer indices must be integers")
            # MSL converts a bool index to 0/1; a raw Python bool would act as
            # a numpy boolean MASK and read/write every element.
            index_bits = int(index_bits)
            buffer = arrays[expression.buffer]
            if index_bits >= buffer.size:
                raise KernelCompileError(
                    "reference execution index out of bounds; the launch "
                    "contract should have prevented this"
                )
            return ("float", np.float32(buffer[index_bits]))
        if isinstance(expression, IRBinaryOp):
            lhs_tag, lhs = eval_expression(expression.lhs, env, intrinsics)
            rhs_tag, rhs = eval_expression(expression.rhs, env, intrinsics)
            if lhs_tag == "bool":
                lhs_tag, lhs = "int", int(lhs)
            if rhs_tag == "bool":
                rhs_tag, rhs = "int", int(rhs)
            if "float" in (lhs_tag, rhs_tag):
                lhs_f, rhs_f = to_float(lhs_tag, lhs), to_float(rhs_tag, rhs)
                if expression.op == "add":
                    return ("float", np.float32(lhs_f + rhs_f))
                if expression.op == "sub":
                    return ("float", np.float32(lhs_f - rhs_f))
                return ("float", np.float32(lhs_f * rhs_f))
            tag = "uint" if "uint" in (lhs_tag, rhs_tag) else "int"
            if expression.op == "add":
                bits = lhs + rhs
            elif expression.op == "sub":
                bits = lhs - rhs
            else:
                bits = lhs * rhs
            return (tag, bits & _UINT32_MASK)
        if isinstance(expression, IRCompare):
            lhs_tag, lhs = eval_expression(expression.lhs, env, intrinsics)
            rhs_tag, rhs = eval_expression(expression.rhs, env, intrinsics)
            # C integral promotion: bool promotes to SIGNED int, so a compare
            # is unsigned only when a genuine uint operand is present.
            lhs = int(lhs) if lhs_tag == "bool" else lhs
            rhs = int(rhs) if rhs_tag == "bool" else rhs
            if "float" in (lhs_tag, rhs_tag):
                lhs_c, rhs_c = to_float(lhs_tag, lhs), to_float(rhs_tag, rhs)
            elif "uint" not in (lhs_tag, rhs_tag):
                lhs_c, rhs_c = as_signed(lhs), as_signed(rhs)
            else:
                lhs_c, rhs_c = lhs, rhs
            comparisons = {
                "lt": lhs_c < rhs_c,
                "lte": lhs_c <= rhs_c,
                "gt": lhs_c > rhs_c,
                "gte": lhs_c >= rhs_c,
                "eq": lhs_c == rhs_c,
                "neq": lhs_c != rhs_c,
            }
            return ("bool", bool(comparisons[expression.op]))
        raise KernelCompileError(
            f"unhandled IR expression: {type(expression).__name__}"
        )

    # Locals carry MSL declared types; msl_context mirrors env in the
    # "uint"/"int"/"float" vocabulary _infer_msl_type uses and is maintained
    # incrementally (rebuilding it per assignment is quadratic in loop trips).
    def declared_tag(expression, msl_context):
        return _infer_msl_type(expression, msl_context)

    def run_block(statements, env, msl_context, intrinsics):
        for statement in statements:
            if isinstance(statement, IRAssign):
                tag, value = eval_expression(statement.value, env, intrinsics)
                if tag != "float" and tag != "bool":
                    tag = declared_tag(statement.value, msl_context)
                env[statement.target] = (tag, value)
                msl_context[statement.target] = tag if tag != "bool" else "uint"
            elif isinstance(statement, IRStore):
                index_tag, index_bits = eval_expression(
                    statement.index, env, intrinsics
                )
                if index_tag == "float":
                    raise KernelCompileError("buffer indices must be integers")
                index_bits = int(index_bits)
                tag, value = eval_expression(statement.value, env, intrinsics)
                buffer = arrays[statement.buffer]
                if index_bits >= buffer.size:
                    raise KernelCompileError(
                        "reference execution index out of bounds; the launch "
                        "contract should have prevented this"
                    )
                buffer[index_bits] = to_float(tag, value)
            elif isinstance(statement, IRIf):
                tag, value = eval_expression(statement.condition, env, intrinsics)
                truthy = value if tag == "bool" else value != 0
                if truthy:
                    run_block(statement.body, env, msl_context, intrinsics)
            elif isinstance(statement, IRFor):
                _, limit_bits = env[statement.limit]
                msl_context[statement.var] = "uint"
                for iteration in range(limit_bits):
                    env[statement.var] = ("uint", iteration)
                    run_block(statement.body, env, msl_context, intrinsics)
            else:
                raise KernelCompileError(
                    f"unhandled IR statement: {type(statement).__name__}"
                )

    base_env = {name: ("uint", value) for name, value in scalars.items()}
    base_msl_context = {name: "uint" for name in scalars}
    for global_index in range(thread_count):
        env = dict(base_env)
        intrinsics = {
            "program_id": global_index // block_size,
            "thread_id": global_index % block_size,
            "block_size": block_size,
        }
        run_block(kernel_ir.body, env, dict(base_msl_context), intrinsics)


def _validate_kernel_function(fn: Callable) -> None:
    if inspect.iscoroutinefunction(fn):
        raise ValueError("experimental kernels do not support async functions")

    signature = inspect.signature(fn)
    for parameter in signature.parameters.values():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            raise ValueError("experimental kernels require explicit positional parameters")
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            raise ValueError("experimental kernels do not support keyword-only parameters")


def _parse_kernel_function(fn: Callable) -> IRKernel:
    try:
        source = inspect.getsource(fn)
    except OSError as exc:
        raise KernelCompileError("kernel source is unavailable") from exc

    module = ast.parse(textwrap.dedent(source))
    function = next(
        (node for node in module.body if isinstance(node, ast.FunctionDef)), None
    )
    if function is None:
        raise KernelCompileError("kernel source does not contain a function definition")

    parameters = tuple(inspect.signature(fn).parameters)
    body = _parse_statement_block(function.body)
    _validate_parameter_shadowing(body, parameters)
    _validate_name_scopes(body, parameters)
    _validate_buffer_references(body, parameters)
    _validate_buffer_scalar_usage(body)
    _validate_single_output(body)
    return IRKernel(
        name=function.name,
        parameters=parameters,
        body=body,
    )


def _parse_statement_block(statements: list[ast.stmt]) -> tuple[IRStatement, ...]:
    parsed: list[IRStatement] = []
    for index, statement in enumerate(statements):
        if _is_docstring(statement, index) or isinstance(statement, ast.Pass):
            continue
        parsed.append(_parse_statement(statement))
    return tuple(parsed)


def _parse_statement(statement: ast.stmt) -> IRStatement:
    if isinstance(statement, ast.Assign):
        return _parse_assign(statement)
    if isinstance(statement, ast.If):
        return _parse_if(statement)
    if isinstance(statement, ast.For):
        return _parse_for(statement)
    _unsupported(statement)


def _parse_for(statement: ast.For) -> IRFor:
    if statement.orelse:
        raise KernelCompileError("unsupported kernel syntax: For with else")
    if not isinstance(statement.target, ast.Name):
        raise KernelCompileError(
            "unsupported kernel syntax: for-loop target must be a simple name"
        )
    call = statement.iter
    if (
        not isinstance(call, ast.Call)
        or not isinstance(call.func, ast.Name)
        or call.func.id != "range"
        or call.keywords
        or len(call.args) != 1
    ):
        raise KernelCompileError(
            "unsupported kernel syntax: for-loops must iterate over "
            "range(scalar_parameter)"
        )
    limit = call.args[0]
    if not isinstance(limit, ast.Name):
        raise KernelCompileError(
            "unsupported kernel syntax: range bound must be a scalar parameter"
        )
    body = _parse_statement_block(statement.body)
    for inner in body:
        if not isinstance(inner, IRAssign):
            raise KernelCompileError(
                "unsupported kernel syntax: for-loop bodies currently support "
                "only local assignments"
            )
    if not body:
        raise KernelCompileError(
            "unsupported kernel syntax: for-loop bodies cannot be empty"
        )
    return IRFor(var=statement.target.id, limit=limit.id, body=body)


def _parse_assign(statement: ast.Assign) -> IRStatement:
    if len(statement.targets) != 1:
        _unsupported(statement)

    target = statement.targets[0]
    value = _parse_expression(statement.value)
    if isinstance(target, ast.Name):
        return IRAssign(target=target.id, value=value)
    if isinstance(target, ast.Subscript):
        buffer_name, index = _parse_subscript_target(target)
        return IRStore(buffer=buffer_name, index=index, value=value)
    _unsupported(target)


def _parse_if(statement: ast.If) -> IRIf:
    if statement.orelse:
        raise KernelCompileError("unsupported kernel syntax: If with else")
    return IRIf(
        condition=_parse_expression(statement.test),
        body=_parse_statement_block(statement.body),
    )


def _parse_expression(expression: ast.expr) -> IRExpression:
    if isinstance(expression, ast.Name):
        return IRName(expression.id)
    if isinstance(expression, ast.Constant):
        if isinstance(expression.value, bool) or not isinstance(
            expression.value, (int, float)
        ):
            _unsupported(expression)
        if isinstance(expression.value, int) and not (
            -(2**31) < expression.value < 2**31
        ):
            # MSL types larger decimal literals as 64-bit, which the 32-bit
            # scalar model (MSL uint locals, reference interpreter, MLIR i32)
            # cannot represent faithfully. (INT32_MIN itself is excluded
            # because unary minus folds after this check.)
            raise KernelCompileError(
                "unsupported kernel syntax: integer constants must fit in "
                "32-bit signed range"
            )
        return IRConstant(expression.value)
    if isinstance(expression, ast.UnaryOp):
        return _parse_unary_op(expression)
    if isinstance(expression, ast.Call):
        return _parse_call(expression)
    if isinstance(expression, ast.BinOp):
        return _parse_binary_op(expression)
    if isinstance(expression, ast.Compare):
        return _parse_compare(expression)
    if isinstance(expression, ast.Subscript):
        buffer_name, index = _parse_subscript_target(expression)
        return IRLoad(buffer=buffer_name, index=index)
    _unsupported(expression)


def _parse_call(call: ast.Call) -> IRCall:
    if call.keywords:
        _unsupported(call)

    call_name = _dotted_name(call.func)
    expected_arg_count = _SUPPORTED_INTRINSICS.get(call_name)
    if expected_arg_count is None:
        _unsupported(call)
    if len(call.args) != expected_arg_count:
        raise KernelCompileError(
            f"unsupported kernel syntax: Call {call_name} expects "
            f"{expected_arg_count} argument(s)"
        )
    if call_name == "cx.experimental.program_id":
        _validate_program_id_axis(call.args[0])

    short_name = call_name.rsplit(".", maxsplit=1)[-1]
    return IRCall(
        name=short_name,
        args=tuple(_parse_expression(argument) for argument in call.args),
    )


def _parse_binary_op(expression: ast.BinOp) -> IRBinaryOp:
    op = _SUPPORTED_BINARY_OPS.get(type(expression.op))
    if op is None:
        _unsupported(expression.op)
    return IRBinaryOp(
        op=op,
        lhs=_parse_expression(expression.left),
        rhs=_parse_expression(expression.right),
    )


def _parse_compare(expression: ast.Compare) -> IRCompare:
    if len(expression.ops) != 1 or len(expression.comparators) != 1:
        _unsupported(expression)
    op = _SUPPORTED_COMPARE_OPS.get(type(expression.ops[0]))
    if op is None:
        _unsupported(expression.ops[0])
    return IRCompare(
        op=op,
        lhs=_parse_expression(expression.left),
        rhs=_parse_expression(expression.comparators[0]),
    )


def _parse_unary_op(expression: ast.UnaryOp) -> IRConstant:
    if not isinstance(expression.op, (ast.UAdd, ast.USub)):
        _unsupported(expression.op)
    operand = _parse_expression(expression.operand)
    if not isinstance(operand, IRConstant):
        _unsupported(expression)
    value = operand.value
    if isinstance(expression.op, ast.USub):
        value = -value
    return IRConstant(value)


def _parse_subscript_target(subscript: ast.Subscript) -> tuple[str, IRExpression]:
    if not isinstance(subscript.value, ast.Name):
        _unsupported(subscript.value)
    return subscript.value.id, _parse_expression(subscript.slice)


def _dotted_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted_name(node.value)}.{node.attr}"
    _unsupported(node)


def _is_docstring(statement: ast.stmt, index: int) -> bool:
    return (
        index == 0
        and isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _validate_program_id_axis(axis: ast.expr) -> None:
    if (
        isinstance(axis, ast.Constant)
        and isinstance(axis.value, int)
        and not isinstance(axis.value, bool)
        and axis.value == 0
    ):
        return
    raise KernelCompileError(
        "unsupported kernel syntax: Call program_id expects literal axis 0"
    )


def _validate_single_output(body: tuple[IRStatement, ...]) -> None:
    buffers = set(_iter_store_buffers(body))
    if len(buffers) > 1:
        raise KernelCompileError(
            "unsupported kernel syntax: multiple output buffers"
        )


def _validate_buffer_references(
    body: tuple[IRStatement, ...],
    parameters: tuple[str, ...],
) -> None:
    parameter_names = set(parameters)
    for buffer_name in _iter_buffer_references(body):
        if buffer_name not in parameter_names:
            raise KernelCompileError(
                "unsupported kernel syntax: buffer references must be parameters"
            )


def _validate_buffer_scalar_usage(body: tuple[IRStatement, ...]) -> None:
    buffer_names = set(_iter_buffer_references(body))
    scalar_names = set(_iter_name_references(body))
    invalid_names = sorted(buffer_names & scalar_names)
    if invalid_names:
        joined = ", ".join(invalid_names)
        raise KernelCompileError(
            "unsupported kernel syntax: buffer parameters cannot be used as "
            f"scalar values: {joined}"
        )


def _validate_parameter_shadowing(
    body: tuple[IRStatement, ...],
    parameters: tuple[str, ...],
) -> None:
    parameter_names = set(parameters)
    for target in _iter_assignment_targets(body):
        if target in parameter_names:
            raise KernelCompileError(
                "unsupported kernel syntax: assignments cannot shadow parameters"
            )


def _validate_name_scopes(
    body: tuple[IRStatement, ...],
    parameters: tuple[str, ...],
) -> None:
    _validate_statement_names(body, defined=set(parameters))


def _validate_statement_names(
    statements: tuple[IRStatement, ...],
    *,
    defined: set[str],
) -> set[str]:
    current = set(defined)
    for statement in statements:
        if isinstance(statement, IRAssign):
            _validate_expression_names(statement.value, current)
            if statement.target in current:
                raise KernelCompileError(
                    "unsupported kernel syntax: local reassignment is not supported"
                )
            current.add(statement.target)
        elif isinstance(statement, IRStore):
            _validate_expression_names(statement.index, current)
            _validate_expression_names(statement.value, current)
        elif isinstance(statement, IRIf):
            _validate_expression_names(statement.condition, current)
            _validate_statement_names(statement.body, defined=set(current))
        elif isinstance(statement, IRFor):
            if statement.limit not in current:
                raise KernelCompileError(
                    f"unsupported kernel syntax: undefined name {statement.limit!r}"
                )
            if statement.var in current:
                raise KernelCompileError(
                    "unsupported kernel syntax: for-loop variable cannot shadow "
                    "an existing name"
                )
            inner = set(current) | {statement.var}
            loop_locals: set[str] = set()
            for inner_statement in statement.body:
                # Parser guarantees assignments only; validate defensively for
                # hand-built IR.
                if not isinstance(inner_statement, IRAssign):
                    raise KernelCompileError(
                        "unsupported kernel syntax: for-loop bodies currently "
                        "support only local assignments"
                    )
                _validate_expression_names(inner_statement.value, inner)
                if inner_statement.target == statement.var:
                    raise KernelCompileError(
                        "unsupported kernel syntax: for-loop variable cannot be "
                        "reassigned"
                    )
                if inner_statement.target in loop_locals:
                    raise KernelCompileError(
                        "unsupported kernel syntax: local reassignment is not supported"
                    )
                # Reassigning a name defined BEFORE the loop is the loop-carried
                # accumulator pattern and is allowed; loop-local names are not
                # reassignable and do not escape the loop.
                if inner_statement.target not in current:
                    loop_locals.add(inner_statement.target)
                inner.add(inner_statement.target)
    return current


def _validate_expression_names(expression: IRExpression, defined: set[str]) -> None:
    if isinstance(expression, IRName):
        if expression.name not in defined:
            raise KernelCompileError(
                f"unsupported kernel syntax: undefined name {expression.name!r}"
            )
    elif isinstance(expression, IRLoad):
        _validate_expression_names(expression.index, defined)
    elif isinstance(expression, IRCall):
        for argument in expression.args:
            _validate_expression_names(argument, defined)
    elif isinstance(expression, IRBinaryOp):
        _validate_expression_names(expression.lhs, defined)
        _validate_expression_names(expression.rhs, defined)
    elif isinstance(expression, IRCompare):
        _validate_expression_names(expression.lhs, defined)
        _validate_expression_names(expression.rhs, defined)


def _iter_store_buffers(statements: tuple[IRStatement, ...]):
    for statement in statements:
        if isinstance(statement, IRStore):
            yield statement.buffer
        elif isinstance(statement, (IRIf, IRFor)):
            yield from _iter_store_buffers(statement.body)


def _iter_assignment_targets(statements: tuple[IRStatement, ...]):
    for statement in statements:
        if isinstance(statement, IRAssign):
            yield statement.target
        elif isinstance(statement, IRIf):
            yield from _iter_assignment_targets(statement.body)
        elif isinstance(statement, IRFor):
            yield statement.var
            yield from _iter_assignment_targets(statement.body)


def _iter_buffer_references(statements: tuple[IRStatement, ...]):
    for statement in statements:
        if isinstance(statement, IRStore):
            yield statement.buffer
            yield from _iter_expression_buffers(statement.index)
            yield from _iter_expression_buffers(statement.value)
        elif isinstance(statement, IRAssign):
            yield from _iter_expression_buffers(statement.value)
        elif isinstance(statement, IRIf):
            yield from _iter_expression_buffers(statement.condition)
            yield from _iter_buffer_references(statement.body)
        elif isinstance(statement, IRFor):
            yield from _iter_buffer_references(statement.body)


def _iter_expression_buffers(expression: IRExpression):
    if isinstance(expression, IRLoad):
        yield expression.buffer
        yield from _iter_expression_buffers(expression.index)
    elif isinstance(expression, IRCall):
        for argument in expression.args:
            yield from _iter_expression_buffers(argument)
    elif isinstance(expression, IRBinaryOp):
        yield from _iter_expression_buffers(expression.lhs)
        yield from _iter_expression_buffers(expression.rhs)
    elif isinstance(expression, IRCompare):
        yield from _iter_expression_buffers(expression.lhs)
        yield from _iter_expression_buffers(expression.rhs)


def _iter_name_references(statements: tuple[IRStatement, ...]):
    for statement in statements:
        if isinstance(statement, IRStore):
            yield from _iter_expression_names(statement.index)
            yield from _iter_expression_names(statement.value)
        elif isinstance(statement, IRAssign):
            yield from _iter_expression_names(statement.value)
        elif isinstance(statement, IRIf):
            yield from _iter_expression_names(statement.condition)
            yield from _iter_name_references(statement.body)
        elif isinstance(statement, IRFor):
            yield statement.limit
            yield from _iter_name_references(statement.body)


def _iter_expression_names(expression: IRExpression):
    if isinstance(expression, IRName):
        yield expression.name
    elif isinstance(expression, IRLoad):
        yield from _iter_expression_names(expression.index)
    elif isinstance(expression, IRCall):
        for argument in expression.args:
            yield from _iter_expression_names(argument)
    elif isinstance(expression, IRBinaryOp):
        yield from _iter_expression_names(expression.lhs)
        yield from _iter_expression_names(expression.rhs)
    elif isinstance(expression, IRCompare):
        yield from _iter_expression_names(expression.lhs)
        yield from _iter_expression_names(expression.rhs)


def _emit_msl(kernel_ir: IRKernel) -> str:
    store_buffers = tuple(dict.fromkeys(_iter_store_buffers(kernel_ir.body)))
    if not store_buffers:
        raise KernelCompileError("MSL emission requires one output buffer")

    buffer_names = set(_iter_buffer_references(kernel_ir.body))
    _validate_msl_parameter_usage(kernel_ir, buffer_names)
    output_names = set(store_buffers)
    # Builtin arguments share the function scope with user names; choose them
    # against every nested local/loop variable too, where shadowing would change
    # the meaning of an intrinsic call inside that scope.
    used_names = set(kernel_ir.parameters) | set(_iter_assignment_targets(kernel_ir.body))
    builtin_names = {}
    for intrinsic, base in (("program_id", "block_position"),
                            ("thread_id", "local_position"),
                            ("block_size", "group_size")):
        name, suffix = base, 0
        while name in used_names:
            suffix += 1
            name = f"{base}_{suffix}"
        builtin_names[intrinsic] = name
        used_names.add(name)
    lines = [
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        f"kernel void {kernel_ir.name}(",
    ]
    lines.extend(_emit_msl_parameters(kernel_ir, buffer_names, output_names, builtin_names))
    lines.extend(
        [
            ") {",
            *_emit_msl_statement_block(
                kernel_ir.body, indent=1, context={}, builtin_names=builtin_names
            ),
            "}",
        ]
    )
    return "\n".join(lines)


def _compile_msl_to_metallib(name: str, msl_source: str) -> bytes:
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        raise KernelCompileError(
            "Metal compiler toolchain is unavailable: xcrun was not found"
        )

    with tempfile.TemporaryDirectory(prefix="tensorcx_kernel_") as temp_dir:
        root = Path(temp_dir)
        source_path = root / f"{name}.metal"
        air_path = root / f"{name}.air"
        metallib_path = root / f"{name}.metallib"
        source_path.write_text(msl_source, encoding="utf-8")
        _run_metal_tool(
            [
                xcrun,
                "-sdk",
                "macosx",
                "metal",
                "-c",
                str(source_path),
                "-o",
                str(air_path),
            ]
        )
        _run_metal_tool(
            [
                xcrun,
                "-sdk",
                "macosx",
                "metallib",
                str(air_path),
                "-o",
                str(metallib_path),
            ]
        )
        return metallib_path.read_bytes()


def _run_metal_tool(command: list[str]) -> None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise KernelCompileError(
            f"Metal compiler tool failed to start: {exc}"
        ) from exc
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        if not message:
            message = f"command exited with status {result.returncode}"
        raise KernelCompileError(f"Metal compiler failed: {message}")


def _emit_msl_parameters(
    kernel_ir: IRKernel,
    buffer_names: set[str],
    output_names: set[str],
    builtin_names: dict[str, str],
) -> list[str]:
    parameters: list[str] = []
    buffer_index = 0
    for parameter in kernel_ir.parameters:
        if parameter in buffer_names:
            qualifier = "device" if parameter in output_names else "const device"
            parameters.append(
                f"    {qualifier} float* {parameter} [[buffer({buffer_index})]],"
            )
        else:
            parameters.append(
                f"    constant uint& {parameter} [[buffer({buffer_index})]],"
            )
        buffer_index += 1

    parameters.extend(
        [
            f"    uint3 {builtin_names['program_id']} [[threadgroup_position_in_grid]],",
            f"    uint3 {builtin_names['thread_id']} [[thread_position_in_threadgroup]],",
            f"    uint3 {builtin_names['block_size']} [[threads_per_threadgroup]]",
        ]
    )
    return parameters


def _validate_msl_parameter_usage(
    kernel_ir: IRKernel,
    buffer_names: set[str],
) -> None:
    parameter_names = set(kernel_ir.parameters)
    used_parameters = buffer_names & parameter_names
    used_parameters.update(
        name for name in _iter_name_references(kernel_ir.body) if name in parameter_names
    )
    unused_parameters = [
        parameter
        for parameter in kernel_ir.parameters
        if parameter not in used_parameters
    ]
    if unused_parameters:
        joined = ", ".join(unused_parameters)
        raise KernelCompileError(
            "MSL emission requires all parameters to be referenced: "
            f"{joined}"
        )


def _emit_msl_statement_block(
    statements: tuple[IRStatement, ...],
    *,
    indent: int,
    context: dict[str, str],
    builtin_names: dict[str, str],
) -> list[str]:
    lines: list[str] = []
    for statement in statements:
        lines.extend(_emit_msl_statement(
            statement, indent=indent, context=context, builtin_names=builtin_names
        ))
    return lines


def _emit_msl_statement(
    statement: IRStatement,
    *,
    indent: int,
    context: dict[str, str],
    builtin_names: dict[str, str],
) -> list[str]:
    prefix = "    " * indent
    if isinstance(statement, IRAssign):
        value_type = _infer_msl_type(statement.value, context)
        if statement.target in context:
            # Loop-carried accumulator reassignment: the parser only allows
            # this for names defined before an enclosing for-loop, and the
            # value must keep the local's declared type.
            if context[statement.target] != value_type:
                raise KernelCompileError(
                    "unsupported kernel syntax: reassignment must preserve the "
                    f"type of {statement.target!r}"
                )
            return [
                f"{prefix}{statement.target} = "
                f"{_emit_msl_expression(statement.value, builtin_names)};"
            ]
        context[statement.target] = value_type
        return [
            f"{prefix}{value_type} {statement.target} = "
            f"{_emit_msl_expression(statement.value, builtin_names)};"
        ]
    if isinstance(statement, IRStore):
        return [
            f"{prefix}{statement.buffer}[{_emit_msl_expression(statement.index, builtin_names)}] = "
            f"{_emit_msl_expression(statement.value, builtin_names)};"
        ]
    if isinstance(statement, IRIf):
        nested_context = dict(context)
        lines = [f"{prefix}if ({_emit_msl_expression(statement.condition, builtin_names)}) {{"]
        lines.extend(
            _emit_msl_statement_block(
                statement.body,
                indent=indent + 1,
                context=nested_context,
                builtin_names=builtin_names,
            )
        )
        lines.append(f"{prefix}}}")
        return lines
    if isinstance(statement, IRFor):
        nested_context = dict(context)
        nested_context[statement.var] = "uint"
        lines = [
            f"{prefix}for (uint {statement.var} = 0u; "
            f"{statement.var} < {statement.limit}; ++{statement.var}) {{"
        ]
        lines.extend(
            _emit_msl_statement_block(
                statement.body,
                indent=indent + 1,
                context=nested_context,
                builtin_names=builtin_names,
            )
        )
        lines.append(f"{prefix}}}")
        return lines
    raise TypeError(f"unhandled IR statement: {type(statement).__name__}")


def _emit_msl_expression(expression: IRExpression, builtin_names: dict[str, str]) -> str:
    if isinstance(expression, IRName):
        return expression.name
    if isinstance(expression, IRConstant):
        return _emit_msl_constant(expression.value)
    if isinstance(expression, IRCall):
        return _emit_msl_call(expression, builtin_names)
    if isinstance(expression, IRBinaryOp):
        op = _MSL_BINARY_OPS[expression.op]
        return (
            f"({_emit_msl_expression(expression.lhs, builtin_names)} {op} "
            f"{_emit_msl_expression(expression.rhs, builtin_names)})"
        )
    if isinstance(expression, IRCompare):
        op = _MSL_COMPARE_OPS[expression.op]
        return (
            f"({_emit_msl_expression(expression.lhs, builtin_names)} {op} "
            f"{_emit_msl_expression(expression.rhs, builtin_names)})"
        )
    if isinstance(expression, IRLoad):
        return f"{expression.buffer}[{_emit_msl_expression(expression.index, builtin_names)}]"
    raise TypeError(f"unhandled IR expression: {type(expression).__name__}")


def _emit_msl_constant(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value!r}f"


def _emit_msl_call(call: IRCall, builtin_names: dict[str, str]) -> str:
    if call.name not in builtin_names:
        raise TypeError(f"unhandled IR call: {call.name}")
    return f"{builtin_names[call.name]}.x"


def _infer_msl_type(expression: IRExpression, context: dict[str, str]) -> str:
    if isinstance(expression, IRConstant):
        if isinstance(expression.value, float):
            return "float"
        return "int" if expression.value < 0 else "uint"
    if isinstance(expression, IRCall):
        return "uint"
    if isinstance(expression, IRLoad):
        return "float"
    if isinstance(expression, IRName):
        return context.get(expression.name, "uint")
    if isinstance(expression, IRCompare):
        return "bool"
    if isinstance(expression, IRBinaryOp):
        lhs_type = _infer_msl_type(expression.lhs, context)
        rhs_type = _infer_msl_type(expression.rhs, context)
        if "float" in {lhs_type, rhs_type}:
            return "float"
        if "int" in {lhs_type, rhs_type}:
            return "int"
        return "uint"
    raise TypeError(f"unhandled IR expression: {type(expression).__name__}")


def _unsupported(node: ast.AST) -> NoReturn:
    raise KernelCompileError(f"unsupported kernel syntax: {type(node).__name__}")


def kernel(fn: Callable | None = None, *, target: str = "auto"):
    """Decorate a Python function as a Phase 7 experimental kernel.

    The decorator records metadata, ``compile(target="metal")`` produces an
    in-memory metallib artifact, and calling the decorated kernel compiles and
    launches the first narrow Metal subset.
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
    "CompiledKernel",
    "IRAssign",
    "IRBinaryOp",
    "IRCall",
    "IRCompare",
    "IRConstant",
    "IRIf",
    "IRKernel",
    "IRLoad",
    "IRName",
    "IRFor",
    "IRStore",
    "Kernel",
    "KernelCompileError",
    "block_size",
    "kernel",
    "program_id",
    "thread_id",
]
