"""Experimental APIs for Cortex Runtime.

Nothing in this module is stable yet. Phase 7 uses it as a controlled place to
prototype the kernel DSL without committing the top-level ``cortex_runtime`` API.
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
from typing import Callable, NoReturn, TypeAlias


_SUPPORTED_TARGETS = {"auto", "cpu", "metal"}
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

        normalized_block_size = _normalize_uint32(
            block_size,
            "block_size must be a positive uint32",
            allow_zero=False,
        )
        if thread_count is None:
            normalized_thread_count = None
        else:
            normalized_thread_count = _normalize_uint32(
                thread_count,
                "thread_count must be a uint32",
                allow_zero=True,
            )
        output_param = _single_output_parameter(self.ir)
        guard_param = _launch_store_guard_parameter(self.ir, output_param)
        (
            native_arguments,
            output_tensor,
            inferred_thread_count,
            scalar_arguments,
        ) = _prepare_launch_arguments(
            self.ir,
            output_param,
            args,
        )
        if normalized_thread_count is None:
            normalized_thread_count = inferred_thread_count
        if normalized_thread_count > inferred_thread_count:
            raise ValueError("thread_count cannot exceed output tensor size")
        if scalar_arguments[guard_param] != normalized_thread_count:
            raise ValueError("kernel guard bound must match thread_count")
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


IRExpression: TypeAlias = (
    IRName | IRConstant | IRCall | IRBinaryOp | IRCompare | IRLoad
)
IRStatement: TypeAlias = IRAssign | IRStore | IRIf


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

    def compile(self, *, target: str | None = None) -> CompiledKernel:
        selected = self.target if target is None else _validate_target(target)
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
        """Parse this kernel into the Phase 7 backend-neutral IR subset."""
        return _parse_kernel_function(self.fn)

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


def _validate_target(target: str) -> str:
    if not isinstance(target, str):
        raise TypeError("experimental kernel target must be a string")
    if target not in _SUPPORTED_TARGETS:
        raise ValueError("experimental kernel target must be 'auto', 'cpu', or 'metal'")
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


def _single_output_parameter(kernel_ir: IRKernel) -> str:
    output_names = tuple(dict.fromkeys(_iter_store_buffers(kernel_ir.body)))
    if len(output_names) != 1:
        raise KernelCompileError("kernel launch requires exactly one output buffer")
    return output_names[0]


def _launch_store_guard_parameter(kernel_ir: IRKernel, output_param: str) -> str:
    buffer_names = set(_iter_buffer_references(kernel_ir.body))
    parameter_names = set(kernel_ir.parameters)
    guard_params: set[str] = set()

    def visit(statements: tuple[IRStatement, ...], guard: IRExpression | None) -> None:
        for statement in statements:
            if isinstance(statement, IRAssign):
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

    visit(kernel_ir.body, guard=None)
    if len(guard_params) != 1:
        raise KernelCompileError(
            "kernel launch requires all stores to share one scalar guard"
        )
    return next(iter(guard_params))


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


def _prepare_launch_arguments(
    kernel_ir: IRKernel,
    output_param: str,
    args: tuple[object, ...],
):
    if len(args) != len(kernel_ir.parameters):
        raise TypeError(
            f"{kernel_ir.name} expects {len(kernel_ir.parameters)} argument(s), "
            f"got {len(args)}"
        )

    from .tensor import Tensor

    buffer_names = set(_iter_buffer_references(kernel_ir.body))
    native_arguments = []
    scalar_arguments: dict[str, int] = {}
    tensor_shapes: list[tuple[int, ...]] = []
    output_tensor = None

    for parameter, argument in zip(kernel_ir.parameters, args, strict=True):
        if parameter in buffer_names:
            if not isinstance(argument, Tensor):
                raise TypeError(
                    "experimental Metal kernel buffer arguments must be Tensor objects"
                )
            if argument.device != "metal":
                raise ValueError("experimental Metal kernel launch requires Metal tensors")
            if argument.dtype != "float32":
                raise ValueError(
                    "experimental Metal kernels currently support float32 tensor buffers"
                )
            tensor_shapes.append(argument.shape)
            native_arguments.append(argument._impl)
            if parameter == output_param:
                output_tensor = argument
        else:
            scalar_value = _normalize_uint32(
                argument,
                "kernel scalar arguments must be uint32",
                allow_zero=True,
            )
            scalar_arguments[parameter] = scalar_value
            native_arguments.append(scalar_value)

    if output_tensor is None:
        raise KernelCompileError("kernel launch requires exactly one output buffer")
    if any(shape != output_tensor.shape for shape in tensor_shapes):
        raise ValueError("experimental Metal kernel tensor shapes must match")

    return native_arguments, output_tensor, _numel(output_tensor.shape), scalar_arguments


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
    _unsupported(statement)


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
        elif isinstance(statement, IRIf):
            yield from _iter_store_buffers(statement.body)


def _iter_assignment_targets(statements: tuple[IRStatement, ...]):
    for statement in statements:
        if isinstance(statement, IRAssign):
            yield statement.target
        elif isinstance(statement, IRIf):
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
    lines = [
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        f"kernel void {kernel_ir.name}(",
    ]
    lines.extend(_emit_msl_parameters(kernel_ir, buffer_names, output_names))
    lines.extend(
        [
            ") {",
            *_emit_msl_statement_block(kernel_ir.body, indent=1, context={}),
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

    with tempfile.TemporaryDirectory(prefix="cortex_runtime_kernel_") as temp_dir:
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
            "    uint3 block_position [[threadgroup_position_in_grid]],",
            "    uint3 local_position [[thread_position_in_threadgroup]],",
            "    uint3 group_size [[threads_per_threadgroup]]",
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
) -> list[str]:
    lines: list[str] = []
    for statement in statements:
        lines.extend(_emit_msl_statement(statement, indent=indent, context=context))
    return lines


def _emit_msl_statement(
    statement: IRStatement,
    *,
    indent: int,
    context: dict[str, str],
) -> list[str]:
    prefix = "    " * indent
    if isinstance(statement, IRAssign):
        value_type = _infer_msl_type(statement.value, context)
        context[statement.target] = value_type
        return [
            f"{prefix}{value_type} {statement.target} = "
            f"{_emit_msl_expression(statement.value)};"
        ]
    if isinstance(statement, IRStore):
        return [
            f"{prefix}{statement.buffer}[{_emit_msl_expression(statement.index)}] = "
            f"{_emit_msl_expression(statement.value)};"
        ]
    if isinstance(statement, IRIf):
        nested_context = dict(context)
        lines = [f"{prefix}if ({_emit_msl_expression(statement.condition)}) {{"]
        lines.extend(
            _emit_msl_statement_block(
                statement.body,
                indent=indent + 1,
                context=nested_context,
            )
        )
        lines.append(f"{prefix}}}")
        return lines
    raise TypeError(f"unhandled IR statement: {type(statement).__name__}")


def _emit_msl_expression(expression: IRExpression) -> str:
    if isinstance(expression, IRName):
        return expression.name
    if isinstance(expression, IRConstant):
        return _emit_msl_constant(expression.value)
    if isinstance(expression, IRCall):
        return _emit_msl_call(expression)
    if isinstance(expression, IRBinaryOp):
        op = _MSL_BINARY_OPS[expression.op]
        return (
            f"({_emit_msl_expression(expression.lhs)} {op} "
            f"{_emit_msl_expression(expression.rhs)})"
        )
    if isinstance(expression, IRCompare):
        op = _MSL_COMPARE_OPS[expression.op]
        return (
            f"{_emit_msl_expression(expression.lhs)} {op} "
            f"{_emit_msl_expression(expression.rhs)}"
        )
    if isinstance(expression, IRLoad):
        return f"{expression.buffer}[{_emit_msl_expression(expression.index)}]"
    raise TypeError(f"unhandled IR expression: {type(expression).__name__}")


def _emit_msl_constant(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value!r}f"


def _emit_msl_call(call: IRCall) -> str:
    if call.name == "program_id":
        return "block_position.x"
    if call.name == "thread_id":
        return "local_position.x"
    if call.name == "block_size":
        return "group_size.x"
    raise TypeError(f"unhandled IR call: {call.name}")


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
    "IRStore",
    "Kernel",
    "KernelCompileError",
    "block_size",
    "kernel",
    "program_id",
    "thread_id",
]
