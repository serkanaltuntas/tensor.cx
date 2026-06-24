"""Experimental APIs for Cortex Runtime.

Nothing in this module is stable yet. Phase 7 uses it as a controlled place to
prototype the kernel DSL without committing the top-level ``cortex_runtime`` API.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import inspect
import textwrap
from types import FunctionType
from typing import Callable, NoReturn, TypeAlias


_SUPPORTED_TARGETS = {"auto", "cpu", "metal"}
_SUPPORTED_BINARY_OPS = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "div",
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


class KernelCompileError(ValueError):
    """Raised when an experimental kernel cannot be parsed into IR."""


@dataclass(frozen=True, slots=True)
class IRKernel:
    """Backend-neutral kernel IR parsed from a restricted Python function."""

    name: str
    parameters: tuple[str, ...]
    body: tuple[IRStatement, ...]


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

    def compile(self, *, target: str | None = None):
        selected = self.target if target is None else _validate_target(target)
        raise NotImplementedError(
            "experimental kernel DSL compilation is not implemented yet "
            f"for target {selected!r}"
        )

    def parse_ir(self) -> IRKernel:
        """Parse this kernel into the Phase 7 backend-neutral IR subset."""
        return _parse_kernel_function(self.fn)

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
    _validate_buffer_references(body, parameters)
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


def _unsupported(node: ast.AST) -> NoReturn:
    raise KernelCompileError(f"unsupported kernel syntax: {type(node).__name__}")


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
