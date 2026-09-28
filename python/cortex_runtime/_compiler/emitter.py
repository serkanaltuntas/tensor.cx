"""Private MLIR emitter shared by the runtime compiler and research scripts.

Mapping (mirrors the semantics of the Phase 7 MSL emitter):

- Buffer parameters become `memref<?xf32>`; scalar parameters become `i32`
  with unsigned semantics, matching the MSL path's `uint` scalars.
- The Metal grid is mapped to a single `scf.for` over the global thread index
  `gi` in `[0, thread_count)`. `program_id(0)` = `gi / block_size`,
  `thread_id()` = `gi % block_size`, `block_size()` = the launch block size.
  Two extra trailing function arguments (`i32 thread_count`, `i32 block_size`)
  carry the launch geometry that Metal would provide implicitly.
- Integer values are tracked as unsigned (`u32`) or possibly-negative (`s32`,
  introduced only by negative constants) the same way the MSL emitter's type
  inference does. Arithmetic is two's-complement and sign-agnostic
  (`arith.addi/subi/muli`). Ordered comparisons are emitted as unsigned
  (`cmpi ult/...`) and are only allowed on `u32` operands; an ordered
  comparison involving an `s32` value is rejected loudly because MSL would
  compile it as a signed compare and the two paths would silently diverge.
  `==`/`!=` are sign-agnostic and stay allowed.
- Float constants are narrowed to float32 first and formatted with a
  guaranteed decimal point so the text is always valid MLIR; non-finite or
  out-of-range constants are rejected.
- `if` maps to `scf.if` without else; assignments are SSA bindings and the
  parser already rejects branch-local names escaping their branch.

Only float32 buffers and the IR node set documented in docs/KERNEL_DSL.md are
supported. Anything else raises MlirEmitError naming the construct, mirroring
the DSL rule that unsupported input must fail loudly rather than miscompile.

All emitter-internal SSA names start with `cortex_`, so kernel parameter names
beginning with `cortex_` are reserved and rejected.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

from cortex_runtime.experimental import (
    IRAssign,
    IRBinaryOp,
    IRCall,
    IRCompare,
    IRConstant,
    IRFor,
    IRIf,
    IRKernel,
    IRLoad,
    IRName,
    IRStore,
)

RESERVED_PREFIX = "cortex_"
THREAD_COUNT_ARG = "cortex_thread_count"
BLOCK_SIZE_ARG = "cortex_block_size"

# Value types tracked by the emitter. "u32" and "s32" both emit as MLIR i32;
# the distinction only gates ordered comparisons (see module docstring).
_INT_TYPES = ("u32", "s32")

# Keyed by the IR's operator names (see _SUPPORTED_BINARY_OPS/_SUPPORTED_
# COMPARE_OPS in cortex_runtime.experimental), not by Python symbols.
_INT_BINARY_OPS = {"add": "arith.addi", "sub": "arith.subi", "mul": "arith.muli"}
_FLOAT_BINARY_OPS = {"add": "arith.addf", "sub": "arith.subf", "mul": "arith.mulf"}
_UNSIGNED_ORDERED_PREDICATES = {"lt": "ult", "lte": "ule", "gt": "ugt", "gte": "uge"}
_EQUALITY_PREDICATES = {"eq": "eq", "neq": "ne"}
_FLOAT_COMPARE_PREDICATES = {
    "lt": "olt",
    "lte": "ole",
    "gt": "ogt",
    "gte": "oge",
    "eq": "oeq",
    "neq": "one",
}


class MlirEmitError(ValueError):
    """Raised when kernel IR cannot be emitted as MLIR by this prototype."""


def format_f32_constant(value: float) -> str:
    """Format a Python float as a valid MLIR f32 literal.

    The value is narrowed to float32 first (so the emitted constant is exactly
    the float32 the kernel will compute with) and the text always carries a
    decimal point — Python's repr may produce forms like ``1e-05`` that MLIR's
    float grammar rejects.
    """
    if not math.isfinite(value):
        raise MlirEmitError(
            f"non-finite float constants are not supported: {value!r}"
        )
    try:
        narrowed = struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError as error:
        raise MlirEmitError(
            f"float constant is out of range for float32: {value!r}"
        ) from error
    if not math.isfinite(narrowed):
        # Finite doubles beyond float32 range narrow to +-inf on platforms
        # where struct.pack does not raise.
        raise MlirEmitError(
            f"float constant is out of range for float32: {value!r}"
        )
    text = repr(narrowed)
    mantissa, exponent_marker, exponent = text.partition("e")
    if "." not in mantissa:
        mantissa += ".0"
    return mantissa + exponent_marker + exponent


@dataclass
class _Emitter:
    lines: list[str] = field(default_factory=list)
    counter: int = 0

    def fresh(self) -> str:
        name = f"%cortex_v{self.counter}"
        self.counter += 1
        return name

    def emit(self, indent: int, text: str) -> None:
        self.lines.append("  " * indent + text)


def _iter_statements(statements):
    for statement in statements:
        yield statement
        if isinstance(statement, (IRIf, IRFor)):
            yield from _iter_statements(statement.body)


def _iter_expressions(statements):
    stack = []
    for statement in _iter_statements(statements):
        if isinstance(statement, IRAssign):
            stack.append(statement.value)
        elif isinstance(statement, IRStore):
            stack.extend((statement.index, statement.value))
        elif isinstance(statement, IRIf):
            stack.append(statement.condition)
    while stack:
        expression = stack.pop()
        yield expression
        if isinstance(expression, (IRBinaryOp, IRCompare)):
            stack.extend((expression.lhs, expression.rhs))
        elif isinstance(expression, IRCall):
            stack.extend(expression.args)
        elif isinstance(expression, IRLoad):
            stack.append(expression.index)


def _buffer_names(kernel_ir: IRKernel) -> tuple[set[str], set[str]]:
    stores: set[str] = set()
    loads: set[str] = set()
    for statement in _iter_statements(kernel_ir.body):
        if isinstance(statement, IRStore):
            stores.add(statement.buffer)
    for expression in _iter_expressions(kernel_ir.body):
        if isinstance(expression, IRLoad):
            loads.add(expression.buffer)
    return stores, loads


def emit_mlir(kernel_ir: IRKernel) -> str:
    """Emit an MLIR module (func/scf/arith/memref dialects) for the kernel."""

    store_buffers, load_buffers = _buffer_names(kernel_ir)
    if not store_buffers:
        raise MlirEmitError("MLIR emission requires one output buffer")
    buffers = store_buffers | load_buffers

    for parameter in kernel_ir.parameters:
        if parameter.startswith(RESERVED_PREFIX):
            raise MlirEmitError(
                "kernel parameter name is reserved by the MLIR harness "
                f"(prefix '{RESERVED_PREFIX}'): {parameter}"
            )

    emitter = _Emitter()
    arguments: list[str] = []
    for parameter in kernel_ir.parameters:
        mlir_type = "memref<?xf32>" if parameter in buffers else "i32"
        arguments.append(f"%{parameter}: {mlir_type}")
    arguments.append(f"%{THREAD_COUNT_ARG}: i32")
    arguments.append(f"%{BLOCK_SIZE_ARG}: i32")

    emitter.emit(0, "module {")
    emitter.emit(
        1,
        f"func.func @{kernel_ir.name}({', '.join(arguments)}) "
        "attributes { llvm.emit_c_interface } {",
    )
    emitter.emit(2, "%cortex_c0 = arith.constant 0 : index")
    emitter.emit(2, "%cortex_c1 = arith.constant 1 : index")
    emitter.emit(
        2,
        f"%cortex_trip_count = arith.index_castui %{THREAD_COUNT_ARG} : i32 to index",
    )
    emitter.emit(
        2,
        "scf.for %cortex_gi = %cortex_c0 to %cortex_trip_count "
        "step %cortex_c1 {",
    )
    emitter.emit(
        3, "%cortex_gi_i32 = arith.index_castui %cortex_gi : index to i32"
    )
    emitter.emit(
        3, f"%cortex_pid = arith.divui %cortex_gi_i32, %{BLOCK_SIZE_ARG} : i32"
    )
    emitter.emit(
        3, f"%cortex_tid = arith.remui %cortex_gi_i32, %{BLOCK_SIZE_ARG} : i32"
    )

    context: dict[str, tuple[str, str]] = {
        parameter: (f"%{parameter}", "u32")
        for parameter in kernel_ir.parameters
        if parameter not in buffers
    }
    _emit_statement_block(emitter, kernel_ir.body, indent=3, context=context)

    emitter.emit(2, "}")
    emitter.emit(2, "return")
    emitter.emit(1, "}")
    emitter.emit(0, "}")
    return "\n".join(emitter.lines) + "\n"


def _emit_statement_block(
    emitter: _Emitter,
    statements,
    *,
    indent: int,
    context: dict[str, tuple[str, str]],
) -> None:
    for statement in statements:
        _emit_statement(emitter, statement, indent=indent, context=context)


def _emit_statement(
    emitter: _Emitter,
    statement,
    *,
    indent: int,
    context: dict[str, tuple[str, str]],
) -> None:
    if isinstance(statement, IRAssign):
        value, value_type = _emit_expression(
            emitter, statement.value, indent=indent, context=context
        )
        context[statement.target] = (value, value_type)
        return
    if isinstance(statement, IRStore):
        index, index_type = _emit_expression(
            emitter, statement.index, indent=indent, context=context
        )
        if index_type not in _INT_TYPES:
            raise MlirEmitError("store index must be an integer expression")
        value, value_type = _emit_expression(
            emitter, statement.value, indent=indent, context=context
        )
        if value_type != "f32":
            raise MlirEmitError("store value must be a float expression")
        cast = emitter.fresh()
        emitter.emit(indent, f"{cast} = arith.index_castui {index} : i32 to index")
        emitter.emit(
            indent,
            f"memref.store {value}, %{statement.buffer}[{cast}] : memref<?xf32>",
        )
        return
    if isinstance(statement, IRIf):
        condition, condition_type = _emit_expression(
            emitter, statement.condition, indent=indent, context=context
        )
        if condition_type != "i1":
            raise MlirEmitError("if condition must be a comparison")
        emitter.emit(indent, f"scf.if {condition} {{")
        nested = dict(context)
        _emit_statement_block(
            emitter, statement.body, indent=indent + 1, context=nested
        )
        emitter.emit(indent, "}")
        # scf.if without results cannot carry values out; a reassignment of an
        # outer name inside the branch would silently diverge from MSL.
        escaped = sorted(
            name for name in context if nested.get(name) != context[name]
        )
        if escaped:
            raise MlirEmitError(
                "values reassigned inside an if do not propagate in the MLIR "
                f"prototype: {', '.join(escaped)}"
            )
        return
    if isinstance(statement, IRFor):
        _emit_for(emitter, statement, indent=indent, context=context)
        return
    raise MlirEmitError(f"unhandled IR statement: {type(statement).__name__}")


def _emit_for(
    emitter: _Emitter,
    statement: IRFor,
    *,
    indent: int,
    context: dict[str, tuple[str, str]],
) -> None:
    """Lower ``for var in range(limit)`` to scf.for with iter_args.

    Names assigned in the loop body that already exist outside it are the
    loop-carried accumulators: they become scf.for iter_args, are rebound via
    scf.yield each iteration, and their loop results replace the outer SSA
    values. Loop-local names do not escape.
    """
    for inner in statement.body:
        if not isinstance(inner, IRAssign):
            raise MlirEmitError(
                "for-loop bodies support only assignments in the MLIR prototype"
            )
    if statement.limit not in context or context[statement.limit] != (
        f"%{statement.limit}",
        "u32",
    ):
        raise MlirEmitError(
            "for-loop range bound must be a scalar parameter"
        )
    carried = list(
        dict.fromkeys(
            inner.target for inner in statement.body if inner.target in context
        )
    )
    for name in carried:
        if context[name][1] == "i1":
            # An i1 accumulator would emit iter_args/yield typed i32 for an i1
            # SSA value — type-invalid MLIR the verifier rejects downstream.
            raise MlirEmitError(
                "bool values cannot be loop-carried in the MLIR prototype: "
                f"{name}"
            )

    limit_index = emitter.fresh()
    emitter.emit(
        indent,
        f"{limit_index} = arith.index_castui %{statement.limit} : i32 to index",
    )
    loop_var = emitter.fresh()
    body_context = dict(context)
    result = emitter.fresh()
    if carried:
        iter_bindings = []
        for position, name in enumerate(carried):
            arg = f"{result}_arg{position}"
            iter_bindings.append(f"{arg} = {context[name][0]}")
            body_context[name] = (arg, context[name][1])
        types = ", ".join(
            "f32" if context[name][1] == "f32" else "i32" for name in carried
        )
        header = (
            f"{result}{':' + str(len(carried)) if len(carried) > 1 else ''} = "
            f"scf.for {loop_var} = %cortex_c0 to {limit_index} step %cortex_c1 "
            f"iter_args({', '.join(iter_bindings)}) -> ({types}) {{"
        )
    else:
        header = (
            f"scf.for {loop_var} = %cortex_c0 to {limit_index} "
            f"step %cortex_c1 {{"
        )
    emitter.emit(indent, header)

    loop_var_i32 = emitter.fresh()
    emitter.emit(
        indent + 1,
        f"{loop_var_i32} = arith.index_castui {loop_var} : index to i32",
    )
    body_context[statement.var] = (loop_var_i32, "u32")
    _emit_statement_block(
        emitter, statement.body, indent=indent + 1, context=body_context
    )
    if carried:
        for name in carried:
            if body_context[name][1] != context[name][1]:
                raise MlirEmitError(
                    f"loop-carried value must preserve its type: {name}"
                )
        yielded = ", ".join(body_context[name][0] for name in carried)
        types = ", ".join(
            "f32" if context[name][1] == "f32" else "i32" for name in carried
        )
        emitter.emit(indent + 1, f"scf.yield {yielded} : {types}")
    emitter.emit(indent, "}")
    for position, name in enumerate(carried):
        result_value = (
            f"{result}#{position}" if len(carried) > 1 else result
        )
        context[name] = (result_value, context[name][1])


def _emit_expression(
    emitter: _Emitter,
    expression,
    *,
    indent: int,
    context: dict[str, tuple[str, str]],
) -> tuple[str, str]:
    if isinstance(expression, IRName):
        if expression.name not in context:
            raise MlirEmitError(f"unbound name in expression: {expression.name}")
        return context[expression.name]
    if isinstance(expression, IRConstant):
        if isinstance(expression.value, bool):
            raise MlirEmitError("bool constants are not supported")
        value = emitter.fresh()
        if isinstance(expression.value, int):
            emitter.emit(
                indent, f"{value} = arith.constant {expression.value} : i32"
            )
            # Negative constants taint the value as possibly-signed, exactly
            # like the MSL emitter's type inference ("int" vs "uint").
            return value, ("s32" if expression.value < 0 else "u32")
        emitter.emit(
            indent,
            f"{value} = arith.constant {format_f32_constant(expression.value)} : f32",
        )
        return value, "f32"
    if isinstance(expression, IRCall):
        return _emit_call(expression)
    if isinstance(expression, IRBinaryOp):
        lhs, lhs_type = _emit_expression(
            emitter, expression.lhs, indent=indent, context=context
        )
        rhs, rhs_type = _emit_expression(
            emitter, expression.rhs, indent=indent, context=context
        )
        if "i1" in (lhs_type, rhs_type):
            raise MlirEmitError(
                "comparison results cannot be used in arithmetic"
            )
        if (lhs_type == "f32") != (rhs_type == "f32"):
            raise MlirEmitError(
                "mixed integer/float arithmetic is not supported by the MLIR "
                f"prototype: {lhs_type} {expression.op} {rhs_type}"
            )
        if lhs_type == "f32":
            ops, result_type = _FLOAT_BINARY_OPS, "f32"
        else:
            ops = _INT_BINARY_OPS
            # Two's-complement arithmetic is sign-agnostic; propagate the
            # signed taint the way the MSL inference does (int wins over uint).
            result_type = "s32" if "s32" in (lhs_type, rhs_type) else "u32"
        if expression.op not in ops:
            raise MlirEmitError(f"unhandled binary operator: {expression.op}")
        value = emitter.fresh()
        emitter.emit(indent, f"{value} = {ops[expression.op]} {lhs}, {rhs} : "
                             f"{'f32' if result_type == 'f32' else 'i32'}")
        return value, result_type
    if isinstance(expression, IRCompare):
        lhs, lhs_type = _emit_expression(
            emitter, expression.lhs, indent=indent, context=context
        )
        rhs, rhs_type = _emit_expression(
            emitter, expression.rhs, indent=indent, context=context
        )
        if "i1" in (lhs_type, rhs_type):
            raise MlirEmitError("comparison results cannot be compared again")
        if (lhs_type == "f32") != (rhs_type == "f32"):
            raise MlirEmitError(
                "mixed integer/float comparison is not supported by the MLIR "
                f"prototype: {lhs_type} {expression.op} {rhs_type}"
            )
        value = emitter.fresh()
        if lhs_type == "f32":
            predicate = _FLOAT_COMPARE_PREDICATES.get(expression.op)
            if predicate is None:
                raise MlirEmitError(
                    f"unhandled comparison operator: {expression.op}"
                )
            emitter.emit(
                indent, f"{value} = arith.cmpf {predicate}, {lhs}, {rhs} : f32"
            )
            return value, "i1"
        if expression.op in _EQUALITY_PREDICATES:
            # Bit equality is sign-agnostic, so s32 operands are fine here.
            predicate = _EQUALITY_PREDICATES[expression.op]
        else:
            predicate = _UNSIGNED_ORDERED_PREDICATES.get(expression.op)
            if predicate is None:
                raise MlirEmitError(
                    f"unhandled comparison operator: {expression.op}"
                )
            if "s32" in (lhs_type, rhs_type):
                # MSL would compile an int-vs-int compare as SIGNED; emitting
                # an unsigned predicate here would silently diverge. Refuse.
                raise MlirEmitError(
                    "ordered comparisons involving negative integer constants "
                    "are not supported by the MLIR prototype (MSL compiles "
                    "them as signed compares)"
                )
        emitter.emit(
            indent, f"{value} = arith.cmpi {predicate}, {lhs}, {rhs} : i32"
        )
        return value, "i1"
    if isinstance(expression, IRLoad):
        index, index_type = _emit_expression(
            emitter, expression.index, indent=indent, context=context
        )
        if index_type not in _INT_TYPES:
            raise MlirEmitError("load index must be an integer expression")
        cast = emitter.fresh()
        emitter.emit(indent, f"{cast} = arith.index_castui {index} : i32 to index")
        value = emitter.fresh()
        emitter.emit(
            indent,
            f"{value} = memref.load %{expression.buffer}[{cast}] : memref<?xf32>",
        )
        return value, "f32"
    raise MlirEmitError(f"unhandled IR expression: {type(expression).__name__}")


def _emit_call(expression: IRCall) -> tuple[str, str]:
    if expression.name == "program_id":
        if (
            len(expression.args) != 1
            or not isinstance(expression.args[0], IRConstant)
            or expression.args[0].value != 0
        ):
            raise MlirEmitError(
                "only program_id(0) is supported by the MLIR prototype"
            )
        return "%cortex_pid", "u32"
    if expression.name == "thread_id":
        if expression.args:
            raise MlirEmitError("thread_id() takes no arguments")
        return "%cortex_tid", "u32"
    if expression.name == "block_size":
        if expression.args:
            raise MlirEmitError("block_size() takes no arguments")
        return f"%{BLOCK_SIZE_ARG}", "u32"
    raise MlirEmitError(f"unhandled IR call: {expression.name}")
