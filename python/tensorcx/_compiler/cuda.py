"""Optional, device-only MLIR CUDA elementwise compiler for the validated sm_52 slice."""
from __future__ import annotations

from pathlib import Path
import re
import tempfile

from .. import _core
from ..experimental import CompiledCudaKernel, IRAssign, IRBinaryOp, IRCall, IRCompare, IRConstant, IRIf, IRLoad, IRName, IRStore, KernelCompileError
from .cpu import _signature as elementwise_signature, _run, toolchain
from .emitter import format_f32_constant
from .cache import artifact_cache, cache_enabled, descriptor

ENTRY = "tensorcx_add_v1"
LOWER_FLAGS = ["--convert-scf-to-cf", "--convert-gpu-to-nvvm=index-bitwidth=64",
               "--reconcile-unrealized-casts"]
SERIALIZE_FLAGS = ["--nvvm-attach-target=chip=sm_52 features=+ptx78 O=2",
                   "--gpu-module-to-binary=format=assembly"]
OPERATIONS = {"add": "arith.addf", "sub": "arith.subf", "mul": "arith.mulf", "div": "arith.divf"}
PREDICATES = {"lt": "olt", "lte": "ole", "gt": "ogt", "gte": "oge", "eq": "oeq", "neq": "une"}


def _emit_tanh(operand, instruction):
    """sm_52 tanh without libdevice or cancellation at the origin."""
    def constant(value):
        return instruction(f"arith.constant {format_f32_constant(value)} : f32")

    def intrinsic(name, values):
        types = ", ".join("f32" for _ in values)
        return instruction(f'llvm.call_intrinsic "llvm.{name}.f32"'
                           f'({", ".join(values)}) : ({types}) -> f32')

    absolute = intrinsic("fabs", (operand,))
    one, ten = constant(1.0), constant(10.0)
    # Clamp both eager branches. NaNs propagate through IEEE minimum.
    small = intrinsic("minimum", (absolute, one))
    large = intrinsic("minimum", (absolute, ten))
    squared = instruction(f"arith.mulf {small}, {small} : f32")

    def polynomial(coefficients):
        result = constant(coefficients[0])
        for coefficient in coefficients[1:]:
            result = intrinsic("fma", (result, squared, constant(coefficient)))
        return result

    # [7/6] Pade approximant, used only for |x| <= 1.
    numerator = polynomial((1.0, 378.0, 17325.0, 135135.0))
    denominator = polynomial((28.0, 3150.0, 62370.0, 135135.0))
    ratio = instruction(f"arith.divf {numerator}, {denominator} : f32")
    rational = instruction(f"arith.mulf {small}, {ratio} : f32")
    scale = constant(-2.8853900817779268)  # -2 * log2(e), rounded to f32.
    exponent = instruction(f"arith.mulf {large}, {scale} : f32")
    exponential = instruction('llvm.call_intrinsic "llvm.nvvm.ex2.approx.f"'
                              f'({exponent}) : (f32) -> f32')
    difference = instruction(f"arith.subf {one}, {exponential} : f32")
    total = instruction(f"arith.addf {one}, {exponential} : f32")
    quotient = instruction(f"arith.divf {difference}, {total} : f32")
    near = instruction(f"arith.cmpf ole, {absolute}, {one} : f32")
    magnitude = instruction(f"arith.select {near}, {rational}, {quotient} : f32")
    saturated = instruction(f"arith.cmpf oge, {absolute}, {ten} : f32")
    magnitude = instruction(f"arith.select {saturated}, {one}, {magnitude} : f32")
    return intrinsic("copysign", (magnitude, operand))


def _emit_sigmoid(operand, instruction):
    """Stable logistic function with non-FTZ exp2 and bounded scaling input."""
    def constant(value):
        return instruction(f"arith.constant {format_f32_constant(value)} : f32")
    absolute = instruction('llvm.call_intrinsic "llvm.fabs.f32"'
                           f'({operand}) : (f32) -> f32')
    limit = constant(128.0)
    absolute = instruction('llvm.call_intrinsic "llvm.minimum.f32"'
                           f'({absolute}, {limit}) : (f32, f32) -> f32')
    scale = constant(-1.4426950408889634)
    exponent = instruction(f"arith.mulf {absolute}, {scale} : f32")
    exponential = instruction('llvm.call_intrinsic "llvm.nvvm.ex2.approx.f"'
                              f'({exponent}) : (f32) -> f32')
    one, zero = constant(1.0), constant(0.0)
    denominator = instruction(f"arith.addf {one}, {exponential} : f32")
    positive = instruction(f"arith.divf {one}, {denominator} : f32")
    negative = instruction(f"arith.divf {exponential}, {denominator} : f32")
    condition = instruction(f"arith.cmpf oge, {operand}, {zero} : f32")
    return instruction(f"arith.select {condition}, {positive}, {negative} : f32")


def _emit_silu(operand, instruction):
    """Non-FTZ half exponentials avoid premature negative-tail underflow."""
    def constant(value):
        return instruction(f"arith.constant {format_f32_constant(value)} : f32")
    absolute = instruction('llvm.call_intrinsic "llvm.fabs.f32"'
                           f'({operand}) : (f32) -> f32')
    scale = constant(-0.7213475204444817)
    exponent = instruction(f"arith.mulf {absolute}, {scale} : f32")
    h = instruction('llvm.call_intrinsic "llvm.nvvm.ex2.approx.f"'
                    f'({exponent}) : (f32) -> f32')
    one, zero = constant(1.0), constant(0.0)
    squared = instruction(f"arith.mulf {h}, {h} : f32")
    denominator = instruction(f"arith.addf {one}, {squared} : f32")
    limit = constant(256.0)
    bounded = instruction('llvm.call_intrinsic "llvm.minimum.f32"'
                          f'({absolute}, {limit}) : (f32, f32) -> f32')
    negative_scale = instruction(f"arith.negf {bounded} : f32")
    tail = instruction(f"arith.mulf {negative_scale}, {h} : f32")
    tail = instruction(f"arith.mulf {tail}, {h} : f32")
    positive = instruction(f"arith.divf {operand}, {denominator} : f32")
    negative = instruction(f"arith.divf {tail}, {denominator} : f32")
    condition = instruction(f"arith.cmpf oge, {operand}, {zero} : f32")
    return instruction(f"arith.select {condition}, {positive}, {negative} : f32")


def entry_name(operation):
    return f"tensorcx_{operation}_v1"


def signature(ir):
    def reject(message):
        raise KernelCompileError("MLIR CUDA expression: " + message)

    # Bound recursion before the shared validator/emitter traverses direct IR.
    if len(ir.body) != 2 or not isinstance(ir.body[1], IRIf):
        reject("requires one canonical index and guard")
    body = ir.body[1].body
    if not body or len(body) > 33 or not isinstance(body[-1], IRStore):
        reject("requires at most 32 locals and one final store")
    output = body[-1].buffer
    locals_, loads = set(), set()
    operators = 0

    def expression(value, depth=1):
        nonlocal operators
        if depth > 16:
            reject("expression depth exceeds 16")
        if isinstance(value, IRLoad):
            if value.buffer == output:
                reject("output reads are unsupported")
            loads.add(value.buffer)
        elif isinstance(value, IRName):
            if value.name not in locals_:
                reject("undefined local")
        elif isinstance(value, IRConstant):
            if type(value.value) is not float:
                reject("arithmetic literals must be float32")
            try:
                format_f32_constant(value.value)
            except ValueError as error:
                reject(str(error))
        elif ((isinstance(value, IRBinaryOp) and value.op in OPERATIONS)
              or (isinstance(value, IRCompare) and value.op in PREDICATES)
              or (isinstance(value, IRCall)
                  and ((value.name == "select" and len(value.args) == 3)
                       or (value.name in {"minimum", "maximum"} and len(value.args) == 2)
                       or (value.name in {"sqrt", "abs", "exp", "log", "rsqrt", "tanh", "sigmoid", "silu"} and len(value.args) == 1)))):
            operators += 1
            if operators > 64:
                reject("expression operation count exceeds 64")
            children = value.args if isinstance(value, IRCall) else (value.lhs, value.rhs)
            for child in children:
                expression(child, depth + 1)
        else:
            reject("unsupported expression")

    index_name = ir.body[0].target if isinstance(ir.body[0], IRAssign) else None
    if index_name in ir.parameters:
        reject("canonical index cannot shadow a parameter")
    for statement in body[:-1]:
        if not isinstance(statement, IRAssign):
            reject("only locals may precede the final store")
        if statement.target in locals_ | set(ir.parameters) | {index_name}:
            reject("local reassignment or shadowing")
        expression(statement.value)
        locals_.add(statement.target)
    expression(body[-1].value)
    # Access indices are also untrusted direct IR; reject them without recursion.
    stack = [statement.value for statement in body]
    while stack:
        value = stack.pop()
        if isinstance(value, IRLoad) and value.index != IRName(index_name):
            reject("loads require the canonical index")
        if isinstance(value, (IRBinaryOp, IRCompare)):
            stack.extend((value.lhs, value.rhs))
        elif isinstance(value, IRCall):
            stack.extend(value.args)
    if body[-1].index != IRName(index_name):
        reject("store requires the canonical index")
    if len(ir.parameters) != 4 or len(set(ir.parameters)) != 4 or len(loads) != 2:
        reject("requires two input parameters, one output and one uint32 bound")
    try:
        kinds, output_index, guard = elementwise_signature(ir)
    except KernelCompileError as error:
        reject(str(error))
    if kinds.count("t") != 3 or kinds.count("u") != 1:
        reject("requires three tensors and one uint32 bound")
    value = body[-1].value
    if (len(body) == 1 and isinstance(value, IRBinaryOp)
            and value.op in {"add", "sub", "mul"}
            and isinstance(value.lhs, IRLoad) and isinstance(value.rhs, IRLoad)):
        return (kinds, output_index, guard, ir.parameters.index(value.lhs.buffer),
                ir.parameters.index(value.rhs.buffer), value.op)
    return kinds, output_index, guard, None, None, "expr"


def manifest(kinds, output, guard, operation="add"):
    return (f"tensorcx.cuda.v1|linux-x86_64|llvm-21.1.8|sm52|ptx78|{operation}-f32-v1|"
            f"{kinds}|{output}|{guard}")


def emit_cuda(ir):
    kinds, output, guard, lhs, rhs, operation = signature(ir)
    if operation == "expr":
        return emit_expression_cuda(ir, kinds, output, guard)
    arguments = ", ".join(f"%p{i}: {'!llvm.ptr' if kind == 't' else 'i32'}"
                          for i, kind in enumerate(kinds))
    return f"""module {{
  gpu.module @device {{
    gpu.func @{entry_name(operation)}({arguments}) kernel {{
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %tid = gpu.thread_id x
      %base = arith.muli %bid, %bdim : index
      %i = arith.addi %base, %tid : index
      %limit = arith.index_castui %p{guard} : i32 to index
      %active = arith.cmpi ult, %i, %limit : index
      scf.if %active {{
        %idx = arith.index_cast %i : index to i64
        %ap = llvm.getelementptr %p{lhs}[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %bp = llvm.getelementptr %p{rhs}[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %op = llvm.getelementptr %p{output}[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %av = llvm.load %ap : !llvm.ptr -> f32
        %bv = llvm.load %bp : !llvm.ptr -> f32
        %v = {OPERATIONS[operation]} %av, %bv : f32
        llvm.store %v, %op : f32, !llvm.ptr
      }}
      gpu.return
    }}
  }}
}}
"""


def emit_expression_cuda(ir, kinds, output, guard):
    arguments = ", ".join(f"%p{i}: {'!llvm.ptr' if kind == 't' else 'i32'}"
                          for i, kind in enumerate(kinds))
    lines = [f"module {{\n  gpu.module @device {{\n    gpu.func @tensorcx_expr_v1({arguments}) kernel {{",
             "      %bid = gpu.block_id x", "      %bdim = gpu.block_dim x",
             "      %tid = gpu.thread_id x", "      %base = arith.muli %bid, %bdim : index",
             "      %i = arith.addi %base, %tid : index",
             f"      %limit = arith.index_castui %p{guard} : i32 to index",
             "      %active = arith.cmpi ult, %i, %limit : index",
             "      scf.if %active {", "        %idx = arith.index_cast %i : index to i64"]
    locals_ = {}
    counter = 0

    def instruction(text):
        nonlocal counter
        name = f"%v{counter}"
        counter += 1
        lines.append(f"        {name} = {text}")
        return name

    def expression(value):
        if isinstance(value, IRName):
            return locals_[value.name]
        if isinstance(value, IRConstant):
            return instruction(f"arith.constant {format_f32_constant(value.value)} : f32")
        if isinstance(value, IRLoad):
            index = ir.parameters.index(value.buffer)
            ptr = instruction(f"llvm.getelementptr %p{index}[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32")
            return instruction(f"llvm.load {ptr} : !llvm.ptr -> f32")
        if isinstance(value, IRCall):
            values = [expression(argument) for argument in value.args]
            if value.name == "silu":
                return _emit_silu(values[0], instruction)
            if value.name == "sigmoid":
                return _emit_sigmoid(values[0], instruction)
            if value.name == "tanh":
                return _emit_tanh(values[0], instruction)
            if value.name in {"minimum", "maximum"}:
                # LLVM expands IEEE minimum/maximum for sm_52, including NaN
                # propagation and zero signs; no libdevice or sm_80 required.
                return instruction(f'llvm.call_intrinsic "llvm.{value.name}.f32"'
                                   f'({values[0]}, {values[1]}) : (f32, f32) -> f32')
            if value.name == "rsqrt":
                # Approximate reciprocal root, with subnormal inputs preserved.
                # Avoid generic GPU lowering's external libdevice dependency.
                return instruction('llvm.call_intrinsic "llvm.nvvm.rsqrt.approx.f"'
                                   f'({values[0]}) : (f32) -> f32')
            if value.name == "log":
                # Non-FTZ base-2 logarithm supports sm_52 subnormal inputs.
                # Round the final conversion to a natural logarithm in f32.
                logarithm = instruction('llvm.call_intrinsic "llvm.nvvm.lg2.approx.f"'
                                        f'({values[0]}) : (f32) -> f32')
                scale = instruction(f"arith.constant {format_f32_constant(0.6931471805599453)} : f32")
                return instruction(f"arith.mulf {logarithm}, {scale} : f32")
            if value.name == "exp":
                # A bounded-error exp implementation without external libdevice.
                # Pin f32 scaling and the non-FTZ sm_52-compatible intrinsic.
                scale = instruction(f"arith.constant {format_f32_constant(1.4426950408889634)} : f32")
                exponent = instruction(f"arith.mulf {values[0]}, {scale} : f32")
                return instruction('llvm.call_intrinsic "llvm.nvvm.ex2.approx.f"'
                                   f'({exponent}) : (f32) -> f32')
            if value.name == "abs":
                # Bypass GPUToNVVM's external libdevice math.absf lowering.
                return instruction('llvm.call_intrinsic "llvm.fabs.f32"'
                                   f'({values[0]}) : (f32) -> f32')
            if value.name == "sqrt":
                # LLVM 21's GPU math.sqrt lowering calls external libdevice.
                # Pin the native RN intrinsic instead: no approximation/FTZ,
                # external library, or change to existing lowering options.
                return instruction('llvm.call_intrinsic "llvm.nvvm.sqrt.rn.f"'
                                   f'({values[0]}) : (f32) -> f32')
            return instruction(f"arith.select {', '.join(values)} : f32")
        lhs, rhs = expression(value.lhs), expression(value.rhs)
        if isinstance(value, IRCompare):
            return instruction(f"arith.cmpf {PREDICATES[value.op]}, {lhs}, {rhs} : f32")
        return instruction(f"{OPERATIONS[value.op]} {lhs}, {rhs} : f32")

    for statement in ir.body[1].body[:-1]:
        locals_[statement.target] = expression(statement.value)
    value = expression(ir.body[1].body[-1].value)
    ptr = instruction(f"llvm.getelementptr %p{output}[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32")
    lines.extend((f"        llvm.store {value}, {ptr} : f32, !llvm.ptr",
                  "      }", "      gpu.return", "    }", "  }", "}", ""))
    return "\n".join(lines)


def extract_ptx(serialized, kinds, operation="add"):
    matches = re.findall(r'assembly = "((?:\\[0-9A-Fa-f]{2}|\\["\\]|[^"\\])*)"', serialized)
    if len(matches) != 1 or serialized.count("assembly =") != 1 or serialized.count("#gpu.object<") != 1:
        raise ValueError("expected one serialized CUDA assembly object")
    ptx = re.sub(r'\\([0-9A-Fa-f]{2}|["\\])',
                 lambda m: chr(int(m[1], 16)) if len(m[1]) == 2 else m[1], matches[0])
    for header in (".version 7.8", ".target sm_52", ".address_size 64"):
        if len(re.findall("^" + re.escape(header) + "$", ptx, re.M)) != 1:
            raise ValueError("CUDA PTX target/header mismatch")
    entries = re.findall(r"\.visible \.entry ([A-Za-z_][A-Za-z_0-9]*)\((.*?)\)", ptx, re.S)
    if len(entries) != 1 or entries[0][0] != entry_name(operation):
        raise ValueError("CUDA entry mismatch")
    params = re.findall(r"\.param\s+\.([a-z0-9]+)([^,]*)", entries[0][1])
    if len(params) != len(kinds):
        raise ValueError("CUDA parameter count mismatch")
    for (width, attributes), kind in zip(params, kinds):
        if width != ("u64" if kind == "t" else "u32") or (kind == "t" and ".ptr" not in attributes):
            raise ValueError("CUDA parameter ABI mismatch")
    if "\x00" in ptx:
        raise ValueError("CUDA PTX contains a null byte")
    return ptx


def lower(ir, *, tools=None):
    kinds, output, guard, _, _, operation = signature(ir)
    source = emit_cuda(ir)
    opt = (toolchain() if tools is None else tools)["mlir-opt"]
    with tempfile.TemporaryDirectory(prefix="tensorcx-cuda-") as directory:
        root = Path(directory)
        original, lowered, serialized = [root / name for name in ("kernel.mlir", "nvvm.mlir", "serialized.mlir")]
        original.write_text(source)
        _run([opt, str(original), *LOWER_FLAGS, "-o", str(lowered)])
        _run([opt, str(lowered), *SERIALIZE_FLAGS, "-o", str(serialized)])
        ptx = extract_ptx(serialized.read_text(), kinds, operation)
    return source, "// " + manifest(kinds, output, guard, operation) + "\n" + ptx


def compile_kernel(ir, *, cache=True):
    kinds, output, guard, _, _, operation = signature(ir)
    if not hasattr(_core, "_cuda_kernel_support"):
        raise RuntimeError("MLIR CUDA requires a CUDA-enabled tensor.cx build")
    _core._cuda_kernel_support()
    source = emit_cuda(ir)
    tools = toolchain()
    options = {"lower": LOWER_FLAGS, "serialize": SERIALIZE_FLAGS}
    description = descriptor("cuda", source, tools=tools, options=options,
                             abi=manifest(kinds, output, guard, operation)) if cache_enabled(cache) else {}
    with artifact_cache(description, enabled=cache) as entry:
        payload = entry.read()
        hit = payload is not None
        if hit:
            ptx = payload.decode("utf-8")
        else:
            _, ptx = lower(ir, tools=tools)
        # Native validation/JIT and device/context ownership remain mandatory.
        module = _core._load_cuda_kernel(ptx, entry_name(operation), kinds, output, guard)
        if not hit:
            entry.store(ptx.encode("utf-8"))
    return CompiledCudaKernel(ir.name, "cuda", ir, source, module, cache_hit=hit)
