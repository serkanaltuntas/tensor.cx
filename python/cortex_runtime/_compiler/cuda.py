"""Optional, device-only MLIR CUDA elementwise compiler for the validated sm_52 slice."""
from __future__ import annotations

from pathlib import Path
import re
import tempfile

from .. import _core
from ..experimental import CompiledCudaKernel, IRAssign, IRBinaryOp, IRConstant, IRIf, IRLoad, IRName, IRStore, KernelCompileError
from .cpu import _signature as elementwise_signature, _run, toolchain
from .emitter import format_f32_constant

ENTRY = "cortex_add_v1"
OPERATIONS = {"add": "arith.addf", "sub": "arith.subf", "mul": "arith.mulf"}


def entry_name(operation):
    return f"cortex_{operation}_v1"


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
                reject("undefined float32 local")
        elif isinstance(value, IRConstant):
            if type(value.value) is not float:
                reject("arithmetic literals must be float32")
            try:
                format_f32_constant(value.value)
            except ValueError as error:
                reject(str(error))
        elif isinstance(value, IRBinaryOp) and value.op in OPERATIONS:
            operators += 1
            if operators > 64:
                reject("binary operation count exceeds 64")
            expression(value.lhs, depth + 1)
            expression(value.rhs, depth + 1)
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
        if isinstance(value, IRBinaryOp):
            stack.extend((value.lhs, value.rhs))
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
            and isinstance(value.lhs, IRLoad) and isinstance(value.rhs, IRLoad)):
        return (kinds, output_index, guard, ir.parameters.index(value.lhs.buffer),
                ir.parameters.index(value.rhs.buffer), value.op)
    return kinds, output_index, guard, None, None, "expr"


def manifest(kinds, output, guard, operation="add"):
    return (f"cortex.cuda.v1|linux-x86_64|llvm-21.1.8|sm52|ptx78|{operation}-f32-v1|"
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
    lines = [f"module {{\n  gpu.module @device {{\n    gpu.func @cortex_expr_v1({arguments}) kernel {{",
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
        lhs, rhs = expression(value.lhs), expression(value.rhs)
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


def lower(ir):
    kinds, output, guard, _, _, operation = signature(ir)
    source = emit_cuda(ir)
    opt = toolchain()["mlir-opt"]
    with tempfile.TemporaryDirectory(prefix="cortex-cuda-") as directory:
        root = Path(directory)
        original, lowered, serialized = [root / name for name in ("kernel.mlir", "nvvm.mlir", "serialized.mlir")]
        original.write_text(source)
        _run([opt, str(original), "--convert-scf-to-cf", "--convert-gpu-to-nvvm=index-bitwidth=64",
              "--reconcile-unrealized-casts", "-o", str(lowered)])
        _run([opt, str(lowered), "--nvvm-attach-target=chip=sm_52 features=+ptx78 O=2",
              "--gpu-module-to-binary=format=assembly", "-o", str(serialized)])
        ptx = extract_ptx(serialized.read_text(), kinds, operation)
    return source, "// " + manifest(kinds, output, guard, operation) + "\n" + ptx


def compile_kernel(ir):
    kinds, output, guard, _, _, operation = signature(ir)
    if not hasattr(_core, "_cuda_kernel_support"):
        raise RuntimeError("MLIR CUDA requires a CUDA-enabled Cortex build")
    _core._cuda_kernel_support()
    source, ptx = lower(ir)
    module = _core._load_cuda_kernel(ptx, entry_name(operation), kinds, output, guard)
    return CompiledCudaKernel(ir.name, "cuda", ir, source, module)
