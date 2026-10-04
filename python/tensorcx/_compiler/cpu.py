"""Explicit CPU compiler adapter. LLVM is discovered only when compiling."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from .. import _core
from ..experimental import (
    CompiledCpuKernel, IRAssign, IRBinaryOp, IRCall, IRCompare, IRConstant,
    IRIf, IRLoad, IRName, IRStore, KernelCompileError, _emit_msl,
    _iter_buffer_references, _single_output_parameter,
)
from .emitter import emit_mlir

LLVM_VERSION = "21.1.8"
COMMAND_TIMEOUT = 60


def _run(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=COMMAND_TIMEOUT, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"MLIR tool failed: {command[0]}: {error}") from error
    if result.returncode:
        raise RuntimeError(
            f"MLIR tool failed ({result.returncode}): {command[0]}\n"
            f"{result.stderr[-4000:]}"
        )
    return result.stdout


def toolchain() -> dict[str, str]:
    explicit = os.environ.get("TENSORCX_LLVM_BIN")
    if explicit == "":
        raise RuntimeError("TENSORCX_LLVM_BIN must name an LLVM directory")
    tools = {}
    for name in ("mlir-opt", "mlir-translate", "clang"):
        path = str(Path(explicit) / name) if explicit is not None else shutil.which(name)
        if not path or not Path(path).is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(f"MLIR requires executable {name}; set TENSORCX_LLVM_BIN to LLVM {LLVM_VERSION}")
        path = str(Path(path).resolve())
        version = _run([path, "--version"])
        if not re.search(r"\bversion\s+" + re.escape(LLVM_VERSION) + r"(?![\d.])", version):
            raise RuntimeError(f"MLIR requires LLVM {LLVM_VERSION}: {name} reported {version[:200]!r}")
        tools[name] = str(Path(path).resolve())
    return tools


def _signature(ir):
    """Prove every memory access uses the guarded canonical global index.

    The research emitter accepts more IR; runtime memory safety deliberately
    requires this smaller, straight-line float32 subset.
    """
    def reject():
        raise KernelCompileError(
            "MLIR CPU runtime requires a canonical global index and one guarded "
            "float32 elementwise body (add/subtract/multiply and local assignments)"
        )

    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", ir.name) or ir.name.startswith("tensorcx_"):
        reject()
    if len(ir.parameters) > 64 or len(ir.body) != 2:
        reject()
    index, guard = ir.body
    canonical = IRBinaryOp(
        "add",
        IRBinaryOp("mul", IRCall("program_id", (IRConstant(0),)),
                   IRCall("block_size", ())),
        IRCall("thread_id", ()),
    )
    if not isinstance(index, IRAssign) or index.value != canonical:
        reject()
    if not isinstance(guard, IRIf) or not isinstance(guard.condition, IRCompare):
        reject()
    condition = guard.condition
    if (condition.op != "lt" or condition.lhs != IRName(index.target)
            or not isinstance(condition.rhs, IRName)
            or condition.rhs.name not in ir.parameters):
        reject()
    guard_name = condition.rhs.name
    buffers = set(_iter_buffer_references(ir.body))
    if guard_name in buffers or set(ir.parameters) != buffers | {guard_name}:
        reject()
    output = _single_output_parameter(ir)
    locals_ = set()

    def expression(value):
        if isinstance(value, IRLoad):
            if value.index != IRName(index.target) or value.buffer not in ir.parameters:
                reject()
        elif isinstance(value, IRConstant):
            if type(value.value) is not float:
                reject()
        elif isinstance(value, IRName):
            if value.name not in locals_:
                reject()
        elif isinstance(value, IRBinaryOp) and value.op in {"add", "sub", "mul"}:
            expression(value.lhs)
            expression(value.rhs)
        else:
            reject()

    for statement in guard.body:
        if isinstance(statement, IRAssign):
            if statement.target == index.target or statement.target in ir.parameters:
                reject()
            expression(statement.value)
            locals_.add(statement.target)
        elif isinstance(statement, IRStore):
            if statement.buffer != output or statement.index != IRName(index.target):
                reject()
            expression(statement.value)
        else:
            reject()
    # Preserve common DSL typing, identifier, and unused-parameter restrictions.
    _emit_msl(ir)
    kinds = "".join("t" if name in buffers else "u" for name in ir.parameters)
    return kinds, ir.parameters.index(output), ir.parameters.index(guard_name)


def _adapter(ir, kinds, output, guard):
    manifest = (f"tensorcx.cpu.v1|linux-x86_64|llvm-{LLVM_VERSION}|elementwise-f32-v1|"
                f"{kinds}|{output}|{guard}")
    types = ["MemRef*" if kind == "t" else "u32" for kind in kinds] + ["u32", "u32"]
    values = [f"(MemRef*)args[{i}]" if kind == "t" else f"*(u32*)args[{i}]"
              for i, kind in enumerate(kinds + "uu")]
    return f"""
typedef __UINT32_TYPE__ u32;
typedef __INT64_TYPE__ i64;
typedef struct {{ float* allocated; float* aligned; i64 offset, size, stride; }} MemRef;
_Static_assert(sizeof(void*) == 8 && sizeof(MemRef) == 40, "unsupported memref ABI");
_Static_assert(__builtin_offsetof(MemRef, stride) == 32, "unsupported memref layout");
extern void _mlir_ciface_{ir.name}({", ".join(types)});
const char* tensorcx_manifest_v1(void) {{ return "{manifest}"; }}
void tensorcx_launch_v1(void** args) {{ _mlir_ciface_{ir.name}({", ".join(values)}); }}
"""


def compile_kernel(ir) -> CompiledCpuKernel:
    signature = _signature(ir)
    if not _core._cpu_kernel_supported():
        raise RuntimeError("MLIR CPU runtime requires Linux x86_64")
    source = emit_mlir(ir)
    tools = toolchain()
    with tempfile.TemporaryDirectory(prefix="tensorcx-mlir-") as directory:
        root = Path(directory)
        mlir, lowered, llvm, adapter, library = [
            root / name for name in ("kernel.mlir", "lowered.mlir", "kernel.ll", "adapter.c", "kernel.so")
        ]
        mlir.write_text(source)
        adapter.write_text(_adapter(ir, *signature))
        _run([tools["mlir-opt"], str(mlir), "--convert-scf-to-cf", "--convert-to-llvm",
              "--reconcile-unrealized-casts", "-o", str(lowered)])
        _run([tools["mlir-translate"], "--mlir-to-llvmir", str(lowered), "-o", str(llvm)])
        _run([tools["clang"], "-O2", "-fPIC", "-shared", "-ffp-contract=off",
              "-Wno-override-module", "-Wl,-Bsymbolic", str(llvm), str(adapter), "-o", str(library)])
        module = _core._load_cpu_kernel(str(library), *signature)
    # Linux keeps the mapped image alive after the private scratch files vanish.
    return CompiledCpuKernel(ir.name, "cpu", ir, source, module)
