"""Explicit CPU compiler adapter. LLVM is discovered only when compiling."""
from __future__ import annotations

from contextlib import ExitStack
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

from .. import _core
from ..experimental import (
    CompiledCpuKernel, IRAssign, IRBinaryOp, IRCall, IRCompare, IRConstant,
    IRIf, IRLoad, IRName, IRStore, KernelCompileError, _emit_msl,
    _iter_buffer_references, _single_output_parameter,
)
from .emitter import emit_mlir
from .cache import artifact_cache, cache_enabled, descriptor, _dependencies

LLVM_VERSION = "21.1.8"
COMMAND_TIMEOUT = 60
LINK_QUERY_WORKERS = 4
OPT_FLAGS = ["--convert-scf-to-cf", "--convert-to-llvm", "--reconcile-unrealized-casts"]
TRANSLATE_FLAGS = ["--mlir-to-llvmir"]
LIBM_OPERATIONS = ("exp", "log", "tanh")
CLANG_FLAGS = ["-O2", "-fPIC", "-shared", "-ffp-contract=off",
               "-Wno-override-module", "-Wl,-Bsymbolic"]


def _run(command: list[str], *, include_stderr=False) -> str:
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
    return result.stdout + (result.stderr if include_stderr else "")


def _run_parallel(commands: list[list[str]]) -> list[str]:
    """Run read-only identity queries in bounded groups, preserving result order.

    Each group shares one deadline. Kill every live child before closing its
    context, which then reaps it even on spawn failure, timeout or interruption.
    """
    outputs = []
    for start in range(0, len(commands), LINK_QUERY_WORKERS):
        batch = commands[start:start + LINK_QUERY_WORKERS]
        processes = []
        with ExitStack() as resources:
            try:
                deadline = time.monotonic() + COMMAND_TIMEOUT
                for command in batch:
                    process = resources.enter_context(subprocess.Popen(
                        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
                    processes.append(process)
                for command, process in zip(batch, processes, strict=True):
                    stdout, stderr = process.communicate(timeout=max(0, deadline - time.monotonic()))
                    if process.returncode:
                        raise RuntimeError(
                            f"MLIR tool failed ({process.returncode}): {command[0]}\n{stderr[-4000:]}"
                        )
                    outputs.append(stdout)
            except (OSError, subprocess.TimeoutExpired) as error:
                raise RuntimeError(f"MLIR tool failed: {command[0]}: {error}") from error
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                # Popen.__exit__ only waits briefly on KeyboardInterrupt. Reap
                # explicitly before propagating it through ExitStack.
                for process in processes:
                    process.wait()
    return outputs


def toolchain(*, names=("mlir-opt", "mlir-translate", "clang")) -> dict[str, str]:
    explicit = os.environ.get("TENSORCX_LLVM_BIN")
    if explicit == "":
        raise RuntimeError("TENSORCX_LLVM_BIN must name an LLVM directory")
    tools = {}
    for name in names:
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
            "float32 elementwise body (arithmetic, sqrt/abs/exp/log/rsqrt/tanh/sigmoid/silu, minimum/maximum, comparisons, select and local assignments)"
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
    locals_ = {}

    def expression(value):
        if isinstance(value, IRLoad):
            if value.index != IRName(index.target) or value.buffer not in ir.parameters:
                reject()
            return "float"
        elif isinstance(value, IRConstant):
            if type(value.value) is not float:
                reject()
            return "float"
        elif isinstance(value, IRName):
            if value.name not in locals_:
                reject()
            return locals_[value.name]
        elif isinstance(value, IRBinaryOp) and value.op in {"add", "sub", "mul", "div"}:
            if (expression(value.lhs), expression(value.rhs)) != ("float", "float"):
                reject()
            return "float"
        elif isinstance(value, IRCompare) and value.op in {"lt", "lte", "gt", "gte", "eq", "neq"}:
            if (expression(value.lhs), expression(value.rhs)) != ("float", "float"):
                reject()
            return "bool"
        elif isinstance(value, IRCall) and value.name in {"minimum", "maximum"} and len(value.args) == 2:
            if any(expression(argument) != "float" for argument in value.args):
                reject()
            return "float"
        elif isinstance(value, IRCall) and value.name in {"sqrt", "abs", "exp", "log", "rsqrt", "tanh", "sigmoid", "silu"} and len(value.args) == 1:
            if expression(value.args[0]) != "float":
                reject()
            return "float"
        elif isinstance(value, IRCall) and value.name == "select" and len(value.args) == 3:
            if tuple(expression(argument) for argument in value.args) != ("bool", "float", "float"):
                reject()
            return "float"
        else:
            reject()

    for statement in guard.body:
        if isinstance(statement, IRAssign):
            if statement.target == index.target or statement.target in ir.parameters:
                reject()
            locals_[statement.target] = expression(statement.value)
        elif isinstance(statement, IRStore):
            if statement.buffer != output or statement.index != IRName(index.target):
                reject()
            if expression(statement.value) != "float":
                reject()
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



def _link_inputs(tools, *, math_library=False):
    names = ("crtbeginS.o", "crtendS.o", "crti.o", "crtn.o", "libgcc.a", "libgcc_s.so", "libc.so")
    if math_library:
        names += ("libm.so", "libm.so.6")
    commands = [[tools["clang"], "-print-prog-name=ld"],
                *[[tools["clang"], "-print-file-name=" + name] for name in names]]
    linker, *paths = (output.strip() for output in _run_parallel(commands))
    if not Path(linker).is_absolute():
        linker = shutil.which(linker)
    if not linker:
        raise RuntimeError("CPU linker identity is unavailable")
    files = [Path(linker), *_dependencies(linker)]
    for name, path in zip(names, paths, strict=True):
        if path == name or not Path(path).is_file():
            raise RuntimeError("CPU link input identity is unavailable")
        files.append(Path(path))
    return files


def compile_kernel(ir, *, cache=True) -> CompiledCpuKernel:
    signature = _signature(ir)
    # LLVM lowers these operations to external libm functions. A kernel with
    # a matching name would capture the call under -Bsymbolic with the wrong ABI.
    # Keep this restriction here: CUDA shares _signature but uses a private entry.
    if ir.name in {name + "f" for name in LIBM_OPERATIONS}:
        raise KernelCompileError(f"MLIR CPU kernel name {ir.name!r} is reserved for libm")
    if not _core._cpu_kernel_supported():
        raise RuntimeError("MLIR CPU runtime requires Linux x86_64")
    source = emit_mlir(ir)
    tools = toolchain()
    adapter_source = _adapter(ir, *signature)
    # Link libm explicitly instead of relying on the embedding process globals.
    math_flags = ["-lm"] if any(f"math.{name} " in source for name in LIBM_OPERATIONS) else []
    link_inputs = (lambda: _link_inputs(tools, math_library=True)) if math_flags else lambda: _link_inputs(tools)
    options = {"opt": OPT_FLAGS, "translate": TRANSLATE_FLAGS, "clang": CLANG_FLAGS + math_flags}
    description = descriptor("cpu", source, tools=tools, options=options,
                             abi=adapter_source, extra_files=link_inputs) if cache_enabled(cache) else {}
    with artifact_cache(description, enabled=cache) as entry, tempfile.TemporaryDirectory(prefix="tensorcx-mlir-") as directory:
        root = Path(directory)
        mlir, lowered, llvm, adapter, library = [
            root / name for name in ("kernel.mlir", "lowered.mlir", "kernel.ll", "adapter.c", "kernel.so")
        ]
        payload = entry.read()
        hit = payload is not None
        if hit:
            library.write_bytes(payload)
        else:
            mlir.write_text(source)
            adapter.write_text(adapter_source)
            _run([tools["mlir-opt"], str(mlir), *OPT_FLAGS, "-o", str(lowered)])
            _run([tools["mlir-translate"], *TRANSLATE_FLAGS, str(lowered), "-o", str(llvm)])
            _run([tools["clang"], *CLANG_FLAGS, str(llvm), str(adapter), *math_flags, "-o", str(library)])
        module = _core._load_cpu_kernel(str(library), *signature)
        if not hit:
            entry.store(library.read_bytes())
    # Linux keeps the mapped image alive after the private scratch files vanish.
    return CompiledCpuKernel(ir.name, "cpu", ir, source, module, cache_hit=hit)
