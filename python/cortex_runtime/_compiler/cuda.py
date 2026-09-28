"""Optional, device-only MLIR CUDA add compiler for the validated sm_52 slice."""
from __future__ import annotations

from pathlib import Path
import re
import tempfile

from .. import _core
from ..experimental import CompiledCudaKernel, IRBinaryOp, IRLoad, IRStore, KernelCompileError
from .cpu import _signature as elementwise_signature, _run, toolchain

ENTRY = "cortex_add_v1"


def signature(ir):
    try:
        kinds, output, guard = elementwise_signature(ir)
    except KernelCompileError as error:
        raise KernelCompileError("MLIR CUDA requires canonical guarded float32 add") from error
    body = ir.body[1].body
    if len(ir.parameters) != 4 or kinds.count("t") != 3 or kinds.count("u") != 1 or len(body) != 1:
        raise KernelCompileError("MLIR CUDA requires two inputs, one output and one uint32 bound")
    store = body[0]
    if not isinstance(store, IRStore) or not isinstance(store.value, IRBinaryOp) or store.value.op != "add":
        raise KernelCompileError("MLIR CUDA currently supports only one float32 add store")
    lhs, rhs = store.value.lhs, store.value.rhs
    if (not isinstance(lhs, IRLoad) or not isinstance(rhs, IRLoad)
            or len({lhs.buffer, rhs.buffer, store.buffer}) != 3):
        raise KernelCompileError("MLIR CUDA requires two distinct input parameters")
    return kinds, output, guard, ir.parameters.index(lhs.buffer), ir.parameters.index(rhs.buffer)


def manifest(kinds, output, guard):
    return (f"cortex.cuda.v1|linux-x86_64|llvm-21.1.8|sm52|ptx78|add-f32-v1|"
            f"{kinds}|{output}|{guard}")


def emit_cuda(ir):
    kinds, output, guard, lhs, rhs = signature(ir)
    arguments = ", ".join(f"%p{i}: {'!llvm.ptr' if kind == 't' else 'i32'}"
                          for i, kind in enumerate(kinds))
    return f"""module {{
  gpu.module @device {{
    gpu.func @{ENTRY}({arguments}) kernel {{
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
        %v = arith.addf %av, %bv : f32
        llvm.store %v, %op : f32, !llvm.ptr
      }}
      gpu.return
    }}
  }}
}}
"""


def extract_ptx(serialized, kinds):
    matches = re.findall(r'assembly = "((?:\\[0-9A-Fa-f]{2}|\\["\\]|[^"\\])*)"', serialized)
    if len(matches) != 1 or serialized.count("assembly =") != 1 or serialized.count("#gpu.object<") != 1:
        raise ValueError("expected one serialized CUDA assembly object")
    ptx = re.sub(r'\\([0-9A-Fa-f]{2}|["\\])',
                 lambda m: chr(int(m[1], 16)) if len(m[1]) == 2 else m[1], matches[0])
    for header in (".version 7.8", ".target sm_52", ".address_size 64"):
        if len(re.findall("^" + re.escape(header) + "$", ptx, re.M)) != 1:
            raise ValueError("CUDA PTX target/header mismatch")
    entries = re.findall(r"\.visible \.entry ([A-Za-z_][A-Za-z_0-9]*)\((.*?)\)", ptx, re.S)
    if len(entries) != 1 or entries[0][0] != ENTRY:
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
    kinds, output, guard, _, _ = signature(ir)
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
        ptx = extract_ptx(serialized.read_text(), kinds)
    return source, "// " + manifest(kinds, output, guard) + "\n" + ptx


def compile_kernel(ir):
    kinds, output, guard, _, _ = signature(ir)
    if not hasattr(_core, "_cuda_kernel_support"):
        raise RuntimeError("MLIR CUDA requires a CUDA-enabled Cortex build")
    _core._cuda_kernel_support()
    source, ptx = lower(ir)
    module = _core._load_cuda_kernel(ptx, ENTRY, kinds, output, guard)
    return CompiledCudaKernel(ir.name, "cuda", ir, source, module)
