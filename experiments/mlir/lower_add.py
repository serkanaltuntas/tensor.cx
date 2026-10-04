"""Phase 10 end-to-end prototype: tensor.cx IR -> MLIR -> native code -> CPU check.

Pipeline:

    @cx.experimental.kernel add
      -> kernel.parse_ir()                     (existing Phase 7 frontend)
      -> emit_mlir()                           (tensorcx_ir_to_mlir.py)
      -> mlir-opt   (scf/arith/memref -> LLVM dialect)
      -> mlir-translate --mlir-to-llvmir       (LLVM IR text)
      -> clang -shared                         (native dylib/so)
      -> ctypes call through _mlir_ciface_*    (memref descriptors)
      -> compare against the tensor.cx CPU reference (cx.tensor add)

Run: `uv run python experiments/mlir/lower_add.py`
Exits 0 and prints PASS lines on success; raises on any mismatch.

Toolchain discovery order: $TENSORCX_LLVM_BIN, the Homebrew llvm@21 keg, PATH.
This stays a research script: nothing under python/tensorcx/ or
cpp/tensorcx/ imports or links MLIR.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tensorcx_ir_to_mlir import emit_mlir  # noqa: E402

import tensorcx as cx  # noqa: E402

HOMEBREW_LLVM_BIN = "/opt/homebrew/opt/llvm@21/bin"
REQUIRED_TOOLS = ("mlir-opt", "mlir-translate", "clang")
LOWERING_PASSES = (
    "--convert-scf-to-cf",
    "--convert-to-llvm",
    "--reconcile-unrealized-casts",
)


def find_llvm_bin() -> Path | None:
    """Locate a directory containing mlir-opt, mlir-translate, and clang.

    All three tools must come from the same directory so the MLIR/LLVM/clang
    versions match.
    """
    candidates = []
    env_bin = os.environ.get("TENSORCX_LLVM_BIN")
    if env_bin:
        env_candidate = Path(env_bin)
        if not all((env_candidate / tool).is_file() for tool in REQUIRED_TOOLS):
            print(
                f"warning: TENSORCX_LLVM_BIN={env_bin} does not contain "
                f"{'/'.join(REQUIRED_TOOLS)}; falling back to auto-discovery",
                file=sys.stderr,
            )
        candidates.append(env_candidate)
    candidates.append(Path(HOMEBREW_LLVM_BIN))
    which = shutil.which("mlir-opt")
    if which:
        candidates.append(Path(which).parent)
    for candidate in candidates:
        if all((candidate / tool).is_file() for tool in REQUIRED_TOOLS):
            return candidate
    return None


def _run_tool(command: list[str]) -> None:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"{Path(command[0]).name} failed: {message}")


def compile_mlir_to_library(mlir_text: str, llvm_bin: Path, work_dir: Path) -> Path:
    """Lower MLIR text to a native shared library; return the library path."""
    mlir_path = work_dir / "kernel.mlir"
    lowered_path = work_dir / "kernel_llvm.mlir"
    llvm_ir_path = work_dir / "kernel.ll"
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    library_path = work_dir / f"kernel{suffix}"

    mlir_path.write_text(mlir_text, encoding="utf-8")
    _run_tool(
        [str(llvm_bin / "mlir-opt"), str(mlir_path), *LOWERING_PASSES,
         "-o", str(lowered_path)]
    )
    _run_tool(
        [str(llvm_bin / "mlir-translate"), "--mlir-to-llvmir",
         str(lowered_path), "-o", str(llvm_ir_path)]
    )
    clang_command = [
        str(llvm_bin / "clang"), "-O2", "-shared", "-Wno-override-module",
        str(llvm_ir_path), "-o", str(library_path),
    ]
    if sys.platform != "darwin":
        # Shared objects on Linux need position-independent code. Untested in
        # this repo's CI (no MLIR toolchain there); validated on macOS arm64.
        clang_command.insert(1, "-fPIC")
    _run_tool(clang_command)
    return library_path


class MemRef1D(ctypes.Structure):
    """Rank-1 MLIR memref descriptor for the C interface ABI."""

    _fields_ = [
        ("allocated", ctypes.POINTER(ctypes.c_float)),
        ("aligned", ctypes.POINTER(ctypes.c_float)),
        ("offset", ctypes.c_int64),
        ("size", ctypes.c_int64),
        ("stride", ctypes.c_int64),
    ]


def _as_memref(array: np.ndarray) -> MemRef1D:
    if array.dtype != np.float32 or array.ndim != 1 or not array.flags.c_contiguous:
        raise ValueError("memref bridge expects contiguous 1-D float32 arrays")
    pointer = array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    return MemRef1D(pointer, pointer, 0, array.size, 1)


def run_add_kernel(
    library_path: Path,
    function_name: str,
    a: np.ndarray,
    b: np.ndarray,
    out: np.ndarray,
    n: int,
    thread_count: int,
    block_size: int,
) -> None:
    library = ctypes.CDLL(str(library_path))
    function = getattr(library, f"_mlir_ciface_{function_name}")
    function.restype = None
    function.argtypes = [ctypes.POINTER(MemRef1D)] * 3 + [ctypes.c_int32] * 3
    a_descriptor, b_descriptor, out_descriptor = map(_as_memref, (a, b, out))
    function(
        ctypes.byref(a_descriptor),
        ctypes.byref(b_descriptor),
        ctypes.byref(out_descriptor),
        ctypes.c_int32(n),
        ctypes.c_int32(thread_count),
        ctypes.c_int32(block_size),
    )


def build_add_kernel():
    @cx.experimental.kernel
    def add_kernel(a, b, out, n):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        if i < n:
            out[i] = a[i] + b[i]

    return add_kernel


def main() -> int:
    llvm_bin = find_llvm_bin()
    if llvm_bin is None:
        print(
            "SKIP: MLIR toolchain not found (set TENSORCX_LLVM_BIN or install "
            "Homebrew llvm@21)"
        )
        return 77

    kernel = build_add_kernel()
    kernel_ir = kernel.parse_ir()
    mlir_text = emit_mlir(kernel_ir)
    print(f"emitted MLIR for kernel '{kernel_ir.name}' "
          f"({len(mlir_text.splitlines())} lines)")

    size = 4096
    block_size = 64
    rng = np.random.default_rng(7)
    a = rng.standard_normal(size).astype(np.float32)
    b = rng.standard_normal(size).astype(np.float32)

    # tensor.cx CPU reference (the mandatory comparison baseline, PROJECT.md §5.4).
    reference = (
        cx.tensor(a, dtype=cx.float32, device="cpu")
        + cx.tensor(b, dtype=cx.float32, device="cpu")
    ).numpy()

    with tempfile.TemporaryDirectory(prefix="tensorcx_mlir_phase10_") as temp_dir:
        library_path = compile_mlir_to_library(mlir_text, llvm_bin, Path(temp_dir))
        print(f"compiled native library: {library_path.name}")

        out = np.zeros(size, dtype=np.float32)
        run_add_kernel(
            library_path, kernel_ir.name, a, b, out,
            n=size, thread_count=size, block_size=block_size,
        )
        np.testing.assert_allclose(out, reference, rtol=1e-6, atol=1e-6)
        print(f"PASS: MLIR-lowered add matches tensor.cx CPU reference (n={size})")

        # Guard semantics: threads with i >= n must not write.
        half = size // 2
        guarded = np.full(size, -1.0, dtype=np.float32)
        run_add_kernel(
            library_path, kernel_ir.name, a, b, guarded,
            n=half, thread_count=size, block_size=block_size,
        )
        np.testing.assert_allclose(
            guarded[:half], reference[:half], rtol=1e-6, atol=1e-6
        )
        if not np.all(guarded[half:] == -1.0):
            raise AssertionError("guard violated: out-of-bounds threads wrote output")
        print(f"PASS: guard semantics preserved (i < n honored for n={half})")

        # Zero-thread launch must be a no-op.
        untouched = np.full(size, -2.0, dtype=np.float32)
        run_add_kernel(
            library_path, kernel_ir.name, a, b, untouched,
            n=0, thread_count=0, block_size=block_size,
        )
        if not np.all(untouched == -2.0):
            raise AssertionError("zero-thread launch wrote output")
        print("PASS: zero-thread launch is a no-op")

    print("PHASE10-PROTOTYPE-OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
