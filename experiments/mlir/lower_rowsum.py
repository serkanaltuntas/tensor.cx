"""Phase 7 DSL extension evidence: a REDUCTION lowered through MLIR.

Same pipeline as lower_add.py, but the kernel uses the new bounded
``for k in range(m)`` + accumulator DSL construct, which lowers to
``scf.for`` with loop-carried ``iter_args``. Compared against the Cortex CPU
backend reduction (cx.sum, §12.3 reduction tolerance).

Run: `uv run python experiments/mlir/lower_rowsum.py`
"""

from __future__ import annotations

import ctypes
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cortex_ir_to_mlir import emit_mlir  # noqa: E402
from lower_add import (  # noqa: E402
    MemRef1D,
    _as_memref,
    compile_mlir_to_library,
    find_llvm_bin,
)

import cortex_runtime as cx  # noqa: E402


def build_rowsum_kernel():
    @cx.experimental.kernel
    def rowsum_kernel(a, out, n, m):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        if i < n:
            acc = 0.0
            for k in range(m):
                acc = acc + a[i * m + k]
            out[i] = acc

    return rowsum_kernel


def main() -> int:
    llvm_bin = find_llvm_bin()
    if llvm_bin is None:
        print(
            "SKIP: MLIR toolchain not found (set CORTEX_LLVM_BIN or install "
            "Homebrew llvm@21)"
        )
        return 77

    kernel = build_rowsum_kernel()
    kernel_ir = kernel.parse_ir()
    mlir_text = emit_mlir(kernel_ir)
    print(f"emitted MLIR for kernel '{kernel_ir.name}' "
          f"({len(mlir_text.splitlines())} lines)")

    rows, cols = 64, 128
    block_size = 16
    rng = np.random.default_rng(11)
    matrix = rng.standard_normal((rows, cols)).astype(np.float32)

    reference = cx.sum(
        cx.tensor(matrix, dtype=cx.float32, device="cpu"), axis=1
    ).numpy()

    with tempfile.TemporaryDirectory(prefix="cortex_mlir_rowsum_") as temp_dir:
        library_path = compile_mlir_to_library(mlir_text, llvm_bin, Path(temp_dir))
        library = ctypes.CDLL(str(library_path))
        function = getattr(library, f"_mlir_ciface_{kernel_ir.name}")
        function.restype = None
        function.argtypes = [ctypes.POINTER(MemRef1D)] * 2 + [ctypes.c_int32] * 4

        flat = np.ascontiguousarray(matrix.reshape(-1))
        out = np.zeros(rows, dtype=np.float32)
        a_descriptor, out_descriptor = _as_memref(flat), _as_memref(out)
        function(
            ctypes.byref(a_descriptor),
            ctypes.byref(out_descriptor),
            ctypes.c_int32(rows),
            ctypes.c_int32(cols),
            ctypes.c_int32(rows),        # thread_count
            ctypes.c_int32(block_size),  # block_size
        )
        np.testing.assert_allclose(out, reference, rtol=1e-5, atol=1e-5)
        print(f"PASS: MLIR-lowered rowsum matches Cortex CPU reference "
              f"({rows}x{cols})")

    print("PHASE7-MLIR-ROWSUM-OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
