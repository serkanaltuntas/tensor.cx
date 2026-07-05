"""Phase 10 tests: Cortex IR -> MLIR emission and end-to-end lowering.

The emitter tests need no MLIR toolchain (text generation only) and run in
CPU CI. The end-to-end test lowers through mlir-opt/mlir-translate/clang and
executes the native code; it skips cleanly when the toolchain is unavailable,
matching the Metal-test convention.

Setting ``CORTEX_REQUIRE_MLIR=1`` turns that clean skip into a hard failure, so
the dedicated CI job (which installs LLVM/MLIR) actually exercises the lowering
evidence instead of silently green-skipping when the toolchain install breaks.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import cortex_runtime as cx
from cortex_runtime.experimental import (
    IRAssign,
    IRBinaryOp,
    IRCompare,
    IRConstant,
    IRKernel,
    IRName,
    IRStore,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "mlir"

sys.path.insert(0, str(EXPERIMENT_DIR))
try:
    from cortex_ir_to_mlir import (  # noqa: E402
        MlirEmitError,
        emit_mlir,
        format_f32_constant,
    )
    from lower_add import find_llvm_bin  # noqa: E402
finally:
    sys.path.remove(str(EXPERIMENT_DIR))


def _mlir_toolchain_or_skip():
    """Return the LLVM/MLIR bin dir, or skip when it is unavailable.

    When ``CORTEX_REQUIRE_MLIR`` is set (the dedicated CI job sets it after
    installing the toolchain), a missing toolchain is a hard failure instead of
    a skip, so a broken toolchain install cannot green-skip the lowering
    evidence.
    """
    llvm_bin = find_llvm_bin()
    if llvm_bin is None:
        message = (
            "MLIR toolchain (mlir-opt/mlir-translate/clang) is unavailable"
        )
        if os.environ.get("CORTEX_REQUIRE_MLIR"):
            pytest.fail(f"CORTEX_REQUIRE_MLIR is set but {message}")
        pytest.skip(message)
    return llvm_bin


# The require-mode conversion below is the load-bearing guard for the dedicated
# `mlir-lowering` CI job (it turns a missing toolchain from a green-skip into a
# hard failure). These tests need no toolchain and run everywhere, so the guard
# itself cannot silently regress.


def test_toolchain_helper_returns_bin_when_present(monkeypatch, tmp_path):
    monkeypatch.setattr(sys.modules[__name__], "find_llvm_bin", lambda: tmp_path)
    assert _mlir_toolchain_or_skip() == tmp_path


def test_toolchain_helper_skips_when_missing_and_not_required(monkeypatch):
    monkeypatch.delenv("CORTEX_REQUIRE_MLIR", raising=False)
    monkeypatch.setattr(sys.modules[__name__], "find_llvm_bin", lambda: None)
    with pytest.raises(pytest.skip.Exception):
        _mlir_toolchain_or_skip()


def test_toolchain_helper_fails_when_missing_and_required(monkeypatch):
    monkeypatch.setenv("CORTEX_REQUIRE_MLIR", "1")
    monkeypatch.setattr(sys.modules[__name__], "find_llvm_bin", lambda: None)
    with pytest.raises(pytest.fail.Exception):
        _mlir_toolchain_or_skip()


def _add_kernel():
    @cx.experimental.kernel
    def add_kernel(a, b, out, n):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        if i < n:
            out[i] = a[i] + b[i]

    return add_kernel


def test_emit_mlir_structure_for_add_kernel():
    mlir_text = emit_mlir(_add_kernel().parse_ir())

    # Buffers become dynamic memrefs, scalars i32, plus the two harness args.
    assert (
        "func.func @add_kernel(%a: memref<?xf32>, %b: memref<?xf32>, "
        "%out: memref<?xf32>, %n: i32, %cortex_thread_count: i32, "
        "%cortex_block_size: i32)" in mlir_text
    )
    assert "attributes { llvm.emit_c_interface }" in mlir_text
    # The Metal grid maps to one scf.for over the global thread index.
    assert "scf.for %cortex_gi" in mlir_text
    assert "arith.divui %cortex_gi_i32, %cortex_block_size" in mlir_text
    assert "arith.remui %cortex_gi_i32, %cortex_block_size" in mlir_text
    # Guard comparison is unsigned, matching the MSL uint semantics.
    assert "arith.cmpi ult," in mlir_text
    assert "scf.if" in mlir_text
    assert "memref.load %a[" in mlir_text
    assert "memref.load %b[" in mlir_text
    assert "arith.addf" in mlir_text
    assert "memref.store" in mlir_text
    # Nothing Metal-specific may leak into the MLIR path.
    assert "metal" not in mlir_text.lower()


def test_emit_mlir_covers_float_sub_and_mul():
    @cx.experimental.kernel
    def sub_mul_kernel(a, b, out, n):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        if i < n:
            out[i] = (a[i] - b[i]) * a[i]

    mlir_text = emit_mlir(sub_mul_kernel.parse_ir())
    assert "arith.subf" in mlir_text
    assert "arith.mulf" in mlir_text


def test_emit_mlir_formats_float_constants_as_valid_mlir():
    # Python repr of 1e-5 is '1e-05', which MLIR's float grammar rejects; the
    # emitter must narrow to f32 and always include a decimal point.
    @cx.experimental.kernel
    def offset_kernel(a, out, n):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        if i < n:
            out[i] = a[i] + 1e-5

    mlir_text = emit_mlir(offset_kernel.parse_ir())
    constant_lines = [
        line for line in mlir_text.splitlines() if ": f32" in line and "arith.constant" in line
    ]
    assert len(constant_lines) == 1
    literal = constant_lines[0].split("arith.constant ")[1].split(" : f32")[0]
    mantissa = literal.partition("e")[0]
    assert "." in mantissa, f"float literal lacks a decimal point: {literal}"
    # The emitted text must parse back to exactly the float32 narrowing of 1e-5.
    import struct

    assert float(literal) == struct.unpack("f", struct.pack("f", 1e-5))[0]


def test_format_f32_constant_edge_cases():
    assert format_f32_constant(0.5) == "0.5"
    assert "." in format_f32_constant(1e20).partition("e")[0]
    with pytest.raises(MlirEmitError, match="non-finite"):
        format_f32_constant(float("inf"))
    with pytest.raises(MlirEmitError, match="non-finite"):
        format_f32_constant(float("nan"))
    with pytest.raises(MlirEmitError, match="out of range for float32"):
        format_f32_constant(1e300)


def test_emit_mlir_rejects_signed_ordered_comparisons():
    # MSL types negative constants as signed int and compiles int-vs-int
    # ordered compares as SIGNED; the MLIR prototype only emits unsigned
    # predicates, so it must refuse instead of silently diverging.
    @cx.experimental.kernel
    def signed_compare_kernel(a, out, n):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        k = -5
        if k < n:
            if i < n:
                out[i] = a[i] + a[i]

    with pytest.raises(MlirEmitError, match="signed compares"):
        emit_mlir(signed_compare_kernel.parse_ir())


def test_emit_mlir_rejects_mixed_type_arithmetic():
    @cx.experimental.kernel
    def mixed_kernel(a, out, n):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        if i < n:
            out[i] = a[i] + n

    with pytest.raises(MlirEmitError, match="mixed integer/float arithmetic"):
        emit_mlir(mixed_kernel.parse_ir())


def test_emit_mlir_rejects_reserved_parameter_names():
    @cx.experimental.kernel
    def clashing_kernel(a, out, cortex_thread_count):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        if i < cortex_thread_count:
            out[i] = a[i] + a[i]

    with pytest.raises(MlirEmitError, match="reserved by the MLIR harness"):
        emit_mlir(clashing_kernel.parse_ir())


def test_emit_mlir_requires_an_output_store():
    kernel_ir = IRKernel(
        name="no_store",
        parameters=("out",),
        body=(IRAssign(target="x", value=IRConstant(1)),),
    )
    with pytest.raises(MlirEmitError, match="requires one output buffer"):
        emit_mlir(kernel_ir)


def test_emit_mlir_rejects_nonzero_program_id_axis():
    # The Python parser already enforces program_id(0); hand-built IR must not
    # silently emit axis-0 code for another axis.
    from cortex_runtime.experimental import IRCall

    kernel_ir = IRKernel(
        name="axis1",
        parameters=("out",),
        body=(
            IRStore(
                buffer="out",
                index=IRCall(name="program_id", args=(IRConstant(1),)),
                value=IRConstant(1.0),
            ),
        ),
    )
    with pytest.raises(MlirEmitError, match="only program_id\\(0\\)"):
        emit_mlir(kernel_ir)


def test_emit_mlir_rejects_bool_results_in_arithmetic():
    kernel_ir = IRKernel(
        name="bool_math",
        parameters=("out",),
        body=(
            IRAssign(
                target="flag",
                value=IRCompare(op="lt", lhs=IRConstant(1), rhs=IRConstant(2)),
            ),
            IRStore(
                buffer="out",
                index=IRBinaryOp(
                    op="add", lhs=IRName("flag"), rhs=IRName("flag")
                ),
                value=IRConstant(1.0),
            ),
        ),
    )
    with pytest.raises(MlirEmitError, match="cannot be used in arithmetic"):
        emit_mlir(kernel_ir)


def test_emitted_float_constant_kernel_parses_with_mlir_opt(tmp_path):
    # Validation-only lowering check for float constants (no execution): the
    # emitted text must be accepted by mlir-opt through the same pass pipeline
    # the end-to-end prototype uses.
    llvm_bin = _mlir_toolchain_or_skip()
    from lower_add import LOWERING_PASSES

    @cx.experimental.kernel
    def offset_kernel(a, out, n):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        if i < n:
            out[i] = a[i] + 1e-5

    mlir_path = tmp_path / "offset.mlir"
    mlir_path.write_text(emit_mlir(offset_kernel.parse_ir()), encoding="utf-8")
    result = subprocess.run(
        [str(llvm_bin / "mlir-opt"), str(mlir_path), *LOWERING_PASSES, "-o", "-"],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_end_to_end_mlir_lowering_matches_cpu_reference():
    _mlir_toolchain_or_skip()
    result = subprocess.run(
        [sys.executable, str(EXPERIMENT_DIR / "lower_add.py")],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"lowering prototype failed\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "PHASE10-PROTOTYPE-OK" in result.stdout


def _rowsum_kernel():
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


def test_emit_mlir_for_loop_uses_scf_for_with_iter_args():
    mlir_text = emit_mlir(_rowsum_kernel().parse_ir())
    assert "scf.for" in mlir_text
    assert "iter_args(" in mlir_text
    assert "-> (f32)" in mlir_text
    assert "scf.yield" in mlir_text
    # The loop bound comes from the scalar parameter m.
    assert "arith.index_castui %m : i32 to index" in mlir_text


def test_end_to_end_mlir_rowsum_matches_cpu_reference():
    _mlir_toolchain_or_skip()
    result = subprocess.run(
        [sys.executable, str(EXPERIMENT_DIR / "lower_rowsum.py")],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"rowsum lowering failed\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "PHASE7-MLIR-ROWSUM-OK" in result.stdout
