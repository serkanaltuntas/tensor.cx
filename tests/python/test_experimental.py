import shutil
import subprocess

import numpy as np
import pytest

import cortex_runtime as cx


def _has_metal_compiler() -> bool:
    if shutil.which("xcrun") is None:
        return False
    for tool in ("metal", "metallib"):
        result = subprocess.run(
            ["xcrun", "-sdk", "macosx", "--find", tool],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False
    return True


def test_experimental_kernel_decorator_records_metadata():
    @cx.experimental.kernel
    def add_kernel(a, b, out, n):
        pass

    assert add_kernel.name == "add_kernel"
    assert add_kernel.parameters == ("a", "b", "out", "n")
    assert add_kernel.target == "auto"


def test_experimental_kernel_decorator_accepts_explicit_target():
    @cx.experimental.kernel(target="metal")
    def add_kernel(a, b, out, n):
        pass

    assert add_kernel.target == "metal"


def test_experimental_kernel_compile_rejects_unsupported_target():
    @cx.experimental.kernel
    def add_kernel(a, b, out, n):
        pass

    with pytest.raises(NotImplementedError, match="only implemented"):
        add_kernel.compile(target="cpu")


def test_experimental_kernel_parses_first_elementwise_ir_shape():
    @cx.experimental.kernel
    def add_kernel(a, b, out, n):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        if i < n:
            out[i] = a[i] + b[i]

    ir = add_kernel.parse_ir()

    assert ir.name == "add_kernel"
    assert ir.parameters == ("a", "b", "out", "n")
    assert len(ir.body) == 2

    assign = ir.body[0]
    assert isinstance(assign, cx.experimental.IRAssign)
    assert assign.target == "i"
    assert isinstance(assign.value, cx.experimental.IRBinaryOp)
    assert assign.value.op == "add"
    assert isinstance(assign.value.lhs, cx.experimental.IRBinaryOp)
    assert assign.value.lhs.op == "mul"
    assert isinstance(assign.value.lhs.lhs, cx.experimental.IRCall)
    assert assign.value.lhs.lhs.name == "program_id"
    assert assign.value.lhs.lhs.args == (cx.experimental.IRConstant(0),)
    assert isinstance(assign.value.lhs.rhs, cx.experimental.IRCall)
    assert assign.value.lhs.rhs.name == "block_size"
    assert isinstance(assign.value.rhs, cx.experimental.IRCall)
    assert assign.value.rhs.name == "thread_id"

    branch = ir.body[1]
    assert isinstance(branch, cx.experimental.IRIf)
    assert isinstance(branch.condition, cx.experimental.IRCompare)
    assert branch.condition.op == "lt"
    assert branch.condition.lhs == cx.experimental.IRName("i")
    assert branch.condition.rhs == cx.experimental.IRName("n")
    assert len(branch.body) == 1

    store = branch.body[0]
    assert isinstance(store, cx.experimental.IRStore)
    assert store.buffer == "out"
    assert store.index == cx.experimental.IRName("i")
    assert isinstance(store.value, cx.experimental.IRBinaryOp)
    assert store.value.op == "add"
    assert store.value.lhs == cx.experimental.IRLoad("a", cx.experimental.IRName("i"))
    assert store.value.rhs == cx.experimental.IRLoad("b", cx.experimental.IRName("i"))


def test_experimental_kernel_emits_text_msl_for_first_elementwise_shape():
    @cx.experimental.kernel
    def add_kernel(a, b, out, n):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        if i < n:
            out[i] = a[i] + b[i]

    assert add_kernel.emit_msl() == """#include <metal_stdlib>
using namespace metal;

kernel void add_kernel(
    const device float* a [[buffer(0)]],
    const device float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint3 block_position [[threadgroup_position_in_grid]],
    uint3 local_position [[thread_position_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]]
) {
    uint i = ((block_position.x * group_size.x) + local_position.x);
    if ((i < n)) {
        out[i] = (a[i] + b[i]);
    }
}"""


def _comparison_scaled_kernel():
    @cx.experimental.kernel
    def compare_scale(a, b, out, n):
        i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
        if i < n:
            out[i] = (a[i] < b[i]) * 2.0

    return compare_scale


def test_experimental_msl_preserves_comparison_precedence():
    compare_scale = _comparison_scaled_kernel()
    assert "out[i] = ((a[i] < b[i]) * 2.0f);" in compare_scale.emit_msl()
    a = cx.tensor([1.0, 3.0], device="cpu")
    b = cx.tensor([2.0, 2.0], device="cpu")
    out = cx.zeros((2,), device="cpu")
    np.testing.assert_array_equal(compare_scale.reference(a, b, out, 2).numpy(), [2.0, 0.0])


@pytest.mark.skipif(not _has_metal_compiler(), reason="Apple Metal compiler unavailable")
def test_metal_comparison_expression_compiles_without_device():
    compiled = _comparison_scaled_kernel().compile(target="metal")
    assert compiled.metallib.startswith(b"MTLB")


@pytest.mark.skipif(not _has_metal_compiler(), reason="Apple Metal compiler unavailable")
@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
@pytest.mark.parametrize("reported_length", [0, 1, 2**60])
def test_metal_comparison_and_sequence_argument_lifetime(reported_length):
    from cortex_runtime import _core

    class MisreportedList(list):
        def __len__(self):
            return reported_length

    kernel = _comparison_scaled_kernel()
    compiled = kernel.compile(target="metal")
    a = cx.tensor([1.0, 3.0], device="metal")
    b = cx.tensor([2.0, 2.0], device="metal")
    out = cx.zeros((2,), device="metal")
    reference = kernel.reference(a.cpu(), b.cpu(), out.cpu(), 2)
    _core.launch_metal_library_function(
        compiled.metallib, compiled.name,
        MisreportedList([a._impl, b._impl, out._impl, 2]), out._impl, 2, 2,
    )
    np.testing.assert_array_equal(out.numpy(), reference.numpy())


def test_experimental_kernel_emit_msl_requires_output_store():
    @cx.experimental.kernel
    def bad_kernel(n):
        i = cx.experimental.program_id(0)
        if i < n:
            value = 1

    with pytest.raises(
        cx.experimental.KernelCompileError,
        match="MSL emission requires one output buffer",
    ):
        bad_kernel.emit_msl()


def test_experimental_kernel_emit_msl_keeps_negative_integer_locals_signed():
    @cx.experimental.kernel
    def signed_kernel(out, n):
        i = cx.experimental.program_id(0)
        value = -1
        if i < n:
            out[i] = value

    emitted = signed_kernel.emit_msl()

    assert "int value = -1;" in emitted
    assert "uint value = -1;" not in emitted


def test_experimental_kernel_emit_msl_keeps_negative_integer_expressions_signed():
    @cx.experimental.kernel
    def signed_kernel(out, n):
        i = cx.experimental.program_id(0)
        value = -1 + 0
        if i < n:
            out[i] = value

    emitted = signed_kernel.emit_msl()

    assert "int value = (-1 + 0);" in emitted
    assert "uint value = (-1 + 0);" not in emitted


def test_experimental_kernel_compile_reports_missing_xcrun(monkeypatch):
    @cx.experimental.kernel
    def add_kernel(a, b, out, n):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        if i < n:
            out[i] = a[i] + b[i]

    monkeypatch.setattr(cx.experimental.shutil, "which", lambda name: None)

    with pytest.raises(
        cx.experimental.KernelCompileError,
        match="xcrun was not found",
    ):
        add_kernel.compile(target="metal")


@pytest.mark.skipif(
    not _has_metal_compiler(),
    reason="Apple Metal command-line compiler tools are unavailable",
)
def test_experimental_kernel_compile_returns_in_memory_metallib_artifact():
    @cx.experimental.kernel
    def add_kernel(a, b, out, n):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        if i < n:
            out[i] = a[i] + b[i]

    compiled = add_kernel.compile()
    explicit = add_kernel.compile(target="metal")

    assert isinstance(compiled, cx.experimental.CompiledKernel)
    assert compiled.name == "add_kernel"
    assert compiled.target == "metal"
    assert compiled.ir == add_kernel.parse_ir()
    assert compiled.msl_source == add_kernel.emit_msl()
    assert compiled.metallib.startswith(b"MTLB")
    assert explicit.target == "metal"
    assert explicit.msl_source == compiled.msl_source
    assert explicit.metallib.startswith(b"MTLB")


def test_experimental_compiled_kernel_validate_rejects_invalid_python_inputs():
    compiled = cx.experimental.CompiledKernel(
        name="add_kernel",
        target="metal",
        ir=cx.experimental.IRKernel("add_kernel", (), ()),
        msl_source="",
        metallib=b"",
    )

    with pytest.raises(TypeError, match="function name must be a string"):
        compiled.validate_metal_function(1)
    with pytest.raises(ValueError, match="function name cannot be empty"):
        compiled.validate_metal_function("")
    with pytest.raises(ValueError, match="function name cannot contain null bytes"):
        compiled.validate_metal_function("add_kernel\x00suffix")

    cpu_compiled = cx.experimental.CompiledKernel(
        name="add_kernel",
        target="cpu",
        ir=cx.experimental.IRKernel("add_kernel", (), ()),
        msl_source="",
        metallib=b"",
    )
    with pytest.raises(ValueError, match="requires target 'metal'"):
        cpu_compiled.validate_metal_function()


def test_experimental_compiled_kernel_launch_rejects_invalid_python_inputs():
    @cx.experimental.kernel
    def add_kernel(a, b, out, n):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        if i < n:
            out[i] = a[i] + b[i]

    compiled = cx.experimental.CompiledKernel(
        name="add_kernel",
        target="metal",
        ir=add_kernel.parse_ir(),
        msl_source="",
        metallib=b"",
    )
    cpu_compiled = cx.experimental.CompiledKernel(
        name="add_kernel",
        target="cpu",
        ir=add_kernel.parse_ir(),
        msl_source="",
        metallib=b"",
    )

    with pytest.raises(ValueError, match="requires target 'metal'"):
        cpu_compiled.launch()
    with pytest.raises(ValueError, match="block_size must be a positive uint32"):
        compiled.launch(block_size=0)
    with pytest.raises(ValueError, match="thread_count must be a uint32"):
        compiled.launch(thread_count=-1)
    with pytest.raises(TypeError, match="expects 4 argument"):
        compiled.launch(block_size=32)

    x = cx.ones((2,), dtype=cx.float32, device="cpu")
    with pytest.raises(ValueError, match="requires Metal tensors"):
        compiled.launch(x, x, x, 2, block_size=32)


def test_experimental_compiled_kernel_launch_requires_guarded_stores():
    @cx.experimental.kernel
    def bad_kernel(out):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        out[i] = 1.0

    compiled = cx.experimental.CompiledKernel(
        name="bad_kernel",
        target="metal",
        ir=bad_kernel.parse_ir(),
        msl_source="",
        metallib=b"",
    )

    with pytest.raises(
        cx.experimental.KernelCompileError,
        match="stores to be guarded by index < scalar_limit",
    ):
        compiled.launch()


def test_experimental_compiled_kernel_launch_requires_guarded_loads():
    @cx.experimental.kernel
    def bad_kernel(a, out, n):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        value = a[i]
        if i < n:
            out[i] = value

    compiled = cx.experimental.CompiledKernel(
        name="bad_kernel",
        target="metal",
        ir=bad_kernel.parse_ir(),
        msl_source="",
        metallib=b"",
    )

    with pytest.raises(
        cx.experimental.KernelCompileError,
        match="buffer loads to be guarded by index < scalar_limit",
    ):
        compiled.launch()


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_experimental_compiled_kernel_launch_rejects_tensor_contract_violations():
    @cx.experimental.kernel
    def add_kernel(a, b, out, n):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        if i < n:
            out[i] = a[i] + b[i]

    compiled = cx.experimental.CompiledKernel(
        name="add_kernel",
        target="metal",
        ir=add_kernel.parse_ir(),
        msl_source="",
        metallib=b"",
    )

    x = cx.ones((2,), dtype=cx.float32, device="metal")
    y = cx.ones((3,), dtype=cx.float32, device="metal")
    out = cx.empty((2,), dtype=cx.float32, device="metal")
    with pytest.raises(ValueError, match="tensor shapes must match"):
        compiled.launch(x, y, out, 2, block_size=32)

    xi = cx.ones((2,), dtype=cx.int32, device="metal")
    with pytest.raises(ValueError, match="support float32 tensor buffers"):
        compiled.launch(xi, xi, xi, 2, block_size=32)
    with pytest.raises(ValueError, match="kernel scalar arguments must be uint32"):
        compiled.launch(x, x, out, True, block_size=32)


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_experimental_compiled_kernel_launch_rejects_guard_contract_violations():
    @cx.experimental.kernel
    def add_kernel(a, b, out, n):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        if i < n:
            out[i] = a[i] + b[i]

    compiled = cx.experimental.CompiledKernel(
        name="add_kernel",
        target="metal",
        ir=add_kernel.parse_ir(),
        msl_source="",
        metallib=b"",
    )

    x = cx.ones((2,), dtype=cx.float32, device="metal")
    out = cx.empty((2,), dtype=cx.float32, device="metal")

    with pytest.raises(ValueError, match="kernel guard bound must match thread_count"):
        compiled.launch(x, x, out, 1, thread_count=2, block_size=32)
    with pytest.raises(ValueError, match="thread_count cannot exceed output tensor size"):
        compiled.launch(x, x, out, 3, thread_count=3, block_size=32)


@pytest.mark.skipif(
    not _has_metal_compiler(),
    reason="Apple Metal command-line compiler tools are unavailable",
)
@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_experimental_kernel_launches_add_and_matches_cpu():
    @cx.experimental.kernel(target="metal")
    def add_kernel(a, b, out, n):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        if i < n:
            out[i] = a[i] + b[i]

    x_cpu = cx.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=cx.float32, device="cpu")
    y_cpu = cx.tensor([5.0, 6.0, 7.0, 8.0, 9.0], dtype=cx.float32, device="cpu")
    x = x_cpu.to("metal")
    y = y_cpu.to("metal")
    out = cx.empty(x.shape, dtype=cx.float32, device="metal")

    returned = add_kernel(x, y, out, x.shape[0], block_size=2)

    assert returned is out
    cx.testing.assert_allclose(out.cpu(), x_cpu + y_cpu, kind="elementwise")

    compiled = add_kernel.compile(target="metal")
    out2 = cx.empty(x.shape, dtype=cx.float32, device="metal")

    returned2 = compiled.launch(
        x,
        y,
        out2,
        x.shape[0],
        thread_count=x.shape[0],
        block_size=2,
    )

    assert returned2 is out2
    cx.testing.assert_allclose(out2.cpu(), x_cpu + y_cpu, kind="elementwise")


@pytest.mark.skipif(
    not _has_metal_compiler(),
    reason="Apple Metal command-line compiler tools are unavailable",
)
@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_experimental_kernel_launch_handles_default_block_size_tail():
    @cx.experimental.kernel(target="metal")
    def add_kernel(a, b, out, n):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        if i < n:
            out[i] = a[i] + b[i]

    values = np.arange(257, dtype=np.float32)
    x_cpu = cx.tensor(values, dtype=cx.float32, device="cpu")
    y_cpu = cx.ones(values.shape, dtype=cx.float32, device="cpu")
    x = x_cpu.to("metal")
    y = y_cpu.to("metal")
    out = cx.empty(x.shape, dtype=cx.float32, device="metal")

    add_kernel(x, y, out, x.shape[0])

    cx.testing.assert_allclose(out.cpu(), x_cpu + y_cpu, kind="elementwise")


@pytest.mark.skipif(
    not _has_metal_compiler(),
    reason="Apple Metal command-line compiler tools are unavailable",
)
@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_experimental_kernel_launch_zero_thread_count_is_noop():
    @cx.experimental.kernel(target="metal")
    def add_kernel(a, b, out, n):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        if i < n:
            out[i] = a[i] + b[i]

    x = cx.empty((0,), dtype=cx.float32, device="metal")
    y = cx.empty((0,), dtype=cx.float32, device="metal")
    out = cx.empty((0,), dtype=cx.float32, device="metal")
    compiled = add_kernel.compile(target="metal")

    returned = compiled.launch(x, y, out, 0, thread_count=0, block_size=32)

    assert returned is out
    np.testing.assert_allclose(out.cpu().numpy(), np.empty((0,), dtype=np.float32))


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_experimental_compiled_kernel_launch_reports_invalid_metallib_errors():
    @cx.experimental.kernel(target="metal")
    def add_kernel(a, b, out, n):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        if i < n:
            out[i] = a[i] + b[i]

    x = cx.ones((2,), dtype=cx.float32, device="metal")
    out = cx.empty((2,), dtype=cx.float32, device="metal")
    empty_artifact = cx.experimental.CompiledKernel(
        name="add_kernel",
        target="metal",
        ir=add_kernel.parse_ir(),
        msl_source="",
        metallib=b"",
    )
    malformed_artifact = cx.experimental.CompiledKernel(
        name="add_kernel",
        target="metal",
        ir=add_kernel.parse_ir(),
        msl_source="",
        metallib=b"not-a-metallib",
    )

    with pytest.raises(ValueError, match="kernel compilation target requires an artifact"):
        empty_artifact.launch(x, x, out, 2, block_size=32)
    with pytest.raises(ValueError, match="failed to load Metal library"):
        malformed_artifact.launch(x, x, out, 2, block_size=32)


@pytest.mark.skipif(
    not _has_metal_compiler(),
    reason="Apple Metal command-line compiler tools are unavailable",
)
@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_experimental_compiled_kernel_launch_reports_missing_function():
    @cx.experimental.kernel(target="metal")
    def add_kernel(a, b, out, n):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        if i < n:
            out[i] = a[i] + b[i]

    compiled = add_kernel.compile(target="metal")
    missing = cx.experimental.CompiledKernel(
        name="missing_kernel",
        target="metal",
        ir=compiled.ir,
        msl_source=compiled.msl_source,
        metallib=compiled.metallib,
    )
    x = cx.ones((2,), dtype=cx.float32, device="metal")
    out = cx.empty((2,), dtype=cx.float32, device="metal")

    with pytest.raises(ValueError, match="does not contain function: missing_kernel"):
        missing.launch(x, x, out, 2, block_size=32)


@pytest.mark.skipif(
    not _has_metal_compiler(),
    reason="Apple Metal command-line compiler tools are unavailable",
)
@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_experimental_compiled_kernel_validates_metal_function_lookup():
    @cx.experimental.kernel
    def add_kernel(a, b, out, n):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        if i < n:
            out[i] = a[i] + b[i]

    compiled = add_kernel.compile(target="metal")

    assert compiled.validate_metal_function() == "add_kernel"
    assert compiled.validate_metal_function("add_kernel") == "add_kernel"
    with pytest.raises(ValueError, match="does not contain function: missing_kernel"):
        compiled.validate_metal_function("missing_kernel")


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_experimental_compiled_kernel_validate_rejects_empty_metallib():
    compiled = cx.experimental.CompiledKernel(
        name="add_kernel",
        target="metal",
        ir=cx.experimental.IRKernel("add_kernel", (), ()),
        msl_source="",
        metallib=b"",
    )

    with pytest.raises(ValueError, match="Metal library data is empty"):
        compiled.validate_metal_function()


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_experimental_compiled_kernel_validate_rejects_malformed_metallib():
    compiled = cx.experimental.CompiledKernel(
        name="add_kernel",
        target="metal",
        ir=cx.experimental.IRKernel("add_kernel", (), ()),
        msl_source="",
        metallib=b"not-a-metallib",
    )

    with pytest.raises(ValueError, match="failed to load Metal library"):
        compiled.validate_metal_function()


def test_experimental_kernel_emit_msl_rejects_unused_parameters():
    @cx.experimental.kernel
    def bad_kernel(a, unused, out, n):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        if i < n:
            out[i] = a[i]

    with pytest.raises(
        cx.experimental.KernelCompileError,
        match="all parameters to be referenced: unused",
    ):
        bad_kernel.emit_msl()


def test_experimental_kernel_parse_rejects_unsupported_statement():
    # for-loops are now supported, but only over range(scalar_parameter) and
    # with assignment-only bodies; this legacy shape violates both.
    @cx.experimental.kernel
    def bad_kernel(out):
        for i in range(1):
            out[i] = i

    with pytest.raises(
        cx.experimental.KernelCompileError,
        match="range bound must be a scalar parameter",
    ):
        bad_kernel.parse_ir()

    @cx.experimental.kernel
    def store_in_loop_kernel(out, n):
        for i in range(n):
            out[i] = 1.0

    with pytest.raises(
        cx.experimental.KernelCompileError,
        match="only local assignments",
    ):
        store_in_loop_kernel.parse_ir()

    @cx.experimental.kernel
    def while_kernel(out, n):
        while n > 0:
            out[0] = 1.0

    with pytest.raises(cx.experimental.KernelCompileError, match="While"):
        while_kernel.parse_ir()


def test_experimental_kernel_parse_rejects_unsupported_subscript_shape():
    @cx.experimental.kernel
    def bad_kernel(a, out):
        i = cx.experimental.program_id(0)
        out[i, 0] = a[i]

    with pytest.raises(cx.experimental.KernelCompileError, match="Tuple"):
        bad_kernel.parse_ir()


def test_experimental_kernel_parse_rejects_unknown_calls():
    @cx.experimental.kernel
    def bad_kernel(out):
        i = range(1)
        out[i] = 1

    with pytest.raises(cx.experimental.KernelCompileError, match="Call"):
        bad_kernel.parse_ir()


def test_experimental_kernel_parse_rejects_true_division():
    @cx.experimental.kernel
    def bad_kernel(out):
        i = cx.experimental.thread_id() / 2
        out[i] = 1

    with pytest.raises(cx.experimental.KernelCompileError, match="Div"):
        bad_kernel.parse_ir()


def _program_id_nonzero_axis_kernel():
    @cx.experimental.kernel
    def bad_kernel(out, axis):
        i = cx.experimental.program_id(1)
        out[i] = 1

    return bad_kernel


def _program_id_dynamic_axis_kernel():
    @cx.experimental.kernel
    def bad_kernel(out, axis):
        i = cx.experimental.program_id(axis)
        out[i] = 1

    return bad_kernel


def _program_id_float_axis_kernel():
    @cx.experimental.kernel
    def bad_kernel(out, axis):
        i = cx.experimental.program_id(0.5)
        out[i] = 1

    return bad_kernel


def _program_id_bool_axis_kernel():
    @cx.experimental.kernel
    def bad_kernel(out, axis):
        i = cx.experimental.program_id(True)
        out[i] = 1

    return bad_kernel


@pytest.mark.parametrize(
    "kernel_factory",
    [
        _program_id_nonzero_axis_kernel,
        _program_id_dynamic_axis_kernel,
        _program_id_float_axis_kernel,
        _program_id_bool_axis_kernel,
    ],
)
def test_experimental_kernel_parse_rejects_unsupported_program_id_axis(
    kernel_factory,
):
    with pytest.raises(
        cx.experimental.KernelCompileError,
        match="program_id expects literal axis 0",
    ):
        kernel_factory().parse_ir()


def test_experimental_kernel_parse_rejects_intrinsic_keywords():
    @cx.experimental.kernel
    def bad_kernel(out):
        i = cx.experimental.program_id(axis=0)
        out[i] = 1

    with pytest.raises(cx.experimental.KernelCompileError, match="Call"):
        bad_kernel.parse_ir()


@pytest.mark.parametrize(
    "kernel_factory, expected",
    [
        (
            lambda: _program_id_missing_axis_kernel(),
            "program_id expects 1 argument",
        ),
        (
            lambda: _thread_id_extra_arg_kernel(),
            "thread_id expects 0 argument",
        ),
        (
            lambda: _block_size_extra_arg_kernel(),
            "block_size expects 0 argument",
        ),
    ],
)
def test_experimental_kernel_parse_rejects_intrinsic_arity(
    kernel_factory,
    expected,
):
    with pytest.raises(cx.experimental.KernelCompileError, match=expected):
        kernel_factory().parse_ir()


def _program_id_missing_axis_kernel():
    @cx.experimental.kernel
    def bad_kernel(out):
        i = cx.experimental.program_id()
        out[i] = 1

    return bad_kernel


def _thread_id_extra_arg_kernel():
    @cx.experimental.kernel
    def bad_kernel(out):
        i = cx.experimental.thread_id(0)
        out[i] = 1

    return bad_kernel


def _block_size_extra_arg_kernel():
    @cx.experimental.kernel
    def bad_kernel(out):
        i = cx.experimental.block_size(1)
        out[i] = 1

    return bad_kernel


def test_experimental_kernel_parse_rejects_if_else():
    @cx.experimental.kernel
    def bad_kernel(out, n):
        i = cx.experimental.program_id(0)
        if i < n:
            out[i] = 1
        else:
            out[i] = 0

    with pytest.raises(cx.experimental.KernelCompileError, match="If with else"):
        bad_kernel.parse_ir()


def test_experimental_kernel_parse_rejects_multiple_output_buffers():
    @cx.experimental.kernel
    def bad_kernel(a, out, other):
        i = cx.experimental.program_id(0)
        out[i] = a[i]
        other[i] = a[i]

    with pytest.raises(
        cx.experimental.KernelCompileError,
        match="multiple output buffers",
    ):
        bad_kernel.parse_ir()


def test_experimental_kernel_parse_accepts_float_constants():
    @cx.experimental.kernel
    def fill_kernel(out):
        i = cx.experimental.program_id(0)
        out[i] = 1.5

    ir = fill_kernel.parse_ir()

    assert isinstance(ir.body[1], cx.experimental.IRStore)
    assert ir.body[1].value == cx.experimental.IRConstant(1.5)


@pytest.mark.parametrize(
    "kernel_factory, expected",
    [
        (lambda: _negative_constant_kernel(), cx.experimental.IRConstant(-1.5)),
        (lambda: _positive_constant_kernel(), cx.experimental.IRConstant(2)),
    ],
)
def test_experimental_kernel_parse_accepts_signed_numeric_constants(
    kernel_factory,
    expected,
):
    ir = kernel_factory().parse_ir()

    assert isinstance(ir.body[1], cx.experimental.IRStore)
    assert ir.body[1].value == expected


def _negative_constant_kernel():
    @cx.experimental.kernel
    def fill_kernel(out):
        i = cx.experimental.program_id(0)
        out[i] = -1.5

    return fill_kernel


def _positive_constant_kernel():
    @cx.experimental.kernel
    def fill_kernel(out):
        i = cx.experimental.program_id(0)
        out[i] = +2

    return fill_kernel


def test_experimental_kernel_parse_rejects_local_buffer_alias_store():
    @cx.experimental.kernel
    def bad_kernel(out):
        i = cx.experimental.program_id(0)
        tmp = out
        tmp[i] = 1

    with pytest.raises(
        cx.experimental.KernelCompileError,
        match="buffer references must be parameters",
    ):
        bad_kernel.parse_ir()


def test_experimental_kernel_parse_rejects_local_buffer_alias_load():
    @cx.experimental.kernel
    def bad_kernel(a, out):
        i = cx.experimental.program_id(0)
        tmp = a
        out[i] = tmp[i]

    with pytest.raises(
        cx.experimental.KernelCompileError,
        match="buffer references must be parameters",
    ):
        bad_kernel.parse_ir()


def test_experimental_kernel_parse_rejects_buffer_scalar_use():
    @cx.experimental.kernel
    def bad_kernel(out):
        i = cx.experimental.program_id(0)
        out[i] = out

    with pytest.raises(
        cx.experimental.KernelCompileError,
        match="buffer parameters cannot be used as scalar values: out",
    ):
        bad_kernel.parse_ir()


def test_experimental_kernel_parse_rejects_buffer_used_as_scalar_and_buffer():
    @cx.experimental.kernel
    def bad_kernel(n, out):
        i = cx.experimental.program_id(0)
        if i < n:
            out[i] = n[i]

    with pytest.raises(
        cx.experimental.KernelCompileError,
        match="buffer parameters cannot be used as scalar values: n",
    ):
        bad_kernel.parse_ir()


def test_experimental_kernel_parse_rejects_local_reassignment():
    @cx.experimental.kernel
    def bad_kernel(a, out):
        i = cx.experimental.program_id(0)
        value = 0
        value = a[i]
        out[i] = value

    with pytest.raises(
        cx.experimental.KernelCompileError,
        match="local reassignment is not supported",
    ):
        bad_kernel.parse_ir()


def test_experimental_kernel_parse_rejects_parameter_shadowing():
    @cx.experimental.kernel
    def bad_kernel(out):
        i = cx.experimental.program_id(0)
        out = 0
        out[i] = 1

    with pytest.raises(
        cx.experimental.KernelCompileError,
        match="assignments cannot shadow parameters",
    ):
        bad_kernel.parse_ir()


def test_experimental_kernel_parse_rejects_undefined_names():
    @cx.experimental.kernel
    def bad_kernel(out):
        i = cx.experimental.program_id(0)
        out[i] = missing

    with pytest.raises(
        cx.experimental.KernelCompileError,
        match="undefined name 'missing'",
    ):
        bad_kernel.parse_ir()


def test_experimental_kernel_parse_rejects_branch_local_escape():
    @cx.experimental.kernel
    def bad_kernel(a, out, n):
        i = cx.experimental.program_id(0)
        if i < n:
            value = a[i]
        out[i] = value

    with pytest.raises(
        cx.experimental.KernelCompileError,
        match="undefined name 'value'",
    ):
        bad_kernel.parse_ir()


def _bool_constant_kernel():
    @cx.experimental.kernel
    def bad_kernel(out):
        i = cx.experimental.program_id(0)
        out[i] = True

    return bad_kernel


def _string_constant_kernel():
    @cx.experimental.kernel
    def bad_kernel(out):
        i = cx.experimental.program_id(0)
        out[i] = "text"

    return bad_kernel


@pytest.mark.parametrize(
    "kernel_factory",
    [
        _bool_constant_kernel,
        _string_constant_kernel,
    ],
)
def test_experimental_kernel_parse_rejects_unsupported_constants(kernel_factory):
    with pytest.raises(cx.experimental.KernelCompileError, match="Constant"):
        kernel_factory().parse_ir()


def test_experimental_kernel_rejects_invalid_inputs():
    with pytest.raises(TypeError, match="expects a Python function"):
        cx.experimental.kernel(123)
    with pytest.raises(ValueError, match="target must be"):
        cx.experimental.kernel(target="unknown")
    with pytest.raises(TypeError, match="target must be a string"):
        cx.experimental.kernel(target=123)


def test_experimental_kernel_rejects_callable_object():
    class CallableKernel:
        def __call__(self, a, b, out, n):
            pass

    with pytest.raises(TypeError, match="expects a Python function"):
        cx.experimental.kernel(CallableKernel())


def test_experimental_kernel_direct_construction_validates_target():
    def add_kernel(a, b, out, n):
        pass

    with pytest.raises(ValueError, match="target must be"):
        cx.experimental.Kernel(add_kernel, target="unknown")


def test_experimental_kernel_rejects_unsupported_signature_shapes():
    with pytest.raises(ValueError, match="explicit positional parameters"):

        @cx.experimental.kernel
        def varargs_kernel(*args):
            pass

    with pytest.raises(ValueError, match="keyword-only parameters"):

        @cx.experimental.kernel
        def keyword_only_kernel(a, *, b):
            pass

    with pytest.raises(ValueError, match="async functions"):

        @cx.experimental.kernel
        async def async_kernel(out):
            pass


def test_experimental_intrinsics_are_kernel_only_placeholders():
    with pytest.raises(NotImplementedError, match="program_id is only valid"):
        cx.experimental.program_id(0)
    with pytest.raises(NotImplementedError, match="thread_id is only valid"):
        cx.experimental.thread_id()
    with pytest.raises(NotImplementedError, match="block_size is only valid"):
        cx.experimental.block_size()


@pytest.mark.skipif(
    not _has_metal_compiler(),
    reason="Apple Metal command-line compiler tools are unavailable",
)
@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_experimental_kernel_repeated_launches_stay_correct():
    # Launches after the first are served by the Metal pipeline cache; every
    # iteration must still produce correct results with fresh buffer bindings.
    @cx.experimental.kernel(target="metal")
    def add_kernel(a, b, out, n):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        if i < n:
            out[i] = a[i] + b[i]

    compiled = add_kernel.compile(target="metal")

    for iteration in range(1, 9):
        values = np.arange(iteration * 8, dtype=np.float32)
        x_cpu = cx.tensor(values, dtype=cx.float32, device="cpu")
        y_cpu = cx.tensor(values * 2.0, dtype=cx.float32, device="cpu")
        x = x_cpu.to("metal")
        y = y_cpu.to("metal")
        out = cx.empty(x.shape, dtype=cx.float32, device="metal")

        compiled.launch(x, y, out, x.shape[0], thread_count=x.shape[0], block_size=32)

        cx.testing.assert_allclose(out.cpu(), x_cpu + y_cpu, kind="elementwise")


def _rowsum_kernel():
    @cx.experimental.kernel(target="metal")
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


def test_experimental_for_loop_parses_and_emits_msl():
    kernel_ir = _rowsum_kernel().parse_ir()
    loop = kernel_ir.body[-1].body[1]
    assert isinstance(loop, cx.experimental.IRFor)
    assert loop.var == "k"
    assert loop.limit == "m"
    assert len(loop.body) == 1

    msl = _rowsum_kernel().emit_msl()
    assert "for (uint k = 0u; k < m; ++k) {" in msl
    # The accumulator is declared once and reassigned without a declarator.
    assert "float acc = 0.0f;" in msl
    assert "acc = (acc + a[((i * m) + k)]);" in msl


def test_experimental_for_loop_parse_rejections():
    @cx.experimental.kernel
    def two_arg_range(a, out, n, m):
        i = cx.experimental.thread_id()
        if i < n:
            acc = 0.0
            for k in range(0, m):
                acc = acc + a[i * m + k]
            out[i] = acc

    with pytest.raises(
        cx.experimental.KernelCompileError, match="range\\(scalar_parameter\\)"
    ):
        two_arg_range.parse_ir()

    @cx.experimental.kernel
    def loop_var_shadow(a, out, n, m):
        i = cx.experimental.thread_id()
        if i < n:
            acc = 0.0
            for i in range(m):
                acc = acc + a[i]
            out[i] = acc

    with pytest.raises(
        cx.experimental.KernelCompileError, match="cannot shadow"
    ):
        loop_var_shadow.parse_ir()

    @cx.experimental.kernel
    def loop_local_escape(a, out, n, m):
        i = cx.experimental.thread_id()
        if i < n:
            acc = 0.0
            for k in range(m):
                acc = acc + a[i * m + k]
            out[i] = acc + k

    with pytest.raises(
        cx.experimental.KernelCompileError, match="undefined name 'k'"
    ):
        loop_local_escape.parse_ir()

    @cx.experimental.kernel
    def top_level_reassign(a, out, n):
        i = cx.experimental.thread_id()
        acc = 0.0
        acc = acc + 1.0
        if i < n:
            out[i] = acc

    with pytest.raises(
        cx.experimental.KernelCompileError, match="local reassignment"
    ):
        top_level_reassign.parse_ir()


def test_experimental_for_loop_launch_contract_rejections():
    @cx.experimental.kernel(target="metal")
    def bad_index_pattern(a, out, n, m):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        if i < n:
            acc = 0.0
            for k in range(m):
                acc = acc + a[k * n + i]
            out[i] = acc

    compiled = cx.experimental.CompiledKernel(
        name="bad_index_pattern",
        target="metal",
        ir=bad_index_pattern.parse_ir(),
        msl_source="",
        metallib=b"",
    )
    with pytest.raises(
        cx.experimental.KernelCompileError,
        match="row-major pattern",
    ):
        compiled.launch()

    @cx.experimental.kernel(target="metal")
    def unguarded_loop(a, out, n, m):
        i = cx.experimental.thread_id()
        acc = 0.0
        for k in range(m):
            acc = acc + a[i * m + k]
        if i < n:
            out[i] = acc

    compiled = cx.experimental.CompiledKernel(
        name="unguarded_loop",
        target="metal",
        ir=unguarded_loop.parse_ir(),
        msl_source="",
        metallib=b"",
    )
    with pytest.raises(
        cx.experimental.KernelCompileError,
        match="for-loops to be inside",
    ):
        compiled.launch()


@pytest.mark.skipif(
    not _has_metal_compiler(),
    reason="Apple Metal command-line compiler tools are unavailable",
)
@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_experimental_for_loop_rowsum_matches_cpu():
    rowsum_kernel = _rowsum_kernel()
    rows, cols = 8, 16
    values = np.arange(rows * cols, dtype=np.float32) / 7.0
    matrix = values.reshape(rows, cols)

    a = cx.tensor(values, dtype=cx.float32, device="metal")
    out = cx.empty((rows,), dtype=cx.float32, device="metal")
    rowsum_kernel(a, out, rows, cols, block_size=4)

    reference = cx.sum(cx.tensor(matrix, dtype=cx.float32, device="cpu"), axis=1)
    cx.testing.assert_allclose(out.cpu(), reference, kind="reduction")

    # Zero-length reduction follows the Phase 6 sum-over-empty convention.
    a_empty = cx.tensor(np.zeros(0, dtype=np.float32), dtype=cx.float32, device="metal")
    out_zero = cx.empty((rows,), dtype=cx.float32, device="metal")
    rowsum_kernel(a_empty, out_zero, rows, 0, block_size=4)
    np.testing.assert_array_equal(
        out_zero.cpu().numpy(), np.zeros(rows, dtype=np.float32)
    )

    # A loop-indexed buffer must be exactly output_size * limit elements.
    bad = cx.tensor(
        np.zeros(rows * cols - 1, dtype=np.float32), dtype=cx.float32, device="metal"
    )
    with pytest.raises(ValueError, match="output size times the loop limit"):
        rowsum_kernel(bad, out, rows, cols, block_size=4)


def test_experimental_for_loop_launch_guard_hardening():
    # Reassigning the guarded row index inside the loop would invalidate the
    # structural bounds proof; conflict/output-load rules are also enforced.
    @cx.experimental.kernel(target="metal")
    def row_reassign(a, out, n, m):
        i = cx.experimental.thread_id()
        if i < n:
            acc = 0.0
            for k in range(m):
                i = i + n
                acc = acc + a[i * m + k]
            out[i] = acc

    compiled = cx.experimental.CompiledKernel(
        name="row_reassign", target="metal", ir=row_reassign.parse_ir(),
        msl_source="", metallib=b"",
    )
    with pytest.raises(
        cx.experimental.KernelCompileError, match="reassigning the guarded index"
    ):
        compiled.launch()

    @cx.experimental.kernel(target="metal")
    def conflict(a, out, n, m):
        i = cx.experimental.thread_id()
        if i < n:
            acc = a[i]
            for k in range(m):
                acc = acc + a[i * m + k]
            out[i] = acc

    compiled = cx.experimental.CompiledKernel(
        name="conflict", target="metal", ir=conflict.parse_ir(),
        msl_source="", metallib=b"",
    )
    with pytest.raises(
        cx.experimental.KernelCompileError,
        match="both elementwise and loop-indexed",
    ):
        compiled.launch()

    @cx.experimental.kernel(target="metal")
    def loads_output(out, n, m):
        i = cx.experimental.thread_id()
        if i < n:
            acc = 0.0
            for k in range(m):
                acc = acc + out[i * m + k]
            out[i] = acc

    compiled = cx.experimental.CompiledKernel(
        name="loads_output", target="metal", ir=loads_output.parse_ir(),
        msl_source="", metallib=b"",
    )
    with pytest.raises(
        cx.experimental.KernelCompileError,
        match="loading the output buffer inside a for-loop",
    ):
        compiled.launch()


def test_experimental_msl_reassignment_must_preserve_type():
    @cx.experimental.kernel
    def type_flip(a, out, n, m):
        i = cx.experimental.thread_id()
        if i < n:
            acc = 0.0
            for k in range(m):
                acc = k
            out[i] = a[i]

    with pytest.raises(
        cx.experimental.KernelCompileError, match="must preserve the type"
    ):
        type_flip.emit_msl()




def test_experimental_reference_add_matches_cpu_backend():
    @cx.experimental.kernel
    def add_kernel(a, b, out, n):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        if i < n:
            out[i] = a[i] + b[i]

    values = np.arange(257, dtype=np.float32)
    x = cx.tensor(values, dtype=cx.float32, device="cpu")
    y = cx.tensor(values * 3.0, dtype=cx.float32, device="cpu")
    out = cx.zeros(values.shape, dtype=cx.float32, device="cpu")

    result = add_kernel.reference(x, y, out, values.shape[0])

    assert result.device == "cpu"
    cx.testing.assert_allclose(result, x + y, kind="elementwise")
    # CPU tensors are immutable values: the out argument is not mutated.
    np.testing.assert_array_equal(out.numpy(), np.zeros_like(values))


def test_experimental_reference_rowsum_and_partial_threads():
    rows, cols = 8, 16
    matrix = np.random.default_rng(3).standard_normal((rows, cols))
    flat = matrix.astype(np.float32).reshape(-1)
    a = cx.tensor(flat, dtype=cx.float32, device="cpu")
    out = cx.tensor(np.full(rows, -1.0, dtype=np.float32), device="cpu")

    kernel = _rowsum_kernel()
    result = kernel.reference(a, out, rows, cols)
    np.testing.assert_allclose(
        result.numpy(), matrix.astype(np.float32).sum(axis=1), rtol=1e-5, atol=1e-5
    )

    # Partial thread_count writes only the guarded prefix; unwritten elements
    # keep the out argument's initial contents. Zero loop limits store 0.0.
    half = rows // 2
    partial = kernel.reference(a, out, half, cols, thread_count=half)
    np.testing.assert_array_equal(partial.numpy()[half:], np.full(half, -1.0))
    untouched = kernel.reference(a, out, 0, cols, thread_count=0)
    np.testing.assert_array_equal(untouched.numpy(), np.full(rows, -1.0))
    empty = kernel.reference(
        cx.tensor(np.zeros(0, dtype=np.float32), dtype=cx.float32, device="cpu"),
        out,
        rows,
        0,
    )
    np.testing.assert_array_equal(empty.numpy(), np.zeros(rows, dtype=np.float32))


def test_experimental_reference_signed_literal_compare_matches_msl_semantics():
    # MSL compiles `int k = -5; if (k < 1)` as a SIGNED compare (true); the
    # reference interpreter must match the C literal/declaration typing.
    @cx.experimental.kernel
    def signed_kernel(a, out, n):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        k = -5
        if k < 1:
            if i < n:
                out[i] = a[i] + a[i]

    values = np.arange(4, dtype=np.float32)
    a = cx.tensor(values, dtype=cx.float32, device="cpu")
    out = cx.zeros(values.shape, dtype=cx.float32, device="cpu")
    result = signed_kernel.reference(a, out, values.shape[0])
    np.testing.assert_array_equal(result.numpy(), values * 2.0)


def test_experimental_reference_validation_parity():
    kernel = _rowsum_kernel()
    a = cx.tensor(np.zeros(8, dtype=np.float32), device="cpu")
    out = cx.tensor(np.zeros(4, dtype=np.float32), device="cpu")

    with pytest.raises(ValueError, match="loop-indexed buffer size"):
        kernel.reference(a, out, 4, 3)
    with pytest.raises(ValueError, match="guard bound must match"):
        kernel.reference(a, out, 3, 2)
    with pytest.raises(TypeError, match="must be Tensor objects"):
        kernel.reference(a, 1, 4, 2)
    if cx.is_available("metal"):
        with pytest.raises(ValueError, match="requires CPU tensors"):
            kernel.reference(a, out.to("metal"), 4, 2)


@pytest.mark.skipif(
    not _has_metal_compiler(),
    reason="Apple Metal command-line compiler tools are unavailable",
)
@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_experimental_reference_matches_metal_launch():
    rows, cols = 8, 16
    matrix = np.random.default_rng(9).standard_normal((rows, cols))
    flat = matrix.astype(np.float32).reshape(-1)
    a_cpu = cx.tensor(flat, dtype=cx.float32, device="cpu")
    out_cpu = cx.zeros((rows,), dtype=cx.float32, device="cpu")

    kernel = _rowsum_kernel()
    reference = kernel.reference(a_cpu, out_cpu, rows, cols)

    a_metal = a_cpu.to("metal")
    out_metal = cx.empty((rows,), dtype=cx.float32, device="metal")
    kernel(a_metal, out_metal, rows, cols, block_size=4)

    cx.testing.assert_allclose(out_metal.cpu(), reference, kind="reduction")


def test_experimental_launch_rejects_outer_guard_index_reassignment():
    # Reassigning an OUTER guard's index inside a loop nested under a
    # different guard must be rejected: the structural bounds proof for the
    # later store/loads was made against the pre-loop value (this was an
    # out-of-bounds write on Metal before the index-name pre-pass).
    @cx.experimental.kernel(target="metal")
    def outer_reassign(a, out, n, m):
        i = cx.experimental.thread_id()
        j = cx.experimental.thread_id()
        if i < n:
            if j < n:
                for k in range(m):
                    i = i + n
            acc = 0.0
            for k2 in range(m):
                acc = acc + a[i * m + k2]
            out[i] = acc

    compiled = cx.experimental.CompiledKernel(
        name="outer_reassign", target="metal", ir=outer_reassign.parse_ir(),
        msl_source="", metallib=b"",
    )
    with pytest.raises(
        cx.experimental.KernelCompileError, match="reassigning the guarded index"
    ):
        compiled.launch()


def test_experimental_reference_bool_index_matches_msl_conversion():
    # MSL converts a bool index to 0/1; numpy would treat a raw Python bool as
    # a boolean MASK selecting every element.
    @cx.experimental.kernel
    def bool_index(a, out, n):
        i = cx.experimental.thread_id()
        t = i < n
        if t < n:
            out[t] = a[t] + 1.0

    values = np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    a = cx.tensor(values, dtype=cx.float32, device="cpu")
    out = cx.zeros(values.shape, dtype=cx.float32, device="cpu")
    result = bool_index.reference(a, out, values.shape[0])
    np.testing.assert_array_equal(
        result.numpy(), np.asarray([0.0, 21.0, 0.0, 0.0], dtype=np.float32)
    )


def test_experimental_reference_aliased_buffers_match_metal_binding():
    # The same tensor bound to two buffer parameters aliases one native buffer
    # on Metal; the reference path must share one array the same way.
    @cx.experimental.kernel
    def alias_kernel(a, out, n):
        i = cx.experimental.thread_id()
        if i < n:
            out[i] = 2.0
            out[i] = a[i] + 1.0

    values = np.asarray([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    t = cx.tensor(values, dtype=cx.float32, device="cpu")
    result = alias_kernel.reference(t, t, values.shape[0])
    np.testing.assert_array_equal(
        result.numpy(), np.full(values.shape, 3.0, dtype=np.float32)
    )
