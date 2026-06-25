import shutil
import subprocess

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


def test_experimental_kernel_compile_rejects_unsupported_target_and_launch_is_disabled():
    @cx.experimental.kernel
    def add_kernel(a, b, out, n):
        pass

    with pytest.raises(NotImplementedError, match="only implemented"):
        add_kernel.compile(target="cpu")
    with pytest.raises(NotImplementedError, match="launch is not implemented"):
        add_kernel(None, None, None, 0)


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
    if (i < n) {
        out[i] = (a[i] + b[i]);
    }
}"""


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
    @cx.experimental.kernel
    def bad_kernel(out):
        for i in range(1):
            out[i] = i

    with pytest.raises(cx.experimental.KernelCompileError, match="For"):
        bad_kernel.parse_ir()


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
        cx.experimental.kernel(target="cuda")
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
        cx.experimental.Kernel(add_kernel, target="cuda")


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
