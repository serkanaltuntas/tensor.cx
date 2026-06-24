import pytest

import cortex_runtime as cx


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


def test_experimental_kernel_compile_and_launch_are_not_implemented():
    @cx.experimental.kernel
    def add_kernel(a, b, out, n):
        pass

    with pytest.raises(NotImplementedError, match="compilation is not implemented"):
        add_kernel.compile(target="metal")
    with pytest.raises(NotImplementedError, match="launch is not implemented"):
        add_kernel(None, None, None, 0)


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


def test_experimental_intrinsics_are_kernel_only_placeholders():
    with pytest.raises(NotImplementedError, match="program_id is only valid"):
        cx.experimental.program_id(0)
    with pytest.raises(NotImplementedError, match="thread_id is only valid"):
        cx.experimental.thread_id()
    with pytest.raises(NotImplementedError, match="block_size is only valid"):
        cx.experimental.block_size()
