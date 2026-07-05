import numpy as np
import pytest

import cortex_runtime as cx


@pytest.mark.backend_capability("copy", include_cpu=False)
@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (cx.int32, np.array([1, 2, 3], dtype=np.int32)),
        (cx.float32, np.array([1.0, 2.0, 3.0], dtype=np.float32)),
    ],
)
def test_accelerator_round_trip_matches_cpu(backend_name, dtype, expected):
    x_cpu = cx.tensor([1, 2, 3], dtype=dtype, device="cpu")

    x_device = x_cpu.to(backend_name)
    x_back = x_device.cpu()

    assert x_device.device == backend_name
    assert x_device.shape == x_cpu.shape
    assert x_device.dtype == dtype
    assert x_device.nbytes == x_cpu.nbytes
    np.testing.assert_array_equal(x_back.numpy(), expected)


@pytest.mark.backend_capability("tensor_factories_float32")
def test_backend_float32_factories_match_cpu(backend_name):
    _assert_factories_match_cpu(backend_name, cx.float32)


@pytest.mark.backend_capability("tensor_factories_int32")
def test_backend_int32_factories_match_cpu(backend_name):
    _assert_factories_match_cpu(backend_name, cx.int32)


def _assert_factories_match_cpu(backend_name, dtype):
    zeros_cpu = cx.zeros((2, 3), dtype=dtype, device="cpu")
    ones_cpu = cx.ones((2, 3), dtype=dtype, device="cpu")

    zeros = cx.zeros((2, 3), dtype=dtype, device=backend_name)
    ones = cx.ones((2, 3), dtype=dtype, device=backend_name)
    empty = cx.empty((2,), dtype=dtype, device=backend_name)

    np.testing.assert_array_equal(zeros.cpu().numpy(), zeros_cpu.numpy())
    np.testing.assert_array_equal(ones.cpu().numpy(), ones_cpu.numpy())
    assert empty.shape == (2,)
    assert empty.dtype == dtype
    assert empty.device == backend_name


@pytest.mark.backend_capability("binary_ops_float32")
def test_backend_float32_binary_ops_match_cpu(backend_name):
    _assert_binary_ops_match_cpu(
        backend_name,
        cx.float32,
        [1.5, 2.0, 3.5],
        [4.0, 5.5, 6.0],
    )


@pytest.mark.backend_capability("binary_ops_int32")
def test_backend_int32_binary_ops_match_cpu(backend_name):
    _assert_binary_ops_match_cpu(
        backend_name,
        cx.int32,
        [1, 2, 3],
        [4, 5, 6],
    )


def _assert_binary_ops_match_cpu(backend_name, dtype, lhs_values, rhs_values):
    lhs_cpu = cx.tensor(lhs_values, dtype=dtype, device="cpu")
    rhs_cpu = cx.tensor(rhs_values, dtype=dtype, device="cpu")
    lhs = lhs_cpu.to(backend_name)
    rhs = rhs_cpu.to(backend_name)

    add = (lhs + rhs).cpu()
    multiply = (lhs * rhs).cpu()

    np.testing.assert_array_equal(add.numpy(), (lhs_cpu + rhs_cpu).numpy())
    np.testing.assert_array_equal(multiply.numpy(), (lhs_cpu * rhs_cpu).numpy())


@pytest.mark.backend_capability("binary_ops_float32")
def test_backend_float32_binary_ops_reject_shape_mismatch(backend_name):
    x = cx.ones((2,), dtype=cx.float32, device=backend_name)
    bad_shape = cx.ones((3,), dtype=cx.float32, device=backend_name)

    with pytest.raises(ValueError, match="shape mismatch"):
        _ = x + bad_shape


@pytest.mark.backend_capability("binary_ops_dtype_mismatch")
def test_backend_binary_ops_reject_dtype_mismatch(backend_name):
    x = cx.ones((2,), dtype=cx.float32, device=backend_name)
    bad_dtype = cx.ones((2,), dtype=cx.int32, device=backend_name)

    with pytest.raises(ValueError, match="dtype mismatch"):
        _ = x * bad_dtype


@pytest.mark.backend_capability("unary_float32")
@pytest.mark.parametrize("op_name", ["exp", "gelu", "silu"])
def test_backend_unary_float32_ops_match_cpu(backend_name, op_name):
    x_cpu = cx.tensor([-4.0, -1.0, 0.0, 0.5, 2.0, 5.0], dtype=cx.float32, device="cpu")
    x = x_cpu.to(backend_name)
    op = getattr(cx, op_name)

    cx.testing.assert_allclose(op(x).cpu(), op(x_cpu), kind="elementwise")


@pytest.mark.backend_capability("reductions_float32")
@pytest.mark.parametrize(
    ("op_name", "axis", "kind"),
    [
        ("sum", 1, "reduction"),
        ("max", 0, "reduction"),
        ("mean", 1, "reduction"),
    ],
)
def test_backend_float32_reductions_match_cpu(backend_name, op_name, axis, kind):
    x_cpu = cx.tensor(
        [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]],
        dtype=cx.float32,
        device="cpu",
    )
    x = x_cpu.to(backend_name)
    op = getattr(cx, op_name)

    cx.testing.assert_allclose(op(x, axis=axis).cpu(), op(x_cpu, axis=axis), kind=kind)


@pytest.mark.backend_capability("reductions_int32")
@pytest.mark.parametrize("op_name", ["sum", "max"])
def test_backend_int32_reductions_match_cpu(backend_name, op_name):
    x_cpu = cx.tensor([[1, 2, 3], [4, 5, 6]], dtype=cx.int32, device="cpu")
    x = x_cpu.to(backend_name)
    op = getattr(cx, op_name)

    np.testing.assert_array_equal(op(x, axis=1).cpu().numpy(), op(x_cpu, axis=1).numpy())


@pytest.mark.backend_capability("normalization_float32")
@pytest.mark.parametrize("op_name", ["softmax", "rmsnorm", "layernorm"])
def test_backend_axis_float32_ops_match_cpu(backend_name, op_name):
    x_cpu = cx.tensor(
        [[1.0, -2.0, 3.0], [4.0, 0.5, -6.0]],
        dtype=cx.float32,
        device="cpu",
    )
    x = x_cpu.to(backend_name)
    op = getattr(cx, op_name)

    cx.testing.assert_allclose(op(x, axis=1).cpu(), op(x_cpu, axis=1), kind="reduction")
