import numpy as np
import pytest

import tensorcx as cx


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


@pytest.mark.backend_capability("binary_ops_float32")
@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (1,), (255,), (256,), (257,), (17, 19), (2, 3, 5), (262145,)])
def test_backend_binary_edges_match_cpu(backend_name, shape):
    rng = np.random.default_rng(9)
    a_cpu = cx.tensor(rng.standard_normal(shape).astype(np.float32))
    b_cpu = cx.tensor(rng.standard_normal(shape).astype(np.float32))
    a, b = a_cpu.to(backend_name), b_cpu.to(backend_name)
    for actual, expected in [(a + b, a_cpu + b_cpu), (a * b, a_cpu * b_cpu)]:
        assert actual.shape == shape
        cx.testing.assert_allclose(actual.cpu(), expected, kind="elementwise")
    np.testing.assert_array_equal(a.cpu().numpy(), a_cpu.numpy())
    np.testing.assert_array_equal(b.cpu().numpy(), b_cpu.numpy())


@pytest.mark.backend_capability("tensor_factories_float32")
@pytest.mark.parametrize("shape", [(), (0,), (2, 0), (257,), (2, 3, 5)])
def test_backend_fill_edges_match_cpu(backend_name, shape):
    for factory in (cx.zeros, cx.ones):
        actual = factory(shape, device=backend_name)
        assert actual.shape == shape
        np.testing.assert_array_equal(actual.numpy(), factory(shape).numpy())


@pytest.mark.backend_capability("copy", include_cpu=False)
@pytest.mark.parametrize("shape", [(), (0,), (2, 0), (3, 5), (2, 3, 5)])
@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_backend_copy_shapes_and_ownership(backend_name, shape, dtype):
    data = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    host = cx.tensor(data)
    device = host.to(backend_name)
    del host
    assert device.to(backend_name) is device
    assert device.strides == cx.tensor(data).strides
    first = device.numpy()
    first[...] = 99
    np.testing.assert_array_equal(device.numpy(), data)


@pytest.mark.backend_capability("binary_ops_float32")
@pytest.mark.parametrize("op", ["add", "multiply"])
def test_backend_binary_rejects_dtype_and_shape_mismatch(backend_name, op):
    import operator

    operation = getattr(operator, op if op == "add" else "mul")
    x = cx.tensor([1.0, 2.0], device=backend_name)
    y = cx.tensor([1, 2], dtype=cx.int32, device=backend_name)
    with pytest.raises(ValueError, match="dtype mismatch"):
        operation(x, y)
    with pytest.raises(ValueError, match="shape mismatch"):
        operation(x, cx.ones((1, 2), device=backend_name))
    if backend_name != "cpu":
        with pytest.raises(ValueError, match="device mismatch"):
            operation(x, cx.ones((2,)))


@pytest.mark.backend_capability("binary_ops_float32")
def test_backend_float_special_values(backend_name):
    a = cx.tensor([np.nan, np.inf, -np.inf, -0.0, 0.0, 1e-30], dtype=cx.float32)
    b = cx.tensor([1.0, 2.0, 3.0, 1.0, -1.0, 1e-10], dtype=cx.float32)
    for x, y in [(a.to(backend_name) + b.to(backend_name), a + b),
                 (a.to(backend_name) * b.to(backend_name), a * b)]:
        np.testing.assert_allclose(x.numpy(), y.numpy(), rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.backend_capability("binary_ops_float32")
def test_backend_threaded_copy_fill_binary(backend_name):
    from concurrent.futures import ThreadPoolExecutor

    def calculate(value):
        a = cx.tensor(np.full(4097, value, dtype=np.float32), device=backend_name)
        b = cx.ones((4097,), device=backend_name)
        return ((a + b) * a).numpy()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(calculate, range(16)))
    for value, result in enumerate(results):
        np.testing.assert_array_equal(result, np.full(4097, (value + 1) * value, dtype=np.float32))
