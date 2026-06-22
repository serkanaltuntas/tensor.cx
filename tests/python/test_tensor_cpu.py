import numpy as np
import pytest

import cortex_runtime as cx


HUGE_SHAPE = (3_037_000_500, 3_037_000_500)
STRIDE_OVERFLOW_SHAPE = (0, 9_223_372_036_854_775_807, 2)


def test_tensor_infers_int32_for_integer_list():
    x = cx.tensor([1, 2, 3], device="cpu")

    assert x.shape == (3,)
    assert x.strides == (1,)
    assert x.dtype == cx.int32
    assert x.device == "cpu"
    assert x.nbytes == 12
    np.testing.assert_array_equal(x.numpy(), np.array([1, 2, 3], dtype=np.int32))


def test_tensor_supports_explicit_float32():
    x = cx.tensor([1, 2, 3], dtype=cx.float32, device="cpu")

    assert x.dtype == cx.float32
    np.testing.assert_allclose(x.numpy(), np.array([1, 2, 3], dtype=np.float32))


def test_zeros_ones_and_empty_metadata():
    zeros = cx.zeros((3,), dtype=cx.float32, device="cpu")
    ones = cx.ones((3,), dtype=cx.int32, device="cpu")
    empty = cx.empty((2,), dtype=cx.float32, device="cpu")

    np.testing.assert_allclose(zeros.numpy(), np.zeros((3,), dtype=np.float32))
    np.testing.assert_array_equal(ones.numpy(), np.ones((3,), dtype=np.int32))
    assert empty.shape == (2,)
    assert empty.dtype == cx.float32
    assert empty.nbytes == 8


def test_factory_rejects_shape_size_overflow():
    with pytest.raises(ValueError, match="shape size overflow"):
        cx.zeros(HUGE_SHAPE, dtype=cx.float32, device="cpu")


def test_factory_rejects_shape_stride_overflow():
    with pytest.raises(ValueError, match="shape stride overflow"):
        cx.zeros(STRIDE_OVERFLOW_SHAPE, dtype=cx.float32, device="cpu")


def test_numpy_preserves_tensor_shape():
    x = cx.zeros((2, 3), dtype=cx.float32, device="cpu")

    assert x.shape == (2, 3)
    np.testing.assert_allclose(x.numpy(), np.zeros((2, 3), dtype=np.float32))


def test_tensor_accepts_nested_rank2_data():
    x = cx.tensor([[1, 2, 3], [4, 5, 6]], dtype=cx.float32, device="cpu")

    assert x.shape == (2, 3)
    assert x.strides == (3, 1)
    np.testing.assert_allclose(
        x.numpy(),
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
    )


def test_add_and_multiply_cpu_int32():
    x = cx.tensor([1, 2, 3], device="cpu")
    y = cx.tensor([4, 5, 6], device="cpu")

    np.testing.assert_array_equal((x + y).numpy(), np.array([5, 7, 9], dtype=np.int32))
    np.testing.assert_array_equal((x * y).numpy(), np.array([4, 10, 18], dtype=np.int32))


def test_add_and_multiply_cpu_float32():
    x = cx.tensor([1.0, 2.0, 3.0], device="cpu")
    y = cx.ones((3,), dtype=cx.float32, device="cpu")

    np.testing.assert_allclose((x + y).numpy(), np.array([2, 3, 4], dtype=np.float32))
    np.testing.assert_allclose((x * y).numpy(), np.array([1, 2, 3], dtype=np.float32))


def test_binary_ops_reject_shape_mismatch():
    x = cx.tensor([1, 2, 3], device="cpu")
    y = cx.tensor([1, 2], device="cpu")

    with pytest.raises(ValueError, match="shape mismatch"):
        _ = x + y


def test_binary_ops_reject_dtype_mismatch():
    x = cx.tensor([1, 2, 3], dtype=cx.int32, device="cpu")
    y = cx.tensor([1, 2, 3], dtype=cx.float32, device="cpu")

    with pytest.raises(ValueError, match="dtype mismatch"):
        _ = x * y


def test_matmul_cpu_float32_matches_numpy():
    x = cx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=cx.float32, device="cpu")
    y = cx.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=cx.float32, device="cpu")

    expected = x.numpy() @ y.numpy()

    np.testing.assert_allclose(cx.matmul(x, y).numpy(), expected)
    np.testing.assert_allclose((x @ y).numpy(), expected)


def test_matmul_cpu_rejects_invalid_inputs():
    x = cx.ones((2, 3), dtype=cx.float32, device="cpu")
    y = cx.ones((2, 3), dtype=cx.float32, device="cpu")
    z = cx.ones((3, 2), dtype=cx.int32, device="cpu")

    with pytest.raises(ValueError, match="matmul shape mismatch"):
        cx.matmul(x, y)
    with pytest.raises(ValueError, match="matmul only supports float32"):
        cx.matmul(x, z)
    with pytest.raises(ValueError, match="matmul requires rank-2"):
        cx.matmul(cx.ones((3,), dtype=cx.float32, device="cpu"), y)


def test_tensor_rejects_unsupported_device():
    with pytest.raises(ValueError, match="unsupported device transfer"):
        cx.tensor([1, 2, 3], device="cuda")


def test_rank0_scalar_numpy_round_trip():
    # Rank-0 tensors are reachable via the low-level factory; numpy() must emit a
    # 0-d array, not crash on the empty shape.
    from cortex_runtime import _core

    for dtype, np_dtype in (("float32", np.float32), ("int32", np.int32)):
        scalar = cx.Tensor(_core.zeros((), dtype=dtype, device="cpu"))
        array = scalar.numpy()
        assert array.shape == ()
        assert array.dtype == np_dtype
        assert array.item() == 0


def test_native_tensor_factory_rejects_nested_sequence():
    # The native _core.tensor is the flat 1-D factory; nested data must route
    # through the public cx.tensor() wrapper instead of mis-flattening here.
    from cortex_runtime import _core

    with pytest.raises(ValueError, match="flat numeric sequence"):
        _core.tensor([[1, 2], [3, 4]])
