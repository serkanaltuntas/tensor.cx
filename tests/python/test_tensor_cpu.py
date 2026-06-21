import numpy as np
import pytest

import cortex_runtime as cx


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


def test_numpy_preserves_tensor_shape():
    x = cx.zeros((2, 3), dtype=cx.float32, device="cpu")

    assert x.shape == (2, 3)
    np.testing.assert_allclose(x.numpy(), np.zeros((2, 3), dtype=np.float32))


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


def test_tensor_rejects_unsupported_device():
    with pytest.raises(ValueError, match="unsupported device transfer"):
        cx.tensor([1, 2, 3], device="cuda")
