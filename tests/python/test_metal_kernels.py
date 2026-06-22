import numpy as np
import pytest

import cortex_runtime as cx


def test_best_device_add_success_snippet():
    device = cx.best_device()

    x = cx.ones((1_000_000,), dtype=cx.float32, device=device)
    y = cx.ones((1_000_000,), dtype=cx.float32, device=device)
    z = x + y

    np.testing.assert_allclose(z.cpu().numpy()[:5], np.array([2, 2, 2, 2, 2], dtype=np.float32))


def test_binary_ops_reject_device_mismatch():
    if not cx.is_available("metal"):
        pytest.skip("Metal is not available")

    x = cx.ones((3,), dtype=cx.float32, device="cpu")
    y = cx.ones((3,), dtype=cx.float32, device="metal")

    with pytest.raises(ValueError, match="device mismatch"):
        _ = x + y


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_add_float32_matches_cpu():
    x_cpu = cx.tensor([1.0, 2.0, 3.0], dtype=cx.float32, device="cpu")
    y_cpu = cx.tensor([4.0, 5.0, 6.0], dtype=cx.float32, device="cpu")

    z_gpu = x_cpu.to("metal") + y_cpu.to("metal")

    np.testing.assert_allclose(z_gpu.cpu().numpy(), (x_cpu + y_cpu).numpy())


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_multiply_float32_matches_cpu():
    x_cpu = cx.tensor([1.5, 2.0, 3.5], dtype=cx.float32, device="cpu")
    y_cpu = cx.tensor([4.0, 5.5, 6.0], dtype=cx.float32, device="cpu")

    z_gpu = x_cpu.to("metal") * y_cpu.to("metal")

    np.testing.assert_allclose(z_gpu.cpu().numpy(), (x_cpu * y_cpu).numpy())


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_fill_float32_matches_cpu():
    zeros_cpu = cx.zeros((2, 3), dtype=cx.float32, device="cpu")
    ones_cpu = cx.ones((2, 3), dtype=cx.float32, device="cpu")
    zeros = cx.zeros((2, 3), dtype=cx.float32, device="metal")
    ones = cx.ones((2, 3), dtype=cx.float32, device="metal")

    np.testing.assert_allclose(zeros.cpu().numpy(), zeros_cpu.numpy())
    np.testing.assert_allclose(ones.cpu().numpy(), ones_cpu.numpy())


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_add_int32_matches_cpu():
    x_cpu = cx.tensor([1, 2, 3], dtype=cx.int32, device="cpu")
    y_cpu = cx.tensor([4, 5, 6], dtype=cx.int32, device="cpu")

    z = x_cpu.to("metal") + y_cpu.to("metal")

    np.testing.assert_array_equal(z.cpu().numpy(), (x_cpu + y_cpu).numpy())


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_multiply_int32_matches_cpu():
    x_cpu = cx.tensor([2, 3, 4], dtype=cx.int32, device="cpu")
    y_cpu = cx.tensor([5, 6, 7], dtype=cx.int32, device="cpu")

    z_gpu = x_cpu.to("metal") * y_cpu.to("metal")

    np.testing.assert_array_equal(z_gpu.cpu().numpy(), (x_cpu * y_cpu).numpy())


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_fill_int32_matches_cpu():
    zeros_cpu = cx.zeros((2, 3), dtype=cx.int32, device="cpu")
    ones_cpu = cx.ones((2, 3), dtype=cx.int32, device="cpu")
    zeros = cx.zeros((2, 3), dtype=cx.int32, device="metal")
    ones = cx.ones((2, 3), dtype=cx.int32, device="metal")

    np.testing.assert_array_equal(zeros.cpu().numpy(), zeros_cpu.numpy())
    np.testing.assert_array_equal(ones.cpu().numpy(), ones_cpu.numpy())
