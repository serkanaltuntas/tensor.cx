import numpy as np
import pytest

import cortex_runtime as cx


def test_devices_always_include_cpu():
    assert "cpu" in cx.devices()
    assert cx.is_available("cpu")
    assert cx.best_device() in {"cpu", "metal"}

    cpu = cx.device("cpu")
    assert cpu.type == "cpu"
    assert cpu.index == 0
    assert cpu.name == "CPU"
    assert str(cpu) == "cpu"
    assert cx.is_available(cpu)

    with pytest.raises(ValueError, match="device is not available"):
        cx.device("cuda")


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_device_discovery():
    assert "metal" in cx.devices()
    assert cx.device_name("metal")

    metal = cx.device("metal")
    assert metal.type == "metal"
    assert metal.index == 0
    assert metal.name == cx.device_name("metal")
    assert cx.is_available(metal)


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (cx.int32, np.array([1, 2, 3], dtype=np.int32)),
        (cx.float32, np.array([1.0, 2.0, 3.0], dtype=np.float32)),
    ],
)
def test_cpu_to_metal_to_cpu_round_trip(dtype, expected):
    x_cpu = cx.tensor([1, 2, 3], dtype=dtype, device="cpu")

    x_gpu = x_cpu.to("metal")
    x_back = x_gpu.cpu()

    assert x_gpu.device == "metal"
    assert x_gpu.shape == (3,)
    assert x_gpu.dtype == dtype
    assert x_gpu.nbytes == x_cpu.nbytes
    np.testing.assert_array_equal(x_back.numpy(), expected)


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_create_tensor_directly_on_metal():
    x = cx.tensor([4, 5, 6], dtype=cx.int32, device=cx.device("metal"))

    assert x.device == "metal"
    np.testing.assert_array_equal(x.cpu().numpy(), np.array([4, 5, 6], dtype=np.int32))


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_tensor_to_cpu_and_numpy_helpers():
    x = cx.ones((2, 3), dtype=cx.float32, device="metal")

    assert x.to("cpu").device == "cpu"
    np.testing.assert_allclose(x.numpy(), np.ones((2, 3), dtype=np.float32))


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_zero_element_tensor_round_trip_on_metal():
    x = cx.zeros((0,), dtype=cx.int32, device="metal")
    x_back = x.cpu()

    assert x.device == "metal"
    assert x.nbytes == 0
    assert x_back.shape == (0,)
    assert x_back.nbytes == 0
    np.testing.assert_array_equal(x_back.numpy(), np.array([], dtype=np.int32))
