"""Factory metadata, direct allocation, routing and initialized storage ownership."""
from dataclasses import replace
import ctypes as C
import gc
import os
import subprocess
import sys

import numpy as np
import pytest
import tensorcx as cx
from tensorcx import backend


@pytest.fixture(params=cx.devices())
def device_name(request):
    return request.param


@pytest.fixture
def cuda():
    if not cx.is_available("cuda"):
        pytest.skip("CUDA is not available")


@pytest.mark.parametrize("dtype", [cx.float32, cx.int32, cx.bool])
@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (1,), (257,), (2, 3, 5), (10**12, 1, 0)])
def test_empty_metadata(device_name, dtype, shape):
    actual = cx.empty(shape, dtype=dtype, device=device_name)
    reference = cx.empty(shape, dtype=dtype)
    assert actual.device == device_name
    assert (actual.shape, actual.strides, actual.dtype, actual.nbytes) == (
        reference.shape, reference.strides, reference.dtype, reference.nbytes)


@pytest.mark.parametrize("shape", [(-1,), (True,), (np.bool_(True),), (1.5,), (2**63,),
                                  (2**62, 4), (0, 2**62, 4), (2**62, 4, 0)])
def test_empty_invalid_metadata_allocates_nothing(device_name, shape):
    with cx.profile() as profile:
        with pytest.raises(ValueError):
            cx.empty(shape, device=device_name)
    assert not any(e["category"] in ("allocation", "memory_pool", "copy_api")
                   for e in profile.report()["events"])


@pytest.mark.parametrize("dtype", ["float16", "complex64", "missing"])
def test_empty_invalid_dtype_allocates_nothing(device_name, dtype):
    with cx.profile() as profile:
        with pytest.raises(ValueError):
            cx.empty((3,), dtype=dtype, device=device_name)
    assert profile.report()["events"] == []


def test_empty_registry_hook_and_legacy_copy_fallback(monkeypatch):
    native = cx._core.empty((2,), dtype=cx.int32)
    calls = []
    route = replace(backend.require_backend("cpu"), name="test-empty",
                    empty=lambda shape, dtype: calls.append((shape, dtype)) or native)
    monkeypatch.setitem(backend._BACKENDS, route.name, route)
    assert cx.empty((2,), dtype=cx.int32, device=route.name)._impl is native
    assert calls == [((2,), cx.int32)]
    copied = []
    monkeypatch.setitem(backend._BACKENDS, route.name, replace(
        route, empty=None, copy_from_cpu=lambda host: copied.append(host) or host))
    result = cx.empty((3,), dtype=cx.bool, device=route.name)
    assert copied == [result._impl] and result.shape == (3,) and result.dtype == cx.bool


@pytest.mark.parametrize("device", ["missing", "cuda:1", cx.Device("cuda", 1)])
def test_empty_checks_device_before_consuming_shape(device):
    def shape():
        pytest.fail("shape consumed before rejecting device")
        yield 1
    with pytest.raises(ValueError):
        cx.empty(shape(), device=device)


def test_empty_hidden_cuda_no_host_allocation():
    code = """
import tensorcx as cx
def fail(*args, **kwargs):
    raise AssertionError('CPU allocation attempted')
cx._core.empty = fail
try:
    cx.empty((2,), device='cuda')
except ValueError as error:
    assert 'not available' in str(error)
else:
    raise AssertionError('unavailable CUDA accepted')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            timeout=30, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("dtype", [cx.float32, cx.int32, cx.bool])
@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (257,)])
def test_cuda_empty_has_no_host_storage_copy_or_fill(cuda, monkeypatch, dtype, shape):
    def fail(*args, **kwargs):
        pytest.fail("CUDA empty used host allocation or transfer")
    monkeypatch.setattr(cx._core, "empty", fail)
    monkeypatch.setattr(cx._core, "cpu_to_cuda", fail)
    with cx.profile() as profile:
        value = cx.empty(shape, dtype=dtype, device="cuda:0")
        del value
    events = profile.report()["events"]
    assert profile.report()["complete"]
    assert all(e["backend"] == "cuda" and e["category"] in
               ("allocation", "deallocation") for e in events)
    nonempty = bool(np.prod(shape, dtype=int))
    assert sum(e["name"] == "cudaMalloc" for e in events) == int(nonempty)
    assert sum(e["name"] == "cudaFree" for e in events) == int(nonempty)


@pytest.mark.parametrize("dtype", [cx.float32, cx.int32, cx.bool])
def test_cuda_empty_pool_reuse_and_escaping_view(cuda, dtype):
    with cx.memory_pool() as pool:
        with cx.profile() as profile:
            for _ in range(3):
                value = cx.empty((257,), dtype=dtype, device="cuda")
                del value
            first = cx.empty((257,), dtype=dtype, device=cx.Device("cuda"))
            second = cx.empty((257,), dtype=dtype, device="cuda")
        stats = pool.stats()
        assert stats["allocation_calls"] == 2 and stats["hits"] == 3
        assert stats["in_use_bytes"] == 2 * first.nbytes
        assert not any(e["backend"] == "cpu" or e["category"] == "copy_api"
                       for e in profile.report()["events"])
        view = first.reshape((1, 257))
        del first, second
        assert pool.stats()["in_use_bytes"] == view.nbytes
    assert pool.stats()["cached_bytes"] == 0
    assert view.shape == (1, 257) and view.device == "cuda"
    del view
    gc.collect()
    assert pool.stats()["in_use_bytes"] == 0


@pytest.mark.parametrize("dtype", [cx.float32, cx.int32, cx.bool])
def test_cuda_empty_initialized_dlpack_storage_and_lifetime(cuda, dtype):
    # Write every byte through the public DLPack pointer before reading values.
    # Never assert anything about the initial contents of an empty allocation.
    from test_dlpack import description
    driver = C.CDLL("libcuda.so.1")
    driver.cuDevicePrimaryCtxRetain.argtypes = [C.POINTER(C.c_void_p), C.c_int]
    driver.cuCtxPushCurrent_v2.argtypes = [C.c_void_p]
    driver.cuCtxPopCurrent_v2.argtypes = [C.POINTER(C.c_void_p)]
    driver.cuDevicePrimaryCtxRelease_v2.argtypes = [C.c_int]
    driver.cuMemcpyHtoD_v2.argtypes = [C.c_uint64, C.c_void_p, C.c_size_t]
    values = np.resize(np.array([0, 1, -17, 16777217], dtype=dtype), (2, 3))
    first = cx.empty((2, 3), dtype=dtype, device="cuda")
    second = cx.empty((2, 3), dtype=dtype, device="cuda")
    cap_a, cap_b = first.__dlpack__(), second.__dlpack__()
    a, b = description(cap_a), description(cap_b)
    assert a.data != b.data
    context, popped = C.c_void_p(), C.c_void_p()
    assert driver.cuDevicePrimaryCtxRetain(C.byref(context), 0) == 0
    try:
        assert driver.cuCtxPushCurrent_v2(context) == 0
        try:
            assert driver.cuMemcpyHtoD_v2(a.data, values.ctypes.data, values.nbytes) == 0
            other = np.logical_not(values).astype(dtype)
            assert driver.cuMemcpyHtoD_v2(b.data, other.ctypes.data, other.nbytes) == 0
        finally:
            assert driver.cuCtxPopCurrent_v2(C.byref(popped)) == 0
    finally:
        assert driver.cuDevicePrimaryCtxRelease_v2(0) == 0
    imported = cx.from_dlpack(cap_a, copy=False)
    view = imported.reshape((6,))
    del first, imported, cap_a
    gc.collect()
    np.testing.assert_array_equal(view.numpy(), values.reshape(-1))
    np.testing.assert_array_equal(second.numpy(), other)
