"""Explicit placement, independent copies and host-staged accelerator pairs."""
from dataclasses import replace
import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import tensorcx as cx
from tensorcx import backend


DEVICES = tuple(cx.devices())
PAIRS = [(source, target) for source in DEVICES for target in DEVICES]


@pytest.fixture(autouse=True)
def validation_clean():
    yield
    if hasattr(cx._core, '_vulkan_validation_errors'):
        assert cx._core._vulkan_validation_errors() == 0


def raw_array(shape, dtype):
    count = int(np.prod(shape))
    if dtype == 'float32':
        bits = np.array([0, 0x80000000, 1, 0x007fffff, 0x7f812345,
                         0xffc12345, 0x7f800000, 0xff800000, 0x3f800001], np.uint32)
        return np.resize(bits, count).view(np.float32).reshape(shape)
    if dtype == 'int32':
        return np.resize(np.array([0, -1, 2**31-1, -2**31, 19], np.int32), count).reshape(shape)
    return (np.arange(count) % 3 == 0).reshape(shape)


def same_bits(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_array_equal(actual.reshape(-1).view(np.uint8), expected.reshape(-1).view(np.uint8))


@pytest.mark.parametrize('source,target', PAIRS)
@pytest.mark.parametrize('dtype', ['float32', 'int32', 'bool'])
@pytest.mark.parametrize('shape', [(), (0,), (2, 0, 3), (1,), (3,), (5,), (257,), (3, 5)])
@pytest.mark.parametrize('force_copy', [False, True])
def test_transfer_matrix_bits_metadata_alias_and_lifetime(source, target, dtype, shape, force_copy):
    expected = raw_array(shape, dtype)
    # Sequence construction can quiet signaling NaNs. Preserve raw CPU bytes.
    original = cx.from_dlpack(expected, copy=False).to(source)
    result = original.to(cx.Device(target.partition(':')[0], int(target.partition(':')[2] or 0)), copy=force_copy)
    assert (result is original) == (source == target and not force_copy)
    assert result.device == target and result.dtype == dtype and result.shape == shape
    assert result.nbytes == original.nbytes
    same_bits(result.numpy(), expected)
    same_bits(original.numpy(), expected)
    if source == target and force_copy and result.nbytes:
        assert not cx._core._shares_storage(original._impl, result._impl)
    del original; gc.collect()
    same_bits(result.numpy(), expected)


@pytest.mark.parametrize('copy', [0, 1, None, 'yes', np.bool_(True)])
def test_copy_flag_is_strict_before_device_access(copy):
    with pytest.raises(TypeError, match='copy must be bool'):
        cx.ones((1,)).to('missing', copy=copy)


def test_cpu_copy_preserves_borrowed_owner_and_mutation_independence():
    array = raw_array((257,), 'float32')
    expected = array.copy(); retained = weakref.ref(array)
    original = cx.from_dlpack(array, copy=False)
    independent = original.to('cpu', copy=True)
    array.view(np.uint32)[:] = 0
    same_bits(independent.numpy(), expected)
    del original, array; gc.collect()
    assert retained() is None
    same_bits(independent.numpy(), expected)


def test_registry_indexed_pair_uses_host_staging_and_retains_it(monkeypatch):
    calls = []; retained = []
    def download(value):
        calls.append(('download', value.device))
        class Host: pass
        host = Host(); retained.append(weakref.ref(host)); return host
    def upload(host):
        assert retained[-1]() is host
        calls.append(('upload', 'vulkan:1'))
        return SimpleNamespace(device='vulkan:1')
    root = backend.Backend('vulkan', 0, lambda: True, lambda: 'mock',
                           indexed=True, copy_to_cpu=download)
    monkeypatch.setitem(backend._BACKENDS, 'vulkan', root)
    monkeypatch.setitem(backend._BACKENDS, 'vulkan:1', replace(root, name='vulkan:1', copy_from_cpu=upload))
    result = cx.Tensor(SimpleNamespace(device='vulkan')).to('vulkan:1')
    assert result.device == 'vulkan:1' and calls == [('download','vulkan'), ('upload','vulkan:1')]
    gc.collect(); assert retained[-1]() is None
    # This is routing evidence; only real two-GPU hardware can prove execution.


@pytest.mark.parametrize('fault', ['unavailable', 'missing_upload', 'missing_download', 'download', 'upload'])
def test_failed_pair_never_publishes_fallback_or_mutates_source(monkeypatch, fault):
    calls = []; value = object()
    def download(source):
        assert source is value; calls.append('download')
        if fault == 'download': raise MemoryError('download allocation failed')
        return object()
    def upload(host):
        calls.append('upload'); raise MemoryError('destination allocation failed')
    src = backend.Backend('src', 0, lambda: True, lambda: 'src', copy_to_cpu=download)
    dst = backend.Backend('dst', 0, lambda: fault != 'unavailable', lambda: 'dst', copy_from_cpu=upload)
    if fault == 'missing_upload': dst = replace(dst, copy_from_cpu=None)
    if fault == 'missing_download': src = replace(src, copy_to_cpu=None)
    monkeypatch.setitem(backend._BACKENDS, 'src', src); monkeypatch.setitem(backend._BACKENDS, 'dst', dst)
    with pytest.raises((ValueError, MemoryError)):
        backend.copy_tensor(value, 'src', 'dst')
    assert calls == (['download','upload'] if fault == 'upload' else ['download'] if fault == 'download' else [])


def test_forced_clone_failure_propagates_without_staging(monkeypatch):
    value = cx.ones((3,)); route = backend._BACKENDS['cpu']
    def fail(impl): raise MemoryError('clone allocation failed')
    monkeypatch.setitem(backend._BACKENDS, 'cpu', replace(route, clone=fail))
    with pytest.raises(MemoryError): value.to('cpu', copy=True)
    np.testing.assert_array_equal(value.numpy(), [1.,1.,1.])
    assert value.to('cpu') is value
