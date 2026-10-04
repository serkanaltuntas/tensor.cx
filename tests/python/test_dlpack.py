"""DLPack ownership, zero-copy, protocol negotiation and native execution."""
import ctypes as C
import gc
import weakref

import numpy as np
import pytest
import tensorcx as cx


@pytest.fixture(params=[d for d in cx.devices() if d in ('cpu', 'cuda')])
def device(request): return request.param


@pytest.mark.parametrize('dtype', ['float32', 'int32', 'bool'])
@pytest.mark.parametrize('shape', [(), (6,), (2, 3), (2, 0, 3), (1, 2, 1)])
@pytest.mark.parametrize('version', [None, (0, 8), (1, 0), (2, 0)])
def test_dlpack_roundtrip_native(device, dtype, shape, version):
    a = (np.arange(np.prod(shape, dtype=int)).reshape(shape) % 2).astype(dtype)
    source = cx.tensor(a, device=device)
    capsule = source.__dlpack__(max_version=version)
    imported = cx.from_dlpack(capsule, copy=False)
    assert imported.device == device and imported.dtype == dtype and imported.shape == shape
    np.testing.assert_array_equal(imported.numpy(), a)
    # Borrowed buffers remain ordinary backend inputs, including bool/int32.
    np.testing.assert_array_equal(imported.transpose().numpy(), a.transpose())
    if dtype == 'float32':
        np.testing.assert_array_equal((imported + imported).numpy(), a + a)
    with pytest.raises(BufferError, match='unconsumed'): cx.from_dlpack(capsule)
    del source, capsule
    gc.collect()
    np.testing.assert_array_equal(imported.reshape((-1,)).numpy(), a.reshape(-1))


@pytest.mark.parametrize('dtype', ['float32', 'int32', 'bool'])
@pytest.mark.parametrize('shape', [(), (5,), (2, 3), (0, 3), (1, 2, 1)])
def test_numpy_zero_copy_and_lifetime(dtype, shape):
    source = np.ones(shape, dtype=dtype)
    owner = weakref.ref(source)
    tensor = cx.from_dlpack(source, copy=False)
    exported = np.from_dlpack(tensor)
    if source.size:
        assert exported.ctypes.data == source.ctypes.data
        source.flat[0] = 0
        np.testing.assert_array_equal(tensor.numpy(), source)
    view = tensor.reshape((-1,))
    del source, tensor, exported
    gc.collect()
    assert owner() is not None
    final = np.from_dlpack(view)
    del view
    gc.collect()
    assert owner() is not None
    del final
    gc.collect()
    assert owner() is None


@pytest.mark.parametrize('copy', [None, False, True])
def test_dlpack_copy_contract(device, copy):
    source = cx.tensor([1., 2., 3.], device=device)
    shared = cx.from_dlpack(source, copy=copy)
    # Inspect addresses via the protocol, without a host data copy.
    a, b = source.__dlpack__(), shared.__dlpack__()
    pa, pb = description(a), description(b)
    assert (pa.data == pb.data) is (copy is not True)
    np.testing.assert_array_equal(shared.numpy(), source.numpy())
    capsule = source.__dlpack__(max_version=(1, 0), copy=True)
    ptr = capsule_pointer(capsule, b'dltensor_versioned')
    t = C.cast(ptr, C.POINTER(Versioned)).contents
    assert t.flags & 2
    assert t.tensor.data != pa.data
    np.testing.assert_array_equal(cx.from_dlpack(capsule).numpy(), source.numpy())


@pytest.mark.parametrize('copy', [0, 1, 'yes', [], np.bool_(True)])
def test_dlpack_bad_copy(copy):
    source = cx.ones((2,))
    with pytest.raises(TypeError): source.__dlpack__(copy=copy)
    with pytest.raises(TypeError): cx.from_dlpack(source, copy=copy)


def test_dlpack_discarded_export_releases_owner():
    source = np.ones(3, np.float32)
    owner = weakref.ref(source)
    tensor = cx.from_dlpack(source)
    capsule = tensor.__dlpack__(max_version=(1, 0))
    del tensor, source
    gc.collect()
    assert owner() is not None
    del capsule
    gc.collect()
    assert owner() is None


class Device(C.Structure):
    _fields_ = [('type', C.c_int32), ('id', C.c_int32)]
class DType(C.Structure):
    _fields_ = [('code', C.c_uint8), ('bits', C.c_uint8), ('lanes', C.c_uint16)]
class Description(C.Structure):
    _fields_ = [('data', C.c_void_p), ('device', Device), ('ndim', C.c_int32),
                ('dtype', DType), ('shape', C.POINTER(C.c_int64)), ('strides', C.POINTER(C.c_int64)),
                ('offset', C.c_uint64)]
Deleter = C.CFUNCTYPE(None, C.c_void_p)
class Legacy(C.Structure):
    _fields_ = [('tensor', Description), ('context', C.c_void_p), ('deleter', Deleter)]
class Version(C.Structure):
    _fields_ = [('major', C.c_uint32), ('minor', C.c_uint32)]
class Versioned(C.Structure):
    _fields_ = [('version', Version), ('context', C.c_void_p), ('deleter', Deleter),
                ('flags', C.c_uint64), ('tensor', Description)]
new_capsule = C.pythonapi.PyCapsule_New
new_capsule.restype = C.py_object
new_capsule.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p]
capsule_pointer = C.pythonapi.PyCapsule_GetPointer
capsule_pointer.restype = C.c_void_p
capsule_pointer.argtypes = [C.py_object, C.c_char_p]


def description(capsule):
    return C.cast(capsule_pointer(capsule, b'dltensor'), C.POINTER(Legacy)).contents.tensor


def fixture_capsule(*, versioned=True, shape=(2, 3), strides=None, offset=0, flags=0):
    state = {'deleted': 0}
    state['data'] = np.arange(64, dtype=np.float32)
    state['shape'] = (C.c_int64 * len(shape))(*shape)
    state['strides'] = None if strides is None else (C.c_int64 * len(strides))(*strides)
    def delete(_): state['deleted'] += 1
    state['delete'] = Deleter(delete)
    t = Description(state['data'].ctypes.data, Device(1, 0), len(shape), DType(2, 32, 1),
                    state['shape'], state['strides'], offset)
    state['managed'] = Versioned(Version(1, 0), None, state['delete'], flags, t) if versioned else Legacy(t, None, state['delete'])
    # The fixture keeps native descriptors alive. No destructor is needed for an
    # unconsumed fixture capsule; all allocated memory belongs to this Python state.
    cap = new_capsule(C.addressof(state['managed']), b'dltensor_versioned' if versioned else b'dltensor', None)
    return cap, state


@pytest.mark.parametrize('versioned', [False, True])
def test_dlpack_offset_and_deleter_exactly_once(versioned):
    cap, state = fixture_capsule(versioned=versioned, offset=4, shape=(2, 1, 3), strides=(3, 99, 1))
    tensor = cx.from_dlpack(cap)
    assert state['deleted'] == 0
    np.testing.assert_array_equal(tensor.numpy(), np.arange(1, 7, dtype=np.float32).reshape(2, 1, 3))
    reexport = tensor.__dlpack__()
    view = tensor.reshape((6,))
    del tensor, cap, view
    gc.collect()
    assert state['deleted'] == 0
    del reexport
    gc.collect()
    assert state['deleted'] == 1


@pytest.mark.parametrize('case', ['version', 'flags', 'device', 'index', 'dtype', 'lanes', 'negative_shape',
                                  'negative_rank', 'null_shape', 'null_data', 'alignment', 'offset_overflow', 'strides'])
def test_dlpack_invalid_descriptor_consumes_once(case):
    cap, state = fixture_capsule()
    t = state['managed'].tensor
    if case == 'version': state['managed'].version.major = 2
    if case == 'flags': state['managed'].flags = 4
    if case == 'device': t.device.type = 8
    if case == 'index': t.device.id = 1
    if case == 'dtype': t.dtype.bits = 64
    if case == 'lanes': t.dtype.lanes = 2
    if case == 'negative_shape': state['shape'][0] = -1
    if case == 'negative_rank': t.ndim = -1
    if case == 'null_shape': t.shape = None
    if case == 'null_data': t.data = None
    if case == 'alignment': t.offset = 1
    if case == 'offset_overflow': t.offset = 2**64 - 1
    if case == 'strides':
        state['strides'] = (C.c_int64 * 2)(1, 2)
        t.strides = state['strides']
    with pytest.raises((BufferError, ValueError)): cx.from_dlpack(cap)
    assert state['deleted'] == 1
    with pytest.raises(BufferError): cx.from_dlpack(cap)
    assert state['deleted'] == 1


@pytest.mark.parametrize('copy', [None, False, True])
def test_dlpack_readonly_policy(copy):
    cap, state = fixture_capsule(flags=1)
    if copy is True:
        tensor = cx.from_dlpack(cap, copy=True)
        assert state['deleted'] == 1
        np.testing.assert_array_equal(tensor.numpy(), state['data'][:6].reshape(2, 3))
        state['data'][0] = -10
        assert tensor.numpy()[0, 0] == 0
    else:
        with pytest.raises(BufferError, match='read-only'): cx.from_dlpack(cap, copy=copy)
        assert state['deleted'] == 1


def test_dlpack_copy_false_rejects_producer_copy():
    cap, state = fixture_capsule(flags=2)
    with pytest.raises(BufferError, match='copied'): cx.from_dlpack(cap, copy=False)
    assert state['deleted'] == 1


def test_dlpack_independent_wrappers_storage_relation(device):
    source = cx.tensor(np.arange(7, dtype=np.float32), device=device)
    a, b = cx.from_dlpack(source), cx.from_dlpack(source)
    assert cx._core._shares_storage(a._impl, b._impl)
    assert cx._core._shares_storage(source._impl, a._impl)
    left, right = source.__dlpack__(), source.__dlpack__()
    description(left).shape[0] = 6
    description(right).shape[0] = 6
    description(right).offset = 4
    a, b = cx.from_dlpack(left), cx.from_dlpack(right)
    np.testing.assert_array_equal(a.numpy(), np.arange(6, dtype=np.float32))
    np.testing.assert_array_equal(b.numpy(), np.arange(1, 7, dtype=np.float32))
    with pytest.raises(ValueError, match='overlap'):
        cx._core._shares_storage(a._impl, b._impl)


def test_dlpack_protocol_negotiation(device):
    x = cx.ones((2, 3), device=device)
    calls = []
    class Producer:
        def __dlpack_device__(self): return x.__dlpack_device__()
        def __dlpack__(self, *, stream=None, max_version=None):
            calls.append((stream, max_version))
            return x.__dlpack__(stream=stream, max_version=max_version)
    result = cx.from_dlpack(Producer())
    np.testing.assert_array_equal(result.numpy(), x.numpy())
    assert calls == [(1 if device == 'cuda' else None, (1, 0))]
    class LegacyProducer:
        def __dlpack_device__(self): return x.__dlpack_device__()
        def __dlpack__(self, stream=None): return x.__dlpack__(stream=stream)
    np.testing.assert_array_equal(cx.from_dlpack(LegacyProducer()).numpy(), x.numpy())
    class Broken(Producer):
        def __dlpack__(self, **kwargs):
            calls.append('broken')
            raise TypeError('producer internal error')
    with pytest.raises(TypeError, match='internal'): cx.from_dlpack(Broken())
    assert calls.count('broken') == 1


def test_dlpack_device_mismatch():
    cap, state = fixture_capsule()
    class Liar:
        def __dlpack_device__(self): return (2, 0)
        def __dlpack__(self, **kwargs): return cap
    with pytest.raises(BufferError, match='differs'): cx.from_dlpack(Liar())
    assert state['deleted'] == 1


def test_dlpack_empty_noncanonical_strides():
    cap, state = fixture_capsule(shape=(2, 0, 3), strides=(3, 3, 1))
    result = cx.from_dlpack(cap)
    assert result.shape == (2, 0, 3)
    assert result.numpy().size == 0


@pytest.mark.parametrize('stream', [-1, 0, 1, 2, True, 'stream'])
def test_dlpack_cpu_rejects_stream(stream):
    with pytest.raises(BufferError): cx.ones((1,)).__dlpack__(stream=stream)


def test_dlpack_cuda_stream_contract(device):
    if device == 'cuda':
        with pytest.raises(BufferError, match='synchronization'):
            cx.ones((2,), device=device).__dlpack__(stream=-1, copy=True)
    if device != 'cuda': return
    x = cx.ones((1,), device=device)
    for stream in (None, -1, 1, 2, 123456):
        assert cx.from_dlpack(x.__dlpack__(stream=stream)).device == device
    for stream in (2, 123456):
        with pytest.raises(BufferError, match='default stream'):
            x.__dlpack__(stream=stream, copy=True)
    for stream in (0, -2):
        with pytest.raises(ValueError): x.__dlpack__(stream=stream)
    with pytest.raises(TypeError): x.__dlpack__(stream=True)


def test_dlpack_no_implicit_copy_for_layout():
    x = np.arange(12, dtype=np.float32).reshape(3, 4).T
    for copy in (None, False, True):
        with pytest.raises(BufferError, match='contiguous'): cx.from_dlpack(x, copy=copy)


def test_dlpack_unsupported_export():
    x = cx.ones((2,))
    with pytest.raises(BufferError): x.__dlpack__(dl_device=(2, 0))
    if cx.is_available('metal'):
        with pytest.raises(BufferError): cx.ones((2,), device='metal').__dlpack__()
