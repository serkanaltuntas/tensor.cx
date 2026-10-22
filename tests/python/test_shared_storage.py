"""Public shared allocation, real CUDA mutation and Vulkan ownership handoffs."""
import ctypes as C
import gc
import os
import json
import subprocess
import sys
import textwrap
import weakref
from dataclasses import replace

import numpy as np
import pytest
import tensorcx as cx
from tensorcx import backend
from test_dlpack import description


@cx.experimental.kernel
def shared_add(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = a[i] + b[i]


@pytest.fixture(autouse=True)
def validation_clean():
    yield
    if hasattr(cx._core, '_vulkan_validation_errors'):
        assert cx._core._vulkan_validation_errors() == 0


@pytest.fixture(params=[('vulkan', 'cuda'), ('cuda', 'vulkan')])
def pair(request):
    required = os.environ.get('TENSORCX_REQUIRE_SHARED_CUDA_VULKAN') == '1'
    if not all(cx.is_available(name) for name in ('cuda', 'vulkan')) or not hasattr(cx._core, '_shared_cuda_vulkan_support'):
        if required:
            pytest.fail('required CUDA/Vulkan sharing build/devices unavailable')
        pytest.skip('requires combined CUDA/Vulkan build and physical GPU')
    try:
        backend.shared_route(*request.param)
    except (RuntimeError, ValueError) as error:
        if required:
            raise
        pytest.skip(f'CUDA/Vulkan device pair cannot share: {error}')
    return request.param


@pytest.fixture
def cudart(pair):
    runtime = C.CDLL('libcudart.so.12')
    runtime.cudaMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
    runtime.cudaMemcpy.restype = C.c_int
    runtime.cudaDeviceSynchronize.restype = C.c_int
    return runtime


@pytest.mark.parametrize('factory', [cx.shared_tensor, cx.shared_empty])
@pytest.mark.parametrize('device', cx.devices())
def test_same_device_alias_reference(factory, device):
    left, right = factory([3, 5] if factory is cx.shared_tensor else (2,), devices=(device, cx.device(device)))
    assert left is not right and left._impl is right._impl
    assert cx._core._shares_storage(left._impl, right._impl)
    if device == 'cpu':
        a, b = np.from_dlpack(left), np.from_dlpack(right)
        assert a.ctypes.data == b.ctypes.data
    del left
    gc.collect()
    assert right.shape == (2,)


@pytest.mark.parametrize('devices', [None, 'cpu', (), ('cpu',), ('cpu', 'cpu', 'cpu'), (None, 'cpu'), (1, 'cpu')])
@pytest.mark.parametrize('factory', [cx.shared_tensor, cx.shared_empty])
def test_bad_pair_precedes_data_or_shape_materialization(devices, factory):
    class Poison:
        def __array__(self, *args, **kwargs):
            pytest.fail('invalid pair materialized input')
        def __iter__(self):
            pytest.fail('invalid pair iterated shape')
    with pytest.raises((TypeError, ValueError)):
        factory(Poison(), devices=devices)


def test_unsupported_pair_has_no_copy_or_data_fallback(monkeypatch):
    root = backend._BACKENDS['cpu']
    monkeypatch.setitem(backend._BACKENDS, 'other', replace(root, name='other'))
    class Poison:
        def __array__(self, *args, **kwargs):
            pytest.fail('unsupported pair materialized input')
    with pytest.raises(ValueError, match='unsupported'):
        cx.shared_tensor(Poison(), devices=('cpu', 'other'))


@pytest.mark.parametrize('dtype', ['float32', 'int32', 'bool'])
@pytest.mark.parametrize('shape', [(), (0,), (2, 0, 3), (1,), (3,), (5,), (257,), (3, 5)])
def test_factories_metadata_bits_views_lifetime(pair, dtype, shape):
    expected = (np.arange(int(np.prod(shape))).reshape(shape) % 3).astype(dtype)
    left, right = cx.shared_tensor(expected, devices=pair)
    assert (left.device, right.device) == pair
    assert left.dtype == right.dtype == dtype
    assert left.shape == right.shape == shape and left.nbytes == expected.nbytes
    np.testing.assert_array_equal(left.numpy(), expected)
    np.testing.assert_array_equal(right.numpy(), expected)
    aliases = (left.reshape((-1,)), right.reshape((-1,)))
    del left, right
    gc.collect()
    for alias in aliases:
        np.testing.assert_array_equal(alias.numpy(), expected.reshape(-1))
    first, second = cx.shared_empty(shape, dtype=dtype, devices=pair)
    assert first.shape == second.shape == shape and first.dtype == second.dtype == dtype
    assert first.nbytes == second.nbytes == expected.nbytes
    # Reading empty contents is valid; nonempty contents are unspecified.
    if not expected.size:
        assert first.numpy().size == second.numpy().size == 0


@pytest.mark.parametrize('dtype', ['float32', 'int32', 'bool'])
def test_foreign_cuda_mutation_visible_via_both_aliases(pair, cudart, dtype):
    expected = (np.arange(257) % 2).astype(dtype)
    aliases = cx.shared_empty(expected.shape, dtype=dtype, devices=pair)
    cuda = aliases[pair.index('cuda')]
    capsule = cuda.__dlpack__(stream=-1)
    pointer = description(capsule).data
    for turn in range(10):
        expected[:] = (np.arange(257) + turn) % 2
        assert cudart.cudaMemcpy(pointer, expected.ctypes.data, expected.nbytes, 1) == 0
        assert cudart.cudaDeviceSynchronize() == 0
        for alias in aliases:
            np.testing.assert_array_equal(alias.numpy(), expected)
    retained = cx.from_dlpack(capsule, copy=False)
    original_wrappers = [weakref.ref(value) for value in aliases]
    del aliases, cuda, capsule, alias
    gc.collect()
    assert all(reference() is None for reference in original_wrappers)
    np.testing.assert_array_equal(retained.numpy(), expected)
    if dtype != 'bool':
        np.testing.assert_array_equal((retained + 2).numpy(), expected + 2)


@pytest.mark.parametrize('dtype,shape,offset', [('float32', 4, 0), ('bool', 4, 0), ('bool', 1, 3)])
def test_dlpack_known_shared_overrun_and_padding_rejected(pair, dtype, shape, offset):
    aliases = cx.shared_tensor(np.ones(3, dtype=dtype), devices=pair)
    cuda = aliases[pair.index('cuda')]
    capsule = cuda.__dlpack__()
    descriptor = description(capsule)
    descriptor.shape[0] = shape
    descriptor.offset = offset
    with pytest.raises((BufferError, ValueError, RuntimeError), match='range'):
        cx.from_dlpack(capsule, copy=False)


def test_dlpack_contiguous_subview_preserves_shared_provenance(pair):
    aliases = cx.shared_tensor(np.arange(5, dtype=np.float32), devices=pair)
    cuda = aliases[pair.index('cuda')]
    capsule = cuda.__dlpack__()
    descriptor = description(capsule)
    descriptor.shape[0] = 3
    descriptor.offset = 4
    view = cx.from_dlpack(capsule, copy=False)
    del aliases, cuda, capsule
    gc.collect()
    np.testing.assert_array_equal((view + view).numpy(), [2., 4., 6.])


def test_nested_composites_and_reversed_input_lock_order(pair):
    # Executor shutdown itself waits forever on a deadlocked worker. Isolate the
    # native lock-order regression so the parent can actually enforce a deadline.
    script = textwrap.dedent('''
        import json, sys
        from concurrent.futures import ThreadPoolExecutor
        import numpy as np
        import tensorcx as cx
        pair = json.loads(sys.argv[1])
        expected = np.arange(15, dtype=np.float32).reshape(3, 5) * 0.125
        a = cx.shared_tensor(expected, devices=pair)
        b = cx.shared_tensor(expected + 1, devices=pair)
        reference = cx.layernorm(cx.tensor(expected), axis=-1,
                                 weight=cx.ones((5,)), bias=cx.ones((5,))).numpy()
        def run(index):
            left, right = (a[index], b[index]) if index == 0 else (b[index], a[index])
            for _ in range(12):
                np.testing.assert_array_equal((left + right).numpy(), 2 * expected + 1)
                np.testing.assert_allclose(cx.layernorm(a[index], axis=-1,
                    weight=cx.ones((5,), device=pair[index]), bias=cx.ones((5,), device=pair[index])).numpy(),
                    reference, rtol=1e-5, atol=1e-6)
                np.testing.assert_array_equal(a[index].numpy(), expected)
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(run, (0, 1)))
        assert cx._core._vulkan_validation_errors() == 0
        print('shared concurrent composites passed')
    ''')
    result = subprocess.run([sys.executable, '-c', script, json.dumps(pair)], capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'shared concurrent composites passed'


def test_to_copy_is_independent_from_explicit_shared_pair(pair, cudart):
    aliases = cx.shared_tensor([1., 2., 3.], devices=pair)
    copies = [alias.to(pair[1-index], copy=True) for index, alias in enumerate(aliases)]
    cuda = aliases[pair.index('cuda')]
    capsule = cuda.__dlpack__()
    changed = np.array([7., 8., 9.], np.float32)
    assert cudart.cudaMemcpy(description(capsule).data, changed.ctypes.data, changed.nbytes, 1) == 0
    assert cudart.cudaDeviceSynchronize() == 0
    for alias in aliases:
        np.testing.assert_array_equal(alias.numpy(), changed)
    for copy in copies:
        np.testing.assert_array_equal(copy.numpy(), [1., 2., 3.])


@pytest.mark.parametrize('target', ['cuda', 'vulkan'])
@pytest.mark.parametrize('threads', [0, 3, 257])
@pytest.mark.parametrize('alias_output', [False, True])
def test_generated_kernel_shared_inputs_private_output_and_suffix(pair, target, threads, alias_output):
    from tensorcx._compiler import cuda, vulkan
    compiler = cuda if target == 'cuda' else vulkan
    try:
        compiler.toolchain()
    except (RuntimeError, OSError) as error:
        if os.environ.get('TENSORCX_REQUIRE_MLIR_' + target.upper()):
            pytest.fail(str(error))
        pytest.skip(str(error))
    host = np.arange(257, dtype=np.float32) * 0.25
    a = cx.shared_tensor(host, devices=pair)
    b = cx.shared_tensor(host + 1, devices=pair)
    template = a if alias_output else cx.shared_tensor(np.full(257, -7, np.float32), devices=pair)
    index = pair.index(target)
    module = shared_add.compile(target=target, compiler='mlir', cache=False)
    result = module.launch(a[index], b[index], template[index], threads, thread_count=threads, block_size=7)
    expected = host.copy() if alias_output else np.full(257, -7, np.float32)
    expected[:threads] = 2 * host[:threads] + 1
    np.testing.assert_array_equal(result.numpy(), expected)
    assert result.device == target
    for index in (0, 1):
        np.testing.assert_array_equal(a[index].numpy(), host)
        np.testing.assert_array_equal(b[index].numpy(), host + 1)
        np.testing.assert_array_equal(template[index].numpy(), host if alias_output else np.full(257, -7, np.float32))
