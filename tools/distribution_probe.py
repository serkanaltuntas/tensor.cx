"""Run with an isolated wheel interpreter, without the source checkout on sys.path."""
import argparse
import ctypes as C
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import weakref

import numpy as np
import tensorcx as cx


@cx.experimental.kernel
def expression(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        total = a[i] + b[i]
        out[i] = total * b[i]


@cx.experimental.kernel
def stable_expression(a, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        value = cx.experimental.silu(a[i])
        out[i] = cx.experimental.select(a[i] < 0.0, value, cx.experimental.sqrt(a[i]))


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['cpu', 'cuda', 'vulkan', 'cuda-vulkan'], required=True)
    parser.add_argument('--mlir', action='store_true')
    parser.add_argument('--hidden-gpu', action='store_true')
    parser.add_argument('--_vulkan-cache-child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    prefix = Path(sys.prefix).resolve()
    for module in (cx, cx._core):
        check(Path(module.__file__).resolve().is_relative_to(prefix), 'package imported outside fresh environment')
    distribution = importlib.metadata.distribution('tensorcx')
    direct = json.loads(distribution.read_text('direct_url.json'))
    check(not direct.get('dir_info', {}).get('editable', False), 'editable install is forbidden')
    check('archive_info' in direct, 'expected installation from a wheel archive')
    if args._vulkan_cache_child:
        check(args.variant in {'vulkan', 'cuda-vulkan'} and args.mlir and not args.hidden_gpu, 'wrong cache-child variant')
        module = expression.compile(target='vulkan', compiler='mlir')
        check(module.cache_hit, 'fresh process did not reuse cached SPIR-V')
        value = cx.ones((3,), device='vulkan')
        np.testing.assert_array_equal(module.launch(value, value, value, 3).numpy(), [2., 2., 2.])
        check(cx._core._vulkan_validation_errors() == 0, 'cache-child Vulkan validation errors')
        print(json.dumps({'cache_hit': True, 'cpu_parity': True}))
        return
    # The mandatory CPU sharing reference is present in every wheel variant.
    shared_cpu = cx.shared_tensor([1., 2., 3.], devices=('cpu', 'cpu'))
    check(np.from_dlpack(shared_cpu[0]).ctypes.data == np.from_dlpack(shared_cpu[1]).ctypes.data,
          'same-CPU shared allocation copied storage')
    if args.variant in {'vulkan', 'cuda-vulkan'}:
        combined = args.variant == 'cuda-vulkan'
        check(cx._core._build_info()['vulkan'] and cx._core._build_info()['cuda'] == combined,
              'wrong Vulkan wheel backend variant')
        check(cx.is_available('vulkan') is not args.hidden_gpu, 'Vulkan visibility differs from required mode')
        device = 'cpu' if args.hidden_gpu else 'vulkan'
        from tensorcx._compiler._vulkan_helpers import SOURCE
        check('tensorcx_silu' in SOURCE, 'wheel missing canonical Vulkan compiler helpers')
        rng = np.random.default_rng(712)
        a = cx.tensor(rng.uniform(0.5, 5, (17, 1, 9)).astype(np.float32))
        b = cx.tensor(rng.uniform(0.5, 5, (1, 7, 1)).astype(np.float32))
        x, y = a.to(device), b.to(device)
        ma = cx.tensor(rng.uniform(-1, 1, (2, 1, 3, 5)).astype(np.float32))
        mb = cx.tensor(rng.uniform(-1, 1, (1, 4, 5, 7)).astype(np.float32))
        bool_source = np.resize(np.array([0, 2, 255, 1, 128], dtype=np.uint8), (3, 5)).view(np.bool_)
        bool_host = cx.from_dlpack(bool_source, copy=False)
        bool_device = bool_host.to(device)
        checks = []
        if combined:
            check(cx.is_available('cuda') is not args.hidden_gpu, 'CUDA visibility differs from required mode')
            if args.hidden_gpu:
                try: cx.shared_empty((3,), devices=('cuda', 'vulkan'))
                except ValueError: pass
                else: raise RuntimeError('hidden GPU shared allocation silently fell back')
                checks.append('hidden_shared_pair_rejected')
            else:
                runtime = C.CDLL('libcudart.so.12')
                runtime.cudaMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
                runtime.cudaMemcpy.restype = C.c_int
                runtime.cudaDeviceSynchronize.restype = C.c_int
                capsule_pointer = C.pythonapi.PyCapsule_GetPointer
                capsule_pointer.restype = C.c_void_p
                capsule_pointer.argtypes = [C.py_object, C.c_char_p]
                for pair in [('vulkan', 'cuda'), ('cuda', 'vulkan')]:
                    for dtype in ('float32', 'int32', 'bool'):
                        expected = (np.arange(257) % 2).astype(dtype)
                        aliases = cx.shared_tensor(expected, devices=pair)
                        independent = aliases[0].to(pair[1], copy=True)
                        cuda = aliases[pair.index('cuda')]
                        capsule = cuda.__dlpack__(stream=-1)
                        # Legacy DLTensor is the first field; its first field is data.
                        pointer = C.cast(capsule_pointer(capsule, b'dltensor'), C.POINTER(C.c_void_p)).contents.value
                        changed = (1 - expected).astype(dtype)
                        check(runtime.cudaMemcpy(pointer, changed.ctypes.data, changed.nbytes, 1) == 0,
                              'installed-wheel foreign shared write failed')
                        check(runtime.cudaDeviceSynchronize() == 0, 'foreign shared write incomplete')
                        for alias in aliases:
                            np.testing.assert_array_equal(alias.numpy(), changed)
                            if dtype != 'bool': np.testing.assert_array_equal((alias + 2).numpy(), changed + 2)
                        np.testing.assert_array_equal(independent.numpy(), expected)
                        borrowed = cx.from_dlpack(capsule, copy=False)
                        original_wrappers = [weakref.ref(value) for value in aliases]
                        del aliases, cuda, capsule, alias
                        gc.collect()
                        check(all(reference() is None for reference in original_wrappers),
                              'original shared wrappers survived DLPack lifetime check')
                        np.testing.assert_array_equal(borrowed.numpy(), changed)
                        for shape in [(), (0,), (3,), (5,)]:
                            values = np.ones(shape, dtype=dtype)
                            small = cx.shared_tensor(values, devices=pair)
                            for alias in small: np.testing.assert_array_equal(alias.numpy(), values)
                        checks.append('shared_' + pair[0] + '_' + dtype)
        for value in (x, bool_device, cx.empty((2,0,3), device=device)):
            forced = value.to(device, copy=True)
            check(forced is not value, 'forced same-device copy returned original')
            if value.nbytes:
                check(not cx._core._shares_storage(value._impl, forced._impl), 'forced copy shares storage')
            np.testing.assert_array_equal(forced.numpy(), value.numpy())
        checks.append('forced_same_device_copy')
        for name, actual, expected in [
            ('copy', x, a), ('fill', cx.ones((65,), device=device), cx.ones((65,))),
            ('add', x + y, a + b), ('subtract', x - y, a - b),
            ('multiply', x * y, a * b), ('divide', x / y, a / b),
            ('scalars', (3 - x) * 2, (3 - a) * 2), ('negative', -x, -a),
            ('bool_fill', cx.ones((7,), dtype=cx.bool, device=device), cx.ones((7,), dtype=cx.bool)),
            ('int32_wrap', cx.tensor([2147483647], device=device) + 1, cx.tensor([2147483647]) + 1),
            ('cast', x.astype(cx.int32), a.astype(cx.int32)),
            ('transpose', x.transpose((2, 0, 1)), a.transpose((2, 0, 1))),
            ('indexing', x[::-1, 0, ::-2], a[::-1, 0, ::-2]),
            ('concat', cx.concat([x, x[:, :, :0], x], axis=-1), cx.concat([a, a[:, :, :0], a], axis=-1)),
            ('stack', cx.stack([x, x], axis=1), cx.stack([a, a], axis=1)),
            ('split', cx.split(x, [3, 3], axis=0)[-1], cx.split(a, [3, 3], axis=0)[-1]),
            ('bool_packed_transform', cx.concat([bool_device.T] * 3, axis=-1),
                                      cx.concat([bool_host.T] * 3, axis=-1)),
            ('noncanonical_bool_cast', bool_device.astype(cx.int32), bool_host.astype(cx.int32)),
            ('compare', x > 2, a > 2),
            ('logical', bool_device & ~bool_device, bool_host & ~bool_host),
            ('where', cx.where(x > 2, x, -x), cx.where(a > 2, a, -a)),
            ('masked_select', bool_device[bool_device], bool_host[bool_host]),
            ('any', bool_device.any(axis=1, keepdims=True), bool_host.any(axis=1, keepdims=True)),
            ('all', bool_device.all(axis=(0, 1)), bool_host.all(axis=(0, 1))),
            ('sum', x.sum((0, 2), keepdims=True), a.sum((0, 2), keepdims=True)),
            ('max', x.max(2), a.max(2)), ('mean', x.mean((2, 0)), a.mean((2, 0))),
            ('min', x.min((2, 0)), a.min((2, 0))), ('argmax', x.argmax(0), a.argmax(0)),
            ('int32_sum', cx.tensor([2147483647, 2], device=device).sum(), cx.tensor([2147483647, 2]).sum()),
            ('matmul_batch_auto', cx.matmul(ma.to(device), mb.to(device)), cx.matmul(ma, mb)),
            ('matmul_vector', cx.matmul(ma[0, 0, 0].to(device), mb[0, 0].to(device)), cx.matmul(ma[0, 0, 0], mb[0, 0])),
            ('matmul_empty_k', cx.matmul(cx.empty((2, 0), device=device), cx.empty((0, 3), device=device)), cx.zeros((2, 3))),
            ('log', x.log(), a.log()), ('sqrt', x.sqrt(), a.sqrt()),
            ('abs', (-x).abs(), (-a).abs()),
            ('clip', cx.clip(x, y, 3.), cx.clip(a, b, 3.)),
            ('topk_values', cx.topk(x, 3, axis=-1, sorted=False)[0], cx.topk(a, 3, axis=-1, sorted=False)[0]),
            ('topk_indices', cx.topk(x, 3, axis=-1, sorted=False)[1], cx.topk(a, 3, axis=-1, sorted=False)[1]),
            ('exp', x.exp(), a.exp()), ('gelu', x.gelu(), a.gelu()), ('silu', x.silu(), a.silu()),
            ('softmax', x.softmax(-1), a.softmax(-1)),
            ('rmsnorm', x.rmsnorm(-1), a.rmsnorm(-1)), ('layernorm', x.layernorm(-1), a.layernorm(-1)),
            ('affine_layernorm', x.layernorm(-1, weight=cx.ones((9,), device=device), bias=cx.ones((9,), device=device)),
                                 a.layernorm(-1, weight=cx.ones((9,)), bias=cx.ones((9,)))),
            ('linear', cx.linear(x, cx.ones((4, 9), device=device)), cx.linear(a, cx.ones((4, 9)))),
            ('embedding', cx.embedding(cx.tensor([1, 0, 2], dtype=cx.int32, device=device), ma[0, 0].to(device)),
                          cx.embedding(cx.tensor([1, 0, 2], dtype=cx.int32), ma[0, 0])),
            ('attention', cx.attention(ma[0, 0].to(device), ma[1, 0].to(device), cx.ones((3, 2), device=device)),
                          cx.attention(ma[0, 0], ma[1, 0], cx.ones((3, 2)))),
        ]:
            check(actual.dtype == expected.dtype, name + ': dtype mismatch')
            if expected.dtype in (cx.int32, cx.bool):
                np.testing.assert_array_equal(actual.numpy(), expected.numpy())
            else:
                np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)
            check(actual.device == device, name + ': wrong execution device')
            checks.append(name)
        boundary_bits = np.array([0, 0x80000000, 1, 3, 0x007fffff, 0x00800000,
                                  0x40000000, 0x7f7fffff, 0x7f800000, 0x7fc12345], np.uint32)
        boundary_a = cx.tensor(boundary_bits.view(np.float32)[:, None])
        boundary_b = cx.tensor(boundary_bits.view(np.float32)[None, :])
        for name, operation in [('add', lambda a, b: a + b), ('subtract', lambda a, b: a - b),
                                ('multiply', lambda a, b: a * b), ('divide', lambda a, b: a / b)]:
            actual = operation(boundary_a.to(device), boundary_b.to(device)).numpy()
            expected = operation(boundary_a, boundary_b).numpy()
            nan = np.isnan(expected)
            np.testing.assert_array_equal(np.isnan(actual), nan)
            np.testing.assert_array_equal(actual.view(np.uint32)[~nan], expected.view(np.uint32)[~nan])
            check(np.all((actual.view(np.uint32)[nan] & np.uint32(0x00400000)) != 0), 'arithmetic NaN not quiet')
            checks.append('float32_rne_' + name)
        positive = cx.tensor(np.array([1, 3, 0x007fffff, 0x00800000, 0x3f800001], np.uint32).view(np.float32))
        for name in ('sqrt', 'log'):
            result = getattr(positive.to(device), name)().numpy()
            reference = getattr(positive, name)().numpy()
            np.testing.assert_allclose(result, reference, rtol=1e-6, atol=0)
            checks.append('subnormal_' + name)
        if not args.hidden_gpu:
            before = cx.diagnostics()['backends'][device]['allocations']['physical_blocks']
            with cx.profile(device_timing=True) as profile:
                with cx.memory_pool(device, max_cached_bytes=8192, max_cached_blocks=16) as pool:
                    for _ in range(3):
                        pooled = (x + 1).softmax(-1)
                        np.testing.assert_allclose(pooled.numpy(), (a + 1).softmax(-1).numpy(), rtol=1e-6, atol=1e-6)
                        del pooled
                    check(pool.stats()['hits'] > 0 and pool.stats()['frame_hits'] > 0, 'Vulkan reuse missing')
            check(pool.stats()['cached_bytes'] == pool.stats()['cached_frames'] == 0, 'Vulkan scope did not drain')
            check(cx.diagnostics()['backends'][device]['allocations']['physical_blocks'] == before, 'Vulkan owned memory leaked')
            recorded = profile.report()
            check(recorded['device_clock'] == 'vulkan_timestamps_elapsed_ms' and recorded['device_events'], 'Vulkan clock missing')
            if cx.diagnostics()['backends'][device]['timestamps']['supported']:
                check(recorded['device_complete'] and recorded['device_errors'] == 0, 'Vulkan timestamps failed')
            checks.extend(['bounded_pool', 'owned_allocations', 'vulkan_profile'])
            check(cx.matmul_backends(device) == ['auto', 'custom'], 'Vulkan matmul preferences')
            np.testing.assert_allclose(cx.matmul(ma.to(device), mb.to(device), backend='custom').numpy(),
                                       cx.matmul(ma, mb).numpy(), rtol=1e-4, atol=1e-4)
            checks.append('matmul_custom')
            try:
                x.__dlpack__()
            except BufferError as error:
                check('CPU and CUDA' in str(error), str(error))
            else:
                raise RuntimeError('unsupported Vulkan DLPack export accepted')
            check(cx._core._vulkan_validation_errors() == 0, 'Vulkan validation errors')
            checks.append('unsupported_dlpack_error')
            invalid_cast = cx.tensor([1., np.nan, 2.], device=device)
            try:
                invalid_cast.astype(cx.int32)
            except ValueError as error:
                check('out of range' in str(error), str(error))
            else:
                raise RuntimeError('invalid float-to-int conversion accepted')
            np.testing.assert_array_equal(invalid_cast.numpy(), np.array([1., np.nan, 2.], dtype=np.float32))
            checks.append('invalid_cast_rejection')
        if args.hidden_gpu or not args.mlir:
            try:
                expression.compile(target='vulkan', compiler='mlir', cache=False)
            except RuntimeError:
                checks.append('compiler_unavailable_error' if args.hidden_gpu else 'compiler_missing_tools_error')
            else: raise RuntimeError('Vulkan compiler unexpectedly available without GPU/tools')
        else:
            host = cx.tensor(np.linspace(-1, 1, 257, dtype=np.float32))
            left, right = host.to(device), cx.ones((257,), device=device)
            output = cx.ones((257,), device=device) * -7
            expected = expression.reference(host, cx.ones((257,)), cx.ones((257,)) * -7,
                                            129, thread_count=129, block_size=7)
            compiled = expression.compile(target='vulkan:0', compiler='mlir', cache=False)
            result = compiled.launch(left, right, output, 129, thread_count=129, block_size=7)
            np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-6, atol=0)
            np.testing.assert_array_equal(output.numpy(), np.full(257, -7, np.float32))
            checks.extend(['compiled_vulkan_kernel', 'compiled_private_prefix_suffix'])
            boundary = cx.tensor([-104., -10., -0., 0., 3., 10.])
            stable = stable_expression.compile(target='vulkan', compiler='mlir', cache=False)
            result = stable.launch(boundary.to(device), boundary.to(device), 6)
            reference = stable_expression.reference(boundary, boundary, 6)
            np.testing.assert_allclose(result.numpy(), reference.numpy(), rtol=1e-6,
                                       atol=float(np.nextafter(np.float32(0), np.float32(1))))
            checks.append('compiled_stable_intrinsics')
            previous = {key: os.environ.get(key) for key in ('TENSORCX_CACHE_DIR', 'TENSORCX_DISABLE_CACHE')}
            try:
                with tempfile.TemporaryDirectory(prefix='tensorcx-vulkan-wheel-cache-') as directory:
                    os.environ['TENSORCX_CACHE_DIR'] = str(Path(directory)/'cache')
                    os.environ['TENSORCX_DISABLE_CACHE'] = '0'
                    cold = expression.compile(target='vulkan', compiler='mlir')
                    warm = expression.compile(target='vulkan', compiler='mlir')
                    check(not cold.cache_hit and warm.cache_hit and cold._module is not warm._module,
                          'Vulkan cache did not reload a fresh native module')
                    child = subprocess.run([sys.executable, '-I', __file__, '--variant', args.variant, '--mlir',
                                            '--_vulkan-cache-child'], text=True, capture_output=True, timeout=60)
                    check(child.returncode == 0, 'installed-wheel cache-child failed')
                    check(json.loads(child.stdout) == {'cache_hit': True, 'cpu_parity': True}, 'cache-child report mismatch')
                    checks.extend(['compiled_cache_fresh_native_load', 'compiled_cache_fresh_process'])
            finally:
                for key, value in previous.items():
                    if value is None: os.environ.pop(key, None)
                    else: os.environ[key] = value
            check(cx._core._vulkan_validation_errors() == 0, 'compiled Vulkan validation errors')
        print(json.dumps({'variant': args.variant, 'device': device, 'mlir': args.mlir, 'checks': checks,
                          'package': str(Path(cx.__file__).relative_to(prefix)),
                          'extension': str(Path(cx._core.__file__).relative_to(prefix)),
                          'version': cx.__version__, 'numpy': np.__version__}))
        return
    cuda_built = hasattr(cx._core, 'CudaTensor')
    check(cuda_built == (args.variant == 'cuda'), 'wrong wheel backend variant')
    if args.hidden_gpu or args.variant == 'cpu':
        check(not cx.is_available('cuda'), 'unexpected available CUDA device')
        device = 'cpu'
    else:
        check(cx.is_available('cuda'), 'CUDA required; cannot skip or fall back')
        device = 'cuda'
    rng = np.random.default_rng(17)
    a = cx.tensor(rng.uniform(0.25, 1, (17, 31)).astype(np.float32), device='cpu')
    b = cx.tensor(rng.uniform(0.25, 1, (31, 9)).astype(np.float32), device='cpu')
    da, db = a.to(device), b.to(device)
    checks = []

    def compare(name, actual, expected, tolerance=1e-5):
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=tolerance, atol=tolerance)
        check(actual.device == device, name + ': wrong execution device')
        checks.append(name)

    compare('copy', da, a)
    forced = da.to(device, copy=True)
    check(forced is not da and not cx._core._shares_storage(da._impl, forced._impl), 'forced same-device copy shares storage')
    compare('forced_same_device_copy', forced, a)
    compare('dlpack_shared', cx.from_dlpack(da, copy=False), a)
    compare('dlpack_copy', cx.from_dlpack(da, copy=True), a)
    array = np.arange(6, dtype=np.float32)
    shared = cx.from_dlpack(array, copy=False)
    check(np.from_dlpack(shared).ctypes.data == array.ctypes.data, 'NumPy DLPack copied storage')
    array[0] = 17
    np.testing.assert_array_equal(shared.numpy(), array)
    checks.append('dlpack_numpy_shared')
    compare('fill', cx.ones((17, 31), device=device), cx.ones((17, 31), device='cpu'))
    compare('add', da + da, a + a)
    compare('multiply', da * da, a * a)
    compare('matmul', cx.matmul(da, db), cx.matmul(a, b), 1e-4)
    for name in ('exp', 'gelu', 'silu'):
        operation = getattr(cx, name)
        compare(name, operation(da), operation(a))
    for name in ('sum', 'max', 'mean', 'softmax', 'rmsnorm', 'layernorm'):
        operation = getattr(cx, name)
        for axis in (0, -1):
            compare(f'{name}:{axis}', operation(da, axis=axis), operation(a, axis=axis))
    if args.mlir:
        artifact = expression.compile(target=device, compiler='mlir')
        compare('mlir_expression', artifact.launch(da, da, da, 17 * 31), (a + a) * a)
        compare('mlir_preserves_input', da, a)
    else:
        # Regular tensor use must work without LLVM. Explicit compilation must
        # fail clearly when the orchestrator hides the external tools.
        try:
            expression.compile(target=device, compiler='mlir')
        except RuntimeError as error:
            check('MLIR requires executable' in str(error), str(error))
        else:
            raise RuntimeError('expected missing LLVM error')
        checks.append('missing_llvm_error')
    print(json.dumps({'variant': args.variant, 'device': device, 'mlir': args.mlir,
                      'checks': checks, 'package': str(Path(cx.__file__).relative_to(prefix)),
                      'extension': str(Path(cx._core.__file__).relative_to(prefix)),
                      'version': cx.__version__, 'numpy': np.__version__}))


if __name__ == '__main__':
    main()
