"""Run with an isolated wheel interpreter, without the source checkout on sys.path."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import tensorcx as cx


@cx.experimental.kernel
def expression(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        total = a[i] + b[i]
        out[i] = total * b[i]


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['cpu', 'cuda', 'vulkan'], required=True)
    parser.add_argument('--mlir', action='store_true')
    parser.add_argument('--hidden-gpu', action='store_true')
    args = parser.parse_args()
    prefix = Path(sys.prefix).resolve()
    for module in (cx, cx._core):
        check(Path(module.__file__).resolve().is_relative_to(prefix), 'package imported outside fresh environment')
    distribution = importlib.metadata.distribution('tensorcx')
    direct = json.loads(distribution.read_text('direct_url.json'))
    check(not direct.get('dir_info', {}).get('editable', False), 'editable install is forbidden')
    check('archive_info' in direct, 'expected installation from a wheel archive')
    if args.variant == 'vulkan':
        check(cx._core._build_info()['vulkan'] and not cx._core._build_info()['cuda'],
              'wrong Vulkan wheel backend variant')
        check(cx.is_available('vulkan') is not args.hidden_gpu, 'Vulkan visibility differs from required mode')
        device = 'cpu' if args.hidden_gpu else 'vulkan'
        rng = np.random.default_rng(712)
        a = cx.tensor(rng.uniform(0.5, 5, (17, 1, 9)).astype(np.float32))
        b = cx.tensor(rng.uniform(0.5, 5, (1, 7, 1)).astype(np.float32))
        x, y = a.to(device), b.to(device)
        bool_source = np.resize(np.array([0, 2, 255, 1, 128], dtype=np.uint8), (3, 5)).view(np.bool_)
        bool_host = cx.from_dlpack(bool_source, copy=False)
        bool_device = bool_host.to(device)
        checks = []
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
        if not args.hidden_gpu:
            try:
                cx.sum(x)
            except ValueError as error:
                check('not supported' in str(error), str(error))
            else:
                raise RuntimeError('unsupported Vulkan reduction accepted')
            check(cx._core._vulkan_validation_errors() == 0, 'Vulkan validation errors')
            checks.append('unsupported_operation_error')
            invalid_cast = cx.tensor([1., np.nan, 2.], device=device)
            try:
                invalid_cast.astype(cx.int32)
            except ValueError as error:
                check('out of range' in str(error), str(error))
            else:
                raise RuntimeError('invalid float-to-int conversion accepted')
            np.testing.assert_array_equal(invalid_cast.numpy(), np.array([1., np.nan, 2.], dtype=np.float32))
            checks.append('invalid_cast_rejection')
        print(json.dumps({'variant': args.variant, 'device': device, 'mlir': False, 'checks': checks,
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
