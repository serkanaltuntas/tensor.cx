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
        checks = []
        for name, actual, expected in [
            ('copy', x, a), ('fill', cx.ones((65,), device=device), cx.ones((65,))),
            ('add', x + y, a + b), ('subtract', x - y, a - b),
            ('multiply', x * y, a * b), ('divide', x / y, a / b),
            ('scalars', (3 - x) * 2, (3 - a) * 2), ('negative', -x, -a),
            ('bool_fill', cx.ones((7,), dtype=cx.bool, device=device), cx.ones((7,), dtype=cx.bool)),
            ('int32_wrap', cx.tensor([2147483647], device=device) + 1, cx.tensor([2147483647]) + 1),
        ]:
            check(actual.dtype == expected.dtype, name + ': dtype mismatch')
            if expected.dtype in (cx.int32, cx.bool):
                np.testing.assert_array_equal(actual.numpy(), expected.numpy())
            else:
                np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)
            check(actual.device == device, name + ': wrong execution device')
            checks.append(name)
        if not args.hidden_gpu:
            try:
                cx.sum(x)
            except ValueError as error:
                check('not supported' in str(error), str(error))
            else:
                raise RuntimeError('unsupported Vulkan reduction accepted')
            check(cx._core._vulkan_validation_errors() == 0, 'Vulkan validation errors')
            checks.append('unsupported_operation_error')
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
