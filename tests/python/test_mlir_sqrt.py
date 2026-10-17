"""Guarded float32 sqrt: rounding, composition, validation and ownership."""
from dataclasses import replace
import importlib.util
import os
import re

import numpy as np
import pytest
import tensorcx as cx
from tensorcx._compiler import cpu, cuda
from tensorcx._compiler.emitter import emit_mlir, MlirEmitError
from tensorcx.experimental import (
    IRAssign, IRBinaryOp, IRCall, IRCompare, IRConstant, IRLoad, IRName,
    KernelCompileError, _emit_msl,
)
from test_mlir_runtime import require_tools
from test_mlir_cuda_runtime import require_device


@cx.experimental.kernel
def root(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.sqrt(a[i] * b[i])


@cx.experimental.kernel
def root_local(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        value = cx.experimental.sqrt(a[i] * a[i] + b[i] * b[i])
        out[i] = cx.experimental.sqrt(value) / 2.0


@cx.experimental.kernel
def root_select(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        value = cx.experimental.select(a[i] < b[i], a[i], b[i])
        out[i] = cx.experimental.select(cx.experimental.sqrt(value) > 1.0, cx.experimental.sqrt(value), cx.experimental.sqrt(0.25))


@cx.experimental.kernel
def root_shadow(sqrt, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.sqrt(sqrt[i] * b[i])


@cx.experimental.kernel
def root_reordered(n, out, b, a):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.sqrt(a[i] * b[i])


KERNELS = [root, root_local, root_select, root_shadow]


def oracle(dsl, a, b):
    with np.errstate(all='ignore'):
        if dsl is root_local:
            value = np.sqrt(np.add(np.multiply(a, a, dtype=np.float32),
                                  np.multiply(b, b, dtype=np.float32), dtype=np.float32), dtype=np.float32)
            return np.divide(np.sqrt(value, dtype=np.float32), np.float32(2), dtype=np.float32)
        if dsl is root_select:
            value = np.sqrt(np.where(a < b, a, b), dtype=np.float32)
            return np.where(value > np.float32(1), value, np.float32(.5))
        return np.sqrt(np.multiply(a, b, dtype=np.float32), dtype=np.float32)


def assert_bits(actual, expected):
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    valid = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[valid].view(np.uint32), expected[valid].view(np.uint32))


@pytest.fixture(scope='module', params=['cpu', 'cuda'])
def target(request):
    (require_tools if request.param == 'cpu' else require_device)()
    return request.param


@pytest.fixture(scope='module', params=KERNELS)
def compiled(request, target):
    return request.param, target, request.param.compile(target=target, compiler='mlir', cache=False)


@pytest.mark.parametrize('shape', [(), (0,), (1,), (255,), (256,), (257,), (2, 3)])
@pytest.mark.parametrize('block', [1, 7, 256])
def test_sqrt_parity(compiled, shape, block):
    dsl, target, native = compiled
    count = int(np.prod(shape))
    a = np.arange(count, dtype=np.float32).reshape(shape) * np.float32(.25) - np.float32(3)
    b = np.full(shape, .75, dtype=np.float32)
    tensors = [cx.tensor(values, device=target) for values in (a, b, np.full(shape, -7, dtype=np.float32))]
    expected = oracle(dsl, a, b)
    assert_bits(native.launch(*tensors, count, block_size=block).numpy(), expected)
    assert_bits(dsl.reference(*(t.cpu() for t in tensors), count, block_size=block).numpy(), expected)
    np.testing.assert_array_equal(tensors[2].numpy(), -7)


def test_sqrt_ieee_value_bits(target):
    native = root.compile(target=target, compiler='mlir', cache=False)
    bits = np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
                     0x00800000, 0x80800000, 0x3f800000, 0xbf800000, 0x40000000,
                     0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
                     0x7fc00001, 0xffc01234, 0x7f800001], dtype=np.uint32)
    rng = np.random.default_rng(1729)
    a = np.concatenate([bits, rng.integers(0, 2**32, 4096, dtype=np.uint32)]).view(np.float32)
    b = np.ones_like(a)
    tensors = [cx.tensor(values, device=target) for values in (a, b, np.zeros_like(a))]
    expected = oracle(root, a, b)
    assert_bits(native.launch(*tensors, a.size, block_size=113).numpy(), expected)
    # The reference is also safe when its caller enables floating-point errors.
    with np.errstate(all='raise'):
        assert_bits(root.reference(*(t.cpu() for t in tensors), a.size).numpy(), expected)


@pytest.mark.parametrize('count', [0, 1, 255, 256, 257])
@pytest.mark.parametrize('alias', ['none', 'left', 'right', 'both'])
def test_sqrt_prefix_alias(target, count, alias):
    native = root.compile(target=target, compiler='mlir', cache=False)
    a = cx.tensor(np.arange(257, dtype=np.float32) * .25, device=target)
    b = a if alias == 'both' else cx.ones((257,), device=target)
    out = {'left': a, 'right': b, 'both': a}.get(alias)
    if out is None:
        out = cx.tensor(np.full(257, -7, dtype=np.float32), device=target)
    originals = [t.numpy() for t in (a, b, out)]
    expected = originals[2].copy()
    expected[:count] = oracle(root, originals[0][:count], originals[1][:count])
    for _ in range(2):
        assert_bits(native.launch(a, b, out, count, thread_count=count, block_size=7).numpy(), expected)
    for tensor, original in zip((a, b, out), originals, strict=True):
        assert_bits(tensor.numpy(), original)


def test_sqrt_reordered_and_no_numpy_fallback(target, monkeypatch):
    a, b, out = (cx.tensor(np.array(x, dtype=np.float32), device=target)
                 for x in ([1, 4, 9, 16], [4, 1, .25, 16], [-7] * 4))
    expected = oracle(root, a.numpy(), b.numpy())
    native = root_reordered.compile(target=target, compiler='mlir', cache=False)
    with monkeypatch.context() as patch:
        patch.setattr(cx.Tensor, 'numpy', lambda *args: pytest.fail('sqrt fell back to NumPy'))
        result = native.launch(4, out, b, a)
    assert_bits(result.numpy(), expected)


def test_gpu_emitter_sqrt_rounding():
    require_tools()
    for dsl in KERNELS:
        source, ptx = cuda.lower(dsl.parse_ir())
        assert 'math.sqrt' in emit_mlir(dsl.parse_ir())
        assert 'llvm.nvvm.sqrt.rn.f' in source
        assert re.search(r'\bsqrt\.rn\.f32\b', ptx)
        assert not re.search(r'\b(?:rsqrt\.|sqrt\.approx|sqrt\.[^;\n]*ftz)', ptx)
        assert '__nv_sqrt' not in ptx and '.extern .func' not in ptx
        assert 'expr-f32-v1' in ptx and 'tensorcx_expr_v1' in ptx
    assert '::metal::sqrt(' in root_shadow.emit_msl()


def with_value(value):
    ir = root.parse_ir()
    guard = ir.body[1]
    return replace(ir, body=(ir.body[0], replace(guard, body=(replace(guard.body[0], value=value),))))


@pytest.mark.parametrize('bad', ['integer', 'bool', 'index', 'arity0', 'arity2', 'offset', 'output_read', 'hidden_buffer'])
def test_sqrt_invalid_before_tools(monkeypatch, bad):
    a, b = IRLoad('a', IRName('i')), IRLoad('b', IRName('i'))
    argument = IRBinaryOp('mul', a, b)
    arguments = (argument,)
    if bad == 'integer': arguments = (IRConstant(1),)
    if bad == 'bool': arguments = (IRCompare('lt', a, b),)
    if bad == 'index': arguments = (IRCall('thread_id', ()),)
    if bad == 'arity0': arguments = ()
    if bad == 'arity2': arguments = (a, b)
    if bad == 'offset': arguments = (IRBinaryOp('mul', a, replace(b, index=IRBinaryOp('add', IRName('i'), IRConstant(1)))),)
    if bad == 'output_read': arguments = (IRBinaryOp('add', argument, IRLoad('out', IRName('i'))),)
    if bad == 'hidden_buffer': arguments = (IRBinaryOp('add', argument, IRLoad('missing', IRName('i'))),)
    ir = with_value(IRCall('sqrt', arguments))
    for adapter in (cpu, cuda):
        if adapter is cpu and bad == 'output_read':
            continue  # Existing CPU alias policy permits this.
        monkeypatch.setattr(adapter, 'toolchain', lambda: pytest.fail('invalid sqrt reached tools'))
        with pytest.raises((KernelCompileError, ValueError)):
            adapter.compile_kernel(ir)
    if bad in {'integer', 'bool', 'index', 'arity0', 'arity2'}:
        with pytest.raises(KernelCompileError): _emit_msl(ir)
        with pytest.raises(MlirEmitError): emit_mlir(ir)


@pytest.mark.parametrize('call', ['cx.experimental.sqrt()', 'cx.experimental.sqrt(a[i], b[i])',
                                  'cx.experimental.sqrt(value=a[i])', 'cx.sqrt(a[i])', 'np.sqrt(a[i])'])
def test_sqrt_parser_arity_keywords(call, tmp_path):
    path = tmp_path / 'sqrt_invalid.py'
    path.write_text('import tensorcx as cx\nimport numpy as np\n@cx.experimental.kernel\ndef bad(a,b,out,n):\n    i=cx.experimental.thread_id()\n    if i<n:\n        out[i]=' + call + '\n')
    spec = importlib.util.spec_from_file_location('sqrt_invalid', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(KernelCompileError): module.bad.parse_ir()


@pytest.mark.parametrize('kind', ['depth', 'operations'])
def test_sqrt_expression_bounds_before_tools(kind, monkeypatch):
    a, b = IRLoad('a', IRName('i')), IRLoad('b', IRName('i'))
    value = IRBinaryOp('mul', a, b)
    if kind == 'depth':
        for _ in range(14): value = IRCall('sqrt', (value,))
        valid = with_value(value)  # 14 roots + multiply + load = depth 16.
        invalid = with_value(IRCall('sqrt', (value,)))
    else:
        ir = with_value(value)
        guard, store = ir.body[1], ir.body[1].body[0]
        body = tuple(IRAssign(f'x{i}', IRCall('sqrt', (value,))) for i in range(32))
        valid = replace(ir, body=(ir.body[0], replace(guard, body=(*body, replace(store, value=IRName('x31'))))))
        first = replace(body[0], value=IRCall('sqrt', (body[0].value,)))
        invalid = replace(valid, body=(ir.body[0], replace(valid.body[1], body=(first, *body[1:], replace(store, value=IRName('x31'))))))
    cuda.signature(valid)
    if os.environ.get('TENSORCX_REQUIRE_MLIR_CUDA'):
        require_device()
        native = cuda.compile_kernel(valid, cache=False)
        tensors = [cx.ones((4,), device='cuda') for _ in range(3)]
        assert_bits(native.launch(*tensors, 4).numpy(), np.ones(4, dtype=np.float32))
    monkeypatch.setattr(cuda, 'toolchain', lambda: pytest.fail('unbounded sqrt reached tools'))
    with pytest.raises(KernelCompileError): cuda.compile_kernel(invalid)


@pytest.mark.parametrize('case', ['shape', 'dtype', 'guard', 'block', 'count'])
def test_sqrt_invalid_launch_preserves_output(target, case):
    native = root.compile(target=target, compiler='mlir', cache=False)
    a = cx.ones((4,), device=target)
    args, block = [a, a, a, 4], 7
    if case == 'shape': args[0] = cx.ones((2, 2), device=target)
    if case == 'dtype': args[0] = cx.tensor(np.ones(4, dtype=np.int32), device=target)
    if case == 'guard': args[-1] = 3
    if case == 'block': block = 0
    if case == 'count': args.pop()
    with pytest.raises((ValueError, TypeError)):
        native.launch(*args, thread_count=4, block_size=block)
    np.testing.assert_array_equal(a.numpy(), 1)


def test_sqrt_cache_operation_separation(target, monkeypatch, tmp_path):
    monkeypatch.setenv('TENSORCX_DISABLE_CACHE', '0')
    monkeypatch.setenv('TENSORCX_CACHE_DIR', str(tmp_path / 'private-cache'))
    first = root.compile(target=target, compiler='mlir')
    second = root.compile(target=target, compiler='mlir')
    ir = root.parse_ir()
    other = (cpu if target == 'cpu' else cuda).compile_kernel(with_value(ir.body[1].body[0].value.args[0]))
    assert not first.cache_hit and second.cache_hit and not other.cache_hit
    a = cx.tensor([4., 9.], device=target)
    b, out = cx.ones((2,), device=target), cx.zeros((2,), device=target)
    assert_bits(second.launch(a, b, out, 2).numpy(), np.array([2, 3], dtype=np.float32))
    assert_bits(other.launch(a, b, out, 2).numpy(), np.array([4, 9], dtype=np.float32))


def test_sqrt_outside_kernel_is_explicit():
    with pytest.raises(NotImplementedError): cx.experimental.sqrt(1.0)
