"""Float32 DSL division: strict lowering, numerical edges and launch ownership."""
from dataclasses import replace
import re

import numpy as np
import pytest

import tensorcx as cx
from tensorcx._compiler import cpu, cuda
from tensorcx._compiler.emitter import emit_mlir, MlirEmitError
from tensorcx.experimental import IRBinaryOp, IRCompare, IRConstant, IRName, KernelCompileError
from test_mlir_runtime import require_tools
from test_mlir_cuda_runtime import require_device


@cx.experimental.kernel
def divide(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = a[i] / b[i]


@cx.experimental.kernel
def quotient_chain(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        quotient = a[i] / b[i]
        out[i] = quotient * 0.5 + b[i]


@cx.experimental.kernel
def nested_divide(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = (a[i] / b[i]) / (b[i] + 0.5)


@cx.experimental.kernel
def reordered_divide(n, out, b, a):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = a[i] / b[i]


@cx.experimental.kernel
def literal_divide(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        reciprocal = 1.0 / b[i]
        out[i] = a[i] / -0.0 + reciprocal


def oracle(dsl, a, b):
    with np.errstate(all='ignore'):
        quotient = np.divide(a, b, dtype=np.float32)
        if dsl is quotient_chain:
            return np.add(np.multiply(quotient, np.float32(.5), dtype=np.float32), b, dtype=np.float32)
        if dsl is nested_divide:
            return np.divide(quotient, np.add(b, np.float32(.5), dtype=np.float32), dtype=np.float32)
        return quotient


def assert_ieee(actual, expected):
    # NaN payload/sign are deliberately not part of the numerical contract.
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    valid = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[valid].view(np.uint32), expected[valid].view(np.uint32))


@pytest.fixture(scope='module', params=['cpu', 'cuda'])
def target(request):
    (require_tools if request.param == 'cpu' else require_device)()
    return request.param


@pytest.fixture(scope='module', params=[divide, quotient_chain, nested_divide])
def compiled(request, target):
    return request.param, target, request.param.compile(target=target, compiler='mlir', cache=False)


@pytest.mark.parametrize('shape', [(), (0,), (1,), (255,), (256,), (257,), (2, 3)])
@pytest.mark.parametrize('block', [1, 7, 256])
def test_division_parity(compiled, shape, block):
    dsl, target, native = compiled
    count = int(np.prod(shape))
    a = np.arange(count, dtype=np.float32).reshape(shape) * np.float32(.125) - np.float32(3)
    b = np.full(shape, np.float32(.75))
    expected = oracle(dsl, a, b)
    inputs = [cx.tensor(values, device=target) for values in (a, b, np.full(shape, -7, dtype=np.float32))]
    actual = native.launch(*inputs, count, block_size=block).numpy()
    reference = dsl.reference(*(t.cpu() for t in inputs), count, block_size=block).numpy()
    assert_ieee(actual, expected)
    assert_ieee(reference, expected)
    np.testing.assert_array_equal(inputs[2].numpy(), -7)


def test_division_ieee_edges_and_random_bits(target):
    native = divide.compile(target=target, compiler='mlir', cache=False)
    bits = np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
                     0x00800000, 0x80800000, 0x3f7fffff, 0x3f800000, 0x3f800001,
                     0xbf800000, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
                     0x7fc00001, 0xffc01234, 0x7f800001], dtype=np.uint32)
    special = bits.view(np.float32)
    a, b = np.repeat(special, len(special)), np.tile(special, len(special))
    rng = np.random.default_rng(42)
    a = np.concatenate([a, rng.integers(0, 2**32, 1024, dtype=np.uint32).view(np.float32)])
    b = np.concatenate([b, rng.integers(0, 2**32, 1024, dtype=np.uint32).view(np.float32)])
    tensors = [cx.tensor(x, device=target) for x in (a, b, np.zeros_like(a))]
    expected = oracle(divide, a, b)
    assert_ieee(native.launch(*tensors, a.size, block_size=7).numpy(), expected)
    # Zero, invalid, overflow and underflow stay IEEE values even if the caller
    # normally asks NumPy scalar operations to raise FloatingPointError.
    with np.errstate(all='raise'):
        reference = divide.reference(*(t.cpu() for t in tensors), a.size).numpy()
    assert_ieee(reference, expected)


@pytest.mark.parametrize('count', [0, 1, 255, 256, 257])
@pytest.mark.parametrize('alias', ['none', 'left', 'right', 'both'])
def test_division_prefix_alias_and_repeated(compiled, count, alias):
    dsl, target, native = compiled
    a = cx.tensor(np.arange(257, dtype=np.float32) * .25, device=target)
    b = a if alias == 'both' else cx.tensor(np.full(257, .75, dtype=np.float32), device=target)
    out = {'left': a, 'right': b, 'both': a}.get(alias)
    if out is None:
        out = cx.tensor(np.full(257, -7, dtype=np.float32), device=target)
    originals = [t.numpy() for t in (a, b, out)]
    expected = originals[2].copy()
    expected[:count] = oracle(dsl, originals[0][:count], originals[1][:count])
    for _ in range(2):
        assert_ieee(native.launch(a, b, out, count, thread_count=count, block_size=7).numpy(), expected)
    for tensor, original in zip((a, b, out), originals, strict=True):
        np.testing.assert_array_equal(tensor.numpy().view(np.uint32), original.view(np.uint32))


def test_division_reordered_arguments(target):
    a, b, out = (cx.tensor(np.array(x, dtype=np.float32), device=target) for x in ([1, -3, .25, 8], [2, 5, -4, .5], [-7]*4))
    native = reordered_divide.compile(target=target, compiler='mlir')
    assert_ieee(native.launch(4, out, b, a).numpy(), oracle(divide, a.numpy(), b.numpy()))


def test_division_float_literals_and_signed_zero(target):
    values = np.array([0., -0., 1., -1., np.inf, np.nan], dtype=np.float32)
    a, b = cx.tensor(values, device=target), cx.tensor(values[::-1].copy(), device=target)
    out = cx.zeros(a.shape, device=target)
    native = literal_divide.compile(target=target, compiler='mlir', cache=False)
    with np.errstate(all='ignore'):
        expected = np.add(np.divide(values, np.float32(-0.), dtype=np.float32),
                          np.divide(np.float32(1.), values[::-1], dtype=np.float32), dtype=np.float32)
    assert_ieee(native.launch(a, b, out, values.size).numpy(), expected)
    assert_ieee(literal_divide.reference(a.cpu(), b.cpu(), out.cpu(), values.size).numpy(), expected)


@pytest.mark.parametrize('case', ['shape', 'dtype', 'guard', 'block', 'count'])
def test_division_failed_launch_preserves_output(compiled, case):
    _, target, native = compiled
    a = cx.ones((4,), device=target)
    args = [a, a, a, 4]
    block = 7
    if case == 'shape': args[0] = cx.ones((2, 2), device=target)
    if case == 'dtype': args[0] = cx.tensor(np.ones(4, dtype=np.int32), device=target)
    if case == 'guard': args[-1] = 3
    if case == 'block': block = 0
    if case == 'count': args.pop()
    with pytest.raises((ValueError, TypeError)):
        native.launch(*args, thread_count=4, block_size=block)
    np.testing.assert_array_equal(a.numpy(), 1)


def test_division_native_launch_has_no_numpy_fallback(target, monkeypatch):
    native = divide.compile(target=target, compiler='mlir', cache=False)
    a, b, out = cx.ones((4,), device=target), cx.ones((4,), device=target), cx.zeros((4,), device=target)
    def forbidden(*args, **kwargs):
        pytest.fail('native division attempted a NumPy fallback')
    with monkeypatch.context() as patch:
        patch.setattr(cx.Tensor, 'numpy', forbidden)
        result = native.launch(a, b, out, 4)
    np.testing.assert_array_equal(result.numpy(), 1)


def test_gpu_emitter_division_rounding():
    require_tools()
    source, ptx = cuda.lower(divide.parse_ir())
    assert 'arith.divf' in source and '@tensorcx_expr_v1' in source
    assert 'div.rn.f32' in ptx
    assert not re.search(r'\b(?:div\.approx|rcp\.|div\.rn\.ftz)', ptx)
    # The existing opaque expression module ABI handles this additional op;
    # entry layout and legacy single-add/sub/mul paths do not change.
    assert ptx.startswith('// ' + cuda.manifest('tttu', 2, 3, 'expr') + '\n')


@pytest.mark.parametrize('location', ['store', 'local', 'compare', 'load_index'])
@pytest.mark.parametrize('kind', ['integer', 'mixed', 'bool'])
def test_division_nonfloat_rejected_before_tools(monkeypatch, location, kind):
    ir = quotient_chain.parse_ir()
    load = ir.body[1].body[0].value.lhs
    lhs, rhs = {'integer': (IRName('n'), IRConstant(2)),
                'mixed': (load, IRConstant(2)),
                'bool': (IRCompare('lt', load, IRConstant(1.0)), IRConstant(2.0))}[kind]
    invalid = IRBinaryOp('div', lhs, rhs)
    body = list(ir.body[1].body)
    if location == 'store':
        body[-1] = replace(body[-1], value=invalid)
    elif location == 'local':
        body[0] = replace(body[0], value=IRBinaryOp('add', invalid, body[0].value))
    elif location == 'compare':
        ir = replace(ir, body=(ir.body[0], replace(ir.body[1], condition=IRCompare('lt', invalid, IRName('n')))))
    else:
        body[-1] = replace(body[-1], value=replace(load, index=invalid))
    if location != 'compare':
        ir = replace(ir, body=(ir.body[0], replace(ir.body[1], body=tuple(body))))
    from tensorcx.experimental import _emit_msl
    with pytest.raises(KernelCompileError, match='two float32 operands'):
        _emit_msl(ir)
    for adapter in (cpu, cuda):
        monkeypatch.setattr(adapter, 'toolchain', lambda: pytest.fail('invalid division reached tools'))
        with pytest.raises(KernelCompileError):
            adapter.compile_kernel(ir)
    with pytest.raises(MlirEmitError):
        emit_mlir(ir)


@pytest.mark.parametrize('source_op', ['//', '%', '**'])
def test_division_does_not_enable_other_operators(source_op, tmp_path):
    # Use a real inspectable source file so the parser follows the public path.
    import importlib.util
    source = tmp_path / 'unsupported.py'
    source.write_text('import tensorcx as cx\n@cx.experimental.kernel\ndef bad(a,b,out,n):\n    i=cx.experimental.thread_id()\n    if i<n:\n        out[i]=a[i] '+source_op+' b[i]\n')
    spec = importlib.util.spec_from_file_location('unsupported_division', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(KernelCompileError):
        module.bad.parse_ir()


def test_division_cache_roundtrip_and_separation(target, monkeypatch, tmp_path):
    monkeypatch.setenv('TENSORCX_DISABLE_CACHE', '0')
    monkeypatch.setenv('TENSORCX_CACHE_DIR', str(tmp_path / 'private-cache'))
    first = quotient_chain.compile(target=target, compiler='mlir')
    second = quotient_chain.compile(target=target, compiler='mlir')
    ir = quotient_chain.parse_ir()
    assignment, store = ir.body[1].body
    other_ir = replace(ir, body=(ir.body[0], replace(ir.body[1], body=(
        replace(assignment, value=replace(assignment.value, op='mul')), store))))
    other = (cpu if target == 'cpu' else cuda).compile_kernel(other_ir)
    assert not first.cache_hit and second.cache_hit and not other.cache_hit
    a = cx.tensor(np.full(4, 6, dtype=np.float32), device=target)
    b = cx.tensor(np.full(4, 2, dtype=np.float32), device=target)
    out = cx.zeros((4,), device=target)
    np.testing.assert_array_equal(second.launch(a, b, out, 4).numpy(), 3.5)
    np.testing.assert_array_equal(other.launch(a, b, out, 4).numpy(), 8)
