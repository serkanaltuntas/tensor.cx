"""IEEE minimum/maximum in guarded kernels, including both zero signs."""
from dataclasses import replace
import importlib.util
import os
import re

import numpy as np
import pytest
import tensorcx as cx
from tensorcx._compiler import cpu, cuda
from tensorcx._compiler.emitter import emit_mlir, MlirEmitError
from tensorcx.experimental import IRAssign, IRBinaryOp, IRCall, IRCompare, IRConstant, IRLoad, IRName, KernelCompileError, _emit_msl
from test_mlir_runtime import require_tools
from test_mlir_cuda_runtime import add as add_kernel, require_device
from test_mlir_abs import absolute as abs_kernel


@cx.experimental.kernel
def lower(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.minimum(a[i], b[i])


@cx.experimental.kernel
def upper(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.maximum(a[i], b[i])


@cx.experimental.kernel
def bounded(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        radius = cx.experimental.abs(b[i])
        capped = cx.experimental.minimum(cx.experimental.maximum(a[i], 0.0), radius)
        out[i] = cx.experimental.sqrt(capped)


@cx.experimental.kernel
def selected(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        value = cx.experimental.select(a[i] < b[i], a[i], b[i])
        out[i] = cx.experimental.maximum(cx.experimental.minimum(value, a[i]), cx.experimental.abs(b[i])) / 2.0


@cx.experimental.kernel
def shadow(minimum, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        maximum = cx.experimental.minimum(minimum[i], b[i])
        out[i] = cx.experimental.maximum(maximum, b[i])


@cx.experimental.kernel
def reordered(n, out, b, a):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.maximum(a[i], b[i])


@cx.experimental.kernel
def helper_collision(tensorcx_minimum_f32, tensorcx_minimum_f32_1, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        tensorcx_maximum_f32 = cx.experimental.minimum(tensorcx_minimum_f32[i], tensorcx_minimum_f32_1[i])
        out[i] = cx.experimental.maximum(tensorcx_maximum_f32, tensorcx_minimum_f32_1[i])


KERNELS = [lower, upper, bounded, selected, shadow]


def ieee(a, b, *, maximum=False):
    """Independent comparison/NaN/zero oracle, without NumPy min/max."""
    with np.errstate(all='ignore'):
        value = np.where(a > b if maximum else a < b, a, b)
        negative = (np.signbit(a) & np.signbit(b)) if maximum else (np.signbit(a) | np.signbit(b))
        zero = np.copysign(np.zeros_like(value), np.where(negative, np.float32(-1), np.float32(1)))
        value = np.where((a == 0) & (b == 0), zero, value)
        return np.where(np.isnan(a) | np.isnan(b), np.float32(np.nan), value)


def oracle(dsl, a, b):
    with np.errstate(all='ignore'):
        if dsl is lower: return ieee(a, b)
        if dsl is upper or dsl is reordered: return ieee(a, b, maximum=True)
        if dsl is bounded:
            return np.sqrt(ieee(ieee(a, np.zeros_like(a), maximum=True), np.abs(b)), dtype=np.float32)
        if dsl is selected:
            return np.divide(ieee(ieee(np.where(a < b, a, b), a), np.abs(b), maximum=True), np.float32(2), dtype=np.float32)
        return ieee(ieee(a, b), b, maximum=True)


def assert_bits(actual, expected):
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    valid = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[valid].view(np.uint32), expected[valid].view(np.uint32))


def special_pairs():
    bits = np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
                     0x00800000, 0x80800000, 0x3f800000, 0xbf800000,
                     0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
                     0x7fc00001, 0xffc01234, 0x7f800001, 0xff800001], dtype=np.uint32)
    values = bits.view(np.float32)
    rng = np.random.default_rng(1729)
    a = np.concatenate([np.repeat(values, values.size), rng.integers(0, 2**32, 4096, dtype=np.uint32).view(np.float32)])
    b = np.concatenate([np.tile(values, values.size), rng.integers(0, 2**32, 4096, dtype=np.uint32).view(np.float32)])
    return a, b


@pytest.fixture(scope='module', params=['cpu', 'cuda'])
def target(request):
    (require_tools if request.param == 'cpu' else require_device)()
    return request.param


@pytest.fixture(scope='module', params=KERNELS)
def compiled(request, target):
    return request.param, target, request.param.compile(target=target, compiler='mlir', cache=False)


@pytest.fixture(scope='module', params=[lower, upper])
def extremum(request, target):
    return request.param, target, request.param.compile(target=target, compiler='mlir', cache=False)


@pytest.mark.parametrize('shape', [(), (0,), (1,), (255,), (256,), (257,), (2, 3)])
@pytest.mark.parametrize('block', [1, 7, 256])
def test_minmax_parity(compiled, shape, block):
    dsl, target, native = compiled
    size = int(np.prod(shape))
    a = np.arange(size, dtype=np.float32).reshape(shape) * np.float32(.25) - np.float32(3)
    b = np.full(shape, .75, dtype=np.float32)
    tensors = [cx.tensor(v, device=target) for v in (a, b, np.full(shape, -7, dtype=np.float32))]
    expected = oracle(dsl, a, b)
    assert_bits(native.launch(*tensors, size, block_size=block).numpy(), expected)
    assert_bits(dsl.reference(*(t.cpu() for t in tensors), size, block_size=block).numpy(), expected)
    np.testing.assert_array_equal(tensors[2].numpy(), -7)


def test_minmax_ieee_pairs(extremum):
    dsl, target, native = extremum
    a, b = special_pairs()
    tensors = [cx.tensor(v, device=target) for v in (a, b, np.zeros_like(a))]
    expected = oracle(dsl, a, b)
    assert_bits(native.launch(*tensors, a.size, block_size=113).numpy(), expected)
    with np.errstate(all='raise'):
        assert_bits(dsl.reference(*(t.cpu() for t in tensors), a.size).numpy(), expected)


@pytest.mark.parametrize('count', [0, 1, 255, 256, 257])
@pytest.mark.parametrize('alias', ['none', 'left', 'right', 'both'])
def test_minmax_prefix_alias(extremum, count, alias):
    dsl, target, native = extremum
    a = cx.tensor(np.arange(257, dtype=np.float32) - np.float32(128), device=target)
    b = a if alias == 'both' else cx.tensor(-a.numpy(), device=target)
    out = {'left': a, 'right': b, 'both': a}.get(alias)
    if out is None: out = cx.tensor(np.full(257, -7, dtype=np.float32), device=target)
    originals = [t.numpy() for t in (a, b, out)]
    expected = originals[2].copy()
    expected[:count] = oracle(dsl, originals[0][:count], originals[1][:count])
    for _ in range(2):
        assert_bits(native.launch(a, b, out, count, thread_count=count, block_size=7).numpy(), expected)
    for tensor, original in zip((a, b, out), originals, strict=True): assert_bits(tensor.numpy(), original)


def test_minmax_reordered_no_numpy_fallback(target, monkeypatch):
    a, b, out = [cx.tensor(v, device=target) for v in ([1., -3., 9., 16.], [4., 1., .25, 8.], [-7.] * 4)]
    expected = oracle(reordered, a.numpy(), b.numpy())
    native = reordered.compile(target=target, compiler='mlir', cache=False)
    with monkeypatch.context() as patch:
        patch.setattr(cx.Tensor, 'numpy', lambda *args: pytest.fail('minmax NumPy fallback'))
        result = native.launch(4, out, b, a)
    assert_bits(result.numpy(), expected)


def test_gpu_emitter_minmax_ieee():
    require_tools()
    for dsl, name, instruction in [(lower, 'minimum', 'min'), (upper, 'maximum', 'max')]:
        source, ptx = cuda.lower(dsl.parse_ir())
        assert f'arith.{name}f' in emit_mlir(dsl.parse_ir())
        assert f'llvm.{name}.f32' in source
        assert re.search(r'\b' + instruction + r'\.f32\b', ptx)
        assert 'setp.nan.f32' in ptx and 'selp.f32' in ptx
        assert not re.search(r'\b(?:min|max)\.[^;\n]*(?:ftz|NaN)', ptx)
        assert '.extern .func' not in ptx and 'expr-f32-v1' in ptx


def with_value(value):
    ir = lower.parse_ir(); guard = ir.body[1]
    return replace(ir, body=(ir.body[0], replace(guard, body=(replace(guard.body[0], value=value),))))


@pytest.mark.parametrize('name', ['minimum', 'maximum'])
@pytest.mark.parametrize('bad', ['left_int', 'right_int', 'left_bool', 'right_bool', 'index', 'arity0', 'arity1', 'arity3', 'offset', 'output_read', 'hidden_buffer', 'store_index'])
def test_minmax_invalid_before_tools(name, bad, monkeypatch):
    a, b = IRLoad('a', IRName('i')), IRLoad('b', IRName('i'))
    args = (a, b)
    if bad == 'left_int': args = (IRConstant(1), b)
    if bad == 'right_int': args = (a, IRConstant(1))
    if bad == 'left_bool': args = (IRCompare('lt', a, b), b)
    if bad == 'right_bool': args = (a, IRCompare('lt', a, b))
    if bad == 'index': args = (a, IRCall('thread_id', ()))
    if bad == 'arity0': args = ()
    if bad == 'arity1': args = (a,)
    if bad == 'arity3': args = (a, b, a)
    if bad == 'offset': args = (a, replace(b, index=IRBinaryOp('add', IRName('i'), IRConstant(1))))
    if bad == 'output_read': args = (IRBinaryOp('add', a, b), IRLoad('out', IRName('i')))
    if bad == 'hidden_buffer': args = (IRBinaryOp('add', a, b), IRLoad('missing', IRName('i')))
    ir = with_value(IRCall(name, args))
    if bad == 'store_index':
        guard = ir.body[1]
        ir = replace(ir, body=(ir.body[0], replace(guard, body=(replace(guard.body[0], index=IRCall(name, args)),))))
    for adapter in (cpu, cuda):
        if adapter is cpu and bad == 'output_read': continue
        monkeypatch.setattr(adapter, 'toolchain', lambda: pytest.fail('invalid minmax reached tools'))
        with pytest.raises((KernelCompileError, ValueError)): adapter.compile_kernel(ir)
    if bad in {'left_int', 'right_int', 'left_bool', 'right_bool', 'index', 'arity0', 'arity1', 'arity3'}:
        with pytest.raises(KernelCompileError): _emit_msl(ir)
        with pytest.raises(MlirEmitError): emit_mlir(ir)


@pytest.mark.parametrize('name', ['minimum', 'maximum'])
@pytest.mark.parametrize('args', ['', 'a[i]', 'a[i],b[i],0.0', 'lhs=a[i],rhs=b[i]'])
def test_minmax_parser_arity_keywords(name, args, tmp_path):
    path = tmp_path / 'bad_minmax.py'
    path.write_text('import tensorcx as cx\n@cx.experimental.kernel\ndef bad(a,b,out,n):\n    i=cx.experimental.thread_id()\n    if i<n:\n        out[i]=cx.experimental.' + name + '(' + args + ')\n')
    spec = importlib.util.spec_from_file_location('bad_minmax', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with pytest.raises(KernelCompileError): module.bad.parse_ir()


@pytest.mark.parametrize('name', ['minimum', 'maximum'])
@pytest.mark.parametrize('kind', ['depth', 'operations'])
def test_minmax_expression_bounds_before_tools(name, kind, monkeypatch):
    a, b = IRLoad('a', IRName('i')), IRLoad('b', IRName('i'))
    if kind == 'depth':
        value = b
        for _ in range(15): value = IRCall(name, (a, value))
        valid = with_value(value)
        invalid = with_value(IRCall(name, (a, value)))
    else:
        value = IRCall(name, (a, IRCall(name, (a, b))))
        ir = with_value(value); guard = ir.body[1]; store = guard.body[0]
        body = tuple(IRAssign(f'x{i}', value) for i in range(32))
        valid = replace(ir, body=(ir.body[0], replace(guard, body=(*body, replace(store, value=IRName('x31'))))))
        extra = replace(body[0], value=IRCall(name, (a, value)))
        invalid = replace(valid, body=(ir.body[0], replace(guard, body=(extra, *body[1:], replace(store, value=IRName('x31'))))))
    cuda.signature(valid)
    if os.environ.get('TENSORCX_REQUIRE_MLIR_CUDA'):
        require_device(); native = cuda.compile_kernel(valid, cache=False)
        tensors = [cx.ones((4,), device='cuda') for _ in range(3)]
        assert_bits(native.launch(*tensors, 4).numpy(), np.ones(4, dtype=np.float32))
    monkeypatch.setattr(cuda, 'toolchain', lambda: pytest.fail('unbounded minmax reached tools'))
    with pytest.raises(KernelCompileError): cuda.compile_kernel(invalid)


@pytest.mark.parametrize('case', ['shape', 'dtype', 'bool_dtype', 'guard', 'block', 'count'])
def test_minmax_invalid_launch(target, case):
    native = lower.compile(target=target, compiler='mlir', cache=False)
    a = cx.ones((4,), device=target); args = [a, a, a, 4]; block = 7
    if case == 'shape': args[0] = cx.ones((2, 2), device=target)
    if case == 'dtype': args[0] = cx.tensor(np.ones(4, dtype=np.int32), device=target)
    if case == 'bool_dtype': args[0] = cx.tensor(np.ones(4, dtype=np.bool_), device=target)
    if case == 'guard': args[-1] = 3
    if case == 'block': block = 0
    if case == 'count': args.pop()
    with pytest.raises((ValueError, TypeError)): native.launch(*args, thread_count=4, block_size=block)
    np.testing.assert_array_equal(a.numpy(), 1)


def test_minmax_cache_separation(target, monkeypatch, tmp_path):
    monkeypatch.setenv('TENSORCX_DISABLE_CACHE', '0'); monkeypatch.setenv('TENSORCX_CACHE_DIR', str(tmp_path / 'private-cache'))
    first = lower.compile(target=target, compiler='mlir'); second = lower.compile(target=target, compiler='mlir')
    value = lower.parse_ir().body[1].body[0].value
    other = (cpu if target == 'cpu' else cuda).compile_kernel(with_value(replace(value, name='maximum')))
    assert not first.cache_hit and second.cache_hit and not other.cache_hit
    a = cx.tensor([-4., 9.], device=target); b = cx.tensor([2., -3.], device=target); out = cx.zeros((2,), device=target)
    assert_bits(second.launch(a, b, out, 2).numpy(), np.array([-4, -3], dtype=np.float32))
    assert_bits(other.launch(a, b, out, 2).numpy(), np.array([2, 9], dtype=np.float32))


def test_minmax_msl_helper_names():
    source = helper_collision.emit_msl()
    assert '::tensorcx_minimum_f32_2(' in source and '::tensorcx_maximum_f32_1(' in source
    # as_type is an MSL language operator, not a metal namespace function.
    assert 'as_type<uint>(a)' in source and 'as_type<float>(' in source
    assert '::as_type<' not in source
    ir = replace(lower.parse_ir(), name='tensorcx_minimum_f32')
    source = _emit_msl(ir)
    assert 'inline float tensorcx_minimum_f32_1(' in source
    assert '::tensorcx_minimum_f32_1(' in source


@pytest.mark.parametrize('name', ['minimum', 'maximum'])
def test_minmax_outside_kernel(name):
    with pytest.raises(NotImplementedError): getattr(cx.experimental, name)(1., 2.)


@pytest.mark.parametrize('dsl,strict', [(lower, True), (upper, True), (bounded, True), (selected, True), (add_kernel, False), (abs_kernel, False)])
def test_minmax_metal_compiler_policy(dsl, strict, monkeypatch):
    from pathlib import Path
    commands = []
    monkeypatch.setattr(cx.experimental.shutil, 'which', lambda _: '/fake/xcrun')
    def run(command):
        commands.append(command)
        Path(command[command.index('-o') + 1]).write_bytes(b'compiled-artifact')
    monkeypatch.setattr(cx.experimental, '_run_metal_tool', run)
    artifact = dsl.compile(target='metal', cache=False)
    assert artifact.metallib == b'compiled-artifact'
    assert ('-fno-fast-math' in commands[0]) is strict
    assert '-fno-fast-math' not in commands[1]


def test_minmax_metal_cache_policy(monkeypatch, tmp_path):
    from tensorcx._compiler import metal
    from tensorcx._compiler.cache import descriptor as make_descriptor
    monkeypatch.setenv('TENSORCX_DISABLE_CACHE', '0')
    monkeypatch.setenv('TENSORCX_CACHE_DIR', str(tmp_path / 'private-cache'))
    monkeypatch.setattr(metal.shutil, 'which', lambda _: '/fake/xcrun')
    monkeypatch.setattr(metal, '_sdk_info', lambda *args: 'fake-sdk')
    def describe(target, source, **kwargs):
        # Keep this a no-SDK cache-key test; real SDK identities have existing
        # coverage. Preserve the actual options and source in the descriptor.
        kwargs['tools'] = {}
        return make_descriptor(target, source, **kwargs)
    monkeypatch.setattr(metal, 'descriptor', describe)
    calls = []
    def compile_(name, source, **kwargs):
        calls.append(kwargs)
        return b'strict' if kwargs.get('strict_math') else b'default'
    monkeypatch.setattr(cx.experimental, '_compile_msl_to_metallib', compile_)
    source = lower.emit_msl()
    normal, hit = metal.compile_artifact('lower', source)
    strict, strict_hit = metal.compile_artifact('lower', source, strict_math=True)
    warm, warm_hit = metal.compile_artifact('lower', source, strict_math=True)
    assert normal == b'default' and strict == warm == b'strict'
    assert not hit and not strict_hit and warm_hit
    assert calls == [{}, {'strict_math': True}]
