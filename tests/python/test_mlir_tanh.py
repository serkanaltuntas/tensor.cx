"""Guarded float32 hyperbolic tangent: bounded error, special values and ownership."""
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
def hyperbolic_tangent(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.tanh(a[i] * b[i])


@cx.experimental.kernel
def hyperbolic_tangent_local(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        tanh = cx.experimental.tanh(a[i] / b[i])
        out[i] = cx.experimental.tanh(cx.experimental.sqrt(tanh + b[i] + 1.0))


@cx.experimental.kernel
def hyperbolic_tangent_select(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        value = cx.experimental.select(a[i] > 0.0, a[i], b[i])
        out[i] = cx.experimental.select(value > 0.5, cx.experimental.tanh(cx.experimental.sqrt(value)), cx.experimental.tanh(0.5))


@cx.experimental.kernel
def hyperbolic_tangent_shadow(tanh, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.tanh(tanh[i] * b[i])


@cx.experimental.kernel
def hyperbolic_tangent_reordered(n, out, b, a):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.tanh(a[i] * b[i])


@cx.experimental.kernel
def hyperbolic_tangent_bounded(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.tanh(cx.experimental.minimum(cx.experimental.maximum(a[i], 0.25), b[i]))


@cx.experimental.kernel
def hyperbolic_tangent_normalize(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = a[i] * cx.experimental.tanh(b[i] + 0.00001)


KERNELS = [hyperbolic_tangent, hyperbolic_tangent_local, hyperbolic_tangent_select,
           hyperbolic_tangent_shadow, hyperbolic_tangent_bounded, hyperbolic_tangent_normalize]


def oracle_tanh(values):
    # Independent float64 computation, rounded once to float32.
    with np.errstate(all='ignore'):
        return np.tanh(values.astype(np.float64)).astype(np.float32)


def oracle(dsl, a, b):
    with np.errstate(all='ignore'):
        if dsl is hyperbolic_tangent_normalize:
            return np.multiply(a, oracle_tanh(np.add(b, np.float32(.00001), dtype=np.float32)), dtype=np.float32)
        if dsl is hyperbolic_tangent_bounded:
            clamped = np.where(a > np.float32(.25), a, np.float32(.25))
            return oracle_tanh(np.where(clamped < b, clamped, b))
        if dsl is hyperbolic_tangent_local:
            value = oracle_tanh(np.divide(a, b, dtype=np.float32))
            value = np.add(np.add(value, b, dtype=np.float32), np.float32(1), dtype=np.float32)
            return oracle_tanh(np.sqrt(value, dtype=np.float32))
        if dsl is hyperbolic_tangent_select:
            value = np.where(a > np.float32(0), a, b)
            return np.where(value > np.float32(.5), oracle_tanh(np.sqrt(value, dtype=np.float32)), oracle_tanh(np.full_like(value, .5)))
        return oracle_tanh(np.multiply(a, b, dtype=np.float32))


def assert_close(actual, expected, target='cpu'):
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    # Relative-only checks retain sensitivity at the origin and for subnormals.
    rtol = 1e-6
    allowance = 0.0
    error = np.abs(actual[finite].astype(np.float64) - expected[finite].astype(np.float64))
    assert np.all(error <= rtol * np.abs(expected[finite].astype(np.float64)) + allowance)
    zero = (actual == 0) & (expected == 0)
    np.testing.assert_array_equal(np.signbit(actual[zero]), np.signbit(expected[zero]))


def assert_bits(actual, expected):
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def numerical_inputs():
    patterns = np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
                         0x00800000, 0x80800000, 0x3f800000, 0xbf800000,
                         0x3f000000, 0xbf000000, 0x40000000, 0xc0000000,
                         0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
                         0x7fc00001, 0xffc01234, 0x7f800001, 0xff800001], dtype=np.uint32).view(np.float32)
    edges = []
    with np.errstate(all='ignore'):
        for value in [np.float32(0), np.float32(1), np.float32(-1), np.float32(10),
                      np.float32(-10), np.float32(2**-12), np.float32(-(2**-12)),
                      np.finfo(np.float32).tiny, -np.finfo(np.float32).tiny,
                      np.finfo(np.float32).max, -np.finfo(np.float32).max]:
            edges.append(value)
            lo = hi = value
            for _ in range(32):
                lo = np.nextafter(lo, np.float32(-np.inf)); hi = np.nextafter(hi, np.float32(np.inf))
                edges.extend((lo, hi))
        powers = np.ldexp(np.ones(277, dtype=np.float32), np.arange(-149, 128, dtype=np.int32))
        power_neighbors = np.concatenate([powers, np.nextafter(powers, np.float32(0)), np.nextafter(powers, np.float32(np.inf))])
        scale_grid = np.logspace(-44.8, 38.5, 4096, dtype=np.float32)
    rng = np.random.default_rng(1729)
    return np.concatenate([patterns, np.array(edges, dtype=np.float32), power_neighbors, -power_neighbors,
                           np.linspace(-12, 12, 32769, dtype=np.float32),
                           np.linspace(-1, 1, 16385, dtype=np.float32), scale_grid, -scale_grid,
                           rng.integers(0, 2**32, 8192, dtype=np.uint32).view(np.float32)])


@pytest.fixture(scope='module', params=['cpu', 'cuda'])
def target(request):
    (require_tools if request.param == 'cpu' else require_device)()
    return request.param


@pytest.fixture(scope='module', params=KERNELS)
def compiled(request, target):
    return request.param, target, request.param.compile(target=target, compiler='mlir', cache=False)


@pytest.mark.parametrize('shape', [(), (0,), (1,), (255,), (256,), (257,), (2, 3)])
@pytest.mark.parametrize('block', [1, 7, 256])
def test_tanh_parity(compiled, shape, block):
    dsl, target, native = compiled
    count = int(np.prod(shape))
    a = np.linspace(.25, 4, count, dtype=np.float32).reshape(shape)
    b = np.full(shape, .75, dtype=np.float32)
    tensors = [cx.tensor(values, device=target) for values in (a, b, np.full(shape, -7, dtype=np.float32))]
    expected = oracle(dsl, a, b)
    assert_close(native.launch(*tensors, count, block_size=block).numpy(), expected, target)
    assert_close(dsl.reference(*(t.cpu() for t in tensors), count, block_size=block).numpy(), expected)
    np.testing.assert_array_equal(tensors[2].numpy(), -7)


def test_tanh_numerical_domain(target):
    a = numerical_inputs()
    tensors = [cx.tensor(values, device=target) for values in (a, np.ones_like(a), np.zeros_like(a))]
    native = hyperbolic_tangent.compile(target=target, compiler='mlir', cache=False)
    expected = oracle_tanh(a)
    result = native.launch(*tensors, a.size, block_size=113).numpy()
    assert_close(result, expected, target)
    assert np.all(np.abs(result[~np.isnan(result)]) <= 1)
    assert_close(hyperbolic_tangent.reference(*(t.cpu() for t in tensors), a.size).numpy(), expected)
    np.testing.assert_array_equal(tensors[2].numpy(), 0)


@pytest.mark.parametrize('count', [0, 1, 255, 256, 257])
@pytest.mark.parametrize('alias', ['none', 'left', 'right', 'both'])
def test_tanh_prefix_alias(target, count, alias):
    native = hyperbolic_tangent.compile(target=target, compiler='mlir', cache=False)
    a = cx.tensor(np.linspace(-2, 2, 257, dtype=np.float32), device=target)
    b = a if alias == 'both' else cx.ones((257,), device=target)
    out = {'left': a, 'right': b, 'both': a}.get(alias)
    if out is None:
        out = cx.tensor(np.full(257, -7, dtype=np.float32), device=target)
    originals = [t.numpy() for t in (a, b, out)]
    expected = originals[2].copy()
    expected[:count] = oracle(hyperbolic_tangent, originals[0][:count], originals[1][:count])
    for _ in range(2):
        result = native.launch(a, b, out, count, thread_count=count, block_size=7)
        assert result is not out
        values = result.numpy()
        assert_close(values, expected, target)
        assert_bits(values[count:], originals[2][count:])
    for tensor, original in zip((a, b, out), originals, strict=True):
        assert_bits(tensor.numpy(), original)


def test_tanh_reordered_and_no_numpy_fallback(target, monkeypatch):
    a, b, out = (cx.tensor(np.array(x, dtype=np.float32), device=target)
                 for x in ([1, 4, 9, 16], [4, 1, .25, 16], [-7] * 4))
    expected = oracle(hyperbolic_tangent, a.numpy(), b.numpy())
    native = hyperbolic_tangent_reordered.compile(target=target, compiler='mlir', cache=False)
    with monkeypatch.context() as patch:
        patch.setattr(cx.Tensor, 'numpy', lambda *args: pytest.fail('tanh fell back to NumPy'))
        result = native.launch(4, out, b, a)
    assert_close(result.numpy(), expected, target)


def test_gpu_emitter_tanh_no_ftz():
    require_tools()
    for dsl in KERNELS:
        source, ptx = cuda.lower(dsl.parse_ir())
        assert 'math.tanh' in emit_mlir(dsl.parse_ir())
        assert 'llvm.nvvm.ex2.approx.f' in source
        assert 'llvm.fma.f32' in source and 'llvm.copysign.f32' in source
        assert re.search(r'\bex2\.approx\.f32\b', ptx)
        assert 'tanh.approx' not in ptx
        assert not re.search(r'\b(?:ex2|mul|fma|div)\.[^;\n]*ftz', ptx)
        assert '__nv_tanh' not in ptx and '.extern .func' not in ptx
        assert 'expr-f32-v1' in ptx and 'tensorcx_expr_v1' in ptx
    assert '::metal::tanh(' in hyperbolic_tangent_shadow.emit_msl()


def with_value(value):
    ir = hyperbolic_tangent.parse_ir()
    guard = ir.body[1]
    return replace(ir, body=(ir.body[0], replace(guard, body=(replace(guard.body[0], value=value),))))


@pytest.mark.parametrize('bad', ['integer', 'bool', 'index', 'arity0', 'arity2', 'offset', 'output_read', 'hidden_buffer'])
def test_tanh_invalid_before_tools(monkeypatch, bad):
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
    ir = with_value(IRCall('tanh', arguments))
    for adapter in (cpu, cuda):
        if adapter is cpu and bad == 'output_read':
            continue  # Existing CPU alias policy permits this.
        monkeypatch.setattr(adapter, 'toolchain', lambda: pytest.fail('invalid tanh reached tools'))
        with pytest.raises((KernelCompileError, ValueError)):
            adapter.compile_kernel(ir)
    if bad in {'integer', 'bool', 'index', 'arity0', 'arity2'}:
        with pytest.raises(KernelCompileError): _emit_msl(ir)
        with pytest.raises(MlirEmitError): emit_mlir(ir)


@pytest.mark.parametrize('call', ['cx.experimental.tanh()', 'cx.experimental.tanh(a[i], b[i])',
                                  'cx.experimental.tanh(value=a[i])', 'cx.tanh(a[i])', 'np.tanh(a[i])', 'tanh(a[i])'])
def test_tanh_parser_arity_keywords(call, tmp_path):
    path = tmp_path / 'tanh_invalid.py'
    path.write_text('import tensorcx as cx\nimport numpy as np\n@cx.experimental.kernel\ndef bad(a,b,out,n):\n    i=cx.experimental.thread_id()\n    if i<n:\n        out[i]=' + call + '\n')
    spec = importlib.util.spec_from_file_location('tanh_invalid', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(KernelCompileError): module.bad.parse_ir()


@pytest.mark.parametrize('kind', ['depth', 'operations'])
def test_tanh_expression_bounds_before_tools(kind, monkeypatch):
    a, b = IRLoad('a', IRName('i')), IRLoad('b', IRName('i'))
    value = IRBinaryOp('mul', a, b)
    if kind == 'depth':
        for _ in range(14): value = IRCall('tanh', (value,))
        valid = with_value(value)  # 14 tanh nodes + multiply + load = depth 16.
        invalid = with_value(IRCall('tanh', (value,)))
    else:
        ir = with_value(value)
        guard, store = ir.body[1], ir.body[1].body[0]
        body = tuple(IRAssign(f'x{i}', IRCall('tanh', (value,))) for i in range(32))
        valid = replace(ir, body=(ir.body[0], replace(guard, body=(*body, replace(store, value=IRName('x31'))))))
        first = replace(body[0], value=IRCall('tanh', (body[0].value,)))
        invalid = replace(valid, body=(ir.body[0], replace(valid.body[1], body=(first, *body[1:], replace(store, value=IRName('x31'))))))
    cuda.signature(valid)
    if os.environ.get('TENSORCX_REQUIRE_MLIR_CUDA'):
        require_device()
        native = cuda.compile_kernel(valid, cache=False)
        tensors = [cx.ones((4,), device='cuda') for _ in range(3)]
        expected = np.ones(4, dtype=np.float32)
        for _ in range(14 if kind == 'depth' else 1): expected = oracle_tanh(expected)
        assert_close(native.launch(*tensors, 4).numpy(), expected, 'cuda')
    monkeypatch.setattr(cuda, 'toolchain', lambda: pytest.fail('unbounded tanh reached tools'))
    with pytest.raises(KernelCompileError): cuda.compile_kernel(invalid)


@pytest.mark.parametrize('case', ['shape', 'dtype', 'bool_dtype', 'guard', 'block', 'count'])
def test_tanh_invalid_launch_preserves_output(target, case):
    native = hyperbolic_tangent.compile(target=target, compiler='mlir', cache=False)
    a = cx.ones((4,), device=target)
    args, block = [a, a, a, 4], 7
    if case == 'shape': args[0] = cx.ones((2, 2), device=target)
    if case == 'dtype': args[0] = cx.tensor(np.ones(4, dtype=np.int32), device=target)
    if case == 'bool_dtype': args[0] = cx.tensor(np.ones(4, dtype=np.bool_), device=target)
    if case == 'guard': args[-1] = 3
    if case == 'block': block = 0
    if case == 'count': args.pop()
    with pytest.raises((ValueError, TypeError)):
        native.launch(*args, thread_count=4, block_size=block)
    np.testing.assert_array_equal(a.numpy(), 1)


def test_tanh_cache_operation_separation(target, monkeypatch, tmp_path):
    monkeypatch.setenv('TENSORCX_DISABLE_CACHE', '0')
    monkeypatch.setenv('TENSORCX_CACHE_DIR', str(tmp_path / 'private-cache'))
    first = hyperbolic_tangent.compile(target=target, compiler='mlir')
    second = hyperbolic_tangent.compile(target=target, compiler='mlir')
    ir = hyperbolic_tangent.parse_ir()
    other_ir = with_value(IRCall('abs', ir.body[1].body[0].value.args))
    other = (cpu if target == 'cpu' else cuda).compile_kernel(other_ir)
    assert not first.cache_hit and second.cache_hit and not other.cache_hit
    a = cx.tensor([4., 9.], device=target)
    b, out = cx.ones((2,), device=target), cx.zeros((2,), device=target)
    assert_close(second.launch(a, b, out, 2).numpy(), oracle_tanh(np.array([4, 9], dtype=np.float32)), target)
    assert_bits(other.launch(a, b, out, 2).numpy(), np.array([4, 9], dtype=np.float32))


def test_tanh_outside_kernel_is_explicit():
    with pytest.raises(NotImplementedError): cx.experimental.tanh(1.0)


@cx.experimental.kernel
def hyperbolic_tangent_one(a, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.tanh(a[i])


def test_tanh_single_input_cpu():
    require_tools()
    values = numerical_inputs()
    a = cx.tensor(values, device='cpu')
    out = cx.zeros(a.shape, device='cpu')
    expected = oracle_tanh(values)
    native = hyperbolic_tangent_one.compile(target='cpu', compiler='mlir', cache=False)
    with np.errstate(all='raise'):
        assert_close(hyperbolic_tangent_one.reference(a, out, values.size).numpy(), expected)
    assert_close(native.launch(a, out, values.size).numpy(), expected)
    np.testing.assert_array_equal(out.numpy(), 0)


def test_tanh_single_input_cuda_unsupported_before_tools(monkeypatch):
    monkeypatch.setattr(cuda, 'toolchain', lambda: pytest.fail('unary CUDA signature reached tools'))
    with pytest.raises(KernelCompileError, match='two input parameters'):
        cuda.compile_kernel(hyperbolic_tangent_one.parse_ir())


def test_tanh_metal_compiler_policy(monkeypatch):
    calls = []
    def compile_(name, source, **options):
        calls.append((source, options))
        return b'compiled-artifact'
    monkeypatch.setattr(cx.experimental, '_compile_msl_to_metallib', compile_)
    hyperbolic_tangent_shadow.compile(target='metal', cache=False)
    assert calls[0][1] == {'strict_math': True}
    assert '::metal::tanh(' in calls[0][0]


@pytest.mark.parametrize('operation', ['tanh', 'exp', 'log', 'abs', 'rsqrt'])
def test_tanh_cpu_link_math_policy(operation, monkeypatch):
    require_tools()
    from test_mlir_abs import absolute
    from test_mlir_exp import exponential
    from test_mlir_log import logarithm
    from test_mlir_rsqrt import reciprocal_root
    commands, llvm_sources = [], []
    run = cpu._run
    def record(command):
        commands.append(command)
        if '-shared' in command:
            from pathlib import Path
            llvm_sources.append(Path(next(x for x in command if x.endswith('kernel.ll'))).read_text())
        return run(command)
    monkeypatch.setattr(cpu, '_run', record)
    {'tanh': hyperbolic_tangent, 'exp': exponential, 'log': logarithm,
     'abs': absolute, 'rsqrt': reciprocal_root}[operation].compile(target='cpu', compiler='mlir', cache=False)
    command, = [cmd for cmd in commands if '-shared' in cmd]
    assert ('-lm' in command) is (operation in {'tanh', 'exp', 'log'})
    if '-lm' in command:
        assert command.index('-lm') > next(i for i, x in enumerate(command) if x.endswith('adapter.c'))
    if operation == 'tanh':
        assert 'llvm.tanh.f32' in llvm_sources[0]
        assert not re.search(r'\b(?:fast|afn|arcp|nsz)\b', llvm_sources[0])


@pytest.mark.parametrize('name', ['tanhf', 'expf', 'logf'])
@pytest.mark.parametrize('cache', [False, True])
def test_tanh_cpu_libm_name_collision_before_tools(name, cache, monkeypatch):
    argument = hyperbolic_tangent.parse_ir().body[1].body[0].value.args
    ir = replace(with_value(IRCall(name[:-1], argument)), name=name)
    monkeypatch.setattr(cpu, 'toolchain', lambda: pytest.fail('libm collision reached tools'))
    monkeypatch.setattr(cpu._core, '_load_cpu_kernel', lambda *args: pytest.fail('libm collision reached native loader'))
    with pytest.raises(KernelCompileError, match=f"name '{name}' is reserved for libm"):
        cpu.compile_kernel(ir, cache=cache)
    cuda.signature(ir)


def test_tanh_exact_specials(target):
    values = np.array([0., -0., np.inf, -np.inf, np.finfo(np.float32).max,
                       -np.finfo(np.float32).max, np.nan], dtype=np.float32)
    tensors = [cx.tensor(x, device=target) for x in (values, np.ones_like(values), np.zeros_like(values))]
    result = hyperbolic_tangent.compile(target=target, compiler='mlir', cache=False).launch(*tensors, values.size).numpy()
    np.testing.assert_array_equal(result[:6].view(np.uint32),
                                  [0, 0x80000000, 0x3f800000, 0xbf800000, 0x3f800000, 0xbf800000])
    assert np.isnan(result[-1])


def test_tanh_small_values_contract(target):
    values = numerical_inputs()
    values = values[np.abs(values) <= np.float32(2**-12)]
    a = cx.tensor(values, device=target)
    b, out = cx.ones(a.shape, device=target), cx.zeros(a.shape, device=target)
    result = hyperbolic_tangent.compile(target=target, compiler='mlir', cache=False).launch(a, b, out, values.size)
    actual = result.numpy()
    assert_close(actual, oracle_tanh(values), target)
    # CPU libm may differ by an ULP for small normal inputs within tolerance.
    # Signed zeros/subnormals stay exact; CUDA's small rational path is identity.
    exact = np.abs(values) < np.finfo(np.float32).tiny
    if target == 'cuda':
        exact = np.ones(values.shape, dtype=bool)
    assert_bits(actual[exact], values[exact])
    assert_bits(a.numpy(), values)
