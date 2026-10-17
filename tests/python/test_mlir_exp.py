"""Guarded float32 exp: bounded error, special values, validation and ownership."""
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
def exponential(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.exp(a[i] * b[i])


@cx.experimental.kernel
def exponential_local(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        exp = cx.experimental.exp(a[i] / b[i])
        out[i] = cx.experimental.exp(exp - b[i])


@cx.experimental.kernel
def exponential_select(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        value = cx.experimental.exp(cx.experimental.select(a[i] < b[i], a[i], b[i]))
        out[i] = cx.experimental.select(value > 0.5, cx.experimental.sqrt(value), cx.experimental.exp(-0.5))


@cx.experimental.kernel
def exponential_shadow(exp, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.exp(exp[i] * b[i])


@cx.experimental.kernel
def exponential_reordered(n, out, b, a):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.exp(a[i] * b[i])


@cx.experimental.kernel
def exponential_bounded(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.exp(cx.experimental.minimum(cx.experimental.maximum(a[i], -2.0), b[i]))


KERNELS = [exponential, exponential_local, exponential_select, exponential_shadow, exponential_bounded]


def oracle_exp(values):
    # Independent float64 computation, rounded once to float32.
    with np.errstate(all='ignore'):
        return np.exp(values.astype(np.float64)).astype(np.float32)


def oracle(dsl, a, b):
    with np.errstate(all='ignore'):
        if dsl is exponential_bounded:
            clamped = np.where(a > np.float32(-2), a, np.float32(-2))
            return oracle_exp(np.where(clamped < b, clamped, b))
        if dsl is exponential_local:
            value = oracle_exp(np.divide(a, b, dtype=np.float32))
            return oracle_exp(np.subtract(value, b, dtype=np.float32))
        if dsl is exponential_select:
            value = oracle_exp(np.where(a < b, a, b))
            return np.where(value > np.float32(.5), np.sqrt(value, dtype=np.float32), oracle_exp(np.full_like(value, -.5)))
        return oracle_exp(np.multiply(a, b, dtype=np.float32))


def assert_close(actual, expected, target='cpu'):
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    # Compute tolerances in float64; f32 arithmetic would round away the
    # subnormal allowance or overflow while comparing large finite results.
    rtol = 1e-5 if target == 'cuda' else 1e-6
    allowance = (2 if target == 'cuda' else 1) * float.fromhex('0x1p-149')
    error = np.abs(actual[finite].astype(np.float64) - expected[finite].astype(np.float64))
    assert np.all(error <= rtol * np.abs(expected[finite].astype(np.float64)) + allowance)
    zero = (actual == 0) & (expected == 0)
    np.testing.assert_array_equal(np.signbit(actual[zero]), np.signbit(expected[zero]))


def assert_bits(actual, expected):
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def numerical_inputs():
    patterns = np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
                         0x00800000, 0x80800000, 0x7f7fffff, 0xff7fffff,
                         0x7f800000, 0xff800000, 0x7fc00001, 0xffc01234,
                         0x7f800001, 0xff800001], dtype=np.uint32).view(np.float32)
    edges = []
    for center in [np.log(np.finfo(np.float32).max), np.log(np.finfo(np.float32).tiny),
                   -150 * np.log(2), -149 * np.log(2)]:
        value = np.float32(center)
        edges.append(value)
        lo = hi = value
        for _ in range(32):
            lo = np.nextafter(lo, np.float32(-np.inf)); hi = np.nextafter(hi, np.float32(np.inf))
            edges.extend((lo, hi))
    rng = np.random.default_rng(1729)
    return np.concatenate([patterns, np.array(edges, dtype=np.float32),
                           np.linspace(-104, 89, 4097, dtype=np.float32),
                           rng.integers(0, 2**32, 4096, dtype=np.uint32).view(np.float32)])


@pytest.fixture(scope='module', params=['cpu', 'cuda'])
def target(request):
    (require_tools if request.param == 'cpu' else require_device)()
    return request.param


@pytest.fixture(scope='module', params=KERNELS)
def compiled(request, target):
    return request.param, target, request.param.compile(target=target, compiler='mlir', cache=False)


@pytest.mark.parametrize('shape', [(), (0,), (1,), (255,), (256,), (257,), (2, 3)])
@pytest.mark.parametrize('block', [1, 7, 256])
def test_exp_parity(compiled, shape, block):
    dsl, target, native = compiled
    count = int(np.prod(shape))
    a = np.linspace(-1, 1, count, dtype=np.float32).reshape(shape)
    b = np.full(shape, .75, dtype=np.float32)
    tensors = [cx.tensor(values, device=target) for values in (a, b, np.full(shape, -7, dtype=np.float32))]
    expected = oracle(dsl, a, b)
    assert_close(native.launch(*tensors, count, block_size=block).numpy(), expected, target)
    assert_close(dsl.reference(*(t.cpu() for t in tensors), count, block_size=block).numpy(), expected)
    np.testing.assert_array_equal(tensors[2].numpy(), -7)


def test_exp_numerical_domain(target):
    a = numerical_inputs()
    tensors = [cx.tensor(values, device=target) for values in (a, np.ones_like(a), np.zeros_like(a))]
    native = exponential.compile(target=target, compiler='mlir', cache=False)
    expected = oracle_exp(a)
    assert_close(native.launch(*tensors, a.size, block_size=113).numpy(), expected, target)
    assert_close(exponential.reference(*(t.cpu() for t in tensors), a.size).numpy(), expected)
    np.testing.assert_array_equal(tensors[2].numpy(), 0)


@pytest.mark.parametrize('count', [0, 1, 255, 256, 257])
@pytest.mark.parametrize('alias', ['none', 'left', 'right', 'both'])
def test_exp_prefix_alias(target, count, alias):
    native = exponential.compile(target=target, compiler='mlir', cache=False)
    a = cx.tensor(np.linspace(-2, 2, 257, dtype=np.float32), device=target)
    b = a if alias == 'both' else cx.ones((257,), device=target)
    out = {'left': a, 'right': b, 'both': a}.get(alias)
    if out is None:
        out = cx.tensor(np.full(257, -7, dtype=np.float32), device=target)
    originals = [t.numpy() for t in (a, b, out)]
    expected = originals[2].copy()
    expected[:count] = oracle(exponential, originals[0][:count], originals[1][:count])
    for _ in range(2):
        result = native.launch(a, b, out, count, thread_count=count, block_size=7)
        assert result is not out
        values = result.numpy()
        assert_close(values, expected, target)
        assert_bits(values[count:], originals[2][count:])
    for tensor, original in zip((a, b, out), originals, strict=True):
        assert_bits(tensor.numpy(), original)


def test_exp_reordered_and_no_numpy_fallback(target, monkeypatch):
    a, b, out = (cx.tensor(np.array(x, dtype=np.float32), device=target)
                 for x in ([1, 4, 9, 16], [4, 1, .25, 16], [-7] * 4))
    expected = oracle(exponential, a.numpy(), b.numpy())
    native = exponential_reordered.compile(target=target, compiler='mlir', cache=False)
    with monkeypatch.context() as patch:
        patch.setattr(cx.Tensor, 'numpy', lambda *args: pytest.fail('exp fell back to NumPy'))
        result = native.launch(4, out, b, a)
    assert_close(result.numpy(), expected, target)


def test_gpu_emitter_exp_no_ftz():
    require_tools()
    for dsl in KERNELS:
        source, ptx = cuda.lower(dsl.parse_ir())
        assert 'math.exp' in emit_mlir(dsl.parse_ir())
        assert 'llvm.nvvm.ex2.approx.f' in source
        assert re.search(r'\bex2\.approx\.f32\b', ptx)
        assert not re.search(r'\b(?:ex2|mul)\.[^;\n]*ftz', ptx)
        assert '__nv_exp' not in ptx and '.extern .func' not in ptx
        assert 'expr-f32-v1' in ptx and 'tensorcx_expr_v1' in ptx
    assert '::metal::exp(' in exponential_shadow.emit_msl()


def with_value(value):
    ir = exponential.parse_ir()
    guard = ir.body[1]
    return replace(ir, body=(ir.body[0], replace(guard, body=(replace(guard.body[0], value=value),))))


@pytest.mark.parametrize('bad', ['integer', 'bool', 'index', 'arity0', 'arity2', 'offset', 'output_read', 'hidden_buffer'])
def test_exp_invalid_before_tools(monkeypatch, bad):
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
    ir = with_value(IRCall('exp', arguments))
    for adapter in (cpu, cuda):
        if adapter is cpu and bad == 'output_read':
            continue  # Existing CPU alias policy permits this.
        monkeypatch.setattr(adapter, 'toolchain', lambda: pytest.fail('invalid exp reached tools'))
        with pytest.raises((KernelCompileError, ValueError)):
            adapter.compile_kernel(ir)
    if bad in {'integer', 'bool', 'index', 'arity0', 'arity2'}:
        with pytest.raises(KernelCompileError): _emit_msl(ir)
        with pytest.raises(MlirEmitError): emit_mlir(ir)


@pytest.mark.parametrize('call', ['cx.experimental.exp()', 'cx.experimental.exp(a[i], b[i])',
                                  'cx.experimental.exp(value=a[i])', 'cx.exp(a[i])', 'np.exp(a[i])', 'exp(a[i])'])
def test_exp_parser_arity_keywords(call, tmp_path):
    path = tmp_path / 'exp_invalid.py'
    path.write_text('import tensorcx as cx\nimport numpy as np\n@cx.experimental.kernel\ndef bad(a,b,out,n):\n    i=cx.experimental.thread_id()\n    if i<n:\n        out[i]=' + call + '\n')
    spec = importlib.util.spec_from_file_location('exp_invalid', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(KernelCompileError): module.bad.parse_ir()


@pytest.mark.parametrize('kind', ['depth', 'operations'])
def test_exp_expression_bounds_before_tools(kind, monkeypatch):
    a, b = IRLoad('a', IRName('i')), IRLoad('b', IRName('i'))
    value = IRBinaryOp('mul', a, b)
    if kind == 'depth':
        for _ in range(14): value = IRCall('exp', (value,))
        valid = with_value(value)  # 14 exp nodes + multiply + load = depth 16.
        invalid = with_value(IRCall('exp', (value,)))
    else:
        ir = with_value(value)
        guard, store = ir.body[1], ir.body[1].body[0]
        body = tuple(IRAssign(f'x{i}', IRCall('exp', (value,))) for i in range(32))
        valid = replace(ir, body=(ir.body[0], replace(guard, body=(*body, replace(store, value=IRName('x31'))))))
        first = replace(body[0], value=IRCall('exp', (body[0].value,)))
        invalid = replace(valid, body=(ir.body[0], replace(valid.body[1], body=(first, *body[1:], replace(store, value=IRName('x31'))))))
    cuda.signature(valid)
    if os.environ.get('TENSORCX_REQUIRE_MLIR_CUDA'):
        require_device()
        native = cuda.compile_kernel(valid, cache=False)
        tensors = [cx.ones((4,), device='cuda') for _ in range(3)]
        assert_close(native.launch(*tensors, 4).numpy(), np.full(4, np.inf, dtype=np.float32) if kind == 'depth' else oracle_exp(np.ones(4, dtype=np.float32)), 'cuda')
    monkeypatch.setattr(cuda, 'toolchain', lambda: pytest.fail('unbounded exp reached tools'))
    with pytest.raises(KernelCompileError): cuda.compile_kernel(invalid)


@pytest.mark.parametrize('case', ['shape', 'dtype', 'bool_dtype', 'guard', 'block', 'count'])
def test_exp_invalid_launch_preserves_output(target, case):
    native = exponential.compile(target=target, compiler='mlir', cache=False)
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


def test_exp_cache_operation_separation(target, monkeypatch, tmp_path):
    monkeypatch.setenv('TENSORCX_DISABLE_CACHE', '0')
    monkeypatch.setenv('TENSORCX_CACHE_DIR', str(tmp_path / 'private-cache'))
    first = exponential.compile(target=target, compiler='mlir')
    second = exponential.compile(target=target, compiler='mlir')
    ir = exponential.parse_ir()
    other_ir = with_value(IRCall('abs', ir.body[1].body[0].value.args))
    other = (cpu if target == 'cpu' else cuda).compile_kernel(other_ir)
    assert not first.cache_hit and second.cache_hit and not other.cache_hit
    a = cx.tensor([4., 9.], device=target)
    b, out = cx.ones((2,), device=target), cx.zeros((2,), device=target)
    assert_close(second.launch(a, b, out, 2).numpy(), oracle_exp(np.array([4, 9], dtype=np.float32)), target)
    assert_bits(other.launch(a, b, out, 2).numpy(), np.array([4, 9], dtype=np.float32))


def test_exp_outside_kernel_is_explicit():
    with pytest.raises(NotImplementedError): cx.experimental.exp(1.0)


@cx.experimental.kernel
def exponential_one(a, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.exp(a[i])


def test_exp_single_input_cpu():
    require_tools()
    values = numerical_inputs()
    a = cx.tensor(values, device='cpu')
    out = cx.zeros(a.shape, device='cpu')
    expected = oracle_exp(values)
    native = exponential_one.compile(target='cpu', compiler='mlir', cache=False)
    with np.errstate(all='raise'):
        assert_close(exponential_one.reference(a, out, values.size).numpy(), expected)
    assert_close(native.launch(a, out, values.size).numpy(), expected)
    np.testing.assert_array_equal(out.numpy(), 0)


def test_exp_single_input_cuda_unsupported_before_tools(monkeypatch):
    monkeypatch.setattr(cuda, 'toolchain', lambda: pytest.fail('unary CUDA signature reached tools'))
    with pytest.raises(KernelCompileError, match='two input parameters'):
        cuda.compile_kernel(exponential_one.parse_ir())


def test_exp_metal_compiler_policy(monkeypatch):
    calls = []
    def compile_(name, source, **options):
        calls.append((source, options))
        return b'compiled-artifact'
    monkeypatch.setattr(cx.experimental, '_compile_msl_to_metallib', compile_)
    exponential_shadow.compile(target='metal', cache=False)
    assert calls[0][1] == {'strict_math': True}
    assert '::metal::exp(' in calls[0][0]


@pytest.mark.parametrize('exponential_math', [True, False])
def test_exp_cpu_link_math_policy(exponential_math, monkeypatch):
    require_tools()
    from test_mlir_abs import absolute
    commands = []
    run = cpu._run
    def record(command):
        commands.append(command)
        return run(command)
    monkeypatch.setattr(cpu, '_run', record)
    (exponential if exponential_math else absolute).compile(target='cpu', compiler='mlir', cache=False)
    command, = [cmd for cmd in commands if '-shared' in cmd]
    assert ('-lm' in command) is exponential_math
    if exponential_math:
        assert command.index('-lm') > next(i for i, value in enumerate(command) if value.endswith('adapter.c'))
