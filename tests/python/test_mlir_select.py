"""Guarded float32 comparison/select, numerical semantics and safe launch paths."""
from dataclasses import replace
import os
import re

import numpy as np
import pytest
import tensorcx as cx
from tensorcx._compiler import cpu, cuda
from tensorcx._compiler.emitter import emit_mlir, MlirEmitError
from tensorcx.experimental import IRBinaryOp, IRCall, IRCompare, IRConstant, IRLoad, IRName, KernelCompileError, _emit_msl
from test_mlir_runtime import require_tools
from test_mlir_cuda_runtime import require_device


@cx.experimental.kernel
def choose_lt(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.select(a[i] < b[i], a[i], b[i])


@cx.experimental.kernel
def choose_lte(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.select(a[i] <= b[i], a[i], b[i])


@cx.experimental.kernel
def choose_gt(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.select(a[i] > b[i], a[i], b[i])


@cx.experimental.kernel
def choose_gte(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.select(a[i] >= b[i], a[i], b[i])


@cx.experimental.kernel
def choose_eq(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.select(a[i] == b[i], a[i], b[i])


@cx.experimental.kernel
def choose_neq(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.select(a[i] != b[i], a[i], b[i])


@cx.experimental.kernel
def choose_local(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        mask = a[i] <= b[i]
        delta = a[i] - b[i]
        out[i] = cx.experimental.select(mask, delta, cx.experimental.select(a[i] != b[i], b[i], a[i])) + 0.5


@cx.experimental.kernel
def choose_division(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        different = a[i] != b[i]
        out[i] = cx.experimental.select(different, a[i] / b[i], 0.0)


@cx.experimental.kernel
def choose_reordered(n, out, b, a):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        mask = a[i] > b[i]
        out[i] = cx.experimental.select(mask, a[i], b[i])


@cx.experimental.kernel
def choose_mask(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.select(a[i] < b[i], 1.0, 0.0)


@cx.experimental.kernel
def choose_shadow(select, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.select(select[i] > b[i], select[i], b[i])


CHOICES = [choose_lt, choose_lte, choose_gt, choose_gte, choose_eq, choose_neq]
COMPARE = dict(zip(CHOICES, [np.less, np.less_equal, np.greater, np.greater_equal, np.equal, np.not_equal], strict=True))


def oracle(dsl, a, b):
    with np.errstate(all='ignore'):
        if dsl is choose_mask:
            return np.where(a < b, np.float32(1.), np.float32(0.))
        if dsl is choose_local:
            return np.add(np.where(a <= b, np.subtract(a, b, dtype=np.float32), np.where(a != b, b, a)), np.float32(.5), dtype=np.float32)
        if dsl is choose_division:
            return np.where(a != b, np.divide(a, b, dtype=np.float32), np.float32(0.))
        return np.where(COMPARE.get(dsl, np.greater)(a, b), a, b)


def assert_bits(actual, expected):
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    valid = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[valid].view(np.uint32), expected[valid].view(np.uint32))


@pytest.fixture(scope='module', params=['cpu', 'cuda'])
def target(request):
    (require_tools if request.param == 'cpu' else require_device)()
    return request.param


@pytest.fixture(scope='module', params=[*CHOICES, choose_local, choose_division, choose_mask, choose_shadow])
def compiled(request, target):
    return request.param, target, request.param.compile(target=target, compiler='mlir', cache=False)


@pytest.mark.parametrize('shape', [(), (0,), (1,), (255,), (256,), (257,), (2, 3)])
@pytest.mark.parametrize('block', [1, 7, 256])
def test_select_parity(compiled, shape, block):
    dsl, target, native = compiled
    count = int(np.prod(shape))
    a = np.arange(count, dtype=np.float32).reshape(shape) * np.float32(.25) - np.float32(3)
    b = np.full(shape, .75, dtype=np.float32)
    tensors = [cx.tensor(values, device=target) for values in (a, b, np.full(shape, -7, dtype=np.float32))]
    assert_bits(native.launch(*tensors, count, block_size=block).numpy(), oracle(dsl, a, b))
    assert_bits(dsl.reference(*(t.cpu() for t in tensors), count, block_size=block).numpy(), oracle(dsl, a, b))
    np.testing.assert_array_equal(tensors[2].numpy(), -7)


def test_select_ieee_predicates_and_value_bits(compiled):
    dsl, target, native = compiled
    bits = np.array([0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,
                     0x80800000,0x3f800000,0xbf800000,0x7f7fffff,0xff7fffff,
                     0x7f800000,0xff800000,0x7fc00001,0xffc01234,0x7f800001], dtype=np.uint32)
    values = bits.view(np.float32)
    rng = np.random.default_rng(1729)
    a = np.concatenate([np.repeat(values, values.size), rng.integers(0, 2**32, 1024, dtype=np.uint32).view(np.float32)])
    b = np.concatenate([np.tile(values, values.size), rng.integers(0, 2**32, 1024, dtype=np.uint32).view(np.float32)])
    tensors = [cx.tensor(values, device=target) for values in (a, b, np.zeros_like(a))]
    expected = oracle(dsl, a, b)
    assert_bits(native.launch(*tensors, a.size, block_size=113).numpy(), expected)
    with np.errstate(all='ignore'):
        assert_bits(dsl.reference(*(t.cpu() for t in tensors), a.size).numpy(), expected)


@pytest.mark.parametrize('count', [0,1,255,256,257])
@pytest.mark.parametrize('alias', ['none','left','right','both'])
def test_select_prefix_alias(compiled, count, alias):
    dsl, target, native = compiled
    a = cx.tensor(np.arange(257, dtype=np.float32) * .25, device=target)
    b = a if alias == 'both' else cx.tensor(np.full(257, .75, dtype=np.float32), device=target)
    out = {'left':a, 'right':b, 'both':a}.get(alias)
    if out is None: out = cx.tensor(np.full(257, -7, dtype=np.float32), device=target)
    originals = [t.numpy() for t in (a,b,out)]
    expected = originals[2].copy()
    expected[:count] = oracle(dsl, originals[0][:count], originals[1][:count])
    for _ in range(2):
        assert_bits(native.launch(a,b,out,count,thread_count=count,block_size=7).numpy(), expected)
    for tensor, original in zip((a,b,out), originals, strict=True):
        np.testing.assert_array_equal(tensor.numpy().view(np.uint32), original.view(np.uint32))


def test_select_reordered_and_no_numpy_fallback(target, monkeypatch):
    a, b, out = (cx.tensor(np.array(x, dtype=np.float32), device=target) for x in ([1,-3,.25,8], [2,5,-4,.5], [-7]*4))
    expected = oracle(choose_reordered, a.numpy(), b.numpy())
    native = choose_reordered.compile(target=target, compiler='mlir', cache=False)
    with monkeypatch.context() as patch:
        patch.setattr(cx.Tensor, 'numpy', lambda *args: pytest.fail('select fell back to NumPy'))
        result = native.launch(4,out,b,a)
    assert_bits(result.numpy(), expected)


def test_gpu_emitter_select_predicates():
    require_tools()
    for dsl, predicate in zip(CHOICES, ['olt','ole','ogt','oge','oeq','une'], strict=True):
        ir = dsl.parse_ir()
        source, ptx = cuda.lower(ir)
        assert f'arith.cmpf {predicate},' in source and 'arith.select' in source
        assert f'arith.cmpf {predicate},' in emit_mlir(ir)
        assert 'tensorcx_expr_v1' in ptx
        assert re.search(r'\bsetp\.', ptx) and re.search(r'\bselp\.', ptx)
        assert not re.search(r'\bsetp\.[^;\n]*\.ftz', ptx)


@pytest.mark.parametrize('bad', ['condition','true','false','arity','bool_math','compare_bool','bool_output','offset','output_read','hidden_buffer'])
def test_select_invalid_before_tools(monkeypatch, bad):
    ir = choose_lt.parse_ir()
    store = ir.body[1].body[0]
    condition, a, b = store.value.args
    value = store.value
    if bad == 'condition': value = replace(value, args=(a,a,b))
    if bad == 'true': value = replace(value, args=(condition,IRConstant(1),b))
    if bad == 'false': value = replace(value, args=(condition,a,condition))
    if bad == 'arity': value = replace(value, args=(condition,a))
    if bad == 'bool_math': value = IRBinaryOp('add',condition,a)
    if bad == 'compare_bool': value = replace(value,args=(IRCompare('eq',condition,condition),a,b))
    if bad == 'bool_output': value = condition
    if bad == 'offset': value = replace(value,args=(condition,a,replace(b,index=IRBinaryOp('add',IRName('i'),IRConstant(1)))))
    if bad == 'output_read': value = replace(value,args=(condition,a,IRLoad('out',IRName('i'))))
    if bad == 'hidden_buffer': value = replace(value,args=(condition,a,IRLoad('missing',IRName('i'))))
    ir = replace(ir,body=(ir.body[0],replace(ir.body[1],body=(replace(store,value=value),))))
    for adapter in (cpu,cuda):
        # CPU retains the earlier general alias policy; CUDA forbids output
        # reads in its bounded expression body even in an unselected arm.
        if adapter is cpu and bad == 'output_read': continue
        monkeypatch.setattr(adapter,'toolchain',lambda:pytest.fail('invalid select reached tools'))
        with pytest.raises((KernelCompileError, ValueError)):
            adapter.compile_kernel(ir)


@pytest.mark.parametrize('bad', ['condition','true','false','arity'])
def test_select_common_typing(bad):
    ir = choose_lt.parse_ir()
    store = ir.body[1].body[0]
    condition,a,b = store.value.args
    args = {'condition':(a,a,b),'true':(condition,IRConstant(1),b),
            'false':(condition,a,condition),'arity':(condition,a)}[bad]
    ir = replace(ir,body=(ir.body[0],replace(ir.body[1],body=(replace(store,value=IRCall('select',args)),))))
    with pytest.raises(KernelCompileError): _emit_msl(ir)
    with pytest.raises(MlirEmitError): emit_mlir(ir)


def test_select_cache_operation_separation(target, monkeypatch, tmp_path):
    monkeypatch.setenv('TENSORCX_DISABLE_CACHE','0')
    monkeypatch.setenv('TENSORCX_CACHE_DIR',str(tmp_path/'private-cache'))
    first = choose_lt.compile(target=target,compiler='mlir')
    second = choose_lt.compile(target=target,compiler='mlir')
    ir = choose_lt.parse_ir()
    store = ir.body[1].body[0]
    condition,a,b = store.value.args
    ir = replace(ir,body=(ir.body[0],replace(ir.body[1],body=(replace(store,value=IRCall('select',(replace(condition,op='gt'),a,b))),))))
    other = (cpu if target=='cpu' else cuda).compile_kernel(ir)
    assert not first.cache_hit and second.cache_hit and not other.cache_hit
    a,b,out = cx.ones((4,),device=target),cx.zeros((4,),device=target),cx.zeros((4,),device=target)
    np.testing.assert_array_equal(second.launch(a,b,out,4).numpy(),0)
    np.testing.assert_array_equal(other.launch(a,b,out,4).numpy(),1)


def test_select_outside_kernel_is_explicit():
    with pytest.raises(NotImplementedError): cx.experimental.select(True,1.0,2.0)


@pytest.mark.parametrize('case',['shape','dtype','guard','block','count'])
def test_select_invalid_launch_preserves_output(target,case):
    native=choose_lt.compile(target=target,compiler='mlir',cache=False)
    a=cx.ones((4,),device=target)
    args=[a,a,a,4]
    block=7
    if case=='shape': args[0]=cx.ones((2,2),device=target)
    if case=='dtype': args[0]=cx.tensor(np.ones(4,dtype=np.int32),device=target)
    if case=='guard': args[-1]=3
    if case=='block': block=0
    if case=='count': args.pop()
    with pytest.raises((ValueError,TypeError)): native.launch(*args,thread_count=4,block_size=block)
    np.testing.assert_array_equal(a.numpy(),1)


@pytest.mark.parametrize('arguments',['a[i] < b[i], a[i]','a[i] < b[i], a[i], b[i], 0.0','condition=a[i] < b[i], on_true=a[i], on_false=b[i]'])
def test_select_parser_arity_keywords(arguments,tmp_path):
    import importlib.util
    path=tmp_path/'select_invalid.py'
    path.write_text('import tensorcx as cx\n@cx.experimental.kernel\ndef bad(a,b,out,n):\n    i=cx.experimental.thread_id()\n    if i<n:\n        out[i]=cx.experimental.select('+arguments+')\n')
    spec=importlib.util.spec_from_file_location('select_invalid',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(KernelCompileError): module.bad.parse_ir()


@pytest.mark.parametrize('kind',['operations','depth'])
def test_select_expression_bounds_before_tools(kind,monkeypatch):
    ir=choose_lt.parse_ir()
    store=ir.body[1].body[0]
    condition,a,b=store.value.args
    if kind=='operations':
        from tensorcx.experimental import IRAssign
        body=tuple(IRAssign(f'x{i}',store.value) for i in range(32))+(replace(store,value=IRName('x31')),)
        valid=replace(ir,body=(ir.body[0],replace(ir.body[1],body=body)))
        first=body[0]
        extra=replace(first,value=IRCall('select',(condition,IRBinaryOp('add',a,IRConstant(.5)),b)))
        invalid=replace(valid,body=(valid.body[0],replace(valid.body[1],body=(extra,*body[1:]))))
    else:
        value=b
        for _ in range(14): value=IRCall('select',(condition,a,value))
        valid=replace(ir,body=(ir.body[0],replace(ir.body[1],body=(replace(store,value=value),))))
        invalid=replace(ir,body=(ir.body[0],replace(ir.body[1],body=(replace(store,value=IRCall('select',(condition,a,value))),))))
    cuda.signature(valid)
    if os.environ.get('TENSORCX_REQUIRE_MLIR_CUDA'):
        require_device()
        native=cuda.compile_kernel(valid,cache=False)
        a,b,out=cx.ones((4,),device='cuda'),cx.zeros((4,),device='cuda'),cx.zeros((4,),device='cuda')
        np.testing.assert_array_equal(native.launch(a,b,out,4).numpy(),0)
    monkeypatch.setattr(cuda,'toolchain',lambda:pytest.fail('unbounded select reached tools'))
    with pytest.raises(KernelCompileError): cuda.compile_kernel(invalid)
