"""Guarded CUDA expressions: syntax bounds, separate rounding and runtime parity."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import os
import re

import numpy as np
import pytest
import cortex_runtime as cx
from cortex_runtime._compiler import cpu, cuda
from cortex_runtime.experimental import IRAssign, IRBinaryOp, IRConstant, IRLoad, IRName, KernelCompileError
from test_mlir_cuda_runtime import require_device


@cx.experimental.kernel
def blend(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        delta = a[i] - b[i]
        scaled = delta * 0.5
        out[i] = scaled + b[i]


@cx.experimental.kernel
def nested(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = (a[i] - b[i]) * 0.5 + b[i]


@cx.experimental.kernel
def reuse(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        delta = a[i] - b[i]
        value = delta * delta
        out[i] = 0.25 * value + (a[i] - delta)


@cx.experimental.kernel
def rounding(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        product = a[i] * b[i]
        out[i] = product - 1.0


@cx.experimental.kernel
def reordered(n, out, b, a):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        value = a[i] - b[i]
        out[i] = value * 0.5


@pytest.fixture(scope='module', params=[blend, nested, reuse, rounding])
def kernels(request):
    require_device()
    dsl = request.param
    return dsl, dsl.compile(target='cpu', compiler='mlir'), dsl.compile(target='cuda', compiler='mlir')


@pytest.mark.parametrize('shape', [(), (0,), (1,), (255,), (256,), (257,), (2, 3)])
@pytest.mark.parametrize('block', [1, 7, 256])
def test_expression_parity(kernels, shape, block):
    dsl, cpu, gpu = kernels
    count = int(np.prod(shape))
    a = cx.tensor((np.arange(count, dtype=np.float32) * 0.25 - 3).reshape(shape))
    b = cx.tensor(np.full(shape, 0.75, dtype=np.float32))
    out = cx.tensor(np.full(shape, -7, dtype=np.float32))
    expected = dsl.reference(a, b, out, count, block_size=block).numpy()
    np.testing.assert_allclose(cpu.launch(a, b, out, count, block_size=block).numpy(), expected, rtol=1e-6, atol=1e-6)
    ga, gb, go = (t.to('cuda') for t in (a, b, out))
    np.testing.assert_allclose(gpu.launch(ga, gb, go, count, block_size=block).numpy(), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(go.numpy(), out.numpy())


@pytest.mark.parametrize('count', [0, 1, 255, 256, 257])
@pytest.mark.parametrize('alias', ['none', 'left', 'right', 'both'])
def test_expression_prefix_alias(kernels, count, alias):
    dsl, _, gpu = kernels
    a = cx.tensor(np.arange(257, dtype=np.float32) * .25, device='cuda')
    b = a if alias == 'both' else cx.tensor(np.full(257, .75, dtype=np.float32), device='cuda')
    out = {'left': a, 'right': b, 'both': a}.get(alias)
    if out is None:
        out = cx.tensor(np.full(257, -7, dtype=np.float32), device='cuda')
    originals = [t.numpy().copy() for t in (a,b,out)]
    expected = dsl.reference(a.cpu(),b.cpu(),out.cpu(),count,thread_count=count).numpy()
    for _ in range(2):
        np.testing.assert_array_equal(gpu.launch(a,b,out,count,thread_count=count).numpy(),expected)
    for t, original in zip((a,b,out), originals):
        np.testing.assert_array_equal(t.numpy(),original)


def test_expression_rounding_and_edges():
    require_device()
    gpu = rounding.compile(target='cuda',compiler='mlir')
    cpu = rounding.compile(target='cpu',compiler='mlir')
    a = cx.tensor(np.array([1+2**-23,0.,-0.,np.inf,-np.inf,np.nan,1e20],dtype=np.float32))
    b = cx.tensor(np.array([1-2**-23,0.,1.,2.,-2.,1.,1e20],dtype=np.float32))
    out = cx.zeros(a.shape)
    with np.errstate(all='ignore'):
        expected = np.subtract(np.multiply(a.numpy(), b.numpy(), dtype=np.float32), np.float32(1), dtype=np.float32)
        actual = gpu.launch(a.to('cuda'),b.to('cuda'),out.to('cuda'),7).numpy()
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(cpu.launch(a,b,out,7).numpy(),expected)
    assert actual[0] == 0 and np.float32(float(a.numpy()[0])*float(b.numpy()[0])-1) == np.float32(-2**-46)


def test_expression_modules_and_reordered():
    require_device()
    ga = cx.tensor(np.array([1,-3,.25,8],dtype=np.float32),device='cuda')
    gb = cx.tensor(np.array([2,5,-4,.5],dtype=np.float32),device='cuda')
    modules = [(dsl, dsl.compile(target='cuda',compiler='mlir')) for dsl in (blend,reuse,rounding)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = [(dsl, pool.submit(module.launch,ga,gb,ga,4)) for dsl,module in modules for _ in range(4)]
        del modules
        for dsl,future in pending:
            np.testing.assert_array_equal(future.result().numpy(),dsl.reference(ga.cpu(),gb.cpu(),ga.cpu(),4).numpy())
    other = reordered.compile(target='cuda',compiler='mlir')
    np.testing.assert_array_equal(other.launch(4,ga,gb,ga).numpy(),reordered.reference(4,ga.cpu(),gb.cpu(),ga.cpu()).numpy())


def body_ir(body):
    ir = blend.parse_ir()
    return replace(ir,body=(ir.body[0],replace(ir.body[1],body=tuple(body))))


@pytest.mark.parametrize('case', ['int','bool','nan','inf','overflow','output_read','undefined','reassign','shadow','extra_store','after_store','loop','offset','div','one_input'])
def test_expression_invalid_before_tools(monkeypatch,case):
    ir = blend.parse_ir()
    body = list(ir.body[1].body)
    if case in ('int','bool','nan','inf','overflow'):
        body[1] = replace(body[1],value=IRConstant({'int':1,'bool':True,'nan':float('nan'),'inf':float('inf'),'overflow':1e100}[case]))
    elif case=='output_read': body[0]=replace(body[0],value=IRLoad('out',IRName('i')))
    elif case=='undefined': body[0]=replace(body[0],value=IRName('missing'))
    elif case=='reassign': body[1]=replace(body[1],target='delta')
    elif case=='shadow': body[0]=replace(body[0],target='a')
    elif case=='extra_store': body.insert(1,body[-1])
    elif case=='after_store': body.append(body[0])
    elif case=='loop': body.insert(0,ir.body[1])
    elif case=='offset': body[0]=replace(body[0],value=IRLoad('a',IRBinaryOp('add',IRName('i'),IRConstant(1))))
    elif case=='div': body[0]=replace(body[0],value=replace(body[0].value,op='div'))
    elif case=='one_input': body[0]=replace(body[0],value=IRLoad('b',IRName('i')))
    monkeypatch.setattr(cuda,'toolchain',lambda:pytest.fail('invalid IR reached tools'))
    with pytest.raises(KernelCompileError): cuda.compile_kernel(body_ir(body))


@pytest.mark.parametrize('kind,limit', [('locals',32),('operators',64),('depth',16)])
def test_expression_bounds(kind,limit,monkeypatch):
    ir = blend.parse_ir()
    base = ir.body[1].body[0].value
    store = ir.body[1].body[-1]
    def make(n):
        if kind=='locals':
            body = [IRAssign(f'x{i}', base if i==0 else IRName(f'x{i-1}')) for i in range(n)]
            return body_ir(body+[replace(store,value=IRName(f'x{n-1}'))])
        if kind=='depth':
            value=base
            for _ in range(n-2): value=IRBinaryOp('add',value,IRConstant(0.5))
            return body_ir([replace(store,value=value)])
        # Four balanced-ish chains keep each expression below depth16 and locals below32.
        body=[]
        remaining=n
        for i in range(8):
            count=min(8,remaining)
            if not count:break
            value=base if i==0 else IRBinaryOp('add',IRName(f'x{i-1}'),IRConstant(0.5))
            for _ in range(count-1):value=IRBinaryOp('add',value,IRConstant(0.5))
            body.append(IRAssign(f'x{i}',value));remaining-=count
        if remaining:body.append(IRAssign('extra',IRBinaryOp('add',IRName('x7'),IRConstant(.5))))
        return body_ir(body+[replace(store,value=IRName(body[-1].target))])
    valid = make(limit)
    cuda.signature(valid)
    if os.environ.get('CORTEX_REQUIRE_MLIR_CUDA'):
        require_device()
        native, reference = cuda.compile_kernel(valid), cpu.compile_kernel(valid)
        a, b, out = cx.ones((4,)), cx.zeros((4,)), cx.zeros((4,))
        expected = reference.launch(a,b,out,4).numpy()
        np.testing.assert_array_equal(native.launch(a.to('cuda'),b.to('cuda'),out.to('cuda'),4).numpy(),expected)
    monkeypatch.setattr(cuda,'toolchain',lambda:pytest.fail('invalid IR reached tools'))
    with pytest.raises(KernelCompileError):cuda.compile_kernel(make(limit+1))


def test_gpu_emitter_expression_fixture():
    try:cuda.toolchain()
    except RuntimeError as error:
        if os.environ.get('CORTEX_REQUIRE_MLIR'):pytest.fail(str(error))
        pytest.skip(str(error))
    source,ptx=cuda.lower(rounding.parse_ir())
    assert 'arith.mulf' in source and 'arith.subf' in source
    assert 'mul.rn.f32' in ptx and not re.search(r'\b(?:fma|mad)\.',ptx)
    assert ptx==(Path(__file__).parents[1]/'cpp/fixtures/cuda_expr_sm52.ptx').read_text()


@pytest.mark.parametrize('case',['shape','dtype','device','guard','block','count'])
def test_expression_failed_launch_preserves_output(kernels,case):
    _,_,gpu=kernels
    a=cx.ones((4,),device='cuda')
    args=[a,a,a,4]
    block=256
    if case=='shape':args[0]=cx.ones((2,2),device='cuda')
    if case=='dtype':args[0]=cx.tensor(np.ones(4,dtype=np.int32),device='cuda')
    if case=='device':args[0]=a.cpu()
    if case=='guard':args[-1]=3
    if case=='block':block=1025
    if case=='count':args.pop()
    with pytest.raises((ValueError,TypeError)):
        gpu.launch(*args,thread_count=4,block_size=block)
    np.testing.assert_array_equal(a.numpy(),1)


def test_expression_signed_zero_subnormal():
    require_device()
    gpu=blend.compile(target='cuda',compiler='mlir')
    tiny=np.finfo(np.float32).tiny
    a=cx.tensor(np.array([0.,-0.,tiny,-tiny],dtype=np.float32))
    b=cx.tensor(np.array([-0.,0.,0.,-0.],dtype=np.float32))
    out=cx.zeros(a.shape)
    expected=np.add(np.multiply(np.subtract(a.numpy(),b.numpy(),dtype=np.float32),np.float32(.5),dtype=np.float32),b.numpy(),dtype=np.float32)
    actual=gpu.launch(a.to('cuda'),b.to('cuda'),out.to('cuda'),4).numpy()
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))


def test_expression_compile_cleanup(monkeypatch,tmp_path):
    require_device()
    monkeypatch.setattr(cuda.tempfile,'tempdir',str(tmp_path))
    blend.compile(target='cuda',compiler='mlir')
    assert list(tmp_path.iterdir())==[]
    monkeypatch.setattr(cuda,'_run',lambda _:(_ for _ in ()).throw(RuntimeError('tool failure')))
    with pytest.raises(RuntimeError,match='tool failure'):
        blend.compile(target='cuda',compiler='mlir')
    assert list(tmp_path.iterdir())==[]


def test_expression_index_shadow_before_tools(monkeypatch):
    ir=blend.parse_ir()
    index=replace(ir.body[0],target='n')
    guard=ir.body[1]
    value=IRBinaryOp('add',IRLoad('a',IRName('n')),IRLoad('b',IRName('n')))
    store=replace(guard.body[-1],index=IRName('n'),value=value)
    guard=replace(guard,condition=replace(guard.condition,lhs=IRName('n')),body=(store,))
    monkeypatch.setattr(cuda,'toolchain',lambda:pytest.fail('invalid IR reached tools'))
    with pytest.raises(KernelCompileError,match='shadow'):
        cuda.compile_kernel(replace(ir,body=(index,guard)))
