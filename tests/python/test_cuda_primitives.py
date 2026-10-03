"""CUDA float32 primitives: shape boundaries, special values and public errors."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
import cortex_runtime as cx

pytestmark=pytest.mark.skipif(not cx.is_available('cuda'),reason='CUDA unavailable')

@pytest.mark.parametrize('m,k,n',[(1,1,1),(2,3,4),(15,17,31),(17,16,15),(32,33,16),(0,3,4),(3,0,4),(3,4,0),(65,129,67)])
@pytest.mark.parametrize('backend',['auto','custom'])
def test_matmul(m,k,n,backend):
    rng=np.random.default_rng(7)
    a=rng.normal(size=(m,k)).astype(np.float32)
    b=rng.normal(size=(k,n)).astype(np.float32)
    ga,gb=cx.tensor(a,device='cuda'),cx.tensor(b,device='cuda')
    actual=cx.matmul(ga,gb,backend=backend)
    cx.testing.assert_allclose(actual,cx.matmul(cx.tensor(a),cx.tensor(b)),kind='matmul')
    np.testing.assert_allclose(actual.numpy(),a@b,rtol=1e-4,atol=1e-4)
    np.testing.assert_array_equal(ga.numpy(),a)
    np.testing.assert_array_equal(gb.numpy(),b)
    assert actual.shape==(m,n) and actual.device=='cuda'

AXIS_OPS=[cx.sum,cx.max,cx.mean,cx.softmax,cx.rmsnorm,cx.layernorm]
@pytest.mark.parametrize('op',AXIS_OPS)
@pytest.mark.parametrize('shape',[(),(1,),(7,),(2,3,5),(2,0,3),(0,2,3)])
def test_axes_and_empty(op,shape):
    rng=np.random.default_rng(11)
    a=rng.normal(size=shape).astype(np.float32)
    cpu,gpu=cx.tensor(a),cx.tensor(a,device='cuda')
    for axis in range(-max(1,len(shape)),max(1,len(shape))):
        try:expected=op(cpu,axis=axis)
        except ValueError:
            with pytest.raises(ValueError):op(gpu,axis=axis)
        else:
            actual=op(gpu,axis=axis)
            cx.testing.assert_allclose(actual,expected,kind='reduction')
    np.testing.assert_array_equal(gpu.numpy(),a)

@pytest.mark.parametrize('op',AXIS_OPS)
@pytest.mark.parametrize('row',[[0.,-0.,1e-38,-1e-38],[1.,np.nan,2.,3.],[np.inf,1.,2.,-np.inf],[1e20]*4])
def test_axis_special_values(op,row):
    a=np.array([row,row],dtype=np.float32).T.copy()
    actual=op(cx.tensor(a,device='cuda'),axis=0).numpy()
    expected=op(cx.tensor(a),axis=0).numpy()
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-5,equal_nan=True)

@pytest.mark.parametrize('op',[cx.exp,cx.gelu,cx.silu])
@pytest.mark.parametrize('shape',[(),(0,),(257,),(2,3)])
def test_unary(op,shape):
    a=np.linspace(-8,8,num=int(np.prod(shape)),dtype=np.float32).reshape(shape)
    actual=op(cx.tensor(a,device='cuda'))
    cx.testing.assert_allclose(actual,op(cx.tensor(a)))

@pytest.mark.parametrize('op',[cx.exp,cx.gelu,cx.silu])
def test_unary_special_values(op):
    a=np.array([0.,-0.,np.inf,-np.inf,np.nan,100,-100,1e-38],dtype=np.float32)
    np.testing.assert_allclose(op(cx.tensor(a,device='cuda')).numpy(),op(cx.tensor(a)).numpy(),rtol=1e-6,atol=1e-6,equal_nan=True)

@pytest.mark.parametrize('op',[cx.rmsnorm,cx.layernorm])
@pytest.mark.parametrize('eps',[-1,np.nan,np.inf,1e100,1e-100])
def test_bad_epsilon(op,eps):
    for shape in [(2,3),(0,3),(2,0)]:
        with pytest.raises(ValueError,match='epsilon'):
            op(cx.ones(shape,device='cuda'),axis=-1,eps=eps)

@pytest.mark.parametrize('op',AXIS_OPS)
def test_bad_axis_dtype(op):
    a=cx.ones((2,3),device='cuda')
    for axis in [-3,2]:
        with pytest.raises(ValueError,match='axis'):op(a,axis=axis)
    with pytest.raises(ValueError,match='float32'):
        op(cx.tensor(np.ones((2,3),dtype=np.int32),device='cuda'),axis=0)


def test_matmul_errors():
    a=cx.ones((2,3),device='cuda')
    for other in [cx.ones((2,3),device='cuda'),cx.ones((3,),device='cuda')]:
        with pytest.raises(ValueError):cx.matmul(a,other)
    with pytest.raises(ValueError):cx.matmul(a,cx.ones((3,2),device='cpu'))
    for name in ['optimized','unknown']:
        with pytest.raises(ValueError):cx.matmul(a,cx.ones((3,2),device='cuda'),backend=name)
    with pytest.raises(ValueError):cx.matmul(cx.ones((2**62,0),device='cuda'),cx.ones((0,4),device='cuda'))
    assert cx.matmul_backends('cuda')==['auto','custom']


def test_concurrent_and_inference_chain():
    rng=np.random.default_rng(19)
    x=cx.tensor(rng.normal(size=(7,17)).astype(np.float32))
    w1=cx.tensor(rng.normal(size=(17,31)).astype(np.float32))
    w2=cx.tensor(rng.normal(size=(31,5)).astype(np.float32))
    def forward(a,b,c):
        return cx.softmax(cx.matmul(cx.gelu(cx.layernorm(cx.matmul(a,b),axis=-1)),c),axis=-1)
    expected=forward(x,w1,w2)
    gx,g1,g2=[t.to('cuda') for t in (x,w1,w2)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(forward,gx,g1,g2) for _ in range(12)]
        for f in futures:cx.testing.assert_allclose(f.result(),expected,kind='matmul')


@pytest.mark.parametrize('op',[cx.rmsnorm,cx.layernorm])
@pytest.mark.parametrize('eps',[0.0,1e-45,1e-5,1e30])
@pytest.mark.parametrize('value',[0.0,0.3,1e38])
def test_normalization_constant_rows(op,eps,value):
    a=np.full((2,33),value,dtype=np.float32)
    expected=op(cx.tensor(a),axis=-1,eps=eps).numpy()
    actual=op(cx.tensor(a,device='cuda'),axis=-1,eps=eps).numpy()
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-5,equal_nan=True)
