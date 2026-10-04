"""Boolean tensors, comparisons, conditional selection and mask indexing."""
import operator

import numpy as np
import pytest

import tensorcx as cx
from tensorcx import _core


@pytest.fixture(params=cx.devices())
def device_name(request):
    return request.param


COMPARE = [("equal", operator.eq), ("not_equal", operator.ne), ("less", operator.lt),
           ("less_equal", operator.le), ("greater", operator.gt), ("greater_equal", operator.ge)]


@pytest.mark.parametrize("dtype", [cx.float32, cx.int32, cx.bool])
@pytest.mark.parametrize("shapes", [((2, 3), (3,)), ((2, 1, 3), (4, 1)), ((), (3,)),
                                    ((0, 3), (1, 3)), ((257,), ()), ((), ())])
@pytest.mark.parametrize("name,op", COMPARE)
def test_comparison_numpy_cpu_parity(device_name, dtype, shapes, name, op):
    def values(shape):
        return ((np.arange(np.prod(shape, dtype=int)) % 5) - 2).reshape(shape).astype(dtype)
    a, b = (values(shape) for shape in shapes)
    x, y = (cx.tensor(v, device=device_name) for v in (a, b))
    for result in (op(x, y), getattr(cx, name)(x, y)):
        assert result.dtype == cx.bool and result.device == device_name
        assert result.nbytes == result.numpy().size
        np.testing.assert_array_equal(result.numpy(), op(a, b))
        np.testing.assert_array_equal(result.numpy(), op(cx.tensor(a), cx.tensor(b)).numpy())


@pytest.mark.parametrize("name,op", COMPARE)
def test_comparison_ieee_and_integer_precision(device_name, name, op):
    a = np.array([np.nan, np.inf, -np.inf, 0., -0., 1e-40, -1e-40], dtype=np.float32)
    b = np.array([np.nan, np.inf, np.inf, -0., 0., 0., 0.], dtype=np.float32)
    np.testing.assert_array_equal(op(cx.tensor(a, device=device_name), cx.tensor(b, device=device_name)).numpy(), op(a,b))
    a = np.array([-(2**31), 2**24, 2**24+1, 2**31-1], dtype=np.int32)
    b = np.array([-(2**31)+1, 2**24+1, 2**24, 2**31-2], dtype=np.int32)
    np.testing.assert_array_equal(op(cx.tensor(a, device=device_name), cx.tensor(b, device=device_name)).numpy(), op(a,b))


@pytest.mark.parametrize("name,op", COMPARE)
def test_comparison_float_bit_patterns(device_name, name, op):
    # Include both signs, all exponent boundaries, NaN payloads and subnormals.
    magnitudes = np.array([0, 1, 2, 0x007fffff, 0x00800000, 0x00800001,
                           0x3f000000, 0x3f800000, 0x7f7fffff, 0x7f800000,
                           0x7f800001, 0x7fc01234, 0x7fffffff], dtype=np.uint32)
    bits = np.concatenate([magnitudes, magnitudes | np.uint32(0x80000000)])
    values = bits.view(np.float32)
    x = cx.tensor(values[:, None], device=device_name)
    y = cx.tensor(values[None, :], device=device_name)
    with np.errstate(invalid="ignore"):
        expected = op(values[:, None], values[None, :])
    np.testing.assert_array_equal(getattr(cx, name)(x, y).numpy(), expected)
    np.testing.assert_array_equal(x.astype(cx.bool).numpy(), (bits & 0x7fffffff != 0)[:, None])


@pytest.mark.parametrize("dtype,scalar", [(cx.float32, 2), (cx.float32, np.float32(1.5)),
                                         (cx.int32, np.int32(2)), (cx.bool, True)])
@pytest.mark.parametrize("name,op", COMPARE)
def test_comparison_scalars(device_name, dtype, scalar, name, op):
    values = np.array([0, 1, 2, 3], dtype=dtype)
    x = cx.tensor(values, device=device_name)
    np.testing.assert_array_equal(op(x, scalar).numpy(), op(values, np.asarray(scalar, dtype=dtype)))
    np.testing.assert_array_equal(getattr(cx, name)(scalar, x).numpy(), op(np.asarray(scalar, dtype=dtype), values))


@pytest.mark.parametrize("shape", [(), (0,), (2, 3), (2, 0, 3)])
def test_bool_construction_copy_shape_ops(device_name, shape):
    values = (np.arange(np.prod(shape, dtype=int)) % 2 == 0).reshape(shape)
    x = cx.tensor(values, device=device_name)
    assert x.dtype == cx.bool and x.nbytes == values.size and x.numpy().dtype == np.bool_
    np.testing.assert_array_equal(x.numpy(), values)
    for fill in (cx.zeros, cx.ones):
        y = fill(shape, dtype=cx.bool, device=device_name)
        np.testing.assert_array_equal(y.numpy(), np.zeros(shape,dtype=bool) if fill is cx.zeros else np.ones(shape,dtype=bool))
    assert cx.empty(shape, dtype=cx.bool, device=device_name).nbytes == values.size
    np.testing.assert_array_equal(x.T.numpy(), values.T)
    np.testing.assert_array_equal(x[...].numpy(), values)
    np.testing.assert_array_equal(cx.stack([x,x]).numpy(), np.stack([values,values]))
    np.testing.assert_array_equal(x.reshape((-1,)).numpy(), values.reshape(-1))
    np.testing.assert_array_equal(x.expand_dims(0).squeeze(0).numpy(), values)
    if shape:
        np.testing.assert_array_equal(x[::-1].numpy(), values[::-1])
        np.testing.assert_array_equal(cx.concat([x,x]).numpy(), np.concatenate([values,values]))
        for actual, expected in zip(x.split([0]),np.split(values,[0])):
            np.testing.assert_array_equal(actual.numpy(),expected)
    exported=x.numpy()
    exported[...] = False
    np.testing.assert_array_equal(x.numpy(),values)


@pytest.mark.parametrize("dtype", [cx.float32, cx.int32, cx.bool])
def test_boolean_cast_contract(device_name, dtype):
    data = np.array([0, -1, 1, 2], dtype=dtype)
    if dtype == cx.float32:
        data=np.array([0,-0.,1e-40,-1e-40,np.nan,np.inf,-np.inf],dtype=dtype)
    x=cx.tensor(data,device=device_name)
    mask=x.astype(cx.bool)
    np.testing.assert_array_equal(mask.numpy(),data.astype(bool))
    for target in (cx.bool,cx.float32,cx.int32):
        actual=mask.astype(target)
        np.testing.assert_array_equal(actual.numpy(),data.astype(bool).astype(target))
        assert actual.dtype==target and not _core._shares_storage(actual._impl,mask._impl)
    assert mask.astype(cx.bool,copy=False) is mask


@pytest.mark.parametrize("name,op,numpy_op", [("logical_and",operator.and_,np.logical_and),
                                             ("logical_or",operator.or_,np.logical_or),
                                             ("logical_xor",operator.xor,np.logical_xor)])
def test_logical_masks(device_name,name,op,numpy_op):
    a=np.array([[True],[False]])
    b=np.array([True,False,True])
    x,y=cx.tensor(a,device=device_name),cx.tensor(b,device=device_name)
    for result in (op(x,y),getattr(cx,name)(x,y)):
        np.testing.assert_array_equal(result.numpy(),numpy_op(a,b))
    np.testing.assert_array_equal(op(True,x).numpy(),numpy_op(True,a))
    np.testing.assert_array_equal((~x).numpy(),np.logical_not(a))
    np.testing.assert_array_equal(cx.logical_not(x).numpy(),np.logical_not(a))


@pytest.mark.parametrize("dtype", [cx.float32,cx.int32,cx.bool])
@pytest.mark.parametrize("shapes", [((2,1),(3,),(2,3)), ((),(),()), ((0,3),(3,),()),
                                    ((2,1,3),(4,1),(2,4,3))])
def test_where_numpy_parity(device_name,dtype,shapes):
    shape_m,shape_x,shape_y=shapes
    m=(np.arange(np.prod(shape_m,dtype=int))%2==0).reshape(shape_m)
    x=np.ones(shape_x,dtype=dtype)
    y=np.zeros(shape_y,dtype=dtype)
    mask,a,b=[cx.tensor(v,device=device_name) for v in (m,x,y)]
    result=cx.where(mask,a,b)
    assert result.dtype==dtype and result.device==device_name
    np.testing.assert_array_equal(result.numpy(),np.where(m,x,y))
    np.testing.assert_array_equal(a.where(mask,b).numpy(),result.numpy())
    assert not _core._shares_storage(a._impl,result._impl)
    np.testing.assert_array_equal(cx.where(mask,a,False if dtype==cx.bool else 0).numpy(),np.where(m,x,0))


@pytest.mark.parametrize("shape,axis", [((2,3,4),None),((2,3,4),(0,-1)),((2,3,4),1),
                                      ((2,3),()),((2,0,3),1),((0,3),0),((0,3),1),
                                      ((),None),((),0),((),-1),((),())])
@pytest.mark.parametrize("name", ["any","all"])
@pytest.mark.parametrize("keepdims", [False,True])
def test_boolean_reductions(device_name,shape,axis,name,keepdims):
    data=(np.arange(np.prod(shape,dtype=int))%3==0).reshape(shape)
    x=cx.tensor(data,device=device_name)
    expected=getattr(np,name)(data,axis=axis,keepdims=keepdims)
    for result in (getattr(cx,name)(x,axis,keepdims),getattr(x,name)(axis,keepdims)):
        assert result.dtype==cx.bool and result.shape==expected.shape
        np.testing.assert_array_equal(result.numpy(),expected)
        assert not _core._shares_storage(x._impl,result._impl)


@pytest.mark.parametrize("dtype", [cx.float32,cx.int32,cx.bool])
@pytest.mark.parametrize("shape,mask_shape", [((3,257),(3,257)),((3,257),(3,)),((3,257),()),
                                            ((2,0,3),(2,)),((2,0,3),(2,0)),((0,3),(0,)),
                                            ((0,3),(0,3)),((),())])
@pytest.mark.parametrize("mode", ["mixed","true","false"])
def test_masked_selection_numpy_parity(device_name,dtype,shape,mask_shape,mode):
    data=np.arange(np.prod(shape,dtype=int)).reshape(shape).astype(dtype)
    mask=(np.arange(np.prod(mask_shape,dtype=int))%3==0).reshape(mask_shape)
    if mode!="mixed": mask[...]=mode=="true"
    x,m=cx.tensor(data,device=device_name),cx.tensor(mask,device=device_name)
    for result in (x[m],x[(m,)],cx.masked_select(x,m)):
        assert result.dtype==dtype and result.device==device_name
        np.testing.assert_array_equal(result.numpy(),data[mask])
        assert not _core._shares_storage(x._impl,result._impl)


def test_where_mask_preserve_bits_no_host_export(device_name,monkeypatch):
    bits=np.array([0,0x80000000,0x7fc12345,0xffc54321,1,0x7f800000],dtype=np.uint32)
    x=cx.tensor(bits.view(np.float32),device=device_name)
    y=cx.tensor(bits[::-1].copy().view(np.float32),device=device_name)
    mask=np.array([True,False,True,False,True,False])
    m=cx.tensor(mask,device=device_name)
    xb,yb=x.numpy().view(np.uint32),y.numpy().view(np.uint32)
    def forbidden(*args,**kwargs): raise AssertionError("host export during native operation")
    with monkeypatch.context() as patch:
        patch.setattr(cx.Tensor,"numpy",forbidden);patch.setattr(cx.Tensor,"cpu",forbidden)
        result=cx.where(m,x,y);selected=x[m];cmp=x>0;logical=m & ~m;reduced=m.any()
    np.testing.assert_array_equal(result.numpy().view(np.uint32),np.where(mask,xb,yb))
    np.testing.assert_array_equal(selected.numpy().view(np.uint32),xb[mask])
    assert not logical.numpy().any() and bool(reduced)
    np.testing.assert_array_equal(cmp.numpy(),xb.view(np.float32)>0)


def test_bool_errors_and_truth(device_name):
    x=cx.tensor([1,2],device=device_name)
    m=cx.tensor([True,False],device=device_name)
    for call in (lambda:m+m,lambda:m*2,lambda:-m,lambda:m.sum(),lambda:m.max(),lambda:m.mean(),
                 lambda:cx.logical_and(x,x),lambda:cx.logical_not(x),lambda:x.any(),lambda:x.all(),
                 lambda:cx.where(m,x,x.astype(cx.float32)),lambda:cx.where(m,cx.ones((3,),device=device_name),x),
                 lambda:cx.masked_select(x,x),lambda:cx.masked_select(x,cx.ones((3,),dtype=cx.bool,device=device_name)),
                 lambda:_core.where(m._impl,x._impl,m._impl),lambda:_core.logical_and(x._impl,x._impl)):
        with pytest.raises(ValueError): call()
    with pytest.raises(TypeError): cx.where(x,x,x)
    with pytest.raises(ValueError): x>1.5
    with pytest.raises(TypeError): m & 1
    with pytest.raises(ValueError): bool(m)
    with pytest.raises(ValueError): bool(cx.tensor([],dtype=cx.bool,device=device_name))
    assert bool(cx.tensor(True,device=device_name)) and not bool(cx.tensor(False,device=device_name))
    for axis in (True,"0",[0,0],2):
        with pytest.raises(ValueError): m.any(axis)
    if device_name!="cpu":
        for call in (lambda:x>x.cpu(),lambda:cx.where(m,x,x.cpu()),lambda:x[m.cpu()]):
            with pytest.raises(ValueError):call()


def test_high_rank_boolean_operations(device_name):
    shape=(1,)*300
    x=cx.ones(shape,dtype=cx.bool,device=device_name)
    assert (x==True).shape==shape
    assert cx.where(x,x,False).shape==shape
    assert bool(x)
    assert not bool(~x)
    assert bool(x.all())
    assert x[x].shape==(1,)


def test_empty_mask_unused_block_overflow(device_name):
    shape = (0, 1, 2**62, 4, 0)
    x = cx.empty(shape, dtype=cx.bool, device=device_name)
    mask = cx.empty((0,), dtype=cx.bool, device=device_name)
    assert x[mask].shape == shape
