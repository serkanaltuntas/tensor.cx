"""Optional real PyTorch integration checks; core tests do not require PyTorch."""
import gc
import numpy as np
import pytest

# CI installs the optional bridge in a dedicated job; ordinary core-only builds
# keep their dependency boundary and skip this file at collection.
torch = pytest.importorskip('torch')
bridge = pytest.importorskip('torch_tensorcx')
import tensorcx as cx


@pytest.fixture(params=['cpu'] + (['cuda'] if torch.cuda.is_available() and cx.is_available('cuda') else []))
def device(request): return request.param


@pytest.mark.parametrize('dtype', [torch.float32, torch.int32, torch.bool])
@pytest.mark.parametrize('shape', [(), (2, 3), (2, 0, 3), (1, 2, 1)])
def test_torch_zero_copy_lifetime_and_offset(device, dtype, shape):
    count = int(np.prod(shape))
    original = torch.arange(count + 3, device=device).to(dtype)[3:].reshape(shape)
    imported = bridge.from_torch(original)
    exported = bridge.to_torch(imported)
    if count:
        assert original.data_ptr() == exported.data_ptr()
        original.fill_(0)
        if device == 'cuda': torch.cuda.synchronize()
        np.testing.assert_array_equal(imported.numpy(), np.zeros(shape, dtype=imported.dtype))
        exported.fill_(1)
        if device == 'cuda': torch.cuda.synchronize()
        np.testing.assert_array_equal(imported.numpy(), np.ones(shape, dtype=imported.dtype))
    view = imported.reshape((-1,))
    del original, imported, exported
    gc.collect()
    final = bridge.to_torch(view)
    del view
    gc.collect()
    assert final.shape == (count,)
    np.testing.assert_array_equal(final.cpu().numpy(), np.ones(count, dtype=str(dtype).split('.')[-1]))


@pytest.mark.parametrize('copy', [False, True])
def test_torch_explicit_copy(device, copy):
    source = torch.arange(6, dtype=torch.float32, device=device)
    imported = bridge.from_torch(source, copy=copy)
    exported = bridge.to_torch(imported, copy=copy)
    assert (exported.data_ptr() == source.data_ptr()) is (not copy)
    source.fill_(-1)
    if device == 'cuda': torch.cuda.synchronize()
    expected = np.arange(6, dtype=np.float32) if copy else -np.ones(6, dtype=np.float32)
    np.testing.assert_array_equal(imported.numpy(), expected)
    np.testing.assert_array_equal(exported.cpu().numpy(), expected)


@pytest.mark.parametrize('shape,out', [((3,), 4), ((2, 3), 5), ((2, 1, 4, 3), 2),
    ((0, 3), 2), ((2, 0), 3), ((2, 3), 0)])
@pytest.mark.parametrize('with_bias', [False, True])
def test_torch_linear_reference(device, shape, out, with_bias):
    torch.manual_seed(31)
    x = torch.randn(shape, device=device)
    w = torch.randn((out, shape[-1]), device=device)
    b = torch.randn((out,), device=device) if with_bias else None
    inputs = [t.clone() for t in (x, w) + (() if b is None else (b,))]
    actual = bridge.linear(x, w, b)
    reference = torch.nn.functional.linear(x, w, b)
    torch.testing.assert_close(actual, reference, rtol=2e-4, atol=2e-5)
    if actual.numel():
        assert actual.data_ptr() != x.data_ptr()
        assert actual.data_ptr() != w.data_ptr()
    for before, after in zip(inputs, (x, w) + (() if b is None else (b,))):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert not actual.requires_grad


def test_torch_custom_op_registration_and_compile(device):
    x = torch.randn((2, 3), device=device)
    w = torch.randn((4, 3), device=device)
    b = torch.randn((4,), device=device)
    result = torch.library.opcheck(torch.ops.tensorcx.linear.default, (x, w, b))
    assert all(status == 'SUCCESS' for status in result.values())
    compiled = torch.compile(bridge.linear, backend='eager', fullgraph=True)
    torch.testing.assert_close(compiled(x, w, b), torch.nn.functional.linear(x, w, b), rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize('case', ['grad', 'dtype', 'layout', 'sparse', 'rank', 'inner', 'bias'])
def test_torch_unsupported_inputs(device, case):
    x = torch.ones((2, 3), device=device)
    w = torch.ones((4, 3), device=device)
    b = None
    if case == 'grad': x.requires_grad_(True)
    if case == 'dtype': x = x.double()
    if case == 'layout': x = torch.ones((3, 2), device=device).T
    if case == 'sparse': x = x.to_sparse()
    if case == 'rank': x = x[0, 0]
    if case == 'inner': w = torch.ones((4, 2), device=device)
    if case == 'bias': b = torch.ones((1, 4), device=device)
    with pytest.raises(ValueError): bridge.linear(x, w, b)


def test_torch_nondefault_cuda_stream(device):
    if device != 'cuda': return
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.arange(6000, dtype=torch.float32, device='cuda').reshape(2000, 3)
        x.mul_(2)
        w = torch.ones((4, 3), device='cuda')
        shared = bridge.from_torch(x)
        # Import must observe queued producer writes without caller synchronization.
        np.testing.assert_array_equal(shared.numpy(), np.arange(6000,dtype=np.float32).reshape(2000,3)*2)
        result = bridge.linear(x, w)
        after = result + 1
    torch.cuda.synchronize()
    expected = (np.arange(6000,dtype=np.float32).reshape(2000,3)*2).sum(axis=-1,keepdims=True)
    np.testing.assert_array_equal(after.cpu().numpy(), np.broadcast_to(expected+1,(2000,4)))


def test_torch_from_dlpack_direct(device):
    source = cx.tensor([1., 2., 3.], device=device)
    exported = torch.from_dlpack(source)
    exported.add_(4)
    if device == 'cuda': torch.cuda.synchronize()
    np.testing.assert_array_equal(source.numpy(), [5., 6., 7.])
    imported = cx.from_dlpack(exported)
    torch.testing.assert_close(bridge.to_torch(imported), exported)
