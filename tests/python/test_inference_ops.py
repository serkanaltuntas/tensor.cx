"""Inference contract, independent references, and native backend parity."""
import numpy as np
import pytest
import tensorcx as cx
from tensorcx import _core


@pytest.fixture(params=cx.devices())
def device(request):
    return request.param


def f32(shape, seed=31):
    return np.random.default_rng(seed).normal(size=shape).astype(np.float32)


def check(result, expected, device):
    expected = np.asarray(expected, dtype=np.float32)
    assert result.device == device
    assert result.dtype == 'float32'
    assert result.shape == expected.shape
    np.testing.assert_allclose(result.numpy(), expected, rtol=2e-4, atol=2e-5, equal_nan=True)


def attention_reference(q, k, v, mask=None, causal=False, scale=None):
    batch = np.broadcast_shapes(q.shape[:-2], k.shape[:-2], v.shape[:-2])
    shape = batch + (q.shape[-2], k.shape[-2])
    scores = np.broadcast_to(q @ np.swapaxes(k, -1, -2), shape).copy()
    scores *= np.float32(1 / np.sqrt(q.shape[-1]) if scale is None else scale)
    allowed = np.ones(shape, dtype=bool)
    if causal:
        allowed &= np.arange(shape[-1]) <= np.arange(shape[-2])[:, None]
    if mask is not None:
        if mask.dtype == np.bool_:
            allowed &= mask
        else:
            allowed &= mask != -np.inf
            scores += mask
    scores = np.where(allowed, scores, -np.inf)
    if shape[-1] == 0:
        return np.zeros(batch + (q.shape[-2], v.shape[-1]), np.float32)
    with np.errstate(all='ignore'):
        maximum = np.max(scores, axis=-1, keepdims=True)
        probabilities = np.exp(scores - maximum)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        probabilities = np.where(maximum == -np.inf, np.float32(0), probabilities)
        return probabilities @ v


@pytest.mark.parametrize('shape,out', [((3,), 4), ((2, 3), 5), ((2, 1, 4, 3), 2),
    ((0, 3), 2), ((2, 0), 3), ((2, 3), 0), ((0,), 2)])
@pytest.mark.parametrize('bias', [False, True])
def test_linear_parity(device, shape, out, bias):
    a, w, b = f32(shape), f32((out, shape[-1]), 4), f32((out,), 9)
    x, weight, offset = [cx.tensor(t, device=device) for t in (a, w, b)]
    expected = a @ w.T + (b if bias else 0)
    y = cx.linear(x, weight, offset if bias else None)
    check(y, expected, device)
    check(x.linear(weight, offset if bias else None), expected, device)
    check(cx.linear(cx.tensor(a), cx.tensor(w), cx.tensor(b) if bias else None), expected, 'cpu')
    assert not _core._shares_storage(y._impl, x._impl)
    np.testing.assert_array_equal(x.numpy(), a)
    np.testing.assert_array_equal(weight.numpy(), w)


@pytest.mark.parametrize('shape,axis', [((2, 3, 4), -1), ((2, 3, 4), 1),
    ((2, 3, 4), 0), ((3,), -1), ((), 0), ((), -1), ((0, 3), 1), ((2, 0, 3), 1)])
@pytest.mark.parametrize('kind,params', [('rmsnorm', 'weight'), ('layernorm', 'weight'), ('layernorm', 'both'), ('layernorm', 'bias')])
def test_affine_norm_parity(device, shape, axis, kind, params):
    a = f32(shape)
    pshape = (shape[axis],) if shape else ()
    w, b = f32(pshape, 5), f32(pshape, 6)
    kwargs = {}
    if params in ('weight', 'both'):
        kwargs['weight'] = cx.tensor(w, device=device)
    if params in ('bias', 'both'):
        kwargs['bias'] = cx.tensor(b, device=device)
    with np.errstate(all='ignore'):
        if a.size or not shape:
            base = a if kind == 'rmsnorm' else a - np.mean(a, axis=axis if shape else None, keepdims=True)
            expected = base / np.sqrt(np.mean(base * base, axis=axis if shape else None, keepdims=True) + np.float32(1e-5))
        else:
            expected = np.empty(shape, np.float32)
        pview = [1] * len(shape)
        if shape:
            pview[axis] = shape[axis]
        if 'weight' in kwargs:
            expected *= w.reshape(pview)
        if 'bias' in kwargs:
            expected += b.reshape(pview)
    x = cx.tensor(a, device=device)
    check(getattr(cx, kind)(x, axis, **kwargs), expected, device)
    check(getattr(x, kind)(axis, **kwargs), expected, device)
    cpu_kwargs = {n: cx.tensor(t.numpy()) for n, t in kwargs.items()}
    check(getattr(cx, kind)(cx.tensor(a), axis, **cpu_kwargs), expected, 'cpu')
    # Explicit None must preserve the old unweighted operation.
    check(getattr(cx, kind)(x, axis, weight=None), getattr(cx, kind)(x, axis).numpy(), device)


@pytest.mark.parametrize('shape', [(), (4,), (2, 3), (2, 0, 3)])
@pytest.mark.parametrize('width', [0, 1, 7])
def test_embedding_parity(device, shape, width):
    indices = np.random.default_rng(7).integers(0, 5, size=shape, dtype=np.int32)
    w = f32((5, width))
    ids, weight = cx.tensor(indices, device=device), cx.tensor(w, device=device)
    y = cx.embedding(ids, weight)
    check(y, w[indices], device)
    check(ids.embedding(weight), w[indices], device)
    check(cx.embedding(cx.tensor(indices), cx.tensor(w)), w[indices], 'cpu')
    assert not _core._shares_storage(y._impl, weight._impl)
    np.testing.assert_array_equal(ids.numpy(), indices)


def test_embedding_preserves_float_bits(device):
    bits = np.array([[0, 0x80000000, 1, 0x7fc01234], [0x7f800000, 0xff800000, 0x80000001, 0x3f800000]], np.uint32)
    ids = np.array([1, 0, 1], np.int32)
    result = cx.embedding(cx.tensor(ids, device=device), cx.tensor(bits.view(np.float32), device=device))
    np.testing.assert_array_equal(result.numpy().view(np.uint32), bits[ids])


@pytest.mark.parametrize('width', [0, 3])
@pytest.mark.parametrize('ids,vocabulary', [([-1], 2), ([2], 2), ([0], 0), ([0, 1, 2147483647], 2)])
def test_embedding_rejects_invalid_index_even_zero_width(device, width, ids, vocabulary):
    with pytest.raises(ValueError):
        cx.embedding(cx.tensor(ids, dtype='int32', device=device), cx.zeros((vocabulary, width), device=device))


ATTENTION_SHAPES = [((3, 4), (5, 4), (5, 2)), ((2, 3, 4), (2, 5, 4), (2, 5, 6)),
    ((2, 1, 3, 4), (1, 3, 5, 4), (2, 3, 5, 2)), ((3, 4), (5, 4), (2, 1, 5, 2)),
    ((0, 4), (5, 4), (5, 2)), ((3, 4), (0, 4), (0, 2)),
    ((0, 3, 4), (1, 5, 4), (1, 5, 2)), ((3, 4), (5, 4), (5, 0))]


@pytest.mark.parametrize('shapes', ATTENTION_SHAPES)
@pytest.mark.parametrize('mask_kind', ['none', 'bool', 'additive', 'scalar_false'])
@pytest.mark.parametrize('causal', [False, True])
def test_attention_parity(device, shapes, mask_kind, causal):
    q, k, v = [f32(shape, i + 10) for i, shape in enumerate(shapes)]
    mshape = (q.shape[-2], k.shape[-2])
    rng = np.random.default_rng(41)
    mask = None
    if mask_kind == 'bool':
        mask = rng.random(mshape) > .4
        if mask.shape[0]: mask[0] = False
    elif mask_kind == 'additive':
        mask = f32(mshape)
        mask[rng.random(mshape) < .3] = -np.inf
        if mask.shape[0]: mask[0] = -np.inf
    elif mask_kind == 'scalar_false':
        mask = np.array(False)
    expected = attention_reference(q, k, v, mask, causal)
    inputs = [cx.tensor(t, device=device) for t in (q, k, v)]
    native_mask = None if mask is None else cx.tensor(mask, device=device)
    check(cx.attention(*inputs, native_mask, is_causal=causal), expected, device)
    check(inputs[0].attention(*inputs[1:], native_mask, is_causal=causal), expected, device)
    check(cx.scaled_dot_product_attention(*[cx.tensor(t) for t in (q, k, v)],
          None if mask is None else cx.tensor(mask), is_causal=causal), expected, 'cpu')


@pytest.mark.parametrize('scale', [0, -0.5, 2, np.float32(.3)])
def test_attention_scale_and_batched_mask(device, scale):
    q, k, v = f32((2, 3, 4)), f32((5, 4), 2), f32((5, 3), 3)
    mask = np.random.default_rng(4).random((2, 1, 5)) > .3
    check(cx.attention(*[cx.tensor(t, device=device) for t in (q, k, v)],
                       cx.tensor(mask, device=device), scale=scale),
          attention_reference(q, k, v, mask, scale=scale), device)


def test_attention_causality(device):
    q, k, v = f32((5, 4)), f32((5, 4), 2), f32((5, 3), 3)
    before = cx.attention(*[cx.tensor(t, device=device) for t in (q, k, v)], is_causal=True)
    k[3:] += 100
    v[3:] -= 500
    after = cx.attention(*[cx.tensor(t, device=device) for t in (q, k, v)], is_causal=True)
    np.testing.assert_array_equal(before.numpy()[:3], after.numpy()[:3])


@pytest.mark.parametrize('exceptional', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('masked', [False, True])
def test_attention_nonfinite_scores(device, exceptional, masked):
    q, k, v = np.ones((1, 2), np.float32), np.ones((2, 2), np.float32), f32((2, 3))
    k[0] = exceptional
    mask = np.array([[not masked, True]])
    with np.errstate(all='ignore'):
        check(cx.attention(*[cx.tensor(t, device=device) for t in (q, k, v)], cx.tensor(mask, device=device)),
              attention_reference(q, k, v, mask), device)


@pytest.mark.parametrize('scale', [np.nan, np.inf, -np.inf, 1e300, 1e-300, 10**400])
def test_attention_invalid_scale_even_empty(device, scale):
    q, k, v = [cx.zeros(s, device=device) for s in ((0, 2), (3, 2), (3, 1))]
    with pytest.raises(ValueError): cx.attention(q, k, v, scale=scale)


@pytest.mark.parametrize('scale', [True, '2', 1j, [1]])
def test_attention_invalid_scale_type(device, scale):
    x = cx.ones((2, 2), device=device)
    with pytest.raises(TypeError): cx.attention(x, x, x, scale=scale)


@pytest.mark.parametrize('kind', ['rmsnorm', 'layernorm'])
@pytest.mark.parametrize('eps', [-1, np.nan, np.inf, 1e300, 1e-300])
def test_affine_norm_invalid_eps_even_empty(device, kind, eps):
    x, w = cx.zeros((0, 3), device=device), cx.ones((3,), device=device)
    with pytest.raises(ValueError): getattr(cx, kind)(x, -1, eps, weight=w)


@pytest.mark.parametrize('case', ['linear_scalar', 'linear_weight_rank', 'linear_inner', 'linear_bias',
    'linear_int', 'embedding_indices', 'embedding_weight', 'embedding_rank',
    'norm_weight_shape', 'norm_weight_int', 'norm_axis', 'attention_rank', 'attention_features',
    'attention_zero_features', 'attention_sequence', 'attention_batch', 'attention_mask_int',
    'attention_mask_shape', 'attention_mask_rank', 'attention_int'])
def test_inference_invalid_contract(device, case):
    def t(shape, dtype='float32'): return cx.zeros(shape, dtype=dtype, device=device)
    actions = {
        'linear_scalar': lambda: cx.linear(t(()), t((2, 1))),
        'linear_weight_rank': lambda: cx.linear(t((2, 3)), t((3,))),
        'linear_inner': lambda: cx.linear(t((0, 3)), t((2, 4))),
        'linear_bias': lambda: cx.linear(t((0, 3)), t((2, 3)), t((1, 2))),
        'linear_int': lambda: cx.linear(t((2, 3), 'int32'), t((2, 3))),
        'embedding_indices': lambda: cx.embedding(t((0,)), t((2, 3))),
        'embedding_weight': lambda: cx.embedding(t((0,), 'int32'), t((2, 3), 'int32')),
        'embedding_rank': lambda: cx.embedding(t((0,), 'int32'), t((3,))),
        'norm_weight_shape': lambda: cx.rmsnorm(t((0, 3)), -1, weight=t((1, 3))),
        'norm_weight_int': lambda: cx.layernorm(t((0, 3)), -1, weight=t((3,), 'int32')),
        'norm_axis': lambda: cx.layernorm(t((0, 3)), 2, bias=t((3,))),
        'attention_rank': lambda: cx.attention(t((3,)), t((2, 3)), t((2, 3))),
        'attention_features': lambda: cx.attention(t((0, 3)), t((2, 4)), t((2, 3))),
        'attention_zero_features': lambda: cx.attention(t((0, 0)), t((2, 0)), t((2, 3))),
        'attention_sequence': lambda: cx.attention(t((0, 3)), t((2, 3)), t((3, 3))),
        'attention_batch': lambda: cx.attention(t((2, 0, 3)), t((3, 2, 3)), t((3, 2, 3))),
        'attention_mask_int': lambda: cx.attention(t((0, 3)), t((2, 3)), t((2, 3)), t((0, 2), 'int32')),
        'attention_mask_shape': lambda: cx.attention(t((0, 3)), t((2, 3)), t((2, 3)), t((0, 3), 'bool')),
        'attention_mask_rank': lambda: cx.attention(t((0, 3)), t((2, 3)), t((2, 3)), t((1, 0, 2), 'bool')),
        'attention_int': lambda: cx.attention(t((0, 3), 'int32'), t((2, 3)), t((2, 3))),
    }
    with pytest.raises(ValueError): actions[case]()


def test_inference_rejects_non_tensors_and_options(device):
    x = cx.ones((2, 2), device=device)
    for call in (lambda: cx.linear(x, [[1, 2]]), lambda: cx.embedding([1], x),
                 lambda: cx.rmsnorm(x, -1, weight=[1, 2]), lambda: cx.attention(x, x, x, [[True]]),
                 lambda: cx.attention(x, x, x, is_causal=1)):
        with pytest.raises(TypeError): call()


@pytest.mark.parametrize('other', cx.devices())
def test_inference_device_mismatch(device, other):
    if device == other:
        return
    x, w = cx.ones((2, 2), device=device), cx.ones((2, 2), device=other)
    for call in (lambda: cx.linear(x, w), lambda: cx.attention(x, x, w),
                 lambda: cx.attention(x, x, x, cx.ones((2, 2), dtype='bool', device=other)),
                 lambda: cx.layernorm(x, -1, weight=cx.ones((2,), device=other)),
                 lambda: cx.embedding(cx.tensor([0], dtype='int32', device=device), w)):
        with pytest.raises(ValueError, match='device'): call()


def test_inference_pipeline_without_python_host_fallback(device, monkeypatch):
    ids_np = np.array([[0, 2, 1], [1, 3, 0]], np.int32)
    table, weight, bias, gain, shift = f32((4, 4)), f32((4, 4), 2), f32((4,), 3), f32((4,), 4), f32((4,), 5)
    ids = cx.tensor(ids_np, device=device)
    ts = [cx.tensor(t, device=device) for t in (table, weight, bias, gain, shift)]
    def fail(*args, **kwargs): raise AssertionError('Python host fallback')
    with monkeypatch.context() as m:
        m.setattr(cx.Tensor, 'numpy', fail)
        m.setattr(cx.Tensor, 'cpu', fail)
        embedded = cx.embedding(ids, ts[0])
        projected = cx.linear(embedded, ts[1], ts[2])
        attended = cx.attention(projected, projected, projected, is_causal=True)
        normalized = cx.layernorm(attended, -1, weight=ts[3], bias=ts[4])
        output = cx.linear(normalized, ts[1])
    projected = table[ids_np] @ weight.T + bias
    attended = attention_reference(projected, projected, projected, causal=True)
    centered = attended - attended.mean(axis=-1, keepdims=True)
    expected = (centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + np.float32(1e-5)) * gain + shift) @ weight.T
    check(output, expected, device)


@pytest.mark.parametrize('scale', [1e-39, -1e-39, 1e-40, -1e-40])
def test_attention_subnormal_scale(device, scale):
    q, k, v = np.array([[1]], np.float32), np.array([[0], [1e38]], np.float32), np.array([[0], [1]], np.float32)
    check(cx.attention(*[cx.tensor(t, device=device) for t in (q, k, v)], scale=scale),
          attention_reference(q, k, v, scale=scale), device)


def test_attention_huge_empty_batch(device):
    q = cx.empty((0, 2**62, 4, 0, 2), device=device)
    v = cx.empty((0, 2**62, 4, 0, 3), device=device)
    result = cx.attention(q, q, v)
    assert result.shape == v.shape and result.device == device


def test_attention_high_rank_metadata(device):
    shape = (1,) * 17
    q, k, v = f32(shape + (3, 4)), f32(shape + (2, 4), 2), f32(shape + (2, 3), 3)
    check(cx.attention(*[cx.tensor(t, device=device) for t in (q, k, v)]),
          attention_reference(q, k, v), device)


@pytest.mark.parametrize('kind', ['rmsnorm', 'layernorm'])
def test_affine_zero_epsilon_and_output_ownership(device, kind):
    x = cx.zeros((2, 3), device=device)
    w = cx.ones((3,), device=device)
    y = getattr(cx, kind)(x, -1, eps=0, weight=w)
    assert np.isnan(y.numpy()).all()
    assert not _core._shares_storage(x._impl, y._impl)
    assert not _core._shares_storage(w._impl, y._impl)
    np.testing.assert_array_equal(x.numpy(), np.zeros((2, 3), np.float32))


def test_embedding_empty_vocabulary(device):
    check(cx.embedding(cx.empty((2, 0), dtype='int32', device=device),
                       cx.empty((0, 4), device=device)), np.empty((2, 0, 4), np.float32), device)


@pytest.mark.parametrize('bias', [np.nan, np.inf, -np.inf])
def test_attention_nonfinite_additive_mask(device, bias):
    q, k, v = f32((2, 3)), f32((4, 3)), f32((4, 2))
    mask = np.array([[bias, 0, -np.inf, 0], [bias, bias, bias, bias]], np.float32)
    with np.errstate(all='ignore'):
        check(cx.attention(*[cx.tensor(t, device=device) for t in (q, k, v)], cx.tensor(mask, device=device)),
              attention_reference(q, k, v, mask), device)


def test_attention_nonfinite_value_under_zero_probability(device):
    q = cx.ones((1, 1), device=device)
    v = cx.tensor([[np.nan]], device=device)
    y = cx.attention(q, q, v, cx.tensor(False, device=device))
    assert np.isnan(y.numpy()).all()


def test_attention_long_sequence_and_ownership(device):
    q, k, v = f32((257, 3)), f32((513, 3), 4), f32((513, 2), 5)
    tensors = [cx.tensor(t, device=device) for t in (q, k, v)]
    result = cx.attention(*tensors, is_causal=True)
    check(result, attention_reference(q, k, v, causal=True), device)
    for source, array in zip(tensors, (q, k, v)):
        assert not _core._shares_storage(source._impl, result._impl)
        np.testing.assert_array_equal(source.numpy(), array)
