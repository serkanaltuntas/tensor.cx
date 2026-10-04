"""Deterministic inference mechanics with CPU parity, without model downloads."""
import argparse
import json

import numpy as np
import tensorcx as cx


def run(device):
    rng = np.random.default_rng(42)
    indices = cx.tensor([[0, 2, 1], [1, 3, 0]], dtype=cx.int32, device=device)
    table = cx.tensor(rng.normal(size=(4, 8)).astype(np.float32), device=device)
    weight = cx.tensor(rng.normal(size=(8, 8)).astype(np.float32), device=device)
    bias = cx.zeros((8,), device=device)
    gain = cx.ones((8,), device=device)
    x = cx.embedding(indices, table)
    x = cx.linear(x, weight, bias)
    x = cx.attention(x, x, x, is_causal=True)
    x = cx.layernorm(x, axis=-1, weight=gain, bias=bias)
    return cx.linear(x, weight)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'metal', 'cuda'])
    args = parser.parse_args()
    actual, reference = run(args.device), run('cpu')
    np.testing.assert_allclose(actual.numpy(), reference.numpy(), rtol=2e-4, atol=2e-5)
    print(json.dumps({'device': args.device, 'shape': actual.shape, 'cpu_parity': True}))
