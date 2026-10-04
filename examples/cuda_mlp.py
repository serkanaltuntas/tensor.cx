"""Deterministic float32 inference example; run with `uv run python examples/cuda_mlp.py`."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import tensorcx as cx


def forward(x, first, second):
    hidden = cx.gelu(cx.layernorm(cx.matmul(x, first), axis=-1))
    return cx.softmax(cx.matmul(hidden, second), axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if not cx.is_available(args.device):
        parser.error(f'required device unavailable: {args.device}')
    rng = np.random.default_rng(17)
    host = [cx.tensor(rng.normal(size=shape).astype(np.float32))
            for shape in [(32, 64), (64, 128), (128, 10)]]
    expected = forward(*host)
    inputs = [t.to(args.device) for t in host]
    # Warm up before synchronous execution timing; transfers and CPU validation
    # are deliberately outside the inference-only measurement.
    actual = forward(*inputs)
    cx.testing.assert_allclose(actual, expected, kind='matmul')
    samples = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        actual = forward(*inputs)
        samples.append((time.perf_counter() - start) * 1000)
    cx.testing.assert_allclose(actual, expected, kind='matmul')
    result = actual.numpy()
    np.testing.assert_allclose(result.sum(axis=-1), 1, rtol=1e-5, atol=1e-5)
    print(json.dumps({'device': args.device, 'device_name': cx.device_name(args.device),
                      'shape': list(result.shape), 'cpu_parity': True,
                      'inference_median_ms': float(np.median(samples)),
                      'repeats': args.repeats}, indent=2))


if __name__ == '__main__':
    main()
