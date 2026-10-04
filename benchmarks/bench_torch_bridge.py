"""Correctness-checked end-to-end linear and DLPack bridge timings."""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch_tensorcx as bridge
import tensorcx as cx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error('repeats must be positive and warmup nonnegative')
    if not cx.is_available(args.device):
        parser.error('requested tensor.cx device is unavailable')
    torch.manual_seed(23)
    torch.set_num_threads(1)

    def sync():
        if args.device == 'cuda':
            torch.cuda.synchronize()

    def measure(fn):
        for _ in range(args.warmup):
            fn()
        sync()
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter_ns()
            result = fn()
            sync()
            samples.append((time.perf_counter_ns() - start) / 1e6)
            del result
        return {'median_ms': statistics.median(samples), 'min_ms': min(samples)}

    rows = []
    for m, k, n in ((1, 32, 16), (32, 64, 32), (128, 256, 128)):
        x = torch.randn((m, k), device=args.device)
        w = torch.randn((n, k), device=args.device)
        b = torch.randn(n, device=args.device)
        torch.testing.assert_close(bridge.linear(x, w, b), torch.nn.functional.linear(x, w, b),
                                   rtol=2e-4, atol=2e-4)
        shared = bridge.from_torch(x)
        assert bridge.to_torch(shared).data_ptr() == x.data_ptr()
        rows.append({'shape_mkn': [m, k, n], 'correct': True,
            'torch_linear': measure(lambda: torch.nn.functional.linear(x, w, b)),
            'tensorcx_linear_bridge': measure(lambda: bridge.linear(x, w, b)),
            'dlpack_roundtrip': measure(lambda: bridge.to_torch(bridge.from_torch(x)))})
    print(json.dumps({'schema': 1, 'device': args.device, 'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda, 'repeats': args.repeats, 'warmup': args.warmup,
        'timing': 'synchronized wall clock; bridge includes exchange and allocation; one CPU thread',
        'results': rows}, indent=2))


if __name__ == '__main__':
    main()
