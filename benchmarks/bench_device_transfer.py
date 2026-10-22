"""Synchronous explicit placement and copy/compute/return wall-clock latency."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np
import tensorcx as cx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIZES = (1024, 16384, 262144, 1048576, 16777216)


def sizes_arg(text):
    try: values = tuple(int(part) for part in text.split(','))
    except ValueError as error: raise argparse.ArgumentTypeError('expected comma-separated element counts') from error
    if not values or min(values) < 0 or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError('sizes must be unique nonnegative element counts')
    return values


def positive(text):
    value = int(text)
    if value < 1: raise argparse.ArgumentTypeError('must be positive')
    return value


def metadata():
    def command(*args):
        return subprocess.run(args, cwd=ROOT, check=True, capture_output=True,
                              text=True, timeout=5).stdout.strip()
    try:
        revision = command('git', 'rev-parse', 'HEAD')
        dirty = bool(command('git', 'status', '--porcelain'))
        diff = hashlib.sha256(command('git', 'diff', 'HEAD').encode()).hexdigest()
    except (OSError, subprocess.SubprocessError):
        revision = dirty = diff = None  # An extracted source archive need not contain Git.
    return {'started_utc': datetime.now(timezone.utc).isoformat(), 'revision': revision,
            'git_dirty': dirty, 'tracked_diff_sha256': diff,
            'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'extension_sha256': hashlib.sha256(Path(cx._core.__file__).read_bytes()).hexdigest(),
            'tensorcx': cx.__version__, 'numpy': np.__version__,
            'python': platform.python_version(), 'system': platform.system(), 'machine': platform.machine()}


def run(*, source, target, sizes=DEFAULT_SIZES, dtype='float32', repeats=5, warmup=2, profile=False):
    source, target = str(cx.device(source)), str(cx.device(target))
    if repeats < 1 or warmup < 0 or dtype not in ('float32', 'int32', 'bool'):
        raise ValueError('invalid benchmark parameters')
    if not sizes or min(sizes) < 0 or len(sizes) != len(set(sizes)):
        raise ValueError('sizes must be unique nonnegative element counts')
    path = ('same_device_clone' if source == target else 'cpu_to_device' if source == 'cpu'
            else 'device_to_cpu' if target == 'cpu' else 'host_staged')
    report = {'schema_version': 1, 'kind': 'device_transfer', 'source': source, 'target': target,
              'transfer_path': path, 'dtype': dtype, 'repeats': repeats, 'warmup': warmup,
              'metadata': metadata(), 'cases': [], 'cpu_parity': True,
              'measurement': {'clock': 'perf_counter_ns_host', 'execution': 'synchronous',
                  'copy': 'public to(target, copy=True), including allocation and staging when required',
                  'pipeline': 'copy to target, add one (logical not for bool), copy back to source',
                  'order': 'alternate copy/pipeline sample order each repeat',
                  'excluded': 'initial input setup, validation and final result destruction',
                  'profile': 'separate captures, not timing samples; device clocks are not host clocks',
                  'limits': 'shared host; no peer, zero-copy, device-only bandwidth or universal speedup claim'}}
    for count in sizes:
        host = (np.arange(count, dtype=np.int32) % 101).astype(dtype)
        resident = cx.tensor(host, device=source)
        expected_pipeline = np.logical_not(host) if dtype == 'bool' else host + np.array(1, dtype=dtype)
        def copy(): return resident.to(target, copy=True)
        def pipeline():
            placed = resident.to(target, copy=True)
            result = ~placed if dtype == 'bool' else placed + 1
            return result.to(source, copy=True)
        operations = {'copy': (copy, host, target), 'pipeline': (pipeline, expected_pipeline, source)}
        samples = {name: [] for name in operations}
        def validate(value, expected, device):
            assert value.device == device and value.dtype == dtype and value.shape == host.shape
            actual = value.numpy()
            np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))
            np.testing.assert_array_equal(resident.numpy().view(np.uint8), host.view(np.uint8))
        for function, expected, device in operations.values():
            for _ in range(warmup):
                value = function(); validate(value, expected, device); del value
        for repeat in range(repeats):
            order = list(operations) if repeat % 2 == 0 else list(reversed(operations))
            for name in order:
                function, expected, device = operations[name]
                started = time.perf_counter_ns(); value = function()
                samples[name].append(time.perf_counter_ns() - started)
                validate(value, expected, device); del value
        row = {'elements': count, 'logical_bytes': host.nbytes, 'cpu_parity': True,
               'timings': {name: {'samples_ns': values, 'median_ns': statistics.median(values),
                                  'min_ns': min(values), 'max_ns': max(values)}
                           for name, values in samples.items()}}
        if profile:
            captures = {}
            for name, (function, expected, device) in operations.items():
                with cx.profile(device_timing=True) as capture: value = function()
                validate(value, expected, device); del value
                captures[name] = {'host_summary': capture.summary(),
                                  'report': capture.report()}
            row['profiles'] = captures
        report['cases'].append(row)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='cuda')
    parser.add_argument('--target', default='vulkan')
    parser.add_argument('--dtype', choices=('float32', 'int32', 'bool'), default='float32')
    parser.add_argument('--sizes', type=sizes_arg, default=DEFAULT_SIZES)
    parser.add_argument('--repeats', type=positive, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists(): parser.error('output already exists')
    if args.warmup < 0: parser.error('warmup must be nonnegative')
    result = run(source=args.source, target=args.target, sizes=args.sizes, dtype=args.dtype,
                 repeats=args.repeats, warmup=args.warmup, profile=args.profile)
    text = json.dumps(result, indent=2) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as output:
            output.write(text)
    print(text, end='')


if __name__ == '__main__': main()
