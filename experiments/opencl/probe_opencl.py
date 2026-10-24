"""Validate an explicitly selected NVIDIA GPU or PoCL CPU using OpenCL 1.2."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import tensorcx as cx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
try:
    from acceptance_support import run
    from publication_report import write_public_report
finally:
    sys.path.pop(0)

SIZES = (0, 1, 63, 64, 65, 255, 256, 257, 1025)
OPERATIONS = ('add', 'multiply')
CYCLES = 2
BLOCK = 64


@cx.experimental.kernel
def guarded_add(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = a[i] + b[i]


@cx.experimental.kernel
def guarded_multiply(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = a[i] * b[i]


def expected_cases():
    return {(cycle, size, active, operation, alias)
            for cycle in range(CYCLES) for size in SIZES for active in {0, size // 2, size}
            for operation in OPERATIONS for alias in (False, True)}


def initial_values(size, active):
    groups = max(1, (active + BLOCK - 1) // BLOCK)
    count = max(size, groups * BLOCK) + 8
    a, b, out = (np.full(count, -1234, np.float32) for _ in range(3))
    a[:size] = ((np.arange(size) % 37) - 18).astype(np.float32) * np.float32(0.25)
    b[:size] = ((np.arange(size) % 7) - 3).astype(np.float32) * np.float32(0.5)
    out[:size] = -11
    return groups, a, b, out


def storage_bits(value, count):
    bits = np.asarray(value)
    if (bits.shape != (count,) or bits.dtype.kind not in 'iu'
            or np.any(bits < 0) or np.any(bits > 0xffffffff)):
        raise ValueError('expected full uint32 storage-bit readback')
    return bits.astype(np.uint32)


def verify_case(case):
    size, active, operation, alias = (case[k] for k in ('size', 'active', 'operation', 'alias'))
    groups, a, b, out = initial_values(size, active)
    if type(case['groups']) is not int or case['groups'] != groups:
        raise ValueError('wrong dispatch group count')
    ta, tb = (cx.tensor(value, device='cpu') for value in (a, b))
    target = ta if alias else cx.tensor(out, device='cpu')
    calculated = (ta + tb if operation == 'add' else ta * tb).numpy()
    numpy_result = a + b if operation == 'add' else a * b
    np.testing.assert_array_equal(calculated.view(np.uint32), numpy_result.view(np.uint32))
    expected = a.copy() if alias else out.copy()
    expected[:active] = calculated[:active]
    kernel = guarded_add if operation == 'add' else guarded_multiply
    reference = kernel.reference(ta, tb, target, active, thread_count=active, block_size=BLOCK)
    np.testing.assert_array_equal(reference.numpy().view(np.uint32), expected.view(np.uint32))
    for key, value in (('result', expected), ('a_after', expected if alias else a), ('b_after', b)):
        np.testing.assert_array_equal(storage_bits(case[key], len(a)), value.view(np.uint32))


def verify_report(report, target, device_name):
    if target not in ('nvidia-gpu', 'pocl-cpu'):
        raise ValueError('unknown execution target')
    nvidia = target == 'nvidia-gpu'
    if (report.get('status') != 'passed' or report.get('target') != target
            or report.get('device_name') != device_name or not device_name
            or type(report.get('device_type')) is not int
            or report['device_type'] != (4 if nvidia else 2)
            or report.get('compiler_available') is not True
            or report.get('device_available') is not True
            or report.get('build_options') != '-cl-std=CL1.2'
            or type(report.get('cycles')) is not int or report['cycles'] != CYCLES
            or type(report.get('release_errors')) is not int or report['release_errors'] != 0):
        raise ValueError('OpenCL identity, compiler or completion contract failed')
    if (nvidia and (report.get('platform_vendor') != 'NVIDIA Corporation'
                    or report.get('vendor_id') != 0x10de)
            or not nvidia and report.get('platform_name') != 'Portable Computing Language'):
        raise ValueError('wrong OpenCL platform/vendor')
    for field in ('platform_version', 'device_version', 'driver_version', 'opencl_c_version'):
        if not isinstance(report.get(field), str) or not report[field]:
            raise ValueError('missing OpenCL version evidence')
    if type(report.get('max_work_group_size')) is not int or report['max_work_group_size'] < BLOCK:
        raise ValueError('insufficient work-group capacity')
    # 2 contexts, queues and programs; 4 kernels; 480 separately owned buffers.
    if any(type(report.get(k)) is not int or report[k] != 490
           for k in ('owned_handles_created', 'owned_handles_released')):
        raise ValueError('owned OpenCL handle cleanup was not established')
    required, seen = expected_cases(), set()
    for case in report['cases']:
        key = tuple(case[k] for k in ('cycle', 'size', 'active', 'operation', 'alias'))
        if (any(type(case[k]) is not int for k in ('cycle', 'size', 'active'))
                or type(case['alias']) is not bool or key not in required or key in seen):
            raise ValueError('unexpected or duplicate OpenCL execution case')
        seen.add(key)
        verify_case(case)
    if seen != required:
        raise ValueError('missing required OpenCL execution cases')
    return {**{k: v for k, v in report.items() if k != 'cases'}, 'cases': len(seen),
            'sizes': list(SIZES), 'operations': list(OPERATIONS),
            'cpu_reference': 'tensorcx CPU primitives, guarded Kernel.reference and NumPy'}


def clean_environment():
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('OCL_', 'OPENCL_', 'POCL_', 'TENSORCX_', 'SKBUILD_', 'CIBW_'))}
    for key in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'PYTHONHOME', 'CFLAGS', 'CXXFLAGS',
                'LDFLAGS', 'CPATH', 'CPLUS_INCLUDE_PATH', 'LIBRARY_PATH'):
        env.pop(key, None)
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=('nvidia-gpu', 'pocl-cpu'), required=True)
    parser.add_argument('--device-name', required=True, help='exact enumerated device name')
    parser.add_argument('--icd', required=True, type=Path, help='explicit ocl-icd vendor file')
    parser.add_argument('--library-dir', action='append', type=Path, default=[])
    parser.add_argument('--cxx', default='g++-13')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = (args.output or ROOT / 'build/opencl-probe' / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'status': 'running', 'target': args.target, 'started_utc': stamp}
    try:
        if not args.icd.is_file() or args.icd.suffix != '.icd':
            raise RuntimeError('explicit OpenCL vendor .icd file is required')
        libraries = [p.resolve() for p in args.library_dir]
        if any(not p.is_dir() for p in libraries):
            raise RuntimeError('explicit OpenCL library directory is missing')
        env = clean_environment()
        env['OCL_ICD_VENDORS'] = str(args.icd.resolve())
        if libraries:
            env['LD_LIBRARY_PATH'] = os.pathsep.join(str(p) for p in libraries)
        if args.target == 'pocl-cpu':
            env.update(POCL_DEVICES='pthread', POCL_MAX_PTHREAD_COUNT='4',
                       POCL_CACHE_DIR=str(output / 'pocl-cache'))
        report['cxx_version'] = run([args.cxx, '--version'], output, env, output / 'cxx-version.log')
        executable = output / 'opencl-probe'
        run([args.cxx, '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror', '-fno-fast-math',
             '-ffp-contract=off', HERE / 'opencl_probe.cpp', '-lOpenCL', '-o', executable],
            output, env, output / 'native-build.log')
        for name, path in (('controller', Path(__file__)), ('native_source', HERE / 'opencl_probe.cpp'),
                           ('kernel_source', HERE / 'guarded.cl'), ('executable', executable)):
            report[name + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        raw = run([executable, args.target, args.device_name, HERE / 'guarded.cl'], output, env,
                  output / 'native.log')
        report.update(verify_report(json.loads(raw), args.target, args.device_name))
        report.update(status='passed', execution='gpu' if args.target == 'nvidia-gpu' else 'software-cpu')
    except BaseException as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_public_report(output / 'result.json', report, root=ROOT)
    print((output / 'result.json').read_text())


if __name__ == '__main__':
    main()
