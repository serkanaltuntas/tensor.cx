"""Run an explicit Mesa CPU Vulkan experiment, without enabling a runtime GPU."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
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
        raise ValueError('expected a full uint32 storage-bit readback')
    return bits.astype(np.uint32)


def verify_case(case):
    size, active, operation, alias = (case[k] for k in ('size', 'active', 'operation', 'alias'))
    groups, a, b, output = initial_values(size, active)
    if case['groups'] != groups:
        raise ValueError('wrong dispatch group count')
    ta, tb = (cx.tensor(value, device='cpu') for value in (a, b))
    target = ta if alias else cx.tensor(output, device='cpu')
    calculated = (ta + tb if operation == 'add' else ta * tb).numpy()
    expected = a.copy() if alias else output.copy()
    expected[:active] = calculated[:active]
    numpy_result = a + b if operation == 'add' else a * b
    np.testing.assert_array_equal(calculated.view(np.uint32), numpy_result.view(np.uint32))
    kernel = guarded_add if operation == 'add' else guarded_multiply
    reference = kernel.reference(ta, tb, target, active, thread_count=active, block_size=BLOCK)
    np.testing.assert_array_equal(reference.numpy().view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(storage_bits(case['result'], len(a)), expected.view(np.uint32))
    np.testing.assert_array_equal(storage_bits(case['a_after'], len(a)),
                                  (expected if alias else a).view(np.uint32))
    np.testing.assert_array_equal(storage_bits(case['b_after'], len(b)), b.view(np.uint32))


def verify_report(report):
    if (report.get('status') != 'passed' or report.get('device_type') != 'cpu'
            or report.get('driver_id') != 13 or report.get('driver_name') != 'llvmpipe'
            or report.get('api_version', 0) < (1 << 22 | 2 << 12)
            or report.get('cycles') != CYCLES or report.get('validation_errors') != 0):
        raise ValueError('software Vulkan identity, validation or completion contract failed')
    allocated, released = (report.get(k) for k in ('owned_handles_created', 'owned_handles_released'))
    if type(allocated) is not int or allocated <= 0 or type(released) is not int or allocated != released:
        raise ValueError('owned Vulkan handle cleanup was not established')
    required = expected_cases()
    seen = set()
    for case in report['cases']:
        key = tuple(case[k] for k in ('cycle', 'size', 'active', 'operation', 'alias'))
        if (any(type(case[k]) is not int for k in ('cycle', 'size', 'active', 'groups'))
                or type(case['alias']) is not bool or key not in required or key in seen):
            raise ValueError('unexpected or duplicate software execution case')
        seen.add(key)
        verify_case(case)
    if seen != required:
        raise ValueError('missing required software execution cases')
    return {'cases': len(seen), 'cycles': CYCLES, 'sizes': list(SIZES),
            'operations': list(OPERATIONS), 'owned_handles_created': allocated,
            'owned_handles_released': released, 'validation_errors': 0,
            **{k: report[k] for k in ('device_type', 'driver_id', 'driver_name', 'driver_info', 'api_version')}}


def clean_environment():
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('VK_', 'TENSORCX_', 'SKBUILD_', 'CIBW_'))}
    for key in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'PYTHONHOME', 'PYTEST_ADDOPTS',
                'PYTEST_PLUGINS', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS'):
        env.pop(key, None)
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--shader-bin', type=Path)
    parser.add_argument('--validation-layer', type=Path, required=True)
    parser.add_argument('--software-icd', type=Path, required=True)
    parser.add_argument('--library-dir', type=Path)
    parser.add_argument('--cxx', default='g++-13')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = (args.output or ROOT / 'build/lavapipe-probe' / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'status': 'running', 'execution': 'software-cpu', 'started_utc': stamp}
    try:
        sdk, layer, icd = (p.resolve() for p in (args.sdk, args.validation_layer, args.software_icd))
        shaders = (args.shader_bin or sdk / 'bin').resolve()
        if not (sdk / 'include/vulkan/vulkan.h').is_file():
            raise RuntimeError('Vulkan headers are required')
        if not icd.is_file() or not any(layer.glob('*.json')):
            raise RuntimeError('explicit software ICD and validation-layer manifests are required')
        env = clean_environment()
        env.update(VK_DRIVER_FILES=str(icd), VK_ICD_FILENAMES=str(icd), VK_LAYER_PATH=str(layer),
                   TENSORCX_VULKAN_VALIDATION='1')
        if args.library_dir:
            library = args.library_dir.resolve()
            if not library.is_dir():
                raise RuntimeError('explicit library directory is missing')
            env['LD_LIBRARY_PATH'] = str(library)
            linking = ['-L' + str(library), '-l:libvulkan.so.1']
        else:
            linking = ['-l:libvulkan.so.1']
        report['cxx_version'] = run([args.cxx, '--version'], output, env, output / 'cxx-version.log')
        report['shader_compiler'] = run([shaders / 'glslangValidator', '--version'], output, env,
                                        output / 'shader-version.log')
        spirv = output / 'lavapipe.spv'
        run([shaders / 'glslangValidator', '-V', '--target-env', 'vulkan1.2',
             HERE / 'lavapipe.comp', '-o', spirv], output, env, output / 'shader-build.log')
        run([shaders / 'spirv-val', '--target-env', 'vulkan1.2', spirv], output, env,
            output / 'spirv-val.log')
        binary = output / 'lavapipe-probe'
        run([args.cxx, '-std=c++20', '-O2', '-fno-fast-math', '-ffp-contract=off',
             '-I' + str(sdk / 'include'), HERE / 'lavapipe_probe.cpp', *linking, '-o', binary],
            output, env, output / 'native-build.log')
        for name, path in (('controller', Path(__file__)), ('native_source', HERE / 'lavapipe_probe.cpp'),
                           ('shader_source', HERE / 'lavapipe.comp'), ('spirv', spirv), ('executable', binary)):
            report[name + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        raw = run([binary, spirv], output, env, output / 'native.log')
        report.update(verify_report(json.loads(raw)))
        # A healthy software device must remain excluded from the production GPU API.
        run([sys.executable, '-I', '-c', '''import tensorcx as cx
assert cx._core._build_info()['vulkan'], 'filter acceptance requires a Vulkan-enabled runtime'
assert not cx.is_available('vulkan'), 'software device leaked into the production GPU registry'
assert not any(str(d).startswith('vulkan') for d in cx.devices())
assert cx.ones((3,), device='cpu').numpy().tolist() == [1., 1., 1.]
assert cx._core._vulkan_validation_errors() == 0
'''], output, env, output / 'production-filter.log')
        report.update(status='passed', production_gpu_filter_verified=True,
                      cpu_reference='tensorcx CPU primitives, guarded Kernel.reference and NumPy')
    except BaseException as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_public_report(output / 'result.json', report, root=ROOT)
    print((output / 'result.json').read_text())


if __name__ == '__main__':
    main()
