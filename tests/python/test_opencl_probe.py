"""Pure CPU tests of the standalone experiment's fail-closed acceptance oracle."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[2] / 'experiments/opencl/probe_opencl.py'
SPEC = importlib.util.spec_from_file_location('tensorcx_opencl_probe', PATH)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def case(size, active, operation, alias, cycle=0):
    groups, a, b, out = probe.initial_values(size, active)
    result = a.copy() if alias else out.copy()
    result[:active] = (a + b if operation == 'add' else a * b)[:active]
    return {'size': size, 'active': active, 'operation': operation, 'alias': alias,
            'groups': groups, 'cycle': cycle, 'result': result.view(np.uint32).tolist(),
            'a_after': (result if alias else a).view(np.uint32).tolist(),
            'b_after': b.view(np.uint32).tolist()}


def report(target='nvidia-gpu'):
    nvidia = target == 'nvidia-gpu'
    return {'status': 'passed', 'target': target, 'device_name': 'fixture device',
            'device_type': 4 if nvidia else 2, 'compiler_available': True, 'device_available': True,
            'platform_vendor': 'NVIDIA Corporation' if nvidia else 'The pocl project',
            'platform_name': 'NVIDIA CUDA' if nvidia else 'Portable Computing Language',
            'vendor_id': 0x10de if nvidia else 0x10006, 'platform_version': 'OpenCL 3.0 fixture',
            'device_version': 'OpenCL 3.0 fixture', 'driver_version': 'fixture driver',
            'opencl_c_version': 'OpenCL C 1.2', 'build_options': '-cl-std=CL1.2',
            'max_work_group_size': 1024, 'cycles': 2, 'release_errors': 0,
            'owned_handles_created': 490, 'owned_handles_released': 490,
            'cases': [case(size, active, op, alias, cycle)
                      for cycle, size, active, op, alias in sorted(probe.expected_cases())]}


@pytest.mark.parametrize('size,active', [(0, 0), (1, 0), (1, 1), (65, 32), (65, 65), (257, 257)])
@pytest.mark.parametrize('operation', probe.OPERATIONS)
@pytest.mark.parametrize('alias', [False, True])
def test_opencl_readback_uses_guarded_cpu_and_numpy(size, active, operation, alias):
    probe.verify_case(case(size, active, operation, alias))


@pytest.mark.parametrize('field,index', [('result', 0), ('result', 31), ('result', 32),
    ('result', 64), ('result', -1), ('a_after', 0), ('a_after', -1), ('b_after', 0), ('b_after', -1)])
@pytest.mark.parametrize('alias', [False, True])
def test_opencl_rejects_corrupt_active_inactive_padding_or_inputs(field, index, alias):
    value = case(65, 32, 'add', alias)
    value[field][index] ^= 1
    with pytest.raises(AssertionError):
        probe.verify_case(value)


@pytest.mark.parametrize('bits', [[True] * 72, [-1] * 72, [2**32] * 72, [0.0] * 72, [0] * 71])
def test_opencl_rejects_invalid_storage_bits(bits):
    with pytest.raises(ValueError, match='storage-bit'):
        probe.storage_bits(bits, 72)


@pytest.mark.parametrize('target', ['nvidia-gpu', 'pocl-cpu'])
def test_opencl_full_scope_validated_independently_for_each_target(target):
    result = probe.verify_report(report(target), target, 'fixture device')
    assert result['cases'] == 192
    assert result['cycles'] == 2
    assert result['owned_handles_created'] == result['owned_handles_released'] == 490


@pytest.mark.parametrize('key,value', [('status', 'running'), ('target', 'pocl-cpu'),
    ('device_name', 'other'), ('device_type', 2), ('device_type', 4.0), ('compiler_available', False),
    ('compiler_available', 1), ('device_available', False), ('build_options', '-cl-fast-relaxed-math'),
    ('cycles', 1), ('release_errors', 1), ('release_errors', False), ('platform_vendor', 'other'),
    ('vendor_id', 0), ('platform_version', ''), ('device_version', None), ('driver_version', ''),
    ('opencl_c_version', 1.2), ('max_work_group_size', 32), ('owned_handles_created', 491),
    ('owned_handles_released', 489), ('owned_handles_released', 490.0)])
def test_opencl_rejects_identity_compiler_or_cleanup_failure(key, value):
    record = report()
    record[key] = value
    with pytest.raises(ValueError):
        probe.verify_report(record, 'nvidia-gpu', 'fixture device')


@pytest.mark.parametrize('key,value', [('platform_name', 'NVIDIA CUDA'), ('device_type', 4)])
def test_opencl_pocl_cannot_substitute_gpu(key, value):
    record = report('pocl-cpu')
    record[key] = value
    with pytest.raises(ValueError):
        probe.verify_report(record, 'pocl-cpu', 'fixture device')


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'unexpected', 'bool-size', 'wrong-groups'])
def test_opencl_rejects_incomplete_or_forged_case_matrix(mutation):
    record = report()
    if mutation == 'missing':
        record['cases'].pop()
    elif mutation == 'duplicate':
        record['cases'][-1] = copy.deepcopy(record['cases'][0])
    elif mutation == 'unexpected':
        record['cases'][0]['operation'] = 'divide'
    elif mutation == 'bool-size':
        record['cases'][0]['size'] = False
    else:
        record['cases'][0]['groups'] = 2
    with pytest.raises(ValueError):
        probe.verify_report(record, 'nvidia-gpu', 'fixture device')


def test_opencl_unknown_target_is_rejected():
    with pytest.raises(ValueError, match='unknown execution target'):
        probe.verify_report(report(), 'automatic', 'fixture device')


def test_opencl_missing_prerequisite_persists_failure(tmp_path):
    output = tmp_path / 'result'
    with pytest.raises(RuntimeError, match='vendor .icd'):
        probe.main(['--target', 'nvidia-gpu', '--device-name', 'fixture device',
                    '--icd', str(tmp_path / 'missing.icd'), '--output', str(output)])
    result = json.loads((output / 'result.json').read_text())
    assert result['status'] == 'failed'
    assert 'cases' not in result and 'execution' not in result


def test_opencl_native_failure_cannot_be_reported_as_passed(tmp_path, monkeypatch):
    icd = tmp_path / 'fixture.icd'
    icd.write_text('fixture.so')
    output = tmp_path / 'failed'
    def failed(*args):
        raise RuntimeError('native fixture failed')
    monkeypatch.setattr(probe, 'run', failed)
    with pytest.raises(RuntimeError, match='native fixture failed'):
        probe.main(['--target', 'pocl-cpu', '--device-name', 'fixture device',
                    '--icd', str(icd), '--output', str(output)])
    result = json.loads((output / 'result.json').read_text())
    assert result['status'] == 'failed' and 'cases' not in result


def test_opencl_sanitizes_driver_compiler_and_visibility_overrides(monkeypatch):
    for key in ('OCL_ICD_VENDORS', 'OCL_ICD_FILENAMES', 'POCL_EXTRA_BUILD_FLAGS', 'POCL_DEVICES',
                'POCL_MAX_PTHREAD_COUNT', 'OPENCL_VENDOR_PATH', 'LD_PRELOAD', 'LD_LIBRARY_PATH',
                'PYTHONPATH', 'CPATH', 'CPLUS_INCLUDE_PATH', 'LIBRARY_PATH', 'CXXFLAGS'):
        monkeypatch.setenv(key, 'injected')
    assert 'injected' not in probe.clean_environment().values()
