"""The software experiment must reject missing cases and incorrect readback."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[2] / 'experiments/vulkan/probe_lavapipe.py'
SPEC = importlib.util.spec_from_file_location('tensorcx_lavapipe_probe', PATH)
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


@pytest.mark.parametrize('size,active', [(0, 0), (1, 0), (1, 1), (65, 32), (65, 65), (257, 257)])
@pytest.mark.parametrize('operation', probe.OPERATIONS)
@pytest.mark.parametrize('alias', [False, True])
def test_lavapipe_readback_against_guarded_cpu_reference(size, active, operation, alias):
    probe.verify_case(case(size, active, operation, alias))


@pytest.mark.parametrize('field,index', [('result', 0), ('result', -1), ('a_after', 0), ('b_after', -1)])
def test_lavapipe_rejects_inactive_output_padding_and_mutated_inputs(field, index):
    record = case(65, 32, 'add', False)
    record[field][index] ^= 1
    with pytest.raises(AssertionError):
        probe.verify_case(record)


@pytest.mark.parametrize('bits', [[True] * 72, [-1] * 72, [2**32] * 72, [0.0] * 72, [0] * 71])
def test_lavapipe_rejects_invalid_storage_bits(bits):
    with pytest.raises(ValueError, match='storage-bit'):
        probe.storage_bits(bits, 72)


def report():
    return {'status': 'passed', 'device_type': 'cpu', 'driver_id': 13, 'driver_name': 'llvmpipe',
            'driver_info': 'Mesa fixture', 'api_version': 1 << 22 | 4 << 12,
            'cycles': 2, 'validation_errors': 0, 'owned_handles_created': 1742,
            'owned_handles_released': 1742,
            'cases': [dict(zip(('cycle', 'size', 'active', 'operation', 'alias'), key), groups=1)
                      for key in sorted(probe.expected_cases())]}


def test_lavapipe_exact_scope_and_two_context_cleanup_cycles(monkeypatch):
    seen = []
    monkeypatch.setattr(probe, 'verify_case', lambda record: seen.append(record))
    value = report()
    assert probe.verify_report(value)['cases'] == 192
    assert len(seen) == 192
    value['cases'].pop()
    with pytest.raises(ValueError, match='missing required'):
        probe.verify_report(value)


def test_lavapipe_duplicate_cannot_replace_missing_case(monkeypatch):
    monkeypatch.setattr(probe, 'verify_case', lambda _: None)
    value = report()
    value['cases'][-1] = value['cases'][0]
    with pytest.raises(ValueError, match='duplicate'):
        probe.verify_report(value)


@pytest.mark.parametrize('key,value', [('device_type', 'gpu'), ('driver_id', 4), ('driver_name', 'other'),
    ('api_version', 1 << 22 | 1 << 12), ('cycles', 1), ('validation_errors', 1), ('status', 'running'),
    ('owned_handles_created', 0), ('owned_handles_released', 1741)])
def test_lavapipe_rejects_wrong_identity_errors_or_unreleased_handles(key, value):
    record = report()
    record[key] = value
    with pytest.raises(ValueError):
        probe.verify_report(record)


def test_lavapipe_failed_prerequisite_persists_failure(tmp_path):
    output = tmp_path / 'result'
    with pytest.raises(RuntimeError, match='headers'):
        probe.main(['--sdk', str(tmp_path / 'no-sdk'), '--validation-layer', str(tmp_path),
                    '--software-icd', str(tmp_path / 'missing.json'), '--output', str(output)])
    result = json.loads((output / 'result.json').read_text())
    assert result['status'] == 'failed' and result['execution'] == 'software-cpu'
    assert 'cases' not in result and 'production_gpu_filter_verified' not in result


def test_lavapipe_environment_rejects_loader_and_visibility_overrides(monkeypatch):
    for key in ('VK_DRIVER_FILES', 'VK_INSTANCE_LAYERS', 'TENSORCX_VULKAN_LIBRARY', 'LD_PRELOAD',
                'LD_LIBRARY_PATH', 'PYTHONPATH', 'PYTEST_ADDOPTS', 'CXXFLAGS'):
        monkeypatch.setenv(key, 'injected')
    env = probe.clean_environment()
    assert not any(value == 'injected' for value in env.values())
