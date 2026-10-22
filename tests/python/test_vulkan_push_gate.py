"""A local GPU gate must reject missing coverage and preserve Git's payload."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

TOOLS = Path(__file__).resolve().parents[2] / 'tools'


def module(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / (name + '.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


gate = module('vulkan_push_gate')
hook = module('pre_push_acceptance')


def test_required_vulkan_scope_and_explicit_combined_mode():
    snapshot = Path('/snapshot')
    pure = gate.selected_tests(snapshot, False)
    combined = gate.selected_tests(snapshot, True)
    assert pure.count('--deselect') == 2
    assert '--deselect' not in combined
    assert snapshot / 'tests/python/test_shared_storage.py' in combined
    assert snapshot / 'tests/python/test_shared_storage.py' not in pure
    for name in gate.TEST_FILES:
        assert snapshot / 'tests/python' / name in pure
        assert snapshot / 'tests/python' / name in combined
    assert len(gate.NATIVE_ANCHORS) == 13
    assert len(gate.NATIVE_ANCHORS + gate.CUDA_NATIVE_ANCHORS) == 41


def test_vulkan_only_deselection_matches_real_snapshot_node_ids(monkeypatch, tmp_path):
    snapshot = tmp_path / 'source'
    tests = snapshot / 'tests/python'
    tests.mkdir(parents=True)
    names = ('test_vulkan_performance.py', 'test_vulkan_transfer.py')
    functions = ('test_vulkan_profile_mixed_cuda_clocks',
                 'test_transfer_threads_pools_and_explicit_dlpack_bridge')
    for name, function in zip(names, functions):
        (tests / name).write_text(f'def {function}():\n    assert False, "CUDA-only test ran"\n'
                                 'def test_vulkan_contract():\n    pass\n')
    monkeypatch.setattr(gate, 'TEST_FILES', names)
    result = subprocess.run([sys.executable, '-I', str(TOOLS / 'cuda_gate_pytest.py'),
                             '-q', '--rootdir', str(snapshot),
                             *map(str, gate.selected_tests(snapshot, False)),
                             '--junitxml', str(tmp_path / 'result.xml')],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '2 deselected' in result.stdout
    assert gate.junit_result(tmp_path / 'result.xml', 2)['tests'] == 2


@pytest.mark.parametrize('with_cuda,minimum', [(False, 4296), (True, 6249)])
def test_measured_mode_specific_floor_rejects_reduced_coverage(tmp_path, with_cuda, minimum):
    assert gate.PYTHON_MINIMUM[with_cuda] == minimum
    report = tmp_path / 'result.xml'
    cases = ''.join(f'<testcase name="test_{index}"/>' for index in range(minimum))
    report.write_text('<testsuite>' + cases + '</testsuite>')
    assert gate.junit_result(report, gate.PYTHON_MINIMUM[with_cuda])['tests'] == minimum
    report.write_text('<testsuite>' + cases.rsplit('<testcase', 1)[0] + '</testsuite>')
    with pytest.raises(RuntimeError, match='expected at least'):
        gate.junit_result(report, gate.PYTHON_MINIMUM[with_cuda])


@pytest.mark.parametrize('local', [None, 'false', 'true'])
def test_repository_local_settings_ignore_global_and_command_scopes(monkeypatch, tmp_path, local):
    repository = tmp_path / 'repository'
    repository.mkdir()
    global_config = tmp_path / 'global.gitconfig'
    global_config.write_text('[tensorcx]\n vulkanGateEnabled = true\n vulkanWithCuda = true\n'
                             ' vulkanSdk = /global-sdk\n')
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', str(global_config))
    monkeypatch.setenv('GIT_CONFIG_SYSTEM', str(tmp_path / 'absent-system-config'))
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    monkeypatch.setenv('GIT_CONFIG_COUNT', '3')
    for index, (key, value) in enumerate((('GateEnabled', 'true'), ('WithCuda', 'true'),
                                         ('Sdk', '/command-sdk'))):
        monkeypatch.setenv(f'GIT_CONFIG_KEY_{index}', 'tensorcx.vulkan' + key)
        monkeypatch.setenv(f'GIT_CONFIG_VALUE_{index}', value)
    real_run = subprocess.run
    real_run(['git', 'init', '--quiet', str(repository)], check=True)
    if local is not None:
        for key in ('GateEnabled', 'WithCuda'):
            real_run(['git', 'config', '--local', 'tensorcx.vulkan' + key, local],
                     cwd=repository, check=True)
        real_run(['git', 'config', '--local', 'tensorcx.vulkanSdk', '/local-sdk'],
                 cwd=repository, check=True)
    monkeypatch.setattr(gate, 'ROOT', repository)
    monkeypatch.setattr(hook, 'ROOT', repository)
    assert gate.configuration('WithCuda', boolean=True) == local
    assert gate.configuration('Sdk') == (None if local is None else '/local-sdk')
    payload = f'HEAD {"a"*40} refs/heads/main {"b"*40}\n'.encode()
    monkeypatch.setattr(hook.sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(payload)))
    seen = []
    def run(command, **kwargs):
        if command[0] == 'git':
            return real_run(command, **kwargs)
        seen.append(Path(command[1]).name)
        assert kwargs['input'] == payload
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(hook.subprocess, 'run', run)
    assert hook.main() == 0
    assert seen == ['cuda_push_gate.py'] + (['vulkan_push_gate.py'] if local == 'true' else [])


def test_external_overrides_cannot_fake_or_skip_acceptance(monkeypatch):
    forbidden = ('TENSORCX_VULKAN_LIBRARY', 'TENSORCX_REQUIRE_BACKENDS', 'TENSORCX_DISABLE_CACHE',
                 'VK_DRIVER_FILES', 'VK_INSTANCE_LAYERS', 'SKBUILD_CMAKE_ARGS', 'CIBW_TEST_SKIP',
                 'PYTHONPATH', 'PYTEST_ADDOPTS', 'PYTEST_PLUGINS', 'LD_PRELOAD', 'LD_LIBRARY_PATH',
                 'CC', 'CXXFLAGS', 'CMAKE_ARGS', 'GIT_DIR', 'GIT_INDEX_FILE')
    for key in forbidden:
        monkeypatch.setenv(key, 'injected override')
    monkeypatch.setattr(gate.subprocess, 'check_output',
                        lambda *a, **k: 'GIT_DIR\nGIT_INDEX_FILE\n')
    env = gate.clean_environment()
    assert not any(key in env for key in forbidden)
    assert env['PATH']


def test_unrelated_remote_ref_needs_no_gpu_configuration(monkeypatch):
    monkeypatch.setattr(gate.sys, 'stdin', io.StringIO(f'HEAD {"a"*40} refs/heads/topic {"b"*40}\n'))
    monkeypatch.setattr(gate, 'configuration', lambda *a, **k: pytest.fail('unrelated push read config'))
    monkeypatch.setattr(gate, 'validate', lambda *a: pytest.fail('unrelated push ran GPU gate'))
    gate.main(['--pre-push'])


def test_invalid_local_configuration_is_not_treated_as_disabled(monkeypatch):
    monkeypatch.setattr(gate.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=128))
    with pytest.raises(RuntimeError, match='invalid local'):
        gate.configuration('WithCuda', boolean=True)


@pytest.mark.parametrize('summary', [
    '', 'driverName = llvmpipe', 'deviceType = PHYSICAL_DEVICE_TYPE_CPU',
    'deviceType = PHYSICAL_DEVICE_TYPE_CPU\ndriverName = other',
    'deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU\ndriverName = llvmpipe',
    'deviceType = PHYSICAL_DEVICE_TYPE_CPU\ndriverName = llvmpipe\n'
    'deviceType = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU\ndriverName = llvmpipe',
    'deviceType = PHYSICAL_DEVICE_TYPE_CPU\ndriverName = llvmpipe\n'
    'deviceType = PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU\ndriverName = llvmpipe',
])
def test_software_discovery_rejects_missing_or_wrong_device(summary):
    with pytest.raises(RuntimeError, match='required Mesa CPU'):
        gate.software_discovery(summary)


def test_software_discovery_requires_real_cpu_and_publishes_only_selected_fields():
    summary = (' deviceType = PHYSICAL_DEVICE_TYPE_CPU\n driverName = llvmpipe\n'
               ' driverInfo = Mesa fixture\n apiVersion = 1.4.335\n deviceUUID = private\n')
    assert gate.software_discovery(summary) == ['deviceType=PHYSICAL_DEVICE_TYPE_CPU',
        'driverName=llvmpipe', 'driverInfo=Mesa fixture', 'apiVersion=1.4.335']


@pytest.mark.parametrize('key,value', [('status', 'failed'), ('execution', 'gpu'),
    ('device_type', 'gpu'), ('driver_id', 4), ('driver_name', 'other'), ('validation_errors', 1),
    ('production_gpu_filter_verified', False), ('cases', 191), ('cycles', 1),
    ('owned_handles_created', 0), ('owned_handles_released', 1741)])
def test_gate_rejects_missing_software_compute_acceptance(key, value):
    report = {'status': 'passed', 'execution': 'software-cpu', 'device_type': 'cpu',
              'driver_id': 13, 'driver_name': 'llvmpipe', 'validation_errors': 0,
              'production_gpu_filter_verified': True, 'cases': 192, 'cycles': 2,
              'owned_handles_created': 1742, 'owned_handles_released': 1742}
    assert gate.software_compute_result(report) is report
    report[key] = value
    with pytest.raises(RuntimeError, match='lavapipe'):
        gate.software_compute_result(report)


def test_missing_prerequisite_records_failure_without_claiming_pass(monkeypatch, tmp_path):
    (tmp_path / 'tools').mkdir()
    for name in ('cuda_gate_pytest.py', 'acceptance_support.py'):
        (tmp_path / 'tools' / name).write_text('# fixture')
    monkeypatch.setattr(gate, 'ROOT', tmp_path)
    monkeypatch.setattr(gate.subprocess, 'check_output',
                        lambda argv, **kw: '' if '--local-env-vars' in argv else 'a' * 40)
    args = SimpleNamespace(with_cuda=False, sdk=tmp_path / 'missing-sdk', llvm_bin=tmp_path,
                           shader_bin=tmp_path, validation_layer=tmp_path, software_icd=tmp_path,
                           library_dir=None, gpu_icd=None, cc='cc', cxx='c++')
    with pytest.raises(RuntimeError, match='header is missing'):
        gate.validate('HEAD', args)
    reports = list(tmp_path.glob('build/vulkan-gate/*/result.json'))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report['status'] == 'failed' and report['revision'] == 'a' * 40
    assert report['validation_required'] is True and report['with_cuda'] is False
    assert 'native' not in report and 'python' not in report
    assert str(tmp_path) not in reports[0].read_text()


@pytest.mark.parametrize('cuda_exit,setting,vulkan_exit,expected,controllers', [
    (17, 'true', 0, 17, ['cuda_push_gate.py']),
    (0, None, 0, 0, ['cuda_push_gate.py']),
    (0, 'false', 0, 0, ['cuda_push_gate.py']),
    (0, 'true', 0, 0, ['cuda_push_gate.py', 'vulkan_push_gate.py']),
    (0, 'true', 19, 19, ['cuda_push_gate.py', 'vulkan_push_gate.py']),
])
def test_hook_keeps_mandatory_cuda_and_exact_stdin(monkeypatch, cuda_exit, setting,
                                                vulkan_exit, expected, controllers):
    payload = (f'refs/heads/topic {"a"*40} refs/heads/main {"b"*40}\n'
               f'refs/heads/main {"c"*40} refs/heads/other {"d"*40}\n').encode()
    monkeypatch.setattr(hook.sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(payload)))
    seen = []
    def run(command, **kwargs):
        if command[0] == 'git':
            return SimpleNamespace(returncode=1 if setting is None else 0, stdout=setting or '')
        name = Path(command[1]).name
        seen.append(name)
        assert kwargs['input'] == payload
        assert command[-1] == '--pre-push'
        return SimpleNamespace(returncode=cuda_exit if name == 'cuda_push_gate.py' else vulkan_exit)
    monkeypatch.setattr(hook.subprocess, 'run', run)
    assert hook.main() == expected
    assert seen == controllers


def test_hook_fails_closed_on_corrupt_opt_in(monkeypatch):
    monkeypatch.setattr(hook.sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(b'')))
    def run(command, **kwargs):
        if command[0] == 'git':
            return SimpleNamespace(returncode=128, stdout='invalid')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(hook.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='invalid tensorcx.vulkanGateEnabled'):
        hook.main()
