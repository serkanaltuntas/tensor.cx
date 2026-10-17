"""Portable release validation must reject incomplete or mislabelled artifacts."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

TOOLS = Path(__file__).resolve().parents[2] / 'tools'
sys.path.insert(0, str(TOOLS))
try:
    spec = importlib.util.spec_from_file_location('build_portable_wheels', TOOLS / 'build_portable_wheels.py')
    portable = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(portable)
finally:
    sys.path.pop(0)


def wheel(directory, *, tag='cp312', platform='manylinux_2_28_x86_64', metadata=None, missing=None):
    path = directory / f'tensorcx-0.1.0-{tag}-{tag}-{platform}.whl'
    members = {
        'tensorcx/_core.cpython-312-x86_64-linux-gnu.so': b'ELF fixture',
        'tensorcx_doctor.py': b'',
        'tensorcx/_compiler/cache.py': b'',
        'tensorcx-0.1.0.dist-info/WHEEL': (metadata or f'Tag: {tag}-{tag}-{platform}').encode(),
        'tensorcx-0.1.0.dist-info/licenses/LICENSE': b'Apache-2.0',
        'tensorcx-0.1.0.dist-info/licenses/THIRD_PARTY_NOTICES.md': (TOOLS.parent / 'THIRD_PARTY_NOTICES.md').read_bytes(),
    }
    members.pop(missing, None)
    with zipfile.ZipFile(path, 'w') as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def test_rejects_missing_abi_and_plain_linux_tags(tmp_path):
    with pytest.raises(RuntimeError, match='missing'):
        portable.inspect_wheels(tmp_path, ['3.12'])
    wheel(tmp_path, platform='linux_x86_64')
    with pytest.raises(RuntimeError, match='target'):
        portable.inspect_wheels(tmp_path, ['3.12'])


def test_rejects_filename_metadata_mismatch(tmp_path):
    wheel(tmp_path, metadata='Tag: cp311-cp311-manylinux_2_28_x86_64')
    with pytest.raises(RuntimeError, match='metadata'):
        portable.inspect_wheels(tmp_path, ['3.12'])


@pytest.mark.parametrize('member', ['tensorcx_doctor.py', 'tensorcx/_compiler/cache.py',
                                   'tensorcx/_core.cpython-312-x86_64-linux-gnu.so',
                                   'tensorcx-0.1.0.dist-info/licenses/LICENSE'])
def test_rejects_incomplete_wheel(tmp_path, member):
    wheel(tmp_path, missing=member)
    with pytest.raises(RuntimeError, match='incomplete'):
        portable.inspect_wheels(tmp_path, ['3.12'])


def test_records_exact_artifact_hash_and_all_requested_abis(tmp_path):
    path = wheel(tmp_path)
    result = portable.inspect_wheels(tmp_path, ['3.12'])
    assert result[0]['sha256'] == portable.sha256(path)
    with pytest.raises(RuntimeError, match='missing'):
        portable.inspect_wheels(tmp_path, ['3.11', '3.12'])


def test_never_overwrites_a_previous_run(tmp_path):
    sentinel = tmp_path / 'manifest.json'
    sentinel.write_text('previous evidence')
    result = subprocess.run([sys.executable, str(TOOLS / 'build_portable_wheels.py'), '--output', str(tmp_path)],
                            text=True, capture_output=True)
    assert result.returncode != 0
    assert sentinel.read_text() == 'previous evidence'
    assert list(tmp_path.iterdir()) == [sentinel]


def test_external_overrides_cannot_skip_tests_or_change_backend(monkeypatch):
    for key in ('CIBW_TEST_SKIP', 'CIBW_TEST_COMMAND', 'CIBW_ENVIRONMENT', 'CIBW_CONTAINER_ENGINE',
                'CMAKE_ARGS', 'CC', 'CXXFLAGS', 'LD_PRELOAD', 'PYTEST_ADDOPTS'):
        monkeypatch.setenv(key, 'unsafe override')
    env = portable.clean_environment()
    assert not any(key.startswith('CIBW_') for key in env)
    assert not any(key in env for key in ('CMAKE_ARGS', 'CC', 'CXXFLAGS', 'LD_PRELOAD', 'PYTEST_ADDOPTS'))


@pytest.mark.parametrize('reported', ['manylinux_2_34_x86_64', 'linux_x86_64', 'manylinux_2_28_aarch64'])
def test_audit_must_report_compatible_platform_not_merely_echo_filename(reported):
    output = ('tensorcx-cp312-cp312-manylinux_2_28_x86_64.whl\n'
              'is consistent with the following platform tag:\n"' + reported + '".')
    with pytest.raises(RuntimeError, match='compatibility'):
        portable.check_audit(output)


def test_older_compatible_policy_and_cpu_dependency_boundary():
    output = 'is consistent with the following platform tag:\n"manylinux_2_27_x86_64".'
    portable.check_audit(output)
    with pytest.raises(RuntimeError, match='GPU/LLVM'):
        portable.check_audit(output + ' libcuda.so.1')


def test_audit_accepts_real_line_wrapped_auditwheel_output():
    portable.check_audit('tensorcx-0.1.0-cp312-cp312-manylinux_2_28_x86_64.whl is consistent\n'
                         'with the following platform tag: "manylinux_2_27_x86_64".\n')
