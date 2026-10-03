"""Failure/report boundaries for the distribution validation runner."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / 'tools/validate_distribution.py'
spec = importlib.util.spec_from_file_location('validate_distribution', SCRIPT)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def test_command_keeps_diagnostics_out_of_json(tmp_path):
    log = tmp_path / 'command.log'
    result = validator.run([sys.executable, '-c',
                            "import sys, json; print(json.dumps(dict(ok=True))); print('warning', file=sys.stderr)"],
                           tmp_path, os.environ.copy(), log)
    assert json.loads(result) == {'ok': True}
    assert 'warning' in log.read_text()


def test_command_failure_retains_diagnostics(tmp_path):
    log = tmp_path / 'failed.log'
    with pytest.raises(RuntimeError, match='failed.*7'):
        validator.run([sys.executable, '-c', 'import sys; print("broken"); sys.exit(7)'],
                      tmp_path, os.environ.copy(), log)
    assert 'broken' in log.read_text()


def test_refuses_to_overwrite_existing_run(tmp_path):
    sentinel = tmp_path / 'manifest.json'
    sentinel.write_text('existing evidence')
    result = subprocess.run([sys.executable, str(SCRIPT), '--output', str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert sentinel.read_text() == 'existing evidence'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['manifest.json']


@pytest.mark.parametrize('path', [str(Path(tempfile.gettempdir()) / 'runtime/libcudart.so.12'),
                                 '/tmp/cuda-driver-stub/libcuda.so.1',
                                 str(validator.ROOT / 'build/libcudart.so.12')])
def test_rejects_contaminated_dependency_resolution(tmp_path, path):
    with pytest.raises(RuntimeError, match='contaminated'):
        validator.audit_linkage(f'libcudart.so.12 => {path} (0x1234)', 'cuda', tmp_path)


def test_allows_system_and_wheel_bundled_dependencies(tmp_path):
    validator.audit_linkage('libc.so.6 => /usr/lib/libc.so.6 (0x1234)', 'cpu', tmp_path)
    validator.audit_linkage(f'libextra.so => {tmp_path}/env/lib/libextra.so (0x1234)', 'cpu', tmp_path)


@pytest.mark.parametrize('linkage', ['libcudart.so.12 => /usr/lib/libcudart.so.12 (0x1234)', 'libextra.so => not found'])
def test_rejects_wrong_or_missing_dependency(linkage, tmp_path):
    with pytest.raises(RuntimeError, match='unexpected or unresolved'):
        validator.audit_linkage(linkage, 'cpu', tmp_path)


@pytest.mark.parametrize('name', [
    'project/python/pkg/__pycache__/module.cpython-312.pyc',
    'project/.env', 'project/.env.production', 'pkg/client.key',
    'project/.git/config', 'pkg/auth.pem', 'project/before.bundle',
    'project/.netrc', 'project/.ssh/config', 'pkg/credentials.json',
])
def test_rejects_private_distribution_members(name):
    with pytest.raises(RuntimeError, match='private/cache'):
        validator.audit_archive_members([name])


def test_accepts_source_and_notice_members():
    validator.audit_archive_members(['project/cpp/core.cpp', 'pkg/_core.so',
                                     'project/THIRD_PARTY_NOTICES.md'])


def test_wheel_requires_embedded_third_party_notices(tmp_path):
    import zipfile
    wheel = tmp_path / 'fixture.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('pkg.dist-info/licenses/LICENSE', 'Apache-2.0')
    with zipfile.ZipFile(wheel) as archive:
        with pytest.raises(RuntimeError, match='missing third-party'):
            validator.audit_wheel_notices(archive)
    with zipfile.ZipFile(wheel, 'a') as archive:
        archive.writestr('pkg.dist-info/licenses/THIRD_PARTY_NOTICES.md',
                         (validator.ROOT / 'THIRD_PARTY_NOTICES.md').read_text())
    with zipfile.ZipFile(wheel) as archive:
        validator.audit_wheel_notices(archive)
