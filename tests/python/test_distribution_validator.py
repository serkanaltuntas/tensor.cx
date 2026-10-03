"""Failure/report boundaries for the distribution validation runner."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

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


@pytest.mark.parametrize('path', ['/tmp/runtime/libcudart.so.12', '/tmp/cuda-driver-stub/libcuda.so.1',
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
