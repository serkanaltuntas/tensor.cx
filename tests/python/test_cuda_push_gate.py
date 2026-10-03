"""A successful command must not hide skipped/missing acceptance tests."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


spec = importlib.util.spec_from_file_location('cuda_push_gate', Path(__file__).resolve().parents[2] / 'tools/cuda_push_gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_push_filters_remote_ref_and_uses_commit_not_branch():
    sha = 'a' * 40
    assert gate.push_revisions(f'refs/heads/topic {sha} refs/heads/main {"b" * 40}') == [sha]
    assert gate.push_revisions(f'refs/heads/main {sha} refs/heads/topic {"b" * 40}') == []
    assert gate.push_revisions('') == []


@pytest.mark.parametrize('value', ['broken', f'(delete) {"0" * 40} refs/heads/main {"a" * 40}',
                                 f'HEAD main refs/heads/main {"a" * 40}'])
def test_rejects_unverifiable_push(value):
    with pytest.raises(RuntimeError):
        gate.push_revisions(value)


@pytest.mark.parametrize('tag', ['skipped', 'failure', 'error'])
def test_rejects_nonpassing_junit(tmp_path, tag):
    report = tmp_path / 'result.xml'
    report.write_text(f'<testsuites><testsuite><testcase name="required"><{tag}/></testcase></testsuite></testsuites>')
    with pytest.raises(RuntimeError, match='non-passing'):
        gate.junit_result(report, 1, ['required'])


def test_accepts_real_cases_and_rejects_count_or_anchor_loss(tmp_path):
    report = tmp_path / 'result.xml'
    report.write_text('<testsuites><testsuite><testcase name="required[cuda]"/></testsuite></testsuites>')
    assert gate.junit_result(report, 1, ['required'])['tests'] == 1
    with pytest.raises(RuntimeError, match='at least'):
        gate.junit_result(report, 2)
    with pytest.raises(RuntimeError, match='missing required'):
        gate.junit_result(report, 1, ['removed_test'])


def test_timeout_preserves_partial_output(monkeypatch, tmp_path):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('compiler', 3600, output=b'partial stdout', stderr=b'partial error')
    monkeypatch.setattr(gate.subprocess, 'run', timeout)
    log = tmp_path / 'timeout.log'
    with pytest.raises(RuntimeError, match='timed out'):
        gate.run(['compiler'], tmp_path, {}, log)
    assert log.read_text() == 'partial stdoutpartial error'


def test_hook_preserves_stdin_and_failure_exit(tmp_path):
    uv = tmp_path / 'uv'
    uv.write_text('#!/bin/sh\ncat\nexit 17\n')
    uv.chmod(0o755)
    hook = gate.ROOT / '.githooks/pre-push'
    line = f'refs/heads/main {"a" * 40} refs/heads/main {"b" * 40}\n'
    env = {**os.environ, 'PATH': str(tmp_path) + os.pathsep + os.environ['PATH']}
    result = subprocess.run([str(hook), 'origin', 'unused'], cwd=gate.ROOT,
                            env=env, input=line, text=True, capture_output=True)
    assert result.returncode == 17
    assert result.stdout == line
    result = subprocess.run([str(hook), 'other-remote', 'unused'], cwd=gate.ROOT,
                            env=env, input=line, text=True, capture_output=True)
    assert result.returncode == 0
    assert result.stdout == ''


@pytest.mark.parametrize('body', [
    'import pytest\n@pytest.mark.xfail(strict=False)\ndef test_case(): pass\n',
    'import pytest\n@pytest.mark.xfail(strict=False)\ndef test_case(): assert False\n',
    'import pytest\ndef test_case(): pytest.skip("unavailable")\n',
    'def test_case(): pass\n',
])
def test_pytest_controller_rejects_non_strict_xpass_and_skip(tmp_path, body):
    test = tmp_path / 'test_case.py'
    test.write_text(body)
    result = subprocess.run([sys.executable, '-I', str(gate.ROOT / 'tools/cuda_gate_pytest.py'),
                             '-q', str(test)], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == (0 if body.startswith('def ') else 1), result.stdout + result.stderr


def test_failed_gate_writes_public_report_without_changing_failure(monkeypatch, tmp_path):
    import json

    (tmp_path / 'tools').mkdir()
    (tmp_path / 'tools/cuda_gate_pytest.py').write_text('# fixture')
    monkeypatch.setattr(gate, 'ROOT', tmp_path)
    monkeypatch.setattr(gate.subprocess, 'check_output',
                        lambda argv, **kwargs: '' if '--local-env-vars' in argv else 'a' * 40)

    def fail(*args, **kwargs):
        raise RuntimeError(f'failed; see {tmp_path}/private-diagnostic.log')

    monkeypatch.setattr(gate, 'run', fail)
    with pytest.raises(RuntimeError, match='failed'):
        gate.validate('a' * 40, tmp_path)
    reports = list(tmp_path.glob('build/cuda-gate/*/result.json'))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report['status'] == 'failed'
    assert report['revision'] == 'a' * 40
    assert report['publication']['format'] == 'cortex-public-report-v1'
    assert str(tmp_path) not in reports[0].read_text()
    assert 'private-diagnostic.log' not in report['error']
