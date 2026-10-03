"""Publication policy: narrow fixture allowances, privacy-safe output and history."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / 'tools/check_publication.py'
spec = importlib.util.spec_from_file_location('check_publication', SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


@pytest.mark.parametrize('filename', [
    '.env', 'nested/.env.production', '.ssh/id_rsa', 'client.key',
    'backup.bundle', '.codex/session.json', 'tests/credentials.json',
])
def test_private_filenames_never_exempted(filename):
    assert 'private-file' in checker.categories(filename, b'')


@pytest.mark.parametrize('filename', [
    'python/pkg/__pycache__/module.pyc', '.cache/data', 'build/result.json',
    'dist/package.whl', 'local.log', 'model.safetensors',
])
def test_cache_build_files(filename):
    assert 'cache-or-build-file' in checker.categories(filename, b'')


def test_template_is_allowed_but_private_key_is_not():
    assert not checker.categories('.env.example', b'SERVICE_TOKEN=')
    key = ('-' * 5 + 'BEGIN OPENSSH PRIVATE KEY' + '-' * 5).encode()
    for filename in ('.env.example', 'tests/python/test_fixture.py', 'tools/check_publication.py'):
        assert 'private-key' in checker.categories(filename, key)


@pytest.mark.parametrize('prefix', ['/' + 'home/', '/' + 'Users/'])
def test_only_explicit_synthetic_test_user_is_allowed(prefix):
    fixture = (prefix + 'example/source').encode()
    personal = (prefix + 'actual-person/source').encode()
    assert 'personal-path' not in checker.categories('tests/python/test_fixture.py', fixture)
    assert 'personal-path' in checker.categories('docs/validation.json', fixture)
    assert 'personal-path' in checker.categories('tests/python/test_fixture.py', personal)
    assert 'personal-path' in checker.categories('tools/publication_report.py', personal)


def test_gpu_uuid_and_process_evidence():
    gpu = ('GPU-' + '12345678-1234-1234-1234-123456789abc').encode()
    assert 'gpu-identifier' in checker.categories('tests/test_fixture.py', gpu)
    assert 'gpu-identifier' in checker.categories('benchmarks/results/run.json', gpu)
    zero = ('GPU-' + '00000000-0000-0000-0000-000000000000').encode()
    assert not checker.categories('tests/test_fixture.py', zero)
    assert 'gpu-identifier' in checker.categories('docs/run.md', zero)
    for value in (b'{"pid": 24680}', b'PID 24680', b'pid=24680'):
        assert 'process-identifier' in checker.categories('docs/run.md', value)
    assert not checker.categories('docs/run.md', b'PID <redacted>')
    assert not checker.categories('tools/example.py', b'pid = os.getpid()')


def repository(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    return tmp_path


def scan(root, *args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=root,
                          capture_output=True, text=True)


def commit_fixture(root, parent=None):
    # Only the temporary test repository gets these synthetic commit objects.
    environment = {**os.environ, 'GIT_AUTHOR_NAME': 'Publication Fixture',
                   'GIT_AUTHOR_EMAIL': 'fixture@example.invalid',
                   'GIT_COMMITTER_NAME': 'Publication Fixture',
                   'GIT_COMMITTER_EMAIL': 'fixture@example.invalid'}
    subprocess.run(['git', 'add', '-A'], cwd=root, check=True)
    tree = subprocess.check_output(['git', 'write-tree'], cwd=root).decode().strip()
    command = ['git', 'commit-tree', tree, '-m', 'publication fixture']
    for parent_commit in (parent if isinstance(parent, tuple) else (parent,) if parent else ()):
        command.extend(['-p', parent_commit])
    commit = subprocess.check_output(command, cwd=root, env=environment).decode().strip()
    subprocess.run(['git', 'update-ref', 'refs/heads/main', commit], cwd=root, check=True)
    return commit


def test_current_scans_untracked_ignores_ignored_and_never_prints_values(tmp_path):
    root = repository(tmp_path)
    (root / '.gitignore').write_text('ignored/\n')
    (root / 'ignored').mkdir()
    private = '/' + 'home/actual-person/private'
    (root / 'ignored/secret.md').write_text(private)
    (root / 'report.md').write_text(private)
    result = scan(root)
    assert result.returncode == 1
    assert result.stdout == 'personal-path: "report.md"\n'
    assert not result.stderr and private not in result.stdout
    (root / 'report.md').unlink()
    assert scan(root).returncode == 0


def test_tracked_ignored_files_are_still_checked(tmp_path):
    root = repository(tmp_path)
    (root / '.gitignore').write_text('.env\n')
    (root / '.env').write_text('SYNTHETIC=1')
    subprocess.run(['git', 'add', '-f', '.env'], cwd=root, check=True)
    assert scan(root).stdout == 'private-file: ".env"\n'


def test_symlink_checks_target_without_reading_external_file(tmp_path):
    root = repository(tmp_path / 'repo')
    external = tmp_path / 'outside'
    external.write_text('/' + 'home/actual-person/secret')
    (root / 'external-link').symlink_to(external)
    assert scan(root).returncode == 0  # Target name is public; contents aren't in Git.
    (root / 'private-link').symlink_to('/' + 'home/actual-person/secret')
    assert scan(root).stdout == 'personal-path: "private-link"\n'


def test_history_finds_deleted_private_file_on_another_ref(tmp_path):
    root = repository(tmp_path)
    (root / '.env').write_text('SYNTHETIC=1')
    old = commit_fixture(root)
    subprocess.run(['git', 'update-ref', 'refs/tags/historical', old], cwd=root, check=True)
    (root / '.env').unlink()
    (root / 'README.md').write_text('clean')
    commit_fixture(root)  # Unrelated clean root; old history survives only on tag.
    assert scan(root).returncode == 0
    result = scan(root, '--history')
    assert result.returncode == 1
    assert result.stdout == 'private-file: ".env"\n'


def test_history_tracks_all_blob_paths_before_allowing_test_fixture(tmp_path):
    root = repository(tmp_path)
    (root / 'tests').mkdir()
    (root / 'tests/fixture.py').write_text('/' + 'home/example/source')
    first = commit_fixture(root)
    (root / 'docs').mkdir()
    (root / 'docs/report.md').write_bytes((root / 'tests/fixture.py').read_bytes())
    second = commit_fixture(root, first)
    (root / 'docs/report.md').unlink()
    commit_fixture(root, second)
    assert scan(root).returncode == 0
    result = scan(root, '--history')
    assert result.returncode == 1
    assert result.stdout == 'personal-path: "docs/report.md"\n'


def test_errors_do_not_echo_git_diagnostics_or_private_paths(tmp_path):
    result = scan(tmp_path)
    assert result.returncode == 2
    assert result.stdout == 'scan-error: "<repository>"\n' and not result.stderr


def test_history_checks_reused_blob_introduced_only_by_merge(tmp_path):
    root = repository(tmp_path)
    (root / 'tests').mkdir()
    (root / 'tests/fixture.py').write_text('/' + 'home/example/source')
    base = commit_fixture(root)
    (root / 'left.txt').write_text('left')
    left = commit_fixture(root, base)
    (root / 'left.txt').unlink()
    (root / 'right.txt').write_text('right')
    right = commit_fixture(root, base)
    (root / 'left.txt').write_text('left')
    (root / 'docs').mkdir()
    (root / 'docs/report.md').write_bytes((root / 'tests/fixture.py').read_bytes())
    (root / '.env').write_bytes((root / 'tests/fixture.py').read_bytes())
    merge = commit_fixture(root, (left, right))
    (root / 'docs/report.md').unlink()
    (root / '.env').unlink()
    commit_fixture(root, merge)
    assert scan(root).returncode == 0
    result = scan(root, '--history')
    assert result.returncode == 1
    assert result.stdout == ('personal-path: ".env"\n'
                             'personal-path: "docs/report.md"\n'
                             'private-file: ".env"\n')
