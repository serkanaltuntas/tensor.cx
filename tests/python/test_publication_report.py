"""Known metadata privacy boundaries, without claiming arbitrary secret removal."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'tools/publication_report.py'
spec = importlib.util.spec_from_file_location('publication_report_tests', SCRIPT)
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


def test_public_export_preserves_measurements_removes_identifiers_and_urls():
    private = {
        'metadata': {'git_status': ' M confidential-name.txt',
                     'gpu_uuid': 'GPU-00000000-0000-0000-0000-000000000000', 'pid': 1234,
                     'cuda_visible_devices': 'GPU-00000000-0000-0000-0000-000000000000',
                     'gpu': 'Model, GPU-00000000-0000-0000-0000-000000000000, 5.2, 580.1',
                     'tool': 'clang version 21.1.8\nInstalledDir: /Users/example/work/llvm/bin'},
        'dependencies': 'numpy==2.5.3\ntensorcx @ file:///home/example/project/wheel.whl\n-e https://user:private@example.invalid/source',
        'results': [{'samples_ms': [1.2, 1.4], 'cpu_parity': True, 'memory_mib': 400}],
    }
    result = publication.public_report(private)
    raw = json.dumps(result)
    for value in ('GPU-00000000-0000-0000-0000-000000000000', 'confidential-name', '1234', '/Users/', '/home/', 'private@', 'example.invalid'):
        assert value not in raw
    assert result['metadata']['git_dirty'] is True
    assert result['metadata']['gpu'] == 'Model, <gpu>, 5.2, 580.1'
    assert 'clang version 21.1.8' in result['metadata']['tool']
    assert 'numpy==2.5.3' in result['dependencies']
    assert result['results'] == private['results']
    assert 'gpu_uuid' in private['metadata']  # Source data is not mutated.
    assert publication.public_report(result) == result


@pytest.mark.parametrize('path', ['/home/example/source/file', '/Users/example/source/file',
                                   '/tmp/tensorcx-build/private', '/private/var/folders/ab/private',
                                   'file:///home/example/source.whl'])
def test_historical_personal_and_temporary_paths(path):
    cleaned = publication.sanitize_text(f'error at {path}\nversion 21.1.8')
    assert path not in cleaned
    assert 'version 21.1.8' in cleaned


def test_custom_roots_and_system_library_paths():
    result = publication.public_report({'error': '/mnt/private-checkout/build/secret.log',
                                        'linkage': '/usr/lib/libcuda.so.1'}, root='/mnt/private-checkout')
    assert 'private-checkout' not in result['error']
    assert 'secret.log' not in result['error']
    assert result['linkage'] == '/usr/lib/libcuda.so.1'


def test_cli_exports_only_json_and_leaves_original(tmp_path):
    source = tmp_path / 'raw.json'
    source.write_text(json.dumps({'status': 'failed', 'error': '/home/example/check.log'}))
    destination = tmp_path / 'public.json'
    result = subprocess.run([sys.executable, str(SCRIPT), str(source), '--output', str(destination)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(destination.read_text())
    assert report['status'] == 'failed'
    assert report['publication']['format'] == 'cortex-public-report-v1'
    assert '/home/example' not in destination.read_text()
    assert '/home/example' in source.read_text()


def test_does_not_claim_to_remove_arbitrary_secrets():
    result = publication.public_report({'diagnostic': 'custom opaque value'})
    assert result['diagnostic'] == 'custom opaque value'
    assert 'arbitrary text requires review' in result['publication']['scope']


def test_tool_installation_location_and_unrelated_prefix():
    assert publication.sanitize_text('clang version 21\nInstalledDir: /opt/private-sdk/bin') == (
        'clang version 21\nInstalledDir: <toolchain>')
    assert publication.sanitize_text('/opt/workhorse', root='/opt/work') == '/opt/workhorse'


def test_gpu_prose_is_preserved_while_uuid_is_redacted():
    message = 'GPU-free CI and GPU-first runtime: GPU-00000000-0000-0000-0000-000000000000'
    assert publication.sanitize_text(message) == 'GPU-free CI and GPU-first runtime: <gpu>'
