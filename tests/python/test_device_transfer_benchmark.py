"""Real placement benchmark validation and failure/report boundaries."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import tensorcx as cx

SCRIPT = Path(__file__).resolve().parents[2] / 'benchmarks/bench_device_transfer.py'
spec = importlib.util.spec_from_file_location('device_transfer_benchmark', SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


@pytest.mark.parametrize('dtype', ['float32', 'int32', 'bool'])
@pytest.mark.parametrize('source,target', [('cpu','cpu'), ('cuda','vulkan'), ('vulkan','cuda'), ('vulkan','vulkan')])
def test_real_transfer_benchmark_and_separate_profiles(source, target, dtype):
    if not cx.is_available(source) or not cx.is_available(target): pytest.skip('required benchmark device unavailable')
    report = benchmark.run(source=source, target=target, dtype=dtype, sizes=(0,3,257), repeats=2, warmup=1, profile=True)
    assert report['cpu_parity'] and report['source'] == source and report['target'] == target
    assert report['transfer_path'] == ('same_device_clone' if source == target else 'host_staged')
    assert report['metadata']['benchmark_sha256'] and report['metadata']['extension_sha256']
    for row in report['cases']:
        assert row['cpu_parity']
        assert set(row['profiles']) == set(row['timings']) == {'copy','pipeline'}
        for timing in row['timings'].values():
            assert len(timing['samples_ns']) == 2 and 0 < timing['min_ns'] <= timing['median_ns'] <= timing['max_ns']
        assert all(profile['report']['host_complete'] for profile in row['profiles'].values())


def test_archive_benchmark_without_git_and_output_preservation(tmp_path, monkeypatch):
    # Keep the archive case independent of TMPDIR being inside a checkout.
    monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path.parent.resolve()))
    monkeypatch.delenv('GIT_DIR', raising=False)
    monkeypatch.delenv('GIT_WORK_TREE', raising=False)
    copied = tmp_path/'bench.py'; copied.write_bytes(SCRIPT.read_bytes())
    output = tmp_path/'report.json'
    args = [sys.executable, '-I', str(copied), '--source', 'cpu', '--target', 'cpu',
            '--sizes', '0,3', '--repeats', '1', '--warmup', '0', '--output', str(output)]
    child = subprocess.run(args, capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stderr
    report = json.loads(output.read_text())
    assert report == json.loads(child.stdout) and report['cpu_parity']
    assert report['metadata']['revision'] is None and report['metadata']['git_dirty'] is None
    before = output.read_bytes()
    repeated = subprocess.run(args, capture_output=True, text=True, timeout=60)
    assert repeated.returncode != 0 and output.read_bytes() == before


def test_unavailable_benchmark_does_not_fall_back_or_publish(tmp_path):
    output = tmp_path/'failure.json'
    child = subprocess.run([sys.executable, str(SCRIPT), '--source', 'missing', '--target', 'cpu',
                            '--repeats','1','--sizes','3','--output',str(output)],
                           capture_output=True, text=True, timeout=60)
    assert child.returncode != 0 and not output.exists() and not child.stdout


def test_report_created_during_benchmark_is_preserved(tmp_path, monkeypatch, capsys):
    output = tmp_path / 'report.json'
    existing = '{"other_run": true}\n'

    def competing_run(**kwargs):
        output.write_text(existing)
        return {'cpu_parity': True}

    monkeypatch.setattr(benchmark, 'run', competing_run)
    with pytest.raises(FileExistsError):
        benchmark.main(['--source', 'cpu', '--target', 'cpu', '--output', str(output)])
    assert output.read_text() == existing
    assert not capsys.readouterr().out
