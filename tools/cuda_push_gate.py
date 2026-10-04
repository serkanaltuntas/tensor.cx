"""Nightblade CUDA acceptance for the exact commit being pushed to origin/main."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
try:
    from publication_report import write_public_report
finally:
    sys.path.pop(0)
TEST_FILES = ('test_cuda.py', 'test_cuda_primitives.py', 'test_mlir_cuda_runtime.py',
              'test_mlir_cuda_expressions.py', 'test_tensor_api.py', 'test_cast_broadcast.py')


def push_revisions(lines):
    revisions = []
    for line in lines.splitlines():
        fields = line.split()
        if len(fields) != 4:
            raise RuntimeError('malformed pre-push input')
        _, local_sha, remote_ref, _ = fields
        if remote_ref != 'refs/heads/main':
            continue
        if not re.fullmatch(r'[0-9a-f]{40}', local_sha) or local_sha == '0' * 40:
            raise RuntimeError('main requires a valid non-deletion commit')
        if local_sha not in revisions:
            revisions.append(local_sha)
    return revisions


def run(command, cwd, env, log):
    try:
        result = subprocess.run([str(x) for x in command], cwd=cwd, env=env,
                                text=True, capture_output=True, timeout=3600)
    except subprocess.TimeoutExpired as error:
        def decoded(value):
            return value.decode(errors='replace') if isinstance(value, bytes) else (value or '')
        log.write_text(decoded(error.stdout) + decoded(error.stderr))
        raise RuntimeError(f'command timed out; partial output: {log}') from error
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f'command failed ({result.returncode}); see {log}')
    return result.stdout.strip()


def junit_result(path, minimum, required_names=()):
    cases = ET.parse(path).getroot().findall('.//testcase')
    if len(cases) < minimum:
        raise RuntimeError(f'{path}: expected at least {minimum} tests, found {len(cases)}')
    for case in cases:
        if any(case.find(tag) is not None for tag in ('skipped', 'failure', 'error')):
            raise RuntimeError(f'{path}: non-passing test {case.get("name")}')
    names = {case.get('name', '').split('[')[0] for case in cases}
    if not set(required_names) <= names:
        raise RuntimeError(f'{path}: missing required acceptance tests')
    return {'tests': len(cases), 'skipped': 0, 'failures': 0, 'errors': 0}


def validate(revision, llvm_bin):
    env = os.environ.copy()
    # Git exports repository-local variables to hooks. Do not let build tools
    # operating on the extracted snapshot rediscover the developer checkout.
    for key in subprocess.check_output(['git', 'rev-parse', '--local-env-vars'], cwd=ROOT, text=True).splitlines():
        env.pop(key, None)
    for key in ('VIRTUAL_ENV', 'PYTHONPATH', 'PYTHONHOME', 'PYTEST_ADDOPTS',
                'PYTEST_PLUGINS', 'CMAKE_ARGS', 'LD_LIBRARY_PATH', 'LD_PRELOAD'):
        env.pop(key, None)
    env.update(CC='gcc-13', CXX='g++-13', CUDACXX='nvcc', CUDAHOSTCXX='g++-13',
               CMAKE_BUILD_PARALLEL_LEVEL='2', TENSORCX_LLVM_BIN=str(llvm_bin),
               TENSORCX_REQUIRE_BACKENDS='cuda', TENSORCX_REQUIRE_CUDA='1',
               TENSORCX_REQUIRE_MLIR='1', TENSORCX_REQUIRE_MLIR_CUDA='1',
               TENSORCX_REQUIRE_BACKEND_CAPABILITIES='cuda:copy,cuda:tensor_factories_float32,'
               'cuda:binary_ops_float32,cuda:unary_float32,cuda:reductions_float32,cuda:normalization_float32')
    resolved = subprocess.check_output(['git', 'rev-parse', '--verify', '--end-of-options', revision + '^{commit}'],
                                       cwd=ROOT, text=True).strip()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = ROOT / 'build/cuda-gate' / (stamp + '-' + resolved[:12])
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'revision': resolved, 'started_utc': stamp,
              'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'pytest_controller_sha256': hashlib.sha256((ROOT / 'tools/cuda_gate_pytest.py').read_bytes()).hexdigest(),
              'status': 'running'}
    print(f'CUDA gate: {resolved}; logs: {output}', flush=True)
    try:
        # Archive the commit, never the mutable working tree. All compilation
        # and test files come from this snapshot in a disposable environment.
        with tempfile.TemporaryDirectory(prefix='tensorcx-cuda-gate-') as temporary:
            work = Path(temporary)
            snapshot = work / 'source'
            snapshot.mkdir()
            archive = work / 'source.tar'
            run(['git', 'archive', '--format=tar', '-o', archive, resolved], ROOT, env, output / 'archive.log')
            report['source_archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
            with tarfile.open(archive) as source:
                source.extractall(snapshot, filter='data')
            python = work / 'env/bin/python'
            run(['uv', 'venv', work / 'env', '--python', sys.executable], snapshot, env, output / 'venv.log')
            build_env = {**env, 'CMAKE_ARGS': '-DTENSORCX_ENABLE_METAL=OFF -DTENSORCX_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=52'}
            run(['uv', 'pip', 'install', '--python', python, str(snapshot) + '[dev]', 'cmake>=3.21', 'ninja'],
                snapshot, build_env, output / 'install.log')
            runner = ['uv', 'run', '--no-project', '--python', python]
            report['dependencies'] = run(['uv', 'pip', 'freeze', '--python', python], snapshot, env, output / 'dependencies.log')
            report['gpu'] = run(['nvidia-smi', '--query-gpu=name,compute_cap,driver_version', '--format=csv'],
                                snapshot, env, output / 'gpu.log')
            report['nvcc'] = run(['nvcc', '--version'], snapshot, env, output / 'nvcc.log')
            # Ignore PYTEST_ADDOPTS and fail on every skip/xfail in the selected
            # CUDA acceptance suite. Anchors prevent accidentally empty scopes.
            pytest_controller = work / 'strict_pytest.py'
            pytest_controller.write_bytes((ROOT / 'tools/cuda_gate_pytest.py').read_bytes())
            run([*runner, 'python', '-I', pytest_controller, '-q', '-o', 'xfail_strict=true',
                 *[snapshot / 'tests/python' / name for name in TEST_FILES],
                 '--junitxml', output / 'pytest.xml'], snapshot, env, output / 'pytest.log')
            report['python'] = junit_result(output / 'pytest.xml', 595,
                ('test_cuda_registry_discovery', 'test_matmul_errors', 'test_parity',
                 'test_gpu_emitter_expression_fixture', 'test_arithmetic_float32_contract',
                 'test_keepdims_matches_numpy_and_methods',
                 'test_astype_values_shape_device_and_storage',
                 'test_broadcast_float32_numpy_and_cpu_parity'))
            nanobind = run([*runner, 'python', '-I', '-c', 'import nanobind; print(nanobind.cmake_dir())'],
                           snapshot, env, output / 'nanobind.log')
            native = work / 'native'
            run([*runner, 'cmake', '-S', snapshot, '-B', native, '-G', 'Ninja',
                 '-DTENSORCX_ENABLE_METAL=OFF', '-DTENSORCX_ENABLE_CUDA=ON', '-DCMAKE_CUDA_ARCHITECTURES=52',
                 '-DTENSORCX_BUILD_TESTS=ON', '-Dnanobind_DIR=' + nanobind, '-DPython_EXECUTABLE=' + str(python)],
                snapshot, env, output / 'configure.log')
            run([*runner, 'cmake', '--build', native], snapshot, env, output / 'build.log')
            run([*runner, 'ctest', '--test-dir', native, '--no-tests=error', '--output-on-failure',
                 '--output-junit', output / 'ctest.xml'], snapshot, env, output / 'ctest.log')
            report['native'] = junit_result(output / 'ctest.xml', 4,
                ('tensorcx_cpu_kernel_tests', 'tensorcx_backend_contract_tests',
                 'tensorcx_cuda_kernel_tests', 'tensorcx_cuda_backend_contract_tests'))
            for tool in ('memcheck', 'racecheck'):
                run(['compute-sanitizer', '--tool', tool, '--error-exitcode', '1',
                     native / 'tensorcx_cuda_backend_contract_tests'], snapshot, env, output / (tool + '.log'))
            report['device_sanitizers'] = ['memcheck', 'racecheck']
            raw = run([*runner, 'python', '-I', snapshot / 'examples/cuda_mlp.py', '--device', 'cuda', '--repeats', '3'],
                      snapshot, env, output / 'mlp.log')
            report['mlp'] = json.loads(raw)
            if (report['mlp'].get('device') != 'cuda' or report['mlp'].get('cpu_parity') is not True
                    or report['mlp'].get('shape') != [32, 10]):
                raise RuntimeError('MLP did not report the required CUDA workload and CPU parity')
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_public_report(output / 'result.json', report, root=ROOT)
    print(f'CUDA gate passed: {resolved}; {output / "result.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--revision')
    mode.add_argument('--pre-push', action='store_true')
    parser.add_argument('--llvm-bin', type=Path)
    args = parser.parse_args()
    revisions = push_revisions(sys.stdin.read()) if args.pre_push else [args.revision]
    if not revisions:
        return
    llvm = args.llvm_bin
    if llvm is None:
        config = subprocess.run(['git', 'config', '--get', 'tensorcx.cudaLlvmBin'],
                                cwd=ROOT, text=True, capture_output=True)
        if config.returncode or not config.stdout.strip():
            parser.error('configure tensorcx.cudaLlvmBin or provide --llvm-bin')
        llvm = Path(config.stdout.strip())
    for revision in revisions:
        validate(revision, llvm.resolve())


if __name__ == '__main__':
    main()
