"""Build CPU/CUDA wheels from an sdist and validate fresh installs on Linux."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
try:
    from publication_report import write_public_report
finally:
    sys.path.pop(0)


def audit_archive_members(names):
    """Reject local caches and private files even if build includes override Git."""
    for name in names:
        path = Path(name)
        private_names = {'id_rsa', 'id_ed25519', '.netrc', '.npmrc', 'credentials',
                         'credentials.json', 'secrets.json'}
        if (any(part in {'__pycache__', '.git', '.venv', '.codex', '.claude', '.ssh', '.aws', '.gnupg'} for part in path.parts)
                or path.suffix.lower() in {'.pyc', '.pyo', '.key', '.pem', '.p12', '.pfx', '.bundle'}
                or path.name.lower() in private_names
                or path.name.lower() == '.env' or path.name.lower().startswith('.env.')):
            raise RuntimeError(f'private/cache file in distribution: {name}')


def audit_wheel_notices(archive):
    notices = [n for n in archive.namelist() if n.endswith('/licenses/THIRD_PARTY_NOTICES.md')]
    if len(notices) != 1:
        raise RuntimeError('wheel missing third-party notices')
    text = archive.read(notices[0]).decode()
    if not all(name in text for name in ('Wenzel Jakob', 'Thibaut Goetghebuer-Planchon', 'Apple Inc.', 'DLPack')):
        raise RuntimeError('wheel has incomplete third-party notices')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv, cwd, env, log):
    result = subprocess.run([str(a) for a in argv], cwd=cwd, env=env,
                            text=True, capture_output=True)
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f'{argv[0]} failed ({result.returncode}); see {log}')
    return result.stdout.strip()


def only(directory, pattern):
    files = list(directory.glob(pattern))
    if len(files) != 1:
        raise RuntimeError(f'expected one {pattern} in {directory}, got {len(files)}')
    return files[0]


def audit_linkage(linkage, variant, work):
    if 'not found' in linkage or (variant == 'cpu' and any(x in linkage.lower() for x in ('cuda', 'llvm', 'mlir'))):
        raise RuntimeError('unexpected or unresolved shared library dependency')
    for value in re.findall(r'=>\s+(/.*?)\s+\(0x[0-9a-fA-F]+\)', linkage):
        path = Path(value).resolve()
        # A library shipped within the fresh environment is legitimate. Other
        # source/temp/stub dependencies can hide an incomplete release artifact.
        bundled = path.is_relative_to(work / 'env')
        if ('stubs' in path.parts or 'cuda-driver-stub' in path.parts or
                (not bundled and (path.is_relative_to(ROOT) or
                                  path.is_relative_to(Path(tempfile.gettempdir()).resolve())))):
            raise RuntimeError(f'contaminated shared library resolution: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new directory; never overwrites a previous run')
    parser.add_argument('--variants', choices=['cpu', 'cuda', 'both'], default='both')
    parser.add_argument('--llvm-bin', type=Path, help='also require MLIR execution with this external LLVM toolchain')
    args = parser.parse_args()
    if platform.system() != 'Linux':
        parser.error('this distribution validation currently targets Linux')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CMAKE_ARGS', 'TENSORCX_LLVM_BIN',
                'LD_LIBRARY_PATH', 'LD_PRELOAD'):
        env.pop(key, None)
    env['CMAKE_BUILD_PARALLEL_LEVEL'] = '2'
    variants = ['cpu', 'cuda'] if args.variants == 'both' else [args.variants]
    manifest = {'schema_version': 1, 'started_utc': datetime.now(timezone.utc).isoformat(),
                'python': sys.version, 'platform': platform.platform(), 'variants': {},
                'validator_sha256': sha256(Path(__file__)),
                'probe_sha256': sha256(ROOT / 'tools/distribution_probe.py')}
    manifest['uv'] = run(['uv', '--version'], ROOT, env, output / 'uv.log')
    run(['uv', 'build', '--sdist', '--python', sys.executable, '--out-dir', output / 'source'],
        ROOT, env, output / 'sdist.log')
    sdist = only(output / 'source', '*.tar.gz')
    manifest['sdist'] = {'file': str(sdist.relative_to(output)), 'sha256': sha256(sdist)}
    with tarfile.open(sdist) as archive:
        members = archive.getnames()
        audit_archive_members(members)
        for required in ('CMakeLists.txt', 'pyproject.toml', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'bindings/python_module.cpp',
                         'cpp/tensorcx/backends/cuda/kernels/primitives.cu',
                         'python/tensorcx/_compiler/cuda.py', 'tools/distribution_probe.py',
                         'examples/cuda_mlp.py', 'bindings/dlpack.cpp',
                         'third_party/dlpack/dlpack.h', 'third_party/dlpack/LICENSE'):
            if not any(member.endswith('/' + required) for member in members):
                raise RuntimeError(f'sdist missing {required}')
        # Read the validation workload from the delivered sdist, not checkout.
        probe = archive.extractfile(next(m for m in members if m.endswith('/tools/distribution_probe.py'))).read()
        example = archive.extractfile(next(m for m in members if m.endswith('/examples/cuda_mlp.py'))).read()
    for variant in variants:
        print(f'building and validating {variant}', flush=True)
        destination = output / variant
        destination.mkdir()
        build_env = {**env, 'CMAKE_ARGS': '-DTENSORCX_ENABLE_METAL=OFF -DTENSORCX_ENABLE_CUDA=' +
                     ('ON -DCMAKE_CUDA_ARCHITECTURES=52' if variant == 'cuda' else 'OFF')}
        run(['uv', 'build', '--wheel', sdist, '--python', sys.executable, '--out-dir', destination],
            ROOT, build_env, destination / 'build.log')
        wheel = only(destination, '*.whl')
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            audit_archive_members(names)
            audit_wheel_notices(archive)
            if any(name.endswith('.pth') or 'editable' in name for name in names):
                raise RuntimeError('editable artifacts in wheel')
            for required in ('tensorcx/__init__.py', 'tensorcx/_compiler/cuda.py'):
                if required not in names:
                    raise RuntimeError(f'wheel missing {required}')
            if len([name for name in names if name.startswith('tensorcx/_core.') and name.endswith('.so')]) != 1:
                raise RuntimeError('wheel must contain one native extension')
            if not any(name.endswith('/licenses/LICENSE') for name in names):
                raise RuntimeError('wheel missing license')
        record = {'wheel': wheel.name, 'sha256': sha256(wheel), 'members': names, 'probes': [],
                  'cmake_args': build_env['CMAKE_ARGS'],
                  'compiler_environment': {key: env.get(key) for key in ('CC', 'CXX', 'CUDACXX', 'CUDAHOSTCXX')}}
        # The test interpreter and working directory are outside the checkout.
        # -I ignores Python path overrides, current directory and user packages.
        with tempfile.TemporaryDirectory(prefix='tensorcx-wheel-') as temporary:
            work = Path(temporary)
            python = work / 'env/bin/python'
            run(['uv', 'venv', work / 'env', '--python', sys.executable], ROOT, env, destination / 'venv.log')
            run(['uv', 'pip', 'install', '--python', python, wheel], ROOT, env, destination / 'install.log')
            run(['uv', 'pip', 'check', '--python', python], ROOT, env, destination / 'dependency-check.log')
            record['installed'] = run(['uv', 'pip', 'freeze', '--python', python], ROOT, env, destination / 'freeze.log')
            (work / 'probe.py').write_bytes(probe)
            (work / 'mlp.py').write_bytes(example)
            runtime_env = {**env, 'TENSORCX_LLVM_BIN': str(work / 'absent-llvm')}
            command = ['uv', 'run', '--no-project', '--python', python, 'python', '-I']
            modes = [('without-llvm', []), ('hidden-gpu', ['--hidden-gpu'])]
            if args.llvm_bin:
                modes.append(('with-llvm', ['--mlir']))
            for mode, flags in modes:
                selected_env = runtime_env.copy()
                if mode == 'hidden-gpu':
                    selected_env['CUDA_VISIBLE_DEVICES'] = ''
                if mode == 'with-llvm':
                    selected_env['TENSORCX_LLVM_BIN'] = str(args.llvm_bin.resolve())
                raw = run([*command, work / 'probe.py', '--variant', variant, *flags],
                          work, selected_env, destination / (mode + '.log'))
                record['probes'].append({'mode': mode, 'result': json.loads(raw)})
            extension = only(work / 'env/lib', 'python*/site-packages/tensorcx/_core*.so')
            linkage = run(['ldd', extension], work, env, destination / 'linkage.log')
            audit_linkage(linkage, variant, work)
            record['linkage'] = linkage
            dynamic = run(['readelf', '--dynamic', extension], work, env, destination / 'elf-dynamic.log')
            record['elf_needed'] = re.findall(r'\(NEEDED\).*?\[(.*?)\]', dynamic)
            record['elf_runpath'] = re.findall(r'\((?:RUNPATH|RPATH)\).*?\[(.*?)\]', dynamic)
            if any('/tmp/' in value or str(ROOT) in value or '/stubs' in value for value in record['elf_runpath']):
                raise RuntimeError('wheel references a temporary/source/stub library path')
            versions = run(['readelf', '--version-info', extension], work, env, destination / 'elf-versions.log')
            record['elf_symbol_versions'] = sorted(set(re.findall(r'\b(?:GLIBC|GLIBCXX|CXXABI)_[0-9.]+', versions)))
            raw = run([*command, work / 'mlp.py', '--device', variant, '--repeats', '3'],
                      work, runtime_env, destination / 'mlp.log')
            record['mlp'] = json.loads(raw)
        manifest['variants'][variant] = record
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_public_report(output / 'manifest.json', manifest, root=ROOT)
    print(f'validated artifacts and evidence: {output}')


if __name__ == '__main__':
    main()
