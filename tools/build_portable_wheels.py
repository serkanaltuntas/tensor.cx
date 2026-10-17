"""Build and audit Linux CPU wheels from an sdist with pinned cibuildwheel/images.

Never publishes a release. The completion manifest is written only after every
selected ABI has passed cibuildwheel's isolated installed-wheel tests and audit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile

from publication_report import write_public_report
from validate_distribution import audit_archive_members, audit_wheel_notices, only, run, sha256

ROOT = Path(__file__).resolve().parents[1]
CIBUILDWHEEL = '4.3.1'
PYTHONS = ('3.11', '3.12', '3.13')
RUNTIME_IMAGE = 'almalinux@sha256:8b469a3a78515e8a18ea8fc727e6a3679e1d0c5ba6f58d5b30be3d0d9b86cffe'


def clean_environment():
    # Overrides must not silently change target/backend, skip tests or inject
    # shell commands into an otherwise accepted portability run.
    env = {k: v for k, v in os.environ.items() if not k.startswith('CIBW_')}
    for key in ('VIRTUAL_ENV', 'PYTHONPATH', 'PYTHONHOME', 'CMAKE_ARGS', 'CC', 'CXX',
                'CFLAGS', 'CXXFLAGS', 'LDFLAGS', 'LD_LIBRARY_PATH', 'LD_PRELOAD',
                'PYTEST_ADDOPTS', 'PYTEST_PLUGINS'):
        env.pop(key, None)
    return env


def inspect_wheels(directory, versions):
    wheels = sorted(directory.glob('*.whl'))
    expected = {'cp' + version.replace('.', '') for version in versions}
    found, records = set(), []
    for wheel in wheels:
        pieces = wheel.name.removesuffix('.whl').split('-')
        if len(pieces) != 5:
            raise RuntimeError('unexpected portable wheel filename')
        _, _, interpreter, abi, platforms = pieces
        if (interpreter not in expected or interpreter in found or abi != interpreter
                or 'manylinux_2_28_x86_64' not in platforms.split('.')):
            raise RuntimeError('unexpected or duplicate portable wheel target')
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            audit_archive_members(names)
            audit_wheel_notices(archive)
            metadata = [name for name in names if name.endswith('.dist-info/WHEEL')]
            required_tag = f'Tag: {interpreter}-{abi}-manylinux_2_28_x86_64'
            if len(metadata) != 1 or required_tag not in archive.read(metadata[0]).decode().splitlines():
                raise RuntimeError('wheel metadata does not match its portability tag')
            if (len([name for name in names if name.startswith('tensorcx/_core.') and name.endswith('.so')]) != 1
                    or 'tensorcx_doctor.py' not in names
                    or 'tensorcx/_compiler/cache.py' not in names
                    or any(name.endswith('.pth') or 'editable' in name for name in names)
                    or not any(name.endswith('/licenses/LICENSE') for name in names)):
                raise RuntimeError('portable wheel is incomplete or editable')
        found.add(interpreter)
        records.append({'wheel': wheel.name, 'sha256': sha256(wheel), 'python_tag': interpreter,
                        'platform': 'manylinux_2_28_x86_64'})
    if found != expected:
        raise RuntimeError('portable run is missing a requested Python ABI')
    return records



def check_audit(audit):
    audit = ' '.join(audit.split())
    match = re.search(r'is consistent with the following platform tag:\s*"manylinux_(\d+)_(\d+)_x86_64"', audit)
    if match is None or tuple(map(int, match.groups())) > (2, 28):
        raise RuntimeError('auditwheel did not confirm glibc 2.28 or older x86_64 compatibility')
    if any(name in audit.lower() for name in ('libcuda', 'libcudart', 'libllvm', 'libmlir')):
        raise RuntimeError('portable CPU wheel has a GPU/LLVM dependency')


def check_clean_runtime(record, wheels, checks, output, env):
    uv = shutil.which('uv')
    if uv is None:
        raise RuntimeError('uv is required for the clean runtime check')
    script = """
uv venv /tmp/consumer --python "$2"
uv pip install --python /tmp/consumer/bin/python "$1"
uv pip check --python /tmp/consumer/bin/python
/tmp/consumer/bin/python -I /checks/probe.py --variant cpu
/tmp/consumer/bin/python -I /checks/mlp.py --device cpu --repeats 2
/tmp/consumer/bin/python -I -m tensorcx_doctor --json
/tmp/consumer/bin/tensorcx-doctor --json
getconf GNU_LIBC_VERSION
ldd /tmp/consumer/lib/python*/site-packages/tensorcx/_core*.so
"""
    version = '3.' + record['python_tag'].removeprefix('cp3')
    run(['docker', 'run', '--rm', '--read-only', '--tmpfs', '/tmp:exec',
         '--mount', f'type=bind,source={wheels},target=/wheels,readonly',
         '--mount', f'type=bind,source={checks},target=/checks,readonly',
         '--mount', f'type=bind,source={Path(uv).resolve()},target=/usr/local/bin/uv,readonly',
         '-e', 'UV_CACHE_DIR=/tmp/uv-cache', '-e', 'UV_PYTHON_INSTALL_DIR=/tmp/pythons',
         '-e', 'UV_PYTHON_PREFERENCE=only-managed', '-e', 'TENSORCX_LLVM_BIN=/missing-llvm',
         RUNTIME_IMAGE, 'bash', '-ec', script, 'tensorcx-runtime', '/wheels/' + record['wheel'], version],
        ROOT, env, output / (record['python_tag'] + '-clean-runtime.log'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new output directory')
    parser.add_argument('--python', choices=PYTHONS, action='append', dest='versions',
                        help='one or more supported CPython versions; default: all three')
    args = parser.parse_args(argv)
    if sys.platform != 'linux' or platform.machine() != 'x86_64':
        parser.error('portable CPU build runner currently requires Linux x86_64 and Docker')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    versions = sorted(set(args.versions or PYTHONS))
    env = clean_environment()
    env['CIBW_BUILD'] = ' '.join('cp' + version.replace('.', '') + '-manylinux_x86_64' for version in versions)
    source, wheels = output / 'source', output / 'wheels'
    source.mkdir()
    wheels.mkdir()
    started = datetime.now(timezone.utc).isoformat()
    run(['uv', 'build', '--sdist', '--out-dir', source], ROOT, env, output / 'sdist.log')
    sdist = only(source, '*.tar.gz')
    with tarfile.open(sdist) as archive:
        audit_archive_members(archive.getnames())
    # Only the immutable source archive enters cibuildwheel's build/test tree.
    run(['uv', 'run', '--no-project', '--python', sys.executable, '--with', 'cibuildwheel==' + CIBUILDWHEEL,
         'python', '-m', 'cibuildwheel', '--platform', 'linux', '--output-dir', wheels, sdist],
        ROOT, env, output / 'cibuildwheel.log')
    records = inspect_wheels(wheels, versions)
    with tarfile.open(sdist) as archive:
        name = next(n for n in archive.getnames() if n.count('/') == 1 and n.endswith('/pyproject.toml'))
        config = tomllib.loads(archive.extractfile(name).read().decode())
    image = config['tool']['cibuildwheel']['linux']['manylinux-x86_64-image']
    if not image.startswith('quay.io/pypa/manylinux_2_28_x86_64@sha256:'):
        raise RuntimeError('portable build image must be pinned by digest')
    checks = output / 'runtime-checks'
    checks.mkdir()
    with tarfile.open(sdist) as archive:
        for source_name, target_name in (('tools/distribution_probe.py', 'probe.py'),
                                         ('examples/cuda_mlp.py', 'mlp.py')):
            member = next(n for n in archive.getnames() if n.partition('/')[2] == source_name)
            (checks / target_name).write_bytes(archive.extractfile(member).read())
    for record in records:
        log = output / (record['python_tag'] + '-auditwheel.log')
        audit = run(['docker', 'run', '--rm', '--mount',
                     f'type=bind,source={wheels},target=/wheels,readonly', image,
                     'auditwheel', 'show', '/wheels/' + record['wheel']], ROOT, env, log)
        check_audit(audit)
        record['auditwheel'] = audit
        check_clean_runtime(record, wheels, checks, output, env)
    write_public_report(output / 'manifest.json', {
        'schema_version': 1, 'started_utc': started,
        'finished_utc': datetime.now(timezone.utc).isoformat(),
        'variant': 'cpu', 'python_versions': versions, 'cibuildwheel': CIBUILDWHEEL,
        'build_image': image, 'runtime_image': RUNTIME_IMAGE, 'runtime_uv': run(['uv', '--version'], ROOT, env, output / 'uv.log'),
        'sdist': {'file': sdist.name, 'sha256': sha256(sdist)},
        'wheels': records, 'checks': ['installed-wheel-full-pytest', 'cpu-probe', 'doctor-module',
                                     'doctor-cli', 'cpu-mlp', 'auditwheel-repair', 'auditwheel-show', 'glibc-2.28-clean-runtime'],
    }, root=ROOT)
    print(f'validated portable CPU wheels: {output}')


if __name__ == '__main__':
    main()
