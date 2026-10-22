"""Build pinned Vulkan shader tools inside the portable wheel build container.

These are build dependencies, never wheel contents. No driver or loader is
installed. Source archives are pinned by both upstream commit and SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request


# SPIRV-Headers is the revision required by SPIRV-Tools v2026.1's DEPS.
SOURCES = {
    'Vulkan-Headers': ('b5c8f996196ba4aa6d8f97e52b5d3b6e70f7e4e2',
                       '7937d8ec21785736436d9ff52564db7d55992070b5ac98324472cce6e425a7ba'),
    'glslang': ('f0bd0257c308b9a26562c1a30c4748a0219cc951',
                'bd58dca4dac67dcf7640292d7d63e0416274d40ee2200f7301878cec11ac6647'),
    'SPIRV-Tools': ('fbe4f3ad913c44fe8700545f8ffe35d1382b7093',
                    'cabb35f4eef0da3ef72ad9edd596af4191d7507a8f35c05df526d2d5ff889f59'),
    'SPIRV-Headers': ('04f10f650d514df88b76d25e83db360142c7b174',
                      '1b220e3eec1714f0451b0e3652979bd280edf10893f617837b88e6359a804ded'),
}


def source(name, directory):
    revision, expected = SOURCES[name]
    archive_path = directory / (name + '.tar.gz')
    url = f'https://codeload.github.com/KhronosGroup/{name}/tar.gz/{revision}'
    with urllib.request.urlopen(url, timeout=60) as response, archive_path.open('xb') as target:
        shutil.copyfileobj(response, target)
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'{name} source archive checksum mismatch')
    target = directory / name
    target.mkdir()
    with tarfile.open(archive_path) as archive:
        archive.extractall(target, filter='data')
    roots = list(target.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise RuntimeError(f'{name} source archive has an unexpected root')
    return roots[0]


def build(source_dir, build_dir, options, target):
    subprocess.run(['cmake', '-S', str(source_dir), '-B', str(build_dir),
                    '-G', 'Unix Makefiles', '-DCMAKE_BUILD_TYPE=Release',
                    '-DCMAKE_C_FLAGS=-O2 -march=x86-64 -mtune=generic',
                    '-DCMAKE_CXX_FLAGS=-O2 -march=x86-64 -mtune=generic', *options], check=True)
    subprocess.run(['cmake', '--build', str(build_dir), '--target', target,
                    '--parallel', '2'], check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, required=True, help='new SDK directory')
    args = parser.parse_args(argv)
    prefix = args.prefix.resolve()
    prefix.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='tensorcx-vulkan-sdk-') as temporary:
        work = Path(temporary)
        headers = source('Vulkan-Headers', work)
        glslang = source('glslang', work)
        tools = source('SPIRV-Tools', work)
        spirv_headers = source('SPIRV-Headers', work)
        glslang_build, tools_build = work / 'glslang-build', work / 'tools-build'
        build(glslang, glslang_build,
              ['-DENABLE_OPT=OFF', '-DGLSLANG_TESTS=OFF', '-DBUILD_EXTERNAL=OFF',
               '-DENABLE_GLSLANG_BINARIES=ON', '-DBUILD_SHARED_LIBS=OFF'],
              'glslang-standalone')
        build(tools, tools_build,
              ['-DSPIRV_SKIP_TESTS=ON', '-DSPIRV_WERROR=OFF', '-DBUILD_SHARED_LIBS=OFF',
               '-DSPIRV_TOOLS_BUILD_STATIC=ON', f'-DSPIRV-Headers_SOURCE_DIR={spirv_headers}'],
              'spirv-val')
        binaries = prefix / 'bin'
        binaries.mkdir()
        shutil.copy2(glslang_build / 'StandAlone/glslang', binaries / 'glslangValidator')
        shutil.copy2(tools_build / 'tools/spirv-val', binaries / 'spirv-val')
        shutil.copytree(headers / 'include', prefix / 'include')
        # Exercise the installed build tools together before wheel compilation.
        shader = work / 'probe.comp'
        shader.write_text('#version 450\nlayout(local_size_x=64) in;\nvoid main() {}\n')
        binary = work / 'probe.spv'
        subprocess.run([str(binaries / 'glslangValidator'), '-V', '--target-env', 'vulkan1.1',
                        str(shader), '-o', str(binary)], check=True)
        subprocess.run([str(binaries / 'spirv-val'), '--target-env', 'vulkan1.1', str(binary)], check=True)
    print('Pinned Vulkan build headers and shader tools ready', flush=True)


if __name__ == '__main__':
    main()
