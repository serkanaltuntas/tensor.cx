"""Check known publication hazards in Git-visible files or reachable history.

This narrow privacy/file-policy check complements Gitleaks; it is not a generic
secret detector. Diagnostics contain categories and repository filenames only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import subprocess


PRIVATE_DIRECTORIES = {'.ssh', '.gnupg', '.aws', '.codex', '.claude'}
CACHE_DIRECTORIES = {'__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache',
                     '.cache', '.venv', 'venv', 'CMakeFiles', '_skbuild', 'build', 'dist'}
PRIVATE_NAMES = {'id_rsa', 'id_ed25519', 'id_dsa', 'id_ecdsa', '.netrc', '.npmrc',
                 'credentials.json', 'credentials', 'secrets.json'}
PRIVATE_SUFFIXES = {'.pem', '.key', '.p12', '.pfx', '.gpg', '.bundle'}
CACHE_SUFFIXES = {'.pyc', '.pyo', '.o', '.a', '.so', '.dylib', '.log', '.gguf', '.safetensors'}
PERSONAL_PATH = re.compile(r'/(?:home|Users)/([^/\s\"\'<>]+)(?=/|$|[\s\"\'<>])')
GPU_UUID = re.compile(r'\bGPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b')
PROCESS_ID = re.compile(r'(?:[\"\']pid[\"\']\s*:\s*|\bPID\s*(?:[:=]\s*)?)([0-9]+)\b', re.I)
PRIVATE_KEY = re.compile(r'-{5}BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-{5}')
SYNTHETIC_USERS = {'example', 'fixture-user', 'test-user'}


def categories(filename: str, data: bytes) -> set[str]:
    path = PurePosixPath(filename)
    parts, name = set(path.parts), path.name.lower()
    found = set()
    env_file = name == '.env' or name.startswith('.env.')
    if ((env_file and name not in {'.env.example', '.env.sample'}) or
            parts & PRIVATE_DIRECTORIES or name in PRIVATE_NAMES or
            path.suffix.lower() in PRIVATE_SUFFIXES):
        found.add('private-file')
    if parts & CACHE_DIRECTORIES or path.suffix.lower() in CACHE_SUFFIXES:
        found.add('cache-or-build-file')
    text = data.decode('utf-8', errors='replace')
    # Fixtures only exempt narrowly named synthetic identifiers, never key
    # headers or forbidden filenames. No whole-file content exclusions exist.
    fixture = path.parts[:1] == ('tests',)
    if any(not (fixture and match[1] in SYNTHETIC_USERS)
           for match in PERSONAL_PATH.finditer(text)):
        found.add('personal-path')
    if any(not (fixture and set(match[0][4:]) <= {'0', '-'})
           for match in GPU_UUID.finditer(text)):
        found.add('gpu-identifier')
    if (path.parts[:1] in {('docs',), ('benchmarks',)} or path.suffix.lower() == '.json'):
        if PROCESS_ID.search(text):
            found.add('process-identifier')
    if PRIVATE_KEY.search(text):
        found.add('private-key')
    return found


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.DEVNULL)


def current_files(root: Path):
    paths = git(root, 'ls-files', '-z', '--cached', '--others', '--exclude-standard')
    for raw in sorted(set(paths.split(b'\0')) - {b''}):
        name = raw.decode('utf-8', errors='surrogateescape')
        path = root / name
        if path.is_symlink():
            # Inspect the published link, never follow it into a private tree.
            yield name, str(path.readlink()).encode('utf-8', errors='surrogateescape')
        elif path.is_file():
            yield name, path.read_bytes()
        elif path.exists():
            raise ValueError('unsupported-git-entry')
        # A deleted tracked file is absent from the current publication tree.


def history_files(root: Path):
    objects = {}
    for row in git(root, 'rev-list', '--objects', '--all').splitlines():
        oid, _, name = row.partition(b' ')
        objects[oid] = {name.decode('utf-8', errors='surrogateescape')} if name else set()
    # rev-list assigns only one representative path to a reused blob. Collect
    # historical paths too, so moving a synthetic fixture into docs cannot hide it.
    changes = git(root, 'log', '--all', '--root', '--diff-merges=first-parent', '--format=', '--raw',
                  '--no-renames', '--no-abbrev', '-z').split(b'\0')
    for index, field in enumerate(changes[:-1]):
        if field.lstrip(b'\n').startswith(b':'):
            metadata = field.strip().split()
            if len(metadata) == 5 and metadata[3] in objects:
                objects[metadata[3]].add(changes[index + 1].decode('utf-8', errors='surrogateescape'))
    with subprocess.Popen(['git', 'cat-file', '--batch'], cwd=root, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as process:
        for oid, names in objects.items():
            process.stdin.write(oid + b'\n')
            process.stdin.flush()
            header = process.stdout.readline().split()
            if len(header) != 3:
                raise ValueError('history-object-unreadable')
            size = int(header[2])
            data = process.stdout.read(size)
            if len(data) != size or process.stdout.read(1) != b'\n':
                raise ValueError('history-object-unreadable')
            if header[1] == b'blob':
                for name in sorted(names or {'<unknown-path>'}):
                    yield name, data
        process.stdin.close()
        if process.wait():
            raise ValueError('history-object-unreadable')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', action='store_true', help='also inspect all reachable historical blobs')
    args = parser.parse_args()
    try:
        root = Path(git(Path.cwd(), 'rev-parse', '--show-toplevel').decode().strip())
        findings = {(category, name) for name, data in current_files(root)
                    for category in categories(name, data)}
        if args.history:
            findings.update((category, name) for name, data in history_files(root)
                            for category in categories(name, data))
        for category, name in sorted(findings):
            print(category + ': ' + json.dumps(name, ensure_ascii=True))
        return 1 if findings else 0
    except (OSError, ValueError, subprocess.CalledProcessError):
        # Git diagnostics and exception strings may contain private paths.
        print('scan-error: "<repository>"')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
