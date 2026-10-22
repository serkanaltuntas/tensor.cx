"""Shared strict command, Git push and JUnit checks for local GPU gates."""
import re
import subprocess
import xml.etree.ElementTree as ET


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
