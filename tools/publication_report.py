"""Export validation metadata without known local identifiers.

This removes known report fields and path/UUID patterns. It is not a generic
secret scanner: arbitrary diagnostic strings still require review before release.
Raw logs belong in ignored local build directories, outside public artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import tempfile


_PRIVATE_KEYS = {'pid', 'gpu_uuid', 'uuid', 'hostname', 'cuda_visible_devices',
                 'cuda_device_order'}
_DEPENDENCY_KEYS = {'dependencies', 'installed'}
_UUID = re.compile(r'\bGPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b')
_PERSONAL_PATH = re.compile(r'(?:file://)?/(?:home|Users)/[^\s/"\'<>]+(?:/[^\s"\'<>]*)?')
_TEMP_PATH = re.compile(r'(?:file://)?/(?:tmp|private/(?:tmp|var/folders)|var/folders)/[^\s"\'<>]+')


def sanitize_text(text, *, root=None, home=None, temporary=None):
    """Redact known local paths and GPU identifiers in a diagnostic string."""
    roots = [(root, '<project>'), (home or Path.home(), '<home>'),
             (temporary or tempfile.gettempdir(), '<temporary>')]
    for path, replacement in sorted(roots, key=lambda item: len(str(item[0] or '')), reverse=True):
        if path is not None:
            # Include descendants, but not unrelated paths sharing a prefix.
            prefix = str(Path(path).resolve()).rstrip('/')
            text = re.sub(r'(?:file://)?' + re.escape(prefix) + r'(?=/|$|[\s"\'<>])(?:/[^\s"\'<>]*)?',
                          replacement, text)
    text = re.sub(r'(?m)^InstalledDir:[^\n]*', 'InstalledDir: <toolchain>', text)
    text = _PERSONAL_PATH.sub('<home>', text)
    text = _TEMP_PATH.sub('<temporary>', text)
    return _UUID.sub('<gpu>', text)


def public_report(data, *, root=None, home=None, temporary=None):
    """Copy a JSON report, retaining measurements but removing known metadata."""
    context = dict(root=root, home=home, temporary=temporary)

    def clean(value, key=''):
        if isinstance(value, dict):
            result = {name: clean(item, name) for name, item in value.items()
                      if name.lower() not in _PRIVATE_KEYS and name != 'git_status'}
            if 'git_status' in value:
                result['git_dirty'] = bool(value['git_status'])
            return result
        if isinstance(value, list):
            return [clean(item, key) for item in value]
        if isinstance(value, str):
            if key in _DEPENDENCY_KEYS:
                # Preserve package/version evidence, never source URLs (which
                # may contain usernames, credentials or private repository names).
                value = '\n'.join(line.split(' @ ', 1)[0] + ' [direct reference omitted]'
                                  if ' @ ' in line else
                                  '[direct reference omitted]' if re.search(r'(?:https?|file|git\+[^:]+)://', line)
                                  else line for line in value.splitlines())
            return sanitize_text(value, **context)
        return value

    if not isinstance(data, dict):
        raise ValueError('public report must be a JSON object')
    result = clean(data)
    result['publication'] = {'format': 'cortex-public-report-v1',
                             'scope': 'known local identifiers removed; arbitrary text requires review'}
    return result


def write_public_report(path, data, **context):
    Path(path).write_text(json.dumps(public_report(data, **context), indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    write_public_report(args.output, json.loads(args.input.read_text()), root=args.root)


if __name__ == '__main__':
    main()
