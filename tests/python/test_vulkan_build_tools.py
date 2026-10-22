"""Untrusted build-tool downloads must fail before their sources are executed."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile

import pytest

TOOLS = Path(__file__).resolve().parents[2] / 'tools'
spec = importlib.util.spec_from_file_location('build_vulkan_sdk', TOOLS / 'build_vulkan_sdk.py')
sdk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sdk)


def archive(name):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as output:
        member = tarfile.TarInfo(name)
        member.size = 6
        output.addfile(member, io.BytesIO(b'source'))
    return stream.getvalue()


def download(monkeypatch, data, checksum=None):
    monkeypatch.setitem(sdk.SOURCES, 'fixture', ('a' * 40, checksum or hashlib.sha256(data).hexdigest()))
    monkeypatch.setattr(sdk.urllib.request, 'urlopen', lambda *args, **kwargs: io.BytesIO(data))


def test_checksum_failure_never_extracts_download(monkeypatch, tmp_path):
    download(monkeypatch, archive('root/CMakeLists.txt'), '0' * 64)
    with pytest.raises(RuntimeError, match='checksum'):
        sdk.source('fixture', tmp_path)
    assert not (tmp_path / 'fixture').exists()


def test_verified_archive_still_rejects_path_escape(monkeypatch, tmp_path):
    download(monkeypatch, archive('../escaped'))
    with pytest.raises(tarfile.FilterError):
        sdk.source('fixture', tmp_path)
    assert not (tmp_path / 'escaped').exists()


def test_verified_single_root_source_is_usable(monkeypatch, tmp_path):
    download(monkeypatch, archive('root/CMakeLists.txt'))
    root = sdk.source('fixture', tmp_path)
    assert root.is_relative_to(tmp_path)
    assert (root / 'CMakeLists.txt').read_bytes() == b'source'
