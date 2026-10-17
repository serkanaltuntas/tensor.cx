"""Opt-in, bounded native host spans for the current thread."""
from __future__ import annotations

import copy
import json
import operator
import os
from pathlib import Path
import tempfile
import threading

import numpy as np

from . import _core


def _atomic_write(path, text):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".tensorcx-profile-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class RuntimeProfile:
    """Single-use context collecting CPU/CUDA native spans, without extra waits."""

    def __init__(self, *, max_events=10000):
        if isinstance(max_events, (bool, np.bool_)):
            raise ValueError("max_events must be an integer in [1, 100000]")
        try:
            limit = operator.index(max_events)
        except TypeError as error:
            raise ValueError("max_events must be an integer in [1, 100000]") from error
        if not 1 <= limit <= 100000:
            raise ValueError("max_events must be an integer in [1, 100000]")
        self._limit = limit
        self._token = None
        self._owner = None
        self._result = None
        self._used = False
        self._lock = threading.Lock()

    def __enter__(self):
        with self._lock:
            if self._used:
                raise RuntimeError("runtime profile contexts are single-use")
            self._token = _core._profile_begin(self._limit)
            self._owner = threading.get_ident()
            self._used = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with self._lock:
            if self._token is None or threading.get_ident() != self._owner:
                raise RuntimeError("runtime profile must exit on its owning thread")
            result = _core._profile_end(self._token)
            self._token = None
        measured = sum(event["self_ns"] for event in result["events"])
        result.update(schema_version=1, clock="steady_clock_host", timing="synchronous_host_spans",
                      complete=result["dropped_events"] == 0,
                      unattributed_ns=result["elapsed_ns"] - measured,
                      coverage={"cpu": "native executors and owned-buffer allocation; sequence imports, clones and NumPy exports combine allocation and copy",
                                "cuda": "native dispatch, allocation/free, copy API, static/generated launch and existing waits",
                                "metal": "not instrumented"})
        self._result = result
        return False

    def __del__(self):
        # Normal with-blocks always close explicitly. Recover an abandoned
        # manual __enter__ on its owner; native TLS also cleans up at thread exit.
        try:
            if getattr(self, "_token", None) is not None and threading.get_ident() == self._owner:
                _core._profile_end(self._token)
        except Exception:
            pass

    def report(self):
        """Return an independent JSON-compatible snapshot after context exit."""
        if self._result is None:
            raise RuntimeError("runtime profile report is available after context exit")
        return copy.deepcopy(self._result)

    def summary(self):
        """Group spans by backend/category/name/operation; add only self time."""
        groups = {}
        for event in self.report()["events"]:
            key = (event["backend"], event["category"], event["name"], event["operation"] or "")
            row = groups.setdefault(key, dict(backend=key[0], category=key[1], name=key[2],
                                             operation=event["operation"], calls=0, self_ns=0,
                                             inclusive_ns=0, bytes=0, byte_records=0))
            row["calls"] += 1
            row["self_ns"] += event["self_ns"]
            row["inclusive_ns"] += event["duration_ns"]
            if event["bytes"] is not None:
                row["bytes"] += event["bytes"]
                row["byte_records"] += 1
        return [groups[key] for key in sorted(groups)]

    def export_chrome_trace(self, path):
        """Explicitly write a host timeline using synthetic process/thread IDs."""
        report = self.report()
        events = [{"name": e["operation"] or e["name"], "cat": e["backend"] + "/" + e["category"],
                   "ph": "X", "pid": 0, "tid": 0, "ts": e["start_ns"] / 1000,
                   "dur": e["duration_ns"] / 1000,
                   "args": {"name": e["name"], "bytes": e["bytes"], "self_ns": e["self_ns"]}}
                  for e in report["events"]]
        _atomic_write(path, json.dumps({"traceEvents": events, "metadata": {
            key: value for key, value in report.items() if key != "events"}}, indent=2))


def profile(*, max_events=10000):
    """Collect current-thread CPU/CUDA native host spans; Metal is not instrumented."""
    return RuntimeProfile(max_events=max_events)
