"""Bounded compiler query concurrency, process ownership and fresh resolver results."""
import subprocess
import sys

import pytest

from tensorcx._compiler import cpu


def fake_processes(monkeypatch, *, failure=None, spawn_error=None, exception=None):
    created, closed, active = [], [], []
    peak = [0]

    class Process:
        def __init__(self, command, **kwargs):
            if len(created) == spawn_error:
                raise OSError("cannot spawn")
            assert kwargs["stdout"] == subprocess.PIPE
            assert kwargs["stderr"] == subprocess.PIPE
            assert kwargs["text"] is True
            assert not kwargs.get("shell", False)
            self.command, self.returncode, self.killed = command, None, False
            self.waited = False
            created.append(self)
            active.append(self)
            peak[0] = max(peak[0], len(active))

        def __enter__(self):
            return self

        def __exit__(self, *error):
            assert self.waited, "child not explicitly reaped before context exit"
            assert self.returncode is not None, "live child not killed before context exit"
            closed.append(self)

        def communicate(self, *, timeout):
            self.timeout = timeout
            assert 0 <= timeout <= cpu.COMMAND_TIMEOUT
            # Every group starts fully before waiting on its first child.
            assert len(created) % cpu.LINK_QUERY_WORKERS == 0
            if exception is not None:
                raise exception
            self.returncode = 7 if self.command[0] == failure else 0
            active.remove(self)
            return self.command[0] + "\n", "query failed"

        def poll(self):
            return self.returncode

        def kill(self):
            self.killed = True
            active.remove(self)

        def wait(self):
            self.waited = True
            if self.killed:
                self.returncode = -9
            assert self.returncode is not None
            return self.returncode

    monkeypatch.setattr(cpu.subprocess, "Popen", Process)
    return created, closed, active, peak


def test_queries_are_bounded_and_results_keep_command_order(monkeypatch):
    created, closed, active, peak = fake_processes(monkeypatch)
    commands = [[f"q{i}"] for i in range(8)]
    assert cpu._run_parallel(commands) == [f"q{i}\n" for i in range(8)]
    assert peak[0] == 4
    assert len(created) == len(closed) == 8
    assert not active and not any(p.killed for p in created)


@pytest.mark.parametrize("kind", ["spawn", "failure", "timeout", "interrupt"])
def test_all_started_children_are_killed_and_reaped_on_error(monkeypatch, kind):
    options = {"spawn_error": 2} if kind == "spawn" else {"failure": "q0"} if kind == "failure" else {
        "exception": subprocess.TimeoutExpired("query", 60) if kind == "timeout" else KeyboardInterrupt()}
    created, closed, active, peak = fake_processes(monkeypatch, **options)
    error = KeyboardInterrupt if kind == "interrupt" else RuntimeError
    with pytest.raises(error):
        cpu._run_parallel([[f"q{i}"] for i in range(8)])
    assert len(created) == len(closed)
    assert not active
    assert all(p.killed for p in created if p.returncode != 7)
    assert len(created) == (2 if kind == "spawn" else 4)


def test_batch_uses_remaining_deadline_not_timeout_per_child(monkeypatch):
    created, *_ = fake_processes(monkeypatch)
    readings = iter([0, 10, 20, 30, 40])
    monkeypatch.setattr(cpu.time, "monotonic", lambda: next(readings))
    assert cpu._run_parallel([[f"q{i}"] for i in range(4)]) == [f"q{i}\n" for i in range(4)]
    assert [p.timeout for p in created] == [50, 40, 30, 20]


def test_real_children_must_overlap_to_complete(tmp_path):
    # A barrier proves overlap without relying on a speed threshold.
    program = '''import pathlib,sys,time
root=pathlib.Path(sys.argv[1]);(root/sys.argv[2]).touch()
deadline=time.monotonic()+20
while len(list(root.iterdir()))<4:
    if time.monotonic()>deadline:raise RuntimeError("queries did not overlap")
    time.sleep(.01)
print(sys.argv[2])
'''
    commands = [[sys.executable, "-c", program, str(tmp_path), str(i)] for i in range(4)]
    assert cpu._run_parallel(commands) == [f"{i}\n" for i in range(4)]


def test_every_link_input_is_queried_again_and_validated(monkeypatch, tmp_path):
    names = ["crtbeginS.o", "crtendS.o", "crti.o", "crtn.o", "libgcc.a", "libgcc_s.so", "libc.so"]
    linker = tmp_path / "ld"
    linker.write_bytes(b"linker")
    files = [tmp_path / name for name in names]
    for path in files:
        path.write_bytes(b"first")
    calls = []

    def query(commands):
        calls.append(commands)
        assert commands == [["clang", "-print-prog-name=ld"], *[["clang", "-print-file-name=" + name] for name in names]]
        return [str(linker), *map(str, files)]

    monkeypatch.setattr(cpu, "_run_parallel", query)
    monkeypatch.setattr(cpu, "_dependencies", lambda path: [])
    assert cpu._link_inputs({"clang": "clang"}) == [linker, *files]
    assert cpu._link_inputs({"clang": "clang"}) == [linker, *files]
    assert len(calls) == 2
    files[-1].unlink()
    with pytest.raises(RuntimeError, match="link input identity"):
        cpu._link_inputs({"clang": "clang"})
    assert len(calls) == 3
