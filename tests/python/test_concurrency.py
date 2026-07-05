"""Concurrency tests for the native execution layer.

Native backend calls release the GIL, so operations from multiple Python
threads genuinely overlap inside the C++/Metal layer. These tests verify that
concurrent execution stays correct: the mutex-guarded pipeline caches (the
``KernelRuntime`` cache for embedded kernels and the experimental-launch cache
in ``metal_library.cpp``), the Metal command queue, and per-op output
allocation must not corrupt results under contention.

Workers align on a barrier immediately before their first device operation so
the racy window (lazy pipeline creation, cache miss/build/insert) is hit under
real contention instead of incidental overlap. Lazy first-use creation is
additionally covered by a cold-start subprocess test, because within this
process earlier tests have usually warmed every pipeline slot already.
"""

import shutil
import subprocess
import sys
import textwrap
import threading

import numpy as np
import pytest

import cortex_runtime as cx

THREADS = 8
ITERATIONS = 25
SIZE = 4096
JOIN_TIMEOUT_SECONDS = 120.0


def _has_metal_compiler() -> bool:
    if shutil.which("xcrun") is None:
        return False
    for tool in ("metal", "metallib"):
        result = subprocess.run(
            ["xcrun", "-sdk", "macosx", "--find", tool],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False
    return True


def _run_in_threads(worker):
    """Run ``worker(index, barrier)`` in THREADS threads and re-raise failures.

    Workers must call ``barrier.wait()`` after their setup and immediately
    before their first device operation. Joins use a timeout so a deadlock
    regression in the mutex/GIL interplay fails the test instead of hanging
    pytest forever.
    """
    barrier = threading.Barrier(THREADS)
    errors = []

    def wrapped(index):
        try:
            worker(index, barrier)
        except Exception as error:  # noqa: BLE001 - re-raised in main thread
            barrier.abort()
            errors.append((index, error))

    threads = [
        threading.Thread(target=wrapped, args=(index,), daemon=True)
        for index in range(THREADS)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=JOIN_TIMEOUT_SECONDS)
    stuck = [thread for thread in threads if thread.is_alive()]
    if stuck:
        raise AssertionError(
            f"{len(stuck)} worker thread(s) did not finish within "
            f"{JOIN_TIMEOUT_SECONDS}s; possible deadlock in the native layer"
        )
    if errors:
        index, error = errors[0]
        raise AssertionError(f"worker thread {index} failed: {error!r}") from error


@pytest.mark.backend_capability("binary_ops_float32")
def test_concurrent_binary_ops_match_numpy(backend_name):
    def worker(index, barrier):
        rng = np.random.default_rng(index)
        lhs = rng.standard_normal(SIZE).astype(np.float32)
        rhs = rng.standard_normal(SIZE).astype(np.float32)
        x = cx.tensor(lhs, dtype=cx.float32, device=backend_name)
        y = cx.tensor(rhs, dtype=cx.float32, device=backend_name)
        expected = (lhs + rhs) * lhs
        barrier.wait()
        for _ in range(ITERATIONS):
            z = (x + y) * x
            np.testing.assert_allclose(
                z.cpu().numpy(), expected, rtol=1e-6, atol=1e-6
            )

    _run_in_threads(worker)


@pytest.mark.backend_capability("copy", include_cpu=False)
def test_concurrent_device_round_trips_are_exact(backend_name):
    def worker(index, barrier):
        rng = np.random.default_rng(1000 + index)
        values = rng.standard_normal(SIZE).astype(np.float32)
        host = cx.tensor(values, dtype=cx.float32, device="cpu")
        barrier.wait()
        for _ in range(ITERATIONS):
            round_tripped = host.to(backend_name).cpu()
            np.testing.assert_array_equal(round_tripped.numpy(), values)

    _run_in_threads(worker)


@pytest.mark.backend_capability("normalization_float32")
def test_concurrent_mixed_ops_match_numpy(backend_name):
    # Softmax exercises the axis-transform kernels while matmul exercises the
    # primitive path; running them together from many threads hits distinct
    # pipelines concurrently.
    rows, cols = 32, 64

    def worker(index, barrier):
        rng = np.random.default_rng(2000 + index)
        matrix = rng.standard_normal((rows, cols)).astype(np.float32)
        x = cx.tensor(matrix, dtype=cx.float32, device=backend_name)
        transpose = cx.tensor(matrix.T, dtype=cx.float32, device=backend_name)
        shifted = matrix - matrix.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        expected_softmax = exp / exp.sum(axis=1, keepdims=True)
        expected_matmul = matrix @ matrix.T @ matrix
        barrier.wait()
        for _ in range(ITERATIONS):
            softmax_result = cx.softmax(x, axis=1)
            np.testing.assert_allclose(
                softmax_result.cpu().numpy(), expected_softmax, rtol=1e-5, atol=1e-5
            )
            matmul_result = cx.matmul(cx.matmul(x, transpose), x)
            # Looser than the single-matmul 1e-4 default (AGENTS.md §Testing):
            # this chains two matmuls, so accumulation-order divergence between
            # the backend and NumPy compounds across the intermediate product.
            np.testing.assert_allclose(
                matmul_result.cpu().numpy(), expected_matmul, rtol=1e-3, atol=1e-3
            )

    _run_in_threads(worker)


@pytest.mark.skipif(
    not _has_metal_compiler(),
    reason="Apple Metal command-line compiler tools are unavailable",
)
@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_concurrent_experimental_kernel_launches_stay_correct():
    # Exercises the mutex-guarded experimental pipeline cache in
    # metal_library.cpp under contention: after the barrier, the first wave of
    # launches races on the cache miss (concurrent build + insert of the same
    # key) and later iterations race on cache hits.
    @cx.experimental.kernel(target="metal")
    def add_kernel(a, b, out, n):
        i = (
            cx.experimental.program_id(0) * cx.experimental.block_size()
            + cx.experimental.thread_id()
        )
        if i < n:
            out[i] = a[i] + b[i]

    compiled = add_kernel.compile(target="metal")
    size = 1024

    def worker(index, barrier):
        base = np.full(size, float(index), dtype=np.float32)
        ones = np.ones(size, dtype=np.float32)
        x = cx.tensor(base, dtype=cx.float32, device="metal")
        y = cx.tensor(ones, dtype=cx.float32, device="metal")
        expected = base + ones
        barrier.wait()
        for _ in range(ITERATIONS):
            out = cx.empty((size,), dtype=cx.float32, device="metal")
            compiled.launch(x, y, out, size, thread_count=size, block_size=32)
            np.testing.assert_array_equal(out.cpu().numpy(), expected)

    _run_in_threads(worker)


@pytest.mark.backend_capability("binary_ops_float32", include_cpu=False)
def test_cold_start_concurrent_first_use_is_correct(backend_name):
    # Lazy pipeline creation only happens on the FIRST use of each kernel per
    # process; inside this pytest process earlier tests have already warmed the
    # slots, so a missing mutex would go unnoticed here. Run the race in a
    # fresh interpreter where the barrier-aligned first ops genuinely contend
    # on pipeline creation.
    script = textwrap.dedent(
        f"""
        import threading
        import numpy as np
        import cortex_runtime as cx

        THREADS = {THREADS}
        SIZE = 1024
        barrier = threading.Barrier(THREADS)
        errors = []

        def worker(index):
            try:
                base = np.full(SIZE, float(index), dtype=np.float32)
                ones = np.ones(SIZE, dtype=np.float32)
                x = cx.tensor(base, dtype=cx.float32, device="{backend_name}")
                y = cx.tensor(ones, dtype=cx.float32, device="{backend_name}")
                barrier.wait()
                z = (x + y) * x
                e = cx.exp(cx.tensor(np.zeros(SIZE, dtype=np.float32),
                                     dtype=cx.float32, device="{backend_name}"))
                np.testing.assert_allclose(
                    z.cpu().numpy(), (base + ones) * base, rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(
                    e.cpu().numpy(), np.ones(SIZE, dtype=np.float32),
                    rtol=1e-6, atol=1e-6)
            except Exception as error:
                barrier.abort()
                errors.append((index, error))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        assert not any(t.is_alive() for t in threads), "worker deadlock"
        assert not errors, errors
        print("COLD-START-OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"cold-start subprocess failed\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "COLD-START-OK" in result.stdout
