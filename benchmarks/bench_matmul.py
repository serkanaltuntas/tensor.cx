"""Matmul benchmark for Cortex Runtime."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterable

import cortex_runtime as cx


DEFAULT_SIZES = ((16, 16, 16), (32, 64, 16), (64, 64, 64))


def parse_sizes(value: str) -> tuple[tuple[int, int, int], ...]:
    sizes = []
    for part in value.split(","):
        spec = part.strip().lower()
        if not spec:
            continue
        pieces = spec.split("x")
        if len(pieces) != 3:
            raise argparse.ArgumentTypeError("sizes must use MxKxN format")
        dims = tuple(int(piece) for piece in pieces)
        if any(dim <= 0 for dim in dims):
            raise argparse.ArgumentTypeError("matmul dimensions must be positive")
        sizes.append(dims)
    if not sizes:
        raise argparse.ArgumentTypeError("at least one matmul size is required")
    return tuple(sizes)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def time_call(fn: Callable[[], object], repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def gflops(m: int, k: int, n: int, seconds: float) -> float:
    return (2.0 * m * k * n) / seconds / 1_000_000_000


def providers_for(lhs_cpu: cx.Tensor, rhs_cpu: cx.Tensor) -> Iterable[tuple[str, Callable[[], object]]]:
    yield "cpu_reference", lambda: cx.matmul(lhs_cpu, rhs_cpu)
    if cx.is_available("metal"):
        lhs_metal = lhs_cpu.to("metal")
        rhs_metal = rhs_cpu.to("metal")
        yield "metal_custom", lambda: cx.matmul(lhs_metal, rhs_metal, backend="custom")
        if "optimized" in cx.matmul_backends("metal"):
            yield "metal_mpsgraph", lambda: cx.matmul(lhs_metal, rhs_metal, backend="optimized")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        type=parse_sizes,
        default=DEFAULT_SIZES,
        help="comma-separated MxKxN sizes",
    )
    parser.add_argument("--repeats", type=positive_int, default=5)
    args = parser.parse_args()

    print(f"cortex_runtime {cx.__version__}")
    print(f"repeats {args.repeats}")
    print(f"{'provider':<15} {'M':>6} {'K':>6} {'N':>6} {'best_ms':>10} {'GFLOP/s':>10}")
    for m, k, n in args.sizes:
        lhs_cpu = cx.randn((m, k), dtype=cx.float32, device="cpu", seed=m + k)
        rhs_cpu = cx.randn((k, n), dtype=cx.float32, device="cpu", seed=k + n)
        for provider, fn in providers_for(lhs_cpu, rhs_cpu):
            seconds = time_call(fn, args.repeats)
            print(
                f"{provider:<15} {m:>6} {k:>6} {n:>6} "
                f"{seconds * 1000:>10.3f} {gflops(m, k, n, seconds):>10.3f}"
            )


if __name__ == "__main__":
    main()
