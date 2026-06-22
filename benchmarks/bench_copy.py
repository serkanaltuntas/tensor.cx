"""CPU/Metal copy benchmark for Cortex Runtime."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import cortex_runtime as cx


DEFAULT_SIZES = (1_024, 16_384, 262_144, 1_048_576, 16_777_216)
FLOAT32_BYTES = 4


def parse_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not sizes:
        raise argparse.ArgumentTypeError("at least one size is required")
    if any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be positive element counts")
    return sizes


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


def gib_per_second(elements: int, seconds: float) -> float:
    bytes_copied = elements * FLOAT32_BYTES
    return bytes_copied / seconds / (1024**3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        type=parse_sizes,
        default=DEFAULT_SIZES,
        help="comma-separated element counts",
    )
    parser.add_argument("--repeats", type=positive_int, default=5)
    args = parser.parse_args()

    print(f"cortex_runtime {cx.__version__}")
    print(f"repeats {args.repeats}")
    if not cx.is_available("metal"):
        print("Metal is not available; copy benchmark requires the Metal backend.")
        return

    print(f"{'direction':<8} {'elements':>10} {'MiB':>10} {'best_ms':>10} {'GiB/s':>10}")
    for elements in args.sizes:
        cpu_tensor = cx.ones((elements,), dtype=cx.float32, device="cpu")
        metal_tensor = cpu_tensor.to("metal")
        mib = elements * FLOAT32_BYTES / (1024**2)

        h2d_seconds = time_call(lambda: cpu_tensor.to("metal"), args.repeats)
        d2h_seconds = time_call(lambda: metal_tensor.cpu(), args.repeats)

        print(
            f"{'H2D':<8} {elements:>10} {mib:>10.3f} "
            f"{h2d_seconds * 1000:>10.3f} {gib_per_second(elements, h2d_seconds):>10.3f}"
        )
        print(
            f"{'D2H':<8} {elements:>10} {mib:>10.3f} "
            f"{d2h_seconds * 1000:>10.3f} {gib_per_second(elements, d2h_seconds):>10.3f}"
        )


if __name__ == "__main__":
    main()
