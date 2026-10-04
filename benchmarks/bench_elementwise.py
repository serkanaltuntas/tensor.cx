"""Elementwise benchmark for tensor.cx."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterable

import tensorcx as cx


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


def effective_gib_per_second(elements: int, seconds: float, tensor_passes: int) -> float:
    bytes_processed = elements * FLOAT32_BYTES * tensor_passes
    return bytes_processed / seconds / (1024**3)


def available_devices() -> list[str]:
    devices = ["cpu"]
    if cx.is_available("metal"):
        devices.append("metal")
    return devices


def bench_device(device: str, elements: int, repeats: int) -> Iterable[tuple[str, float, float]]:
    x = cx.ones((elements,), dtype=cx.float32, device=device)
    y = cx.ones((elements,), dtype=cx.float32, device=device)

    operations = [
        # NOTE: fill_ones includes the output allocation in the timed region, so
        # its GiB/s is not directly comparable to add/multiply (which reuse
        # pre-allocated operands). It is reported as an allocation+fill figure.
        ("fill_ones", lambda: cx.ones((elements,), dtype=cx.float32, device=device), 1),
        ("add", lambda: x + y, 3),
        ("multiply", lambda: x * y, 3),
    ]

    for name, fn, tensor_passes in operations:
        seconds = time_call(fn, repeats)
        yield name, seconds, effective_gib_per_second(elements, seconds, tensor_passes)


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

    print(f"tensorcx {cx.__version__}")
    print(f"repeats {args.repeats}")
    print(f"{'device':<8} {'operation':<10} {'elements':>10} {'best_ms':>10} {'GiB/s':>10}")
    for elements in args.sizes:
        for device in available_devices():
            for operation, seconds, gib_per_second in bench_device(device, elements, args.repeats):
                print(
                    f"{device:<8} {operation:<10} {elements:>10} "
                    f"{seconds * 1000:>10.3f} {gib_per_second:>10.3f}"
                )


if __name__ == "__main__":
    main()
