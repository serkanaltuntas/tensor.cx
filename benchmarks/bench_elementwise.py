"""Basic elementwise benchmark for Cortex Runtime."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import cortex_runtime as cx


def time_call(fn: Callable[[], object], repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        if hasattr(result, "cpu"):
            result.cpu().numpy()
        best = min(best, time.perf_counter() - start)
    return best


def bench_device(device: str, elements: int, repeats: int) -> list[tuple[str, float]]:
    x = cx.ones((elements,), dtype=cx.float32, device=device)
    y = cx.ones((elements,), dtype=cx.float32, device=device)

    return [
        ("fill_ones", time_call(lambda: cx.ones((elements,), dtype=cx.float32, device=device), repeats)),
        ("add", time_call(lambda: x + y, repeats)),
        ("multiply", time_call(lambda: x * y, repeats)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    devices = ["cpu"]
    if cx.is_available("metal"):
        devices.append("metal")

    print(f"cortex_runtime {cx.__version__}")
    print(f"elements {args.elements}")
    print("device,operation,best_ms")
    for device in devices:
        for operation, seconds in bench_device(device, args.elements, args.repeats):
            print(f"{device},{operation},{seconds * 1000:.3f}")


if __name__ == "__main__":
    main()
