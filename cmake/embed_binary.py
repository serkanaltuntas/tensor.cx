#!/usr/bin/env python3
"""Embed a binary file as a C++ byte array."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--array", required=True)
    parser.add_argument("--size", required=True)
    args = parser.parse_args()

    data = Path(args.input).read_bytes()
    values = [f"0x{byte:02x}" for byte in data]
    lines = [
        f'#include "{args.include}"',
        "",
        f"namespace {args.namespace} {{",
        "",
        f"alignas(16) const std::uint8_t {args.array}[] = {{",
    ]
    for index in range(0, len(values), 12):
        lines.append(f"  {', '.join(values[index:index + 12])},")
    lines.extend(
        [
            "};",
            f"const std::size_t {args.size} = sizeof({args.array});",
            "",
            f"}}  // namespace {args.namespace}",
            "",
        ]
    )
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
