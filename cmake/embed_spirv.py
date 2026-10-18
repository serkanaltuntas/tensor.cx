"""Embed a build-time validated SPIR-V module as aligned C++ words."""
import pathlib
import struct
import sys

source, output = map(pathlib.Path, sys.argv[1:])
data = source.read_bytes()
if len(data) % 4 or len(data) < 20:
    raise ValueError("invalid SPIR-V module length")
words = struct.unpack(f"<{len(data) // 4}I", data)
if words[0] != 0x07230203:
    raise ValueError("invalid SPIR-V magic")
output.write_text(
    "#pragma once\n#include <cstdint>\n"
    "namespace tensorcx::vulkan {\ninline constexpr std::uint32_t elementwise_spirv[] = {\n"
    + ",\n".join("  " + ", ".join(f"0x{x:08x}u" for x in words[i:i + 8])
                 for i in range(0, len(words), 8))
    + "\n};\n}\n", encoding="utf-8")
