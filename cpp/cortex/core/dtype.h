#pragma once

#include <cstddef>
#include <string_view>

namespace cortex {

enum class DType {
  kFloat32,
  kInt32,
};

std::string_view dtype_name(DType dtype);
std::size_t dtype_size(DType dtype);

// True if `value` is finite and its truncated integer part is exactly
// representable as int32. Used to reject fill values that would otherwise be an
// undefined double->int32 narrowing cast (NaN/inf or out of range). Shared by
// the CPU and Metal fill paths so both reject identically.
bool is_int32_representable(double value);

}  // namespace cortex
