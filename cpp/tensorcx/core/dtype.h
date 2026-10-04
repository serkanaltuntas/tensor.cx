#pragma once

#include <cstddef>
#include <string_view>

namespace tensorcx {

enum class DType {
  kFloat32,
  kInt32,
  kBool,
};

std::string_view dtype_name(DType dtype);
std::size_t dtype_size(DType dtype);

// True if `value` is finite, integral-valued, and exactly representable as
// int32. Used to reject fill values that would otherwise be an undefined or
// silently truncating double->int32 narrowing cast. Shared by backend fill paths
// so all implementations reject identically.
bool is_int32_representable(double value);

}  // namespace tensorcx
