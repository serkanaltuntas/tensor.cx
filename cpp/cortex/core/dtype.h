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

}  // namespace cortex
