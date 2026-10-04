#include "tensorcx/core/dtype.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace tensorcx {

bool is_int32_representable(double value) {
  if (!std::isfinite(value)) {
    return false;
  }
  // Both int32 bounds are exactly representable as double. Use INT32_MIN as the
  // inclusive lower bound and 2^31 (= INT32_MAX + 1) as the *exclusive* upper
  // bound, which is the cleanest way to admit exactly [INT32_MIN, INT32_MAX].
  constexpr double kMin = static_cast<double>(std::numeric_limits<std::int32_t>::min());
  constexpr double kUpperExclusive = 2147483648.0;  // 2^31
  if (std::trunc(value) != value) {
    return false;
  }
  return value >= kMin && value < kUpperExclusive;
}

std::string_view dtype_name(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return "float32";
    case DType::kInt32:
      return "int32";
    case DType::kBool:
      return "bool";
  }
  throw std::invalid_argument("unknown dtype");
}

std::size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return sizeof(float);
    case DType::kInt32:
      return sizeof(std::int32_t);
    case DType::kBool:
      return sizeof(std::uint8_t);
  }
  throw std::invalid_argument("unknown dtype");
}

}  // namespace tensorcx
