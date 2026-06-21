#include "cortex/core/dtype.h"

#include <cstdint>
#include <stdexcept>

namespace cortex {

std::string_view dtype_name(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return "float32";
    case DType::kInt32:
      return "int32";
  }
  throw std::invalid_argument("unknown dtype");
}

std::size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return sizeof(float);
    case DType::kInt32:
      return sizeof(std::int32_t);
  }
  throw std::invalid_argument("unknown dtype");
}

}  // namespace cortex
