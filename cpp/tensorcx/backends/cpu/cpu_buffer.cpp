#include "tensorcx/backends/cpu/cpu_buffer.h"

#include <stdexcept>
#include <utility>

#include "tensorcx/core/dtype.h"

namespace tensorcx::cpu {

CpuBuffer::CpuBuffer(DType dtype, std::size_t elements) : dtype_(dtype) {
  switch (dtype_) {
    case DType::kFloat32:
      data_ = std::vector<float>(elements);
      break;
    case DType::kInt32:
      data_ = std::vector<std::int32_t>(elements);
      break;
    case DType::kBool:
      data_ = std::vector<std::uint8_t>(elements);
      break;
    default:
      throw std::invalid_argument("unknown CPU buffer dtype");
  }
}

CpuBuffer::CpuBuffer(std::vector<float> values)
    : dtype_(DType::kFloat32), data_(std::move(values)) {}

CpuBuffer::CpuBuffer(std::vector<std::int32_t> values)
    : dtype_(DType::kInt32), data_(std::move(values)) {}

CpuBuffer::CpuBuffer(std::vector<std::uint8_t> values)
    : dtype_(DType::kBool), data_(std::move(values)) {}

const std::vector<std::uint8_t>& CpuBuffer::bool_data() const {
  return std::get<std::vector<std::uint8_t>>(data_);
}
std::vector<std::uint8_t>& CpuBuffer::mutable_bool_data() {
  return std::get<std::vector<std::uint8_t>>(data_);
}
const void* CpuBuffer::data() const {
  return std::visit([](const auto& values) -> const void* { return values.data(); }, data_);
}
void* CpuBuffer::mutable_data() {
  return std::visit([](auto& values) -> void* { return values.data(); }, data_);
}

std::size_t CpuBuffer::nbytes() const {
  switch (dtype_) {
    case DType::kFloat32:
      return float_data().size() * dtype_size(dtype_);
    case DType::kInt32:
      return int32_data().size() * dtype_size(dtype_);
    case DType::kBool:
      return bool_data().size();
  }
  throw std::invalid_argument("unknown CPU buffer dtype");
}

const std::vector<float>& CpuBuffer::float_data() const {
  return std::get<std::vector<float>>(data_);
}

const std::vector<std::int32_t>& CpuBuffer::int32_data() const {
  return std::get<std::vector<std::int32_t>>(data_);
}

std::vector<float>& CpuBuffer::mutable_float_data() {
  return std::get<std::vector<float>>(data_);
}

std::vector<std::int32_t>& CpuBuffer::mutable_int32_data() {
  return std::get<std::vector<std::int32_t>>(data_);
}

}  // namespace tensorcx::cpu
