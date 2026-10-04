#include "tensorcx/backends/cpu/cpu_tensor.h"

#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include "tensorcx/core/shape.h"

namespace tensorcx::cpu {

CpuTensor::CpuTensor(DType dtype, Shape shape)
    : dtype_(dtype),
      shape_(std::move(shape)),
      size_(numel(shape_)),
      strides_(contiguous_strides(shape_)) {
  buffer_ = std::make_shared<CpuBuffer>(dtype_, static_cast<std::size_t>(size_));
}

CpuTensor::CpuTensor(DType dtype, Shape shape, std::shared_ptr<CpuBuffer> buffer)
    : dtype_(dtype),
      shape_(std::move(shape)),
      size_(numel(shape_)),
      strides_(contiguous_strides(shape_)),
      buffer_(std::move(buffer)) {
  if (!buffer_) {
    throw std::invalid_argument("CPU tensor requires a buffer");
  }
  if (buffer_->dtype() != dtype_) {
    throw std::invalid_argument("CPU buffer dtype mismatch");
  }
  const auto element_bytes = dtype_size(dtype_);
  if (static_cast<std::uint64_t>(size_) >
      std::numeric_limits<std::size_t>::max() / element_bytes) {
    throw std::invalid_argument("CPU tensor byte size overflow");
  }
  if (buffer_->nbytes() != static_cast<std::size_t>(size_) * element_bytes) {
    throw std::invalid_argument("CPU buffer size does not match tensor shape");
  }
}

CpuTensor::CpuTensor(Shape shape, std::vector<float> values)
    : dtype_(DType::kFloat32),
      shape_(std::move(shape)),
      size_(numel(shape_)),
      strides_(contiguous_strides(shape_)),
      buffer_(std::make_shared<CpuBuffer>(std::move(values))) {
  if (static_cast<std::int64_t>(float_data().size()) != size_) {
    throw std::invalid_argument("tensor data length does not match shape");
  }
}

CpuTensor::CpuTensor(Shape shape, std::vector<std::int32_t> values)
    : dtype_(DType::kInt32),
      shape_(std::move(shape)),
      size_(numel(shape_)),
      strides_(contiguous_strides(shape_)),
      buffer_(std::make_shared<CpuBuffer>(std::move(values))) {
  if (static_cast<std::int64_t>(int32_data().size()) != size_) {
    throw std::invalid_argument("tensor data length does not match shape");
  }
}

CpuTensor::CpuTensor(Shape shape, std::vector<std::uint8_t> values)
    : CpuTensor(DType::kBool, std::move(shape), std::make_shared<CpuBuffer>(std::move(values))) {}

std::span<const std::uint8_t> CpuTensor::bool_data() const { return buffer_->bool_data(); }
std::span<std::uint8_t> CpuTensor::mutable_bool_data() { return buffer_->mutable_bool_data(); }
const void* CpuTensor::data() const { return buffer_->data(); }
void* CpuTensor::mutable_data() { return buffer_->mutable_data(); }

std::span<const float> CpuTensor::float_data() const {
  return buffer_->float_data();
}

std::span<const std::int32_t> CpuTensor::int32_data() const {
  return buffer_->int32_data();
}

std::span<float> CpuTensor::mutable_float_data() {
  return buffer_->mutable_float_data();
}

std::span<std::int32_t> CpuTensor::mutable_int32_data() {
  return buffer_->mutable_int32_data();
}

}  // namespace tensorcx::cpu
