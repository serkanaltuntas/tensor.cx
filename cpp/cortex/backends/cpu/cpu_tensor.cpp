#include "cortex/backends/cpu/cpu_tensor.h"

#include <memory>
#include <stdexcept>
#include <utility>

#include "cortex/core/shape.h"

namespace cortex::cpu {

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
  if (buffer_->nbytes() != static_cast<std::size_t>(size_) * dtype_size(dtype_)) {
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

const std::vector<float>& CpuTensor::float_data() const {
  return buffer_->float_data();
}

const std::vector<std::int32_t>& CpuTensor::int32_data() const {
  return buffer_->int32_data();
}

std::vector<float>& CpuTensor::mutable_float_data() {
  return buffer_->mutable_float_data();
}

std::vector<std::int32_t>& CpuTensor::mutable_int32_data() {
  return buffer_->mutable_int32_data();
}

}  // namespace cortex::cpu
