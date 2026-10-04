#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "tensorcx/backends/cpu/cpu_buffer.h"
#include "tensorcx/core/device.h"
#include "tensorcx/core/dtype.h"
#include "tensorcx/core/shape.h"

namespace tensorcx::cpu {

class CpuTensor {
 public:
  CpuTensor(DType dtype, Shape shape);
  CpuTensor(DType dtype, Shape shape, std::shared_ptr<CpuBuffer> buffer);
  CpuTensor(Shape shape, std::vector<float> values);
  CpuTensor(Shape shape, std::vector<std::int32_t> values);

  DType dtype() const { return dtype_; }
  const Shape& shape() const { return shape_; }
  const Shape& strides() const { return strides_; }
  const Device& device() const { return device_; }
  std::int64_t size() const { return size_; }
  const std::shared_ptr<CpuBuffer>& buffer() const { return buffer_; }

  const std::vector<float>& float_data() const;
  const std::vector<std::int32_t>& int32_data() const;

  std::vector<float>& mutable_float_data();
  std::vector<std::int32_t>& mutable_int32_data();

 private:
  DType dtype_;
  Shape shape_;
  std::int64_t size_{0};
  Shape strides_;
  Device device_{"cpu", 0};
  std::shared_ptr<CpuBuffer> buffer_;
};

}  // namespace tensorcx::cpu
