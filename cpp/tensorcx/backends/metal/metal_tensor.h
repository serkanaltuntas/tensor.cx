#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>

#include "tensorcx/core/device.h"
#include "tensorcx/core/dtype.h"
#include "tensorcx/core/shape.h"

namespace tensorcx::metal {

class MetalBuffer;

class MetalTensor {
 public:
  MetalTensor(DType dtype, Shape shape, std::shared_ptr<MetalBuffer> buffer);

  DType dtype() const { return dtype_; }
  const Shape& shape() const { return shape_; }
  const Shape& strides() const { return strides_; }
  const Device& device() const { return device_; }
  std::int64_t size() const { return size_; }
  std::size_t nbytes() const;
  const std::shared_ptr<MetalBuffer>& buffer() const { return buffer_; }

 private:
  DType dtype_;
  Shape shape_;
  std::int64_t size_{0};
  Shape strides_;
  Device device_{"metal", 0};
  std::shared_ptr<MetalBuffer> buffer_;
};

}  // namespace tensorcx::metal
