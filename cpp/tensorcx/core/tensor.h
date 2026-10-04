#pragma once

#include <memory>

#include "tensorcx/core/buffer.h"
#include "tensorcx/core/device.h"
#include "tensorcx/core/dtype.h"
#include "tensorcx/core/shape.h"

namespace tensorcx {

struct Tensor {
  DType dtype;
  Shape shape;
  Shape strides;
  Device device;
  std::shared_ptr<Buffer> buffer;
  std::size_t offset{0};
};

}  // namespace tensorcx
