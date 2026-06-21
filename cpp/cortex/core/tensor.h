#pragma once

#include <memory>

#include "cortex/core/buffer.h"
#include "cortex/core/device.h"
#include "cortex/core/dtype.h"
#include "cortex/core/shape.h"

namespace cortex {

struct Tensor {
  DType dtype;
  Shape shape;
  Shape strides;
  Device device;
  std::shared_ptr<Buffer> buffer;
  std::size_t offset{0};
};

}  // namespace cortex
