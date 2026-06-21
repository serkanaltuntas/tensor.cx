#include "cortex/backends/metal/metal_backend.h"

#include <memory>
#include <stdexcept>
#include <vector>

#include "cortex/backends/metal/metal_buffer.h"
#include "cortex/backends/metal/metal_context.h"

namespace cortex::metal {

bool available() {
  return is_available();
}

std::vector<std::string> devices() {
  if (!available()) {
    return {};
  }
  return {default_context().device_name()};
}

MetalTensor from_cpu(const cpu::CpuTensor& tensor) {
  auto buffer = std::make_shared<MetalBuffer>(
      tensor.dtype(), static_cast<std::size_t>(tensor.size()));

  switch (tensor.dtype()) {
    case DType::kFloat32:
      buffer->copy_from_host(tensor.float_data().data(), buffer->nbytes());
      break;
    case DType::kInt32:
      buffer->copy_from_host(tensor.int32_data().data(), buffer->nbytes());
      break;
  }

  return MetalTensor(tensor.dtype(), tensor.shape(), std::move(buffer));
}

cpu::CpuTensor to_cpu(const MetalTensor& tensor) {
  switch (tensor.dtype()) {
    case DType::kFloat32: {
      std::vector<float> values(static_cast<std::size_t>(tensor.size()));
      tensor.buffer()->copy_to_host(values.data(), tensor.nbytes());
      return cpu::CpuTensor(tensor.shape(), std::move(values));
    }
    case DType::kInt32: {
      std::vector<std::int32_t> values(static_cast<std::size_t>(tensor.size()));
      tensor.buffer()->copy_to_host(values.data(), tensor.nbytes());
      return cpu::CpuTensor(tensor.shape(), std::move(values));
    }
  }
  throw std::invalid_argument("unsupported Metal tensor dtype");
}

}  // namespace cortex::metal
