#include "cortex/backends/metal/metal_backend.h"

#include <memory>
#include <vector>

#include "cortex/backends/metal/metal_buffer.h"
#include "cortex/backends/metal/metal_context.h"

namespace cortex::metal {

bool available() {
  return default_context().ready();
}

std::vector<std::string> devices() {
  if (!available()) {
    return {};
  }
  auto& context = default_context();
  if (!context.ready()) {
    return {};
  }
  return {context.device_name()};
}

Expected<MetalTensor> from_cpu(const cpu::CpuTensor& tensor) {
  auto buffer_result = MetalBuffer::create(
      tensor.dtype(), static_cast<std::size_t>(tensor.size()));
  if (!buffer_result) {
    return buffer_result.status();
  }
  auto buffer = buffer_result.move_value();

  switch (tensor.dtype()) {
    case DType::kFloat32: {
      const Status status = buffer->copy_from_host(tensor.float_data().data(), buffer->nbytes());
      if (!status.ok()) {
        return status;
      }
      break;
    }
    case DType::kInt32: {
      const Status status = buffer->copy_from_host(tensor.int32_data().data(), buffer->nbytes());
      if (!status.ok()) {
        return status;
      }
      break;
    }
  }

  return MetalTensor(tensor.dtype(), tensor.shape(), std::move(buffer));
}

Expected<cpu::CpuTensor> to_cpu(const MetalTensor& tensor) {
  switch (tensor.dtype()) {
    case DType::kFloat32: {
      std::vector<float> values(static_cast<std::size_t>(tensor.size()));
      const Status status = tensor.buffer()->copy_to_host(values.data(), tensor.nbytes());
      if (!status.ok()) {
        return status;
      }
      return cpu::CpuTensor(tensor.shape(), std::move(values));
    }
    case DType::kInt32: {
      std::vector<std::int32_t> values(static_cast<std::size_t>(tensor.size()));
      const Status status = tensor.buffer()->copy_to_host(values.data(), tensor.nbytes());
      if (!status.ok()) {
        return status;
      }
      return cpu::CpuTensor(tensor.shape(), std::move(values));
    }
  }
  return Status(StatusCode::kInvalidArgument, "unsupported Metal tensor dtype");
}

}  // namespace cortex::metal
