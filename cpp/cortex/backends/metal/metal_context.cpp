#include "cortex/backends/metal/metal_context.h"

#include <stdexcept>

namespace cortex::metal {

MetalContext::MetalContext() {
  device_ = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device_) {
    throw std::runtime_error("Metal is not available on this system");
  }

  command_queue_ = NS::TransferPtr(device_->newCommandQueue());
  if (!command_queue_) {
    throw std::runtime_error("failed to create Metal command queue");
  }
}

const char* MetalContext::device_name() const {
  return device_->name()->utf8String();
}

bool is_available() {
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  return static_cast<bool>(device);
}

MetalContext& default_context() {
  static MetalContext context;
  return context;
}

}  // namespace cortex::metal
