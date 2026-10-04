#include "tensorcx/backends/metal/metal_context.h"

namespace tensorcx::metal {

MetalContext::MetalContext() {
  device_ = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device_) {
    status_ = Status(StatusCode::kUnavailable, "Metal is not available on this system");
    return;
  }

  command_queue_ = NS::TransferPtr(device_->newCommandQueue());
  if (!command_queue_) {
    status_ = Status(StatusCode::kInternal, "failed to create Metal command queue");
    return;
  }

  status_ = Status::Ok();
}

const char* MetalContext::device_name() const {
  if (!device_) {
    return "";
  }
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

}  // namespace tensorcx::metal
