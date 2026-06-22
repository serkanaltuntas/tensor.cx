#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "cortex/core/status.h"

namespace cortex::metal {

class MetalContext {
 public:
  MetalContext();

  MTL::Device& device() const { return *device_.get(); }
  MTL::CommandQueue& command_queue() const { return *command_queue_.get(); }
  const char* device_name() const;
  const Status& status() const { return status_; }
  bool ready() const { return status_.ok(); }

 private:
  NS::SharedPtr<MTL::Device> device_;
  NS::SharedPtr<MTL::CommandQueue> command_queue_;
  Status status_;
};

bool is_available();
MetalContext& default_context();

}  // namespace cortex::metal
