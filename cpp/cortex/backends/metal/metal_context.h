#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

namespace cortex::metal {

class MetalContext {
 public:
  MetalContext();

  MTL::Device& device() const { return *device_.get(); }
  MTL::CommandQueue& command_queue() const { return *command_queue_.get(); }
  const char* device_name() const;

 private:
  NS::SharedPtr<MTL::Device> device_;
  NS::SharedPtr<MTL::CommandQueue> command_queue_;
};

bool is_available();
MetalContext& default_context();

}  // namespace cortex::metal
