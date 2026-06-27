#include "cortex/backends/metal/metal_library.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <cstdint>

#include "cortex/backends/metal/metal_buffer.h"
#include "cortex/backends/metal/metal_context.h"
#include "cortex/backends/metal/metal_dispatch_data.h"
#include "cortex/backends/metal/metal_tensor.h"
#include "cortex/core/dtype.h"
#include "cortex/core/status.h"

namespace cortex::metal {
namespace {

std::string error_message(const char* prefix, NS::Error* error) {
  if (error && error->localizedDescription()) {
    return std::string(prefix) + ": " + error->localizedDescription()->utf8String();
  }
  return prefix;
}

}  // namespace

Expected<std::string> validate_library_function(
    const std::vector<std::uint8_t>& metallib,
    const std::string& function_name) {
  if (metallib.empty()) {
    return Status(StatusCode::kInvalidArgument, "Metal library data is empty");
  }
  if (function_name.empty()) {
    return Status(StatusCode::kInvalidArgument, "Metal function name is empty");
  }
  if (function_name.find('\0') != std::string::npos) {
    return Status(StatusCode::kInvalidArgument, "Metal function name cannot contain null bytes");
  }

  NS::SharedPtr<NS::AutoreleasePool> pool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  auto& context = default_context();
  if (!context.ready()) {
    return context.status();
  }

  detail::DispatchData library_data(dispatch_data_create(
      metallib.data(),
      metallib.size(),
      nullptr,
      DISPATCH_DATA_DESTRUCTOR_DEFAULT));
  if (!library_data.get()) {
    return Status(StatusCode::kInternal, "failed to create Metal library data");
  }

  NS::Error* error = nullptr;
  NS::SharedPtr<MTL::Library> library =
      NS::TransferPtr(context.device().newLibrary(library_data.get(), &error));
  if (!library) {
    return Status(
        StatusCode::kInvalidArgument,
        error_message("failed to load Metal library", error));
  }

  NS::SharedPtr<NS::String> name =
      NS::TransferPtr(NS::String::alloc()->init(function_name.c_str(), NS::UTF8StringEncoding));
  if (!name) {
    return Status(StatusCode::kInternal, "failed to allocate Metal function name");
  }

  NS::SharedPtr<MTL::Function> function =
      NS::TransferPtr(library->newFunction(name.get()));
  if (!function) {
    return Status(
        StatusCode::kInvalidArgument,
        "Metal library does not contain function: " + function_name);
  }

  return function_name;
}

Expected<std::string> launch_library_function(
    const std::vector<std::uint8_t>& metallib,
    const std::string& function_name,
    const std::vector<KernelArgument>& arguments,
    std::uint32_t thread_count,
    std::uint32_t threads_per_threadgroup) {
  if (metallib.empty()) {
    return Status(StatusCode::kInvalidArgument, "Metal library data is empty");
  }
  if (function_name.empty()) {
    return Status(StatusCode::kInvalidArgument, "Metal function name is empty");
  }
  if (function_name.find('\0') != std::string::npos) {
    return Status(StatusCode::kInvalidArgument, "Metal function name cannot contain null bytes");
  }
  if (threads_per_threadgroup == 0) {
    return Status(StatusCode::kInvalidArgument, "threads per threadgroup must be positive");
  }

  for (const auto& argument : arguments) {
    if (argument.kind != KernelArgument::Kind::kTensor) {
      continue;
    }
    if (!argument.tensor) {
      return Status(StatusCode::kInternal, "Metal kernel tensor argument is null");
    }
    if (argument.tensor->dtype() != DType::kFloat32) {
      return Status(
          StatusCode::kInvalidArgument,
          "experimental Metal kernels currently support float32 tensor buffers");
    }
    if (!argument.tensor->buffer()) {
      return Status(StatusCode::kInternal, "Metal kernel tensor buffer is missing");
    }
    if (thread_count > 0 && !argument.tensor->buffer()->native()) {
      return Status(StatusCode::kInternal, "Metal kernel tensor buffer is not allocated");
    }
  }

  NS::SharedPtr<NS::AutoreleasePool> pool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  auto& context = default_context();
  if (!context.ready()) {
    return context.status();
  }

  detail::DispatchData library_data(dispatch_data_create(
      metallib.data(),
      metallib.size(),
      nullptr,
      DISPATCH_DATA_DESTRUCTOR_DEFAULT));
  if (!library_data.get()) {
    return Status(StatusCode::kInternal, "failed to create Metal library data");
  }

  NS::Error* error = nullptr;
  NS::SharedPtr<MTL::Library> library =
      NS::TransferPtr(context.device().newLibrary(library_data.get(), &error));
  if (!library) {
    return Status(
        StatusCode::kInvalidArgument,
        error_message("failed to load Metal library", error));
  }

  NS::SharedPtr<NS::String> name =
      NS::TransferPtr(NS::String::alloc()->init(function_name.c_str(), NS::UTF8StringEncoding));
  if (!name) {
    return Status(StatusCode::kInternal, "failed to allocate Metal function name");
  }

  NS::SharedPtr<MTL::Function> function =
      NS::TransferPtr(library->newFunction(name.get()));
  if (!function) {
    return Status(
        StatusCode::kInvalidArgument,
        "Metal library does not contain function: " + function_name);
  }

  NS::SharedPtr<MTL::ComputePipelineState> pipeline =
      NS::TransferPtr(context.device().newComputePipelineState(function.get(), &error));
  if (!pipeline) {
    return Status(
        StatusCode::kInternal,
        error_message("failed to create Metal compute pipeline", error));
  }

  if (threads_per_threadgroup > pipeline->maxTotalThreadsPerThreadgroup()) {
    return Status(
        StatusCode::kInvalidArgument,
        "threads per threadgroup exceeds the Metal pipeline limit");
  }
  if (thread_count == 0) {
    return function_name;
  }

  NS::SharedPtr<MTL::CommandBuffer> command_buffer =
      NS::RetainPtr(context.command_queue().commandBuffer());
  if (!command_buffer) {
    return Status(StatusCode::kInternal, "failed to create Metal command buffer");
  }

  NS::SharedPtr<MTL::ComputeCommandEncoder> encoder =
      NS::RetainPtr(command_buffer->computeCommandEncoder());
  if (!encoder) {
    return Status(StatusCode::kInternal, "failed to create Metal compute command encoder");
  }

  encoder->setComputePipelineState(pipeline.get());
  for (std::size_t index = 0; index < arguments.size(); ++index) {
    const auto& argument = arguments[index];
    if (argument.kind == KernelArgument::Kind::kTensor) {
      MTL::Buffer* buffer = argument.tensor->buffer()->native();
      encoder->setBuffer(buffer, 0, index);
    } else {
      const std::uint32_t value = argument.uint32_value;
      encoder->setBytes(&value, sizeof(value), index);
    }
  }

  const std::uint64_t threadgroups =
      (static_cast<std::uint64_t>(thread_count) + threads_per_threadgroup - 1) /
      threads_per_threadgroup;
  encoder->dispatchThreadgroups(
      MTL::Size::Make(threadgroups, 1, 1),
      MTL::Size::Make(threads_per_threadgroup, 1, 1));
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();

  if (command_buffer->status() == MTL::CommandBufferStatusError) {
    return Status(
        StatusCode::kInternal,
        error_message("Metal command buffer failed", command_buffer->error()));
  }
  return function_name;
}

}  // namespace cortex::metal
