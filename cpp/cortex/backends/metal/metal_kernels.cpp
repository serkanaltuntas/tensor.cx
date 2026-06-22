#include "cortex/backends/metal/metal_kernels.h"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "cortex/backends/metal/metal_buffer.h"
#include "cortex/backends/metal/metal_context.h"
#include "cortex/backends/metal/metal_kernels_data.h"

namespace cortex::metal {
namespace {

std::string error_message(const char* prefix, NS::Error* error) {
  if (error && error->localizedDescription()) {
    return std::string(prefix) + ": " + error->localizedDescription()->utf8String();
  }
  return prefix;
}

Expected<const char*> binary_kernel_name(OpKind kind, DType dtype) {
  switch (kind) {
    case OpKind::kAdd:
      return dtype == DType::kFloat32 ? "add_f32" : "add_i32";
    case OpKind::kMultiply:
      return dtype == DType::kFloat32 ? "mul_f32" : "mul_i32";
    default:
      return Status(StatusCode::kInvalidArgument, "unsupported Metal binary operation");
  }
}

Expected<const char*> fill_kernel_name(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return "fill_f32";
    case DType::kInt32:
      return "fill_i32";
  }
  return Status(StatusCode::kInvalidArgument, "unsupported Metal fill dtype");
}

struct MatmulDims {
  std::uint32_t m;
  std::uint32_t k;
  std::uint32_t n;
  std::int64_t output_elements;
};

Expected<std::int64_t> checked_numel_for_metal(const Shape& shape) {
  std::int64_t total = 1;
  for (Dim dim : shape) {
    if (dim < 0) {
      return Status(StatusCode::kInvalidArgument, "shape dimensions must be non-negative");
    }
    if (total != 0 && dim > std::numeric_limits<Dim>::max() / total) {
      return Status(StatusCode::kInvalidArgument, "shape size overflow");
    }
    total *= dim;
  }
  return total;
}

Expected<std::int64_t> checked_shape_for_metal(const Shape& shape) {
  auto element_count_result = checked_numel_for_metal(shape);
  if (!element_count_result) {
    return element_count_result.status();
  }

  try {
    (void)cortex::contiguous_strides(shape);
  } catch (const std::invalid_argument& error) {
    return Status(StatusCode::kInvalidArgument, error.what());
  }

  return element_count_result.move_value();
}

Expected<std::uint32_t> checked_thread_count(std::int64_t size) {
  if (size < 0) {
    return Status(StatusCode::kInvalidArgument, "Metal tensor size must be non-negative");
  }
  if (size > std::numeric_limits<std::uint32_t>::max()) {
    return Status(
        StatusCode::kInvalidArgument,
        "Metal kernels currently support at most 2^32 - 1 elements");
  }
  return static_cast<std::uint32_t>(size);
}

Expected<std::uint32_t> checked_dim_for_metal(Dim dim, const char* name) {
  if (dim < 0) {
    return Status(StatusCode::kInvalidArgument, "shape dimensions must be non-negative");
  }
  if (dim > std::numeric_limits<std::uint32_t>::max()) {
    return Status(
        StatusCode::kInvalidArgument,
        std::string("Metal matmul dimension exceeds 2^32 - 1: ") + name);
  }
  return static_cast<std::uint32_t>(dim);
}

Expected<MatmulDims> checked_matmul_dims(const MetalTensor& lhs, const MetalTensor& rhs) {
  if (lhs.dtype() != DType::kFloat32 || rhs.dtype() != DType::kFloat32) {
    return Status(StatusCode::kInvalidArgument, "Metal matmul only supports float32 tensors");
  }
  if (lhs.shape().size() != 2 || rhs.shape().size() != 2) {
    return Status(StatusCode::kInvalidArgument, "matmul requires rank-2 tensors");
  }
  if (lhs.shape()[1] != rhs.shape()[0]) {
    return Status(StatusCode::kInvalidArgument, "matmul shape mismatch");
  }

  auto m_result = checked_dim_for_metal(lhs.shape()[0], "M");
  if (!m_result) {
    return m_result.status();
  }
  auto k_result = checked_dim_for_metal(lhs.shape()[1], "K");
  if (!k_result) {
    return k_result.status();
  }
  auto n_result = checked_dim_for_metal(rhs.shape()[1], "N");
  if (!n_result) {
    return n_result.status();
  }

  const auto m = m_result.move_value();
  const auto k = k_result.move_value();
  const auto n = n_result.move_value();
  if (m != 0 && n > std::numeric_limits<std::uint32_t>::max() / m) {
    return Status(
        StatusCode::kInvalidArgument,
        "Metal kernels currently support at most 2^32 - 1 elements");
  }
  const std::int64_t output_elements =
      static_cast<std::int64_t>(m) * static_cast<std::int64_t>(n);
  auto thread_count_result = checked_thread_count(output_elements);
  if (!thread_count_result) {
    return thread_count_result.status();
  }
  return MatmulDims{m, k, n, output_elements};
}

class KernelRuntime {
 public:
  KernelRuntime() : status_(initialize()) {}

  Expected<MTL::ComputePipelineState*> pipeline(const char* name) {
    if (!status_.ok()) {
      return status_;
    }
    if (std::strcmp(name, "add_f32") == 0) {
      return pipeline_slot(add_f32_, name);
    }
    if (std::strcmp(name, "mul_f32") == 0) {
      return pipeline_slot(mul_f32_, name);
    }
    if (std::strcmp(name, "fill_f32") == 0) {
      return pipeline_slot(fill_f32_, name);
    }
    if (std::strcmp(name, "add_i32") == 0) {
      return pipeline_slot(add_i32_, name);
    }
    if (std::strcmp(name, "mul_i32") == 0) {
      return pipeline_slot(mul_i32_, name);
    }
    if (std::strcmp(name, "fill_i32") == 0) {
      return pipeline_slot(fill_i32_, name);
    }
    if (std::strcmp(name, "matmul_f32") == 0) {
      return pipeline_slot(matmul_f32_, name);
    }
    return Status(StatusCode::kInvalidArgument, "unknown Metal kernel name");
  }

 private:
  Status initialize() {
    auto& context = default_context();
    if (!context.ready()) {
      return context.status();
    }
    library_data_ = dispatch_data_create(
        kElementwiseMetallib,
        kElementwiseMetallibSize,
        nullptr,
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    if (!library_data_) {
      return Status(StatusCode::kInternal, "failed to create embedded Metal library data");
    }

    NS::Error* error = nullptr;
    library_ = NS::TransferPtr(context.device().newLibrary(library_data_, &error));
    if (!library_) {
      return Status(
          StatusCode::kInternal,
          error_message("failed to load embedded Metal library", error));
    }
    return Status::Ok();
  }

  Expected<MTL::ComputePipelineState*> pipeline_slot(
      NS::SharedPtr<MTL::ComputePipelineState>& slot,
      const char* name) {
    if (slot) {
      return slot.get();
    }

    NS::SharedPtr<NS::String> function_name =
        NS::TransferPtr(NS::String::alloc()->init(name, NS::UTF8StringEncoding));
    if (!function_name) {
      return Status(StatusCode::kInternal, "failed to allocate Metal kernel name");
    }
    NS::SharedPtr<MTL::Function> function =
        NS::TransferPtr(library_->newFunction(function_name.get()));
    if (!function) {
      return Status(StatusCode::kInternal, std::string("failed to load Metal kernel: ") + name);
    }

    NS::Error* error = nullptr;
    slot = NS::TransferPtr(default_context().device().newComputePipelineState(function.get(), &error));
    if (!slot) {
      return Status(
          StatusCode::kInternal,
          error_message("failed to create Metal compute pipeline", error));
    }
    return slot.get();
  }

  NS::SharedPtr<MTL::Library> library_;
  dispatch_data_t library_data_{nullptr};
  Status status_;
  NS::SharedPtr<MTL::ComputePipelineState> add_f32_;
  NS::SharedPtr<MTL::ComputePipelineState> mul_f32_;
  NS::SharedPtr<MTL::ComputePipelineState> fill_f32_;
  NS::SharedPtr<MTL::ComputePipelineState> add_i32_;
  NS::SharedPtr<MTL::ComputePipelineState> mul_i32_;
  NS::SharedPtr<MTL::ComputePipelineState> fill_i32_;
  NS::SharedPtr<MTL::ComputePipelineState> matmul_f32_;
};

KernelRuntime& runtime() {
  static KernelRuntime instance;
  return instance;
}

Status validate_same_metadata(const MetalTensor& lhs, const MetalTensor& rhs) {
  if (lhs.dtype() != rhs.dtype()) {
    return Status(StatusCode::kInvalidArgument, "dtype mismatch for Metal binary operation");
  }
  if (lhs.shape() != rhs.shape()) {
    return Status(StatusCode::kInvalidArgument, "shape mismatch for Metal binary operation");
  }
  return Status::Ok();
}

Status run_threads(MTL::ComputePipelineState& pipeline,
                   const std::uint32_t thread_count,
                   const std::function<void(MTL::ComputeCommandEncoder&)>& bind) {
  if (thread_count == 0) {
    return Status::Ok();
  }

  auto& context = default_context();
  if (!context.ready()) {
    return context.status();
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

  encoder->setComputePipelineState(&pipeline);
  bind(*encoder.get());
  const auto width = std::max<NS::UInteger>(1, pipeline.threadExecutionWidth());
  encoder->dispatchThreads(MTL::Size::Make(thread_count, 1, 1), MTL::Size::Make(width, 1, 1));
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();

  if (command_buffer->status() == MTL::CommandBufferStatusError) {
    return Status(
        StatusCode::kInternal,
        error_message("Metal command buffer failed", command_buffer->error()));
  }
  return Status::Ok();
}

}  // namespace

Expected<MetalTensor> execute_binary(const OpDesc& op, const MetalTensor& lhs, const MetalTensor& rhs) {
  const Status metadata_status = validate_same_metadata(lhs, rhs);
  if (!metadata_status.ok()) {
    return metadata_status;
  }

  auto thread_count_result = checked_thread_count(lhs.size());
  if (!thread_count_result) {
    return thread_count_result.status();
  }
  const auto thread_count = thread_count_result.move_value();
  auto output_buffer_result =
      MetalBuffer::create(lhs.dtype(), static_cast<std::size_t>(lhs.size()));
  if (!output_buffer_result) {
    return output_buffer_result.status();
  }
  auto output_buffer = output_buffer_result.move_value();
  MetalTensor output(lhs.dtype(), lhs.shape(), output_buffer);
  if (thread_count == 0) {
    return output;
  }
  auto kernel_name_result = binary_kernel_name(op.kind, lhs.dtype());
  if (!kernel_name_result) {
    return kernel_name_result.status();
  }
  auto pipeline_result = runtime().pipeline(kernel_name_result.move_value());
  if (!pipeline_result) {
    return pipeline_result.status();
  }
  const Status run_status = run_threads(*pipeline_result.move_value(), thread_count, [&](MTL::ComputeCommandEncoder& encoder) {
    encoder.setBuffer(lhs.buffer()->native(), 0, 0);
    encoder.setBuffer(rhs.buffer()->native(), 0, 1);
    encoder.setBuffer(output_buffer->native(), 0, 2);
    encoder.setBytes(&thread_count, sizeof(thread_count), 3);
  });
  if (!run_status.ok()) {
    return run_status;
  }
  return output;
}

Expected<MetalTensor> fill(const OpDesc& op, Shape shape, DType dtype, double value) {
  if (op.kind != OpKind::kFill) {
    return Status(StatusCode::kInvalidArgument, "unsupported Metal fill operation");
  }

  auto element_count_result = checked_shape_for_metal(shape);
  if (!element_count_result) {
    return element_count_result.status();
  }
  const auto element_count = element_count_result.move_value();
  auto thread_count_result = checked_thread_count(element_count);
  if (!thread_count_result) {
    return thread_count_result.status();
  }
  const auto thread_count = thread_count_result.move_value();
  auto output_buffer_result =
      MetalBuffer::create(dtype, static_cast<std::size_t>(element_count));
  if (!output_buffer_result) {
    return output_buffer_result.status();
  }
  auto output_buffer = output_buffer_result.move_value();
  MetalTensor output(dtype, std::move(shape), output_buffer);
  if (thread_count == 0) {
    return output;
  }
  auto kernel_name_result = fill_kernel_name(dtype);
  if (!kernel_name_result) {
    return kernel_name_result.status();
  }
  auto pipeline_result = runtime().pipeline(kernel_name_result.move_value());
  if (!pipeline_result) {
    return pipeline_result.status();
  }
  const Status run_status = run_threads(*pipeline_result.move_value(), thread_count, [&](MTL::ComputeCommandEncoder& encoder) {
    encoder.setBuffer(output_buffer->native(), 0, 0);
    switch (dtype) {
      case DType::kFloat32: {
        const float fill_value = static_cast<float>(value);
        encoder.setBytes(&fill_value, sizeof(fill_value), 1);
        break;
      }
      case DType::kInt32: {
        const std::int32_t fill_value = static_cast<std::int32_t>(value);
        encoder.setBytes(&fill_value, sizeof(fill_value), 1);
        break;
      }
    }
    encoder.setBytes(&thread_count, sizeof(thread_count), 2);
  });
  if (!run_status.ok()) {
    return run_status;
  }
  return output;
}

Expected<MetalTensor> matmul_custom(const MetalTensor& lhs, const MetalTensor& rhs) {
  auto dims_result = checked_matmul_dims(lhs, rhs);
  if (!dims_result) {
    return dims_result.status();
  }
  const auto dims = dims_result.move_value();

  auto output_buffer_result =
      MetalBuffer::create(DType::kFloat32, static_cast<std::size_t>(dims.output_elements));
  if (!output_buffer_result) {
    return output_buffer_result.status();
  }
  auto output_buffer = output_buffer_result.move_value();
  MetalTensor output(DType::kFloat32, Shape{dims.m, dims.n}, output_buffer);

  auto thread_count_result = checked_thread_count(dims.output_elements);
  if (!thread_count_result) {
    return thread_count_result.status();
  }
  const auto thread_count = thread_count_result.move_value();
  if (thread_count == 0) {
    return output;
  }

  auto pipeline_result = runtime().pipeline("matmul_f32");
  if (!pipeline_result) {
    return pipeline_result.status();
  }
  const Status run_status = run_threads(*pipeline_result.move_value(), thread_count, [&](MTL::ComputeCommandEncoder& encoder) {
    encoder.setBuffer(lhs.buffer()->native(), 0, 0);
    encoder.setBuffer(rhs.buffer()->native(), 0, 1);
    encoder.setBuffer(output_buffer->native(), 0, 2);
    encoder.setBytes(&dims.m, sizeof(dims.m), 3);
    encoder.setBytes(&dims.k, sizeof(dims.k), 4);
    encoder.setBytes(&dims.n, sizeof(dims.n), 5);
  });
  if (!run_status.ok()) {
    return run_status;
  }
  return output;
}

}  // namespace cortex::metal
