#include "tensorcx/backends/metal/metal_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensorcx/backends/metal/metal_buffer.h"
#include "tensorcx/backends/metal/metal_context.h"
#include "tensorcx/backends/metal/metal_dispatch_data.h"
#include "tensorcx/backends/metal/metal_kernels_data.h"
#include "tensorcx/core/dtype.h"
#include "tensorcx/core/predicate.h"
#include "tensorcx/core/math.h"
#include "tensorcx/core/inference.h"
#include "tensorcx/backends/metal/metal_backend.h"

namespace tensorcx::metal {
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
    case OpKind::kSubtract:
      return dtype == DType::kFloat32 ? "sub_f32" : "sub_i32";
    case OpKind::kMultiply:
      return dtype == DType::kFloat32 ? "mul_f32" : "mul_i32";
    case OpKind::kDivide:
      if (dtype != DType::kFloat32) {
        return Status(StatusCode::kInvalidArgument, "division only supports float32 tensors");
      }
      return "div_f32";
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
    case DType::kBool:
      return "fill_bool";
  }
  return Status(StatusCode::kInvalidArgument, "unsupported Metal fill dtype");
}

Expected<const char*> unary_kernel_name(OpKind kind, DType dtype) {
  switch (kind) {
    case OpKind::kNegate:
      return dtype == DType::kFloat32 ? "neg_f32" : "neg_i32";
    case OpKind::kExp:
      if (dtype != DType::kFloat32) {
        return Status(StatusCode::kInvalidArgument, "exp only supports float32 tensors");
      }
      return "exp_f32";
    case OpKind::kGelu:
      if (dtype != DType::kFloat32) {
        return Status(StatusCode::kInvalidArgument, "gelu only supports float32 tensors");
      }
      return "gelu_f32";
    case OpKind::kSilu:
      if (dtype != DType::kFloat32) {
        return Status(StatusCode::kInvalidArgument, "silu only supports float32 tensors");
      }
      return "silu_f32";
    default:
      return Status(StatusCode::kInvalidArgument, "unsupported Metal unary operation");
  }
}

Expected<std::uint32_t> arithmetic_operation_code(OpKind kind) {
  switch (kind) {
    case OpKind::kAdd:
    case OpKind::kAddScalar: return std::uint32_t{0};
    case OpKind::kSubtract:
    case OpKind::kSubtractScalar: return std::uint32_t{1};
    case OpKind::kMultiply:
    case OpKind::kMultiplyScalar: return std::uint32_t{2};
    case OpKind::kDivide:
    case OpKind::kDivideScalar: return std::uint32_t{3};
    default:
      return Status(StatusCode::kInvalidArgument, "unsupported Metal arithmetic operation");
  }
}

Expected<const char*> softmax_kernel_name(DType dtype) {
  if (dtype != DType::kFloat32) {
    return Status(StatusCode::kInvalidArgument, "softmax only supports float32 tensors");
  }
  return "softmax_f32";
}

Expected<const char*> rmsnorm_kernel_name(DType dtype) {
  if (dtype != DType::kFloat32) {
    return Status(StatusCode::kInvalidArgument, "rmsnorm only supports float32 tensors");
  }
  return "rmsnorm_f32";
}

Expected<const char*> layernorm_kernel_name(DType dtype) {
  if (dtype != DType::kFloat32) {
    return Status(StatusCode::kInvalidArgument, "layernorm only supports float32 tensors");
  }
  return "layernorm_f32";
}

Expected<float> checked_epsilon(double epsilon) {
  if (!std::isfinite(epsilon) || epsilon < 0.0 ||
      epsilon > static_cast<double>(std::numeric_limits<float>::max())) {
    return Status(
        StatusCode::kInvalidArgument,
        "epsilon must be finite and non-negative and representable as float32");
  }
  const float rounded = static_cast<float>(epsilon);
  if (epsilon > 0.0 && rounded == 0.0F) {
    return Status(
        StatusCode::kInvalidArgument,
        "epsilon must be finite and non-negative and representable as float32");
  }
  return rounded;
}

Expected<const char*> reduction_kernel_name(OpKind kind, DType dtype) {
  switch (kind) {
    case OpKind::kSum:
      return dtype == DType::kFloat32 ? "reduce_sum_f32" : "reduce_sum_i32";
    case OpKind::kMax:
      return dtype == DType::kFloat32 ? "reduce_max_f32" : "reduce_max_i32";
    case OpKind::kMean:
      if (dtype != DType::kFloat32) {
        return Status(StatusCode::kInvalidArgument, "mean only supports float32 tensors");
      }
      return "reduce_mean_f32";
    default:
      return Status(StatusCode::kInvalidArgument, "unsupported Metal reduction operation");
  }
}

Status fill_empty_reduction_output(
    OpKind kind,
    DType dtype,
    std::uint32_t output_elements,
    const std::shared_ptr<MetalBuffer>& output_buffer) {
  switch (kind) {
    case OpKind::kSum: {
      switch (dtype) {
        case DType::kFloat32: {
          const std::vector<float> zeros(output_elements, 0.0F);
          return output_buffer->copy_from_host(zeros.data(), output_buffer->nbytes());
        }
        case DType::kInt32: {
          const std::vector<std::int32_t> zeros(output_elements, 0);
          return output_buffer->copy_from_host(zeros.data(), output_buffer->nbytes());
        }
      }
      break;
    }
    case OpKind::kMean: {
      if (dtype != DType::kFloat32) {
        return Status(StatusCode::kInvalidArgument, "mean only supports float32 tensors");
      }
      const std::vector<float> values(
          output_elements, std::numeric_limits<float>::quiet_NaN());
      return output_buffer->copy_from_host(values.data(), output_buffer->nbytes());
    }
    default:
      break;
  }
  return Status(StatusCode::kInvalidArgument, "unsupported empty reduction operation");
}

struct ReductionDims {
  Shape output_shape;
  std::uint32_t output_elements;
  std::uint32_t reduce_elements;
  std::uint32_t inner_elements;
};

struct AxisTransformDims {
  std::uint32_t total_elements;
  std::uint32_t reduce_elements;
  std::uint32_t inner_elements;
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
    (void)tensorcx::contiguous_strides(shape);
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

Expected<std::uint32_t> checked_reduction_extent_for_metal(Dim dim) {
  if (dim < 0) {
    return Status(StatusCode::kInvalidArgument, "shape dimensions must be non-negative");
  }
  if (dim > std::numeric_limits<std::uint32_t>::max()) {
    return Status(
        StatusCode::kInvalidArgument,
        "Metal reduction axis exceeds 2^32 - 1 elements");
  }
  return static_cast<std::uint32_t>(dim);
}

Expected<std::int64_t> normalized_axis(std::int64_t axis, const std::size_t rank) {
  if (rank == 0) {
    if (axis == 0 || axis == -1) {
      return 0;
    }
    return Status(StatusCode::kInvalidArgument, "reduction axis is out of range");
  }
  const auto signed_rank = static_cast<std::int64_t>(rank);
  if (axis < 0) {
    axis += signed_rank;
  }
  if (axis < 0 || axis >= signed_rank) {
    return Status(StatusCode::kInvalidArgument, "reduction axis is out of range");
  }
  return axis;
}

Expected<ReductionDims> checked_reduction_dims(const MetalTensor& input, const OpDesc& op) {
  if (op.kind != OpKind::kSum && op.kind != OpKind::kMax && op.kind != OpKind::kMean) {
    return Status(StatusCode::kInvalidArgument, "unsupported Metal reduction operation");
  }

  auto input_count_result = checked_thread_count(input.size());
  if (!input_count_result) {
    return input_count_result.status();
  }

  auto axis_result = normalized_axis(op.axis, input.shape().size());
  if (!axis_result) {
    return axis_result.status();
  }
  const auto axis = static_cast<std::size_t>(axis_result.move_value());

  if (input.shape().empty()) {
    return ReductionDims{Shape{}, 1, 1, 1};
  }

  if (op.kind == OpKind::kMax && input.shape()[axis] == 0) {
    return Status(StatusCode::kInvalidArgument, "max reduction requires a non-empty axis");
  }

  Shape output_shape;
  output_shape.reserve(input.shape().size() - 1);
  for (std::size_t index = 0; index < input.shape().size(); ++index) {
    if (index != axis) {
      output_shape.push_back(input.shape()[index]);
    }
  }

  auto output_count_result = checked_shape_for_metal(output_shape);
  if (!output_count_result) {
    return output_count_result.status();
  }
  auto output_thread_count_result = checked_thread_count(output_count_result.move_value());
  if (!output_thread_count_result) {
    return output_thread_count_result.status();
  }

  // Reuse the validated contiguous suffix product, including empty shapes.
  const auto inner = input.strides()[axis];
  auto inner_result = checked_thread_count(inner);
  if (!inner_result) {
    return inner_result.status();
  }
  auto reduce_result = checked_reduction_extent_for_metal(input.shape()[axis]);
  if (!reduce_result) {
    return reduce_result.status();
  }

  const auto inner_elements = inner_result.move_value();
  if (inner_elements == 0 && output_thread_count_result.value() > 0) {
    return Status(StatusCode::kInvalidArgument, "reduction inner dimension must be non-zero");
  }

  return ReductionDims{
      std::move(output_shape),
      output_thread_count_result.move_value(),
      reduce_result.move_value(),
      inner_elements};
}

Expected<AxisTransformDims> checked_axis_transform_dims(const MetalTensor& input, const OpDesc& op) {
  if (op.kind != OpKind::kSoftmax && op.kind != OpKind::kRmsNorm &&
      op.kind != OpKind::kLayerNorm) {
    return Status(StatusCode::kInvalidArgument, "unsupported Metal axis transform operation");
  }

  auto total_count_result = checked_thread_count(input.size());
  if (!total_count_result) {
    return total_count_result.status();
  }

  auto axis_result = normalized_axis(op.axis, input.shape().size());
  if (!axis_result) {
    return axis_result.status();
  }
  const auto axis = static_cast<std::size_t>(axis_result.move_value());

  if (input.shape().empty()) {
    return AxisTransformDims{1, 1, 1};
  }

  const auto inner = input.strides()[axis];
  auto inner_result = checked_thread_count(inner);
  if (!inner_result) {
    return inner_result.status();
  }
  auto reduce_result = checked_reduction_extent_for_metal(input.shape()[axis]);
  if (!reduce_result) {
    return reduce_result.status();
  }

  return AxisTransformDims{
      total_count_result.move_value(),
      reduce_result.move_value(),
      inner_result.move_value()};
}

class KernelRuntime {
 public:
  KernelRuntime() : status_(initialize()) {}

  Expected<MTL::ComputePipelineState*> pipeline(const char* name) {
    // Callers reach this from Python threads that run with the GIL released,
    // so the lazy pipeline_slot initialization below must be serialized. The
    // returned raw pointer stays valid: slots are never cleared and the
    // singleton KernelRuntime lives for the whole process.
    const std::lock_guard<std::mutex> lock(mutex_);
    if (!status_.ok()) {
      return status_;
    }
    if (std::strcmp(name, "add_f32") == 0) {
      return pipeline_slot(add_f32_, name);
    }
    if (std::strcmp(name, "mul_f32") == 0) {
      return pipeline_slot(mul_f32_, name);
    }
    if (std::strcmp(name, "sub_f32") == 0) {
      return pipeline_slot(sub_f32_, name);
    }
    if (std::strcmp(name, "div_f32") == 0) {
      return pipeline_slot(div_f32_, name);
    }
    if (std::strcmp(name, "neg_f32") == 0) {
      return pipeline_slot(neg_f32_, name);
    }
    if (std::strcmp(name, "scalar_f32") == 0) {
      return pipeline_slot(scalar_f32_, name);
    }
    if (std::strcmp(name, "broadcast_f32") == 0) {
      return pipeline_slot(broadcast_f32_, name);
    }
    if (std::strcmp(name, "broadcast_i32") == 0) {
      return pipeline_slot(broadcast_i32_, name);
    }
    if (std::strcmp(name, "copy_bits") == 0) {
      return pipeline_slot(copy_bits_, name);
    }
    if (std::strcmp(name, "concat_bits") == 0) {
      return pipeline_slot(concat_bits_, name);
    }
    if (std::strcmp(name, "predicate_values") == 0) {
      return pipeline_slot(predicate_values_, name);
    }
    if (std::strcmp(name, "embedding_validate") == 0) return pipeline_slot(embedding_validate_, name);
    if (std::strcmp(name, "embedding_gather") == 0) return pipeline_slot(embedding_gather_, name);
    if (std::strcmp(name, "attention_softmax") == 0) return pipeline_slot(attention_softmax_, name);
    if (std::strcmp(name, "math_values") == 0) {
      return pipeline_slot(math_values_, name);
    }
    if (std::strcmp(name, "reduce_bool") == 0) {
      return pipeline_slot(reduce_bool_, name);
    }
    if (std::strcmp(name, "cast_bool") == 0) {
      return pipeline_slot(cast_bool_, name);
    }
    if (std::strcmp(name, "fill_bool") == 0) {
      return pipeline_slot(fill_bool_, name);
    }
    if (std::strcmp(name, "mask_prefix") == 0) {
      return pipeline_slot(mask_prefix_, name);
    }
    if (std::strcmp(name, "mask_scan") == 0) {
      return pipeline_slot(mask_scan_, name);
    }
    if (std::strcmp(name, "mask_copy") == 0) {
      return pipeline_slot(mask_copy_, name);
    }
    if (std::strcmp(name, "transpose_bool") == 0) {
      return pipeline_slot(transpose_bool_, name);
    }
    if (std::strcmp(name, "concat_bool") == 0) {
      return pipeline_slot(concat_bool_, name);
    }
    if (std::strcmp(name, "transpose_bits") == 0) {
      return pipeline_slot(transpose_bits_, name);
    }
    if (std::strcmp(name, "cast_i32_f32") == 0) {
      return pipeline_slot(cast_i32_f32_, name);
    }
    if (std::strcmp(name, "cast_f32_i32") == 0) {
      return pipeline_slot(cast_f32_i32_, name);
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
    if (std::strcmp(name, "sub_i32") == 0) {
      return pipeline_slot(sub_i32_, name);
    }
    if (std::strcmp(name, "neg_i32") == 0) {
      return pipeline_slot(neg_i32_, name);
    }
    if (std::strcmp(name, "scalar_i32") == 0) {
      return pipeline_slot(scalar_i32_, name);
    }
    if (std::strcmp(name, "fill_i32") == 0) {
      return pipeline_slot(fill_i32_, name);
    }
    if (std::strcmp(name, "matmul_f32") == 0) {
      return pipeline_slot(matmul_f32_, name);
    }
    if (std::strcmp(name, "exp_f32") == 0) {
      return pipeline_slot(exp_f32_, name);
    }
    if (std::strcmp(name, "gelu_f32") == 0) {
      return pipeline_slot(gelu_f32_, name);
    }
    if (std::strcmp(name, "silu_f32") == 0) {
      return pipeline_slot(silu_f32_, name);
    }
    if (std::strcmp(name, "softmax_f32") == 0) {
      return pipeline_slot(softmax_f32_, name);
    }
    if (std::strcmp(name, "rmsnorm_f32") == 0) {
      return pipeline_slot(rmsnorm_f32_, name);
    }
    if (std::strcmp(name, "layernorm_f32") == 0) {
      return pipeline_slot(layernorm_f32_, name);
    }
    if (std::strcmp(name, "reduce_sum_f32") == 0) {
      return pipeline_slot(reduce_sum_f32_, name);
    }
    if (std::strcmp(name, "reduce_axes_f32") == 0) {
      return pipeline_slot(reduce_axes_f32_, name);
    }
    if (std::strcmp(name, "reduce_axes_i32") == 0) {
      return pipeline_slot(reduce_axes_i32_, name);
    }
    if (std::strcmp(name, "reduce_max_f32") == 0) {
      return pipeline_slot(reduce_max_f32_, name);
    }
    if (std::strcmp(name, "reduce_mean_f32") == 0) {
      return pipeline_slot(reduce_mean_f32_, name);
    }
    if (std::strcmp(name, "reduce_sum_i32") == 0) {
      return pipeline_slot(reduce_sum_i32_, name);
    }
    if (std::strcmp(name, "reduce_max_i32") == 0) {
      return pipeline_slot(reduce_max_i32_, name);
    }
    return Status(StatusCode::kInvalidArgument, "unknown Metal kernel name");
  }

 private:
  Status initialize() {
    auto& context = default_context();
    if (!context.ready()) {
      return context.status();
    }
    // First use may happen on a pool-less Python worker thread; drain any
    // autoreleased temporaries (including failure-path NS::Error objects).
    NS::SharedPtr<NS::AutoreleasePool> pool =
        NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    library_data_.reset(dispatch_data_create(
        kElementwiseMetallib,
        kElementwiseMetallibSize,
        nullptr,
        DISPATCH_DATA_DESTRUCTOR_DEFAULT));
    if (!library_data_.get()) {
      return Status(StatusCode::kInternal, "failed to create embedded Metal library data");
    }

    NS::Error* error = nullptr;
    library_ = NS::TransferPtr(context.device().newLibrary(library_data_.get(), &error));
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

    // Pipeline creation may run on a pool-less Python worker thread; drain
    // autoreleased temporaries from the Metal calls below.
    NS::SharedPtr<NS::AutoreleasePool> pool =
        NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

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

  std::mutex mutex_;
  NS::SharedPtr<MTL::Library> library_;
  detail::DispatchData library_data_;
  Status status_;
  NS::SharedPtr<MTL::ComputePipelineState> add_f32_;
  NS::SharedPtr<MTL::ComputePipelineState> mul_f32_;
  NS::SharedPtr<MTL::ComputePipelineState> sub_f32_;
  NS::SharedPtr<MTL::ComputePipelineState> div_f32_;
  NS::SharedPtr<MTL::ComputePipelineState> neg_f32_;
  NS::SharedPtr<MTL::ComputePipelineState> scalar_f32_;
  NS::SharedPtr<MTL::ComputePipelineState> broadcast_f32_;
  NS::SharedPtr<MTL::ComputePipelineState> broadcast_i32_;
  NS::SharedPtr<MTL::ComputePipelineState> copy_bits_;
  NS::SharedPtr<MTL::ComputePipelineState> predicate_values_;
  NS::SharedPtr<MTL::ComputePipelineState> embedding_validate_;
  NS::SharedPtr<MTL::ComputePipelineState> embedding_gather_;
  NS::SharedPtr<MTL::ComputePipelineState> attention_softmax_;
  NS::SharedPtr<MTL::ComputePipelineState> math_values_;
  NS::SharedPtr<MTL::ComputePipelineState> reduce_bool_;
  NS::SharedPtr<MTL::ComputePipelineState> cast_bool_;
  NS::SharedPtr<MTL::ComputePipelineState> fill_bool_;
  NS::SharedPtr<MTL::ComputePipelineState> mask_prefix_;
  NS::SharedPtr<MTL::ComputePipelineState> mask_scan_;
  NS::SharedPtr<MTL::ComputePipelineState> mask_copy_;
  NS::SharedPtr<MTL::ComputePipelineState> transpose_bool_;
  NS::SharedPtr<MTL::ComputePipelineState> concat_bool_;
  NS::SharedPtr<MTL::ComputePipelineState> transpose_bits_;
  NS::SharedPtr<MTL::ComputePipelineState> concat_bits_;
  NS::SharedPtr<MTL::ComputePipelineState> cast_i32_f32_;
  NS::SharedPtr<MTL::ComputePipelineState> cast_f32_i32_;
  NS::SharedPtr<MTL::ComputePipelineState> fill_f32_;
  NS::SharedPtr<MTL::ComputePipelineState> add_i32_;
  NS::SharedPtr<MTL::ComputePipelineState> mul_i32_;
  NS::SharedPtr<MTL::ComputePipelineState> sub_i32_;
  NS::SharedPtr<MTL::ComputePipelineState> neg_i32_;
  NS::SharedPtr<MTL::ComputePipelineState> scalar_i32_;
  NS::SharedPtr<MTL::ComputePipelineState> fill_i32_;
  NS::SharedPtr<MTL::ComputePipelineState> matmul_f32_;
  NS::SharedPtr<MTL::ComputePipelineState> exp_f32_;
  NS::SharedPtr<MTL::ComputePipelineState> gelu_f32_;
  NS::SharedPtr<MTL::ComputePipelineState> silu_f32_;
  NS::SharedPtr<MTL::ComputePipelineState> softmax_f32_;
  NS::SharedPtr<MTL::ComputePipelineState> rmsnorm_f32_;
  NS::SharedPtr<MTL::ComputePipelineState> layernorm_f32_;
  NS::SharedPtr<MTL::ComputePipelineState> reduce_sum_f32_;
  NS::SharedPtr<MTL::ComputePipelineState> reduce_axes_f32_;
  NS::SharedPtr<MTL::ComputePipelineState> reduce_axes_i32_;
  NS::SharedPtr<MTL::ComputePipelineState> reduce_max_f32_;
  NS::SharedPtr<MTL::ComputePipelineState> reduce_mean_f32_;
  NS::SharedPtr<MTL::ComputePipelineState> reduce_sum_i32_;
  NS::SharedPtr<MTL::ComputePipelineState> reduce_max_i32_;
};

KernelRuntime& runtime() {
  static KernelRuntime instance;
  return instance;
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

  // Drain Metal's internally-autoreleased temporaries at function scope. (The
  // command buffer and encoder below are explicitly RetainPtr-managed, but encode
  // /commit create other +0 autoreleased objects.) A Python C-extension call has
  // no implicit autorelease pool, and worker threads never get one, so without
  // this they would accumulate under sustained or multi-threaded dispatch.
  NS::SharedPtr<NS::AutoreleasePool> pool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

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
  // Threadgroup width = execution width, clamped to the pipeline's max so a
  // future heavier kernel can never request more threads per group than allowed.
  const auto exec_width = std::max<NS::UInteger>(1, pipeline.threadExecutionWidth());
  const auto max_width = std::max<NS::UInteger>(1, pipeline.maxTotalThreadsPerThreadgroup());
  const auto width = std::min<NS::UInteger>(exec_width, max_width);
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

namespace {

Expected<MetalTensor> execute_cast(const OpDesc& op, const MetalTensor& input) {
  if (op.target_dtype != DType::kFloat32 && op.target_dtype != DType::kInt32 && op.target_dtype != DType::kBool) {
    return Status(StatusCode::kInvalidArgument, "unsupported Metal cast dtype");
  }
  auto count_result = checked_thread_count(input.size());
  if (!count_result) return count_result.status();
  const auto count = count_result.move_value();
  auto buffer_result = MetalBuffer::create(op.target_dtype, static_cast<std::size_t>(input.size()));
  if (!buffer_result) return buffer_result.status();
  auto buffer = buffer_result.move_value();
  MetalTensor result(op.target_dtype, input.shape(), buffer);
  if (count == 0) return result;

  const bool checked_conversion = input.dtype() == DType::kFloat32 &&
                                  op.target_dtype == DType::kInt32;
  const bool bool_cast = input.dtype() == DType::kBool || op.target_dtype == DType::kBool;
  const char* name = bool_cast ? "cast_bool" : input.dtype() == op.target_dtype ? "copy_bits" :
                     checked_conversion ? "cast_f32_i32" : "cast_i32_f32";
  std::shared_ptr<MetalBuffer> invalid_buffer;
  std::uint32_t invalid = 0;
  if (checked_conversion) {
    auto flag_result = MetalBuffer::create(DType::kInt32, 1);
    if (!flag_result) return flag_result.status();
    invalid_buffer = flag_result.move_value();
    const auto status = invalid_buffer->copy_from_host(&invalid, sizeof(invalid));
    if (!status.ok()) return status;
  }
  auto pipeline = runtime().pipeline(name);
  if (!pipeline) return pipeline.status();
  const auto status = run_threads(*pipeline.move_value(), count,
      [&](MTL::ComputeCommandEncoder& encoder) {
        encoder.setBuffer(input.buffer()->native(), 0, 0);
        encoder.setBuffer(buffer->native(), 0, 1);
        encoder.setBytes(&count, sizeof(count), 2);
        if (invalid_buffer) encoder.setBuffer(invalid_buffer->native(), 0, 3);
        if (bool_cast) {
          const std::uint32_t source = input.dtype() == DType::kFloat32 ? 0 : input.dtype() == DType::kInt32 ? 1 : 2;
          const std::uint32_t target = op.target_dtype == DType::kFloat32 ? 0 : op.target_dtype == DType::kInt32 ? 1 : 2;
          encoder.setBytes(&source, sizeof(source), 3); encoder.setBytes(&target, sizeof(target), 4);
        }
      });
  if (!status.ok()) return status;
  if (invalid_buffer) {
    const auto copied = invalid_buffer->copy_to_host(&invalid, sizeof(invalid));
    if (!copied.ok()) return copied;
    if (invalid) {
      return Status(StatusCode::kInvalidArgument,
                    "float32 value is out of range for int32 cast");
    }
  }
  // The caller publishes this independent buffer only on success. No input or
  // supplied output is mutated when the GPU reports an invalid conversion.
  return result;
}

Expected<MetalTensor> execute_transpose(const OpDesc& op, const MetalTensor& input) {
  TransposePlan plan;
  try {
    plan = op.kind == OpKind::kTranspose
        ? make_transpose_plan(input.shape(), op.axes)
        : make_slice_plan(input.shape(), op.slice_starts, op.slice_steps, op.slice_shape);
  } catch (const std::invalid_argument& error) {
    return Status(StatusCode::kInvalidArgument, error.what());
  }
  // A scalar still returns independent storage, with no permutation metadata.
  if (plan.output_shape.empty()) {
    OpDesc copy{OpKind::kCast}; copy.target_dtype = input.dtype();
    return execute_cast(copy, input);
  }
  auto count_result = checked_thread_count(numel(plan.output_shape));
  if (!count_result) return count_result.status();
  const auto count = count_result.move_value();
  auto allocated = MetalBuffer::create(input.dtype(), static_cast<std::size_t>(numel(plan.output_shape)));
  if (!allocated) return allocated.status();
  auto buffer = allocated.move_value();
  MetalTensor result(input.dtype(), plan.output_shape, buffer);
  if (count == 0) return result;

  const std::uint64_t rank = plan.output_shape.size();
  if (rank > std::numeric_limits<std::size_t>::max() / (2 * sizeof(std::uint64_t))) {
    return Status(StatusCode::kInvalidArgument, "transpose metadata byte size overflow");
  }
  // Device metadata avoids imposing a rank cap from Metal's inline-data limit.
  std::vector<Dim> metadata;
  metadata.reserve(static_cast<std::size_t>(rank) * 2);
  for (std::size_t axis = 0; axis < rank; ++axis) {
    metadata.push_back(static_cast<std::uint64_t>(plan.output_shape[axis]));
    metadata.push_back(plan.input_strides[axis]);
  }
  auto metadata_result = MetalBuffer::create(DType::kInt32, metadata.size() * 2);
  if (!metadata_result) return metadata_result.status();
  auto metadata_buffer = metadata_result.move_value();
  auto status = metadata_buffer->copy_from_host(metadata.data(), metadata_buffer->nbytes());
  if (!status.ok()) return status;
  auto pipeline = runtime().pipeline(input.dtype() == DType::kBool ? "transpose_bool" : "transpose_bits");
  if (!pipeline) return pipeline.status();
  status = run_threads(*pipeline.move_value(), count, [&](MTL::ComputeCommandEncoder& encoder) {
    encoder.setBuffer(input.buffer()->native(), 0, 0);
    encoder.setBuffer(buffer->native(), 0, 1);
    encoder.setBytes(&count, sizeof(count), 2);
    encoder.setBuffer(metadata_buffer->native(), 0, 3);
    encoder.setBytes(&rank, sizeof(rank), 4);
    encoder.setBytes(&plan.offset, sizeof(plan.offset), 5);
  });
  if (!status.ok()) return status;
  return result;
}

Expected<MetalTensor> execute_scalar(const OpDesc& op, const MetalTensor& input) {
  auto operation_result = arithmetic_operation_code(op.kind);
  if (!operation_result) {
    return operation_result.status();
  }
  const auto operation = operation_result.move_value();
  // Validate before allocation and before the empty fast path. An int32
  // operation must never silently truncate a fractional scalar or divide.
  if (input.dtype() == DType::kInt32) {
    if (op.kind == OpKind::kDivideScalar) {
      return Status(StatusCode::kInvalidArgument, "division only supports float32 tensors");
    }
    if (!is_int32_representable(op.scalar_value) ||
        std::trunc(op.scalar_value) != op.scalar_value) {
      return Status(StatusCode::kInvalidArgument, "scalar must be an integer in the int32 range");
    }
  }
  auto thread_count_result = checked_thread_count(input.size());
  if (!thread_count_result) {
    return thread_count_result.status();
  }
  const auto thread_count = thread_count_result.move_value();
  auto output_buffer_result =
      MetalBuffer::create(input.dtype(), static_cast<std::size_t>(input.size()));
  if (!output_buffer_result) {
    return output_buffer_result.status();
  }
  auto output_buffer = output_buffer_result.move_value();
  MetalTensor output(input.dtype(), input.shape(), output_buffer);
  if (thread_count == 0) {
    return output;
  }
  auto pipeline_result = runtime().pipeline(
      input.dtype() == DType::kFloat32 ? "scalar_f32" : "scalar_i32");
  if (!pipeline_result) {
    return pipeline_result.status();
  }
  const std::uint32_t scalar_left = op.scalar_left ? 1 : 0;
  const Status run_status = run_threads(
      *pipeline_result.move_value(), thread_count, [&](MTL::ComputeCommandEncoder& encoder) {
        encoder.setBuffer(input.buffer()->native(), 0, 0);
        encoder.setBuffer(output_buffer->native(), 0, 1);
        encoder.setBytes(&thread_count, sizeof(thread_count), 2);
        if (input.dtype() == DType::kFloat32) {
          const float scalar = static_cast<float>(op.scalar_value);
          encoder.setBytes(&scalar, sizeof(scalar), 3);
        } else {
          const auto scalar = static_cast<std::int32_t>(op.scalar_value);
          encoder.setBytes(&scalar, sizeof(scalar), 3);
        }
        encoder.setBytes(&operation, sizeof(operation), 4);
        encoder.setBytes(&scalar_left, sizeof(scalar_left), 5);
      });
  if (!run_status.ok()) {
    return run_status;
  }
  return output;
}

Expected<MetalTensor> execute_softmax(const OpDesc& op, const MetalTensor& input) {
  auto dims_result = checked_axis_transform_dims(input, op);
  if (!dims_result) {
    return dims_result.status();
  }
  const auto dims = dims_result.move_value();

  auto kernel_name_result = softmax_kernel_name(input.dtype());
  if (!kernel_name_result) {
    return kernel_name_result.status();
  }

  auto output_buffer_result =
      MetalBuffer::create(input.dtype(), static_cast<std::size_t>(input.size()));
  if (!output_buffer_result) {
    return output_buffer_result.status();
  }
  auto output_buffer = output_buffer_result.move_value();
  MetalTensor output(input.dtype(), input.shape(), output_buffer);
  if (dims.total_elements == 0) {
    return output;
  }

  auto pipeline_result = runtime().pipeline(kernel_name_result.move_value());
  if (!pipeline_result) {
    return pipeline_result.status();
  }
  const Status run_status =
      run_threads(*pipeline_result.move_value(), dims.total_elements, [&](MTL::ComputeCommandEncoder& encoder) {
        encoder.setBuffer(input.buffer()->native(), 0, 0);
        encoder.setBuffer(output_buffer->native(), 0, 1);
        encoder.setBytes(&dims.total_elements, sizeof(dims.total_elements), 2);
        encoder.setBytes(&dims.reduce_elements, sizeof(dims.reduce_elements), 3);
        encoder.setBytes(&dims.inner_elements, sizeof(dims.inner_elements), 4);
      });
  if (!run_status.ok()) {
    return run_status;
  }
  return output;
}

Expected<MetalTensor> execute_rmsnorm(const OpDesc& op, const MetalTensor& input) {
  auto dims_result = checked_axis_transform_dims(input, op);
  if (!dims_result) {
    return dims_result.status();
  }
  const auto dims = dims_result.move_value();

  auto epsilon_result = checked_epsilon(op.epsilon);
  if (!epsilon_result) {
    return epsilon_result.status();
  }
  const float epsilon = epsilon_result.move_value();

  auto kernel_name_result = rmsnorm_kernel_name(input.dtype());
  if (!kernel_name_result) {
    return kernel_name_result.status();
  }

  auto output_buffer_result =
      MetalBuffer::create(input.dtype(), static_cast<std::size_t>(input.size()));
  if (!output_buffer_result) {
    return output_buffer_result.status();
  }
  auto output_buffer = output_buffer_result.move_value();
  MetalTensor output(input.dtype(), input.shape(), output_buffer);
  if (dims.total_elements == 0 || dims.reduce_elements == 0) {
    return output;
  }

  auto pipeline_result = runtime().pipeline(kernel_name_result.move_value());
  if (!pipeline_result) {
    return pipeline_result.status();
  }
  const Status run_status =
      run_threads(*pipeline_result.move_value(), dims.total_elements, [&](MTL::ComputeCommandEncoder& encoder) {
        encoder.setBuffer(input.buffer()->native(), 0, 0);
        encoder.setBuffer(output_buffer->native(), 0, 1);
        encoder.setBytes(&dims.total_elements, sizeof(dims.total_elements), 2);
        encoder.setBytes(&dims.reduce_elements, sizeof(dims.reduce_elements), 3);
        encoder.setBytes(&dims.inner_elements, sizeof(dims.inner_elements), 4);
        encoder.setBytes(&epsilon, sizeof(epsilon), 5);
      });
  if (!run_status.ok()) {
    return run_status;
  }
  return output;
}

Expected<MetalTensor> execute_layernorm(const OpDesc& op, const MetalTensor& input) {
  auto dims_result = checked_axis_transform_dims(input, op);
  if (!dims_result) {
    return dims_result.status();
  }
  const auto dims = dims_result.move_value();

  auto epsilon_result = checked_epsilon(op.epsilon);
  if (!epsilon_result) {
    return epsilon_result.status();
  }
  const float epsilon = epsilon_result.move_value();

  auto kernel_name_result = layernorm_kernel_name(input.dtype());
  if (!kernel_name_result) {
    return kernel_name_result.status();
  }

  auto output_buffer_result =
      MetalBuffer::create(input.dtype(), static_cast<std::size_t>(input.size()));
  if (!output_buffer_result) {
    return output_buffer_result.status();
  }
  auto output_buffer = output_buffer_result.move_value();
  MetalTensor output(input.dtype(), input.shape(), output_buffer);
  if (dims.total_elements == 0 || dims.reduce_elements == 0) {
    return output;
  }

  auto pipeline_result = runtime().pipeline(kernel_name_result.move_value());
  if (!pipeline_result) {
    return pipeline_result.status();
  }
  const Status run_status =
      run_threads(*pipeline_result.move_value(), dims.total_elements, [&](MTL::ComputeCommandEncoder& encoder) {
        encoder.setBuffer(input.buffer()->native(), 0, 0);
        encoder.setBuffer(output_buffer->native(), 0, 1);
        encoder.setBytes(&dims.total_elements, sizeof(dims.total_elements), 2);
        encoder.setBytes(&dims.reduce_elements, sizeof(dims.reduce_elements), 3);
        encoder.setBytes(&dims.inner_elements, sizeof(dims.inner_elements), 4);
        encoder.setBytes(&epsilon, sizeof(epsilon), 5);
      });
  if (!run_status.ok()) {
    return run_status;
  }
  return output;
}

}  // namespace

Expected<MetalTensor> execute_concat(const OpDesc& op, const std::vector<MetalTensor>& inputs) {
  if (inputs.empty()) return Status(StatusCode::kInvalidArgument, "concat requires inputs");
  std::vector<Shape> shapes;
  const auto dtype = inputs[0].dtype();
  for (const auto& input : inputs) {
    if (input.dtype() != dtype)
      return Status(StatusCode::kInvalidArgument, "concat dtypes must match");
    shapes.push_back(input.shape());
  }
  ConcatPlan plan;
  try {
    plan = make_concat_plan(shapes, op.axis);
  } catch (const std::invalid_argument& error) {
    return Status(StatusCode::kInvalidArgument, error.what());
  }
  auto count_result = checked_thread_count(numel(plan.output_shape));
  if (!count_result) return count_result.status();
  auto allocated = MetalBuffer::create(dtype, static_cast<std::size_t>(numel(plan.output_shape)));
  if (!allocated) return allocated.status();
  auto buffer = allocated.move_value();
  MetalTensor result(dtype, plan.output_shape, buffer);
  if (count_result.value() == 0) return result;
  auto pipeline = runtime().pipeline(dtype == DType::kBool ? "concat_bool" : "concat_bits");
  if (!pipeline) return pipeline.status();
  const std::uint64_t output_block = plan.output_shape[plan.axis] * plan.inner;
  std::uint64_t offset = 0;
  for (const auto& input : inputs) {
    const auto count = static_cast<std::uint32_t>(input.size());
    const std::uint64_t block = input.shape()[plan.axis] * plan.inner;
    if (count) {
      auto status = run_threads(*pipeline.value(), count, [&](MTL::ComputeCommandEncoder& encoder) {
        encoder.setBuffer(input.buffer()->native(), 0, 0);
        encoder.setBuffer(buffer->native(), 0, 1);
        encoder.setBytes(&count, sizeof(count), 2);
        encoder.setBytes(&block, sizeof(block), 3);
        encoder.setBytes(&output_block, sizeof(output_block), 4);
        encoder.setBytes(&offset, sizeof(offset), 5);
      });
      if (!status.ok()) return status;
    }
    offset += block;
  }
  return result;
}

namespace {
std::uint32_t predicate_operation_code(OpKind kind) {
  switch (kind) {
    case OpKind::kEqual: return 0; case OpKind::kNotEqual: return 1;
    case OpKind::kLess: return 2; case OpKind::kLessEqual: return 3;
    case OpKind::kGreater: return 4; case OpKind::kGreaterEqual: return 5;
    case OpKind::kLogicalAnd: return 6; case OpKind::kLogicalOr: return 7;
    case OpKind::kLogicalXor: return 8; case OpKind::kLogicalNot: return 9;
    case OpKind::kWhere: return 10;
    default: throw std::invalid_argument("unknown predicate operation");
  }
}
std::uint32_t predicate_dtype_code(DType dtype) {
  return dtype == DType::kFloat32 ? 0 : dtype == DType::kInt32 ? 1 : 2;
}
Expected<std::shared_ptr<MetalBuffer>> index_metadata_buffer(Shape metadata) {
  // Metal requires a bound buffer even when scalar kernels never index it.
  if (metadata.empty()) metadata.push_back(0);
  if (metadata.size() > std::numeric_limits<std::size_t>::max() / sizeof(Dim))
    return Status(StatusCode::kInvalidArgument, "predicate metadata size overflow");
  auto buffer = MetalBuffer::create(DType::kInt32, metadata.size() * 2);
  if (!buffer) return buffer.status();
  const auto status = buffer.value()->copy_from_host(metadata.data(), metadata.size() * sizeof(Dim));
  if (!status.ok()) return status;
  return buffer;
}
}

Expected<MetalTensor> execute_inference_primitive(const OpDesc& op,const std::vector<MetalTensor>& inputs) {
  std::vector<Tensor> descriptors;for(const auto& t:inputs)descriptors.push_back(to_core_tensor(t));
  if(op.kind==OpKind::kEmbedding) {
    const auto p=make_embedding_plan(descriptors);
    auto count_result=checked_thread_count(numel(p.shape));if(!count_result)return count_result.status();
    auto indices_result=checked_thread_count(p.indices);if(!indices_result)return indices_result.status();
    if(p.indices) {
      auto flag=MetalBuffer::create(DType::kInt32,1);if(!flag)return flag.status();
      std::uint32_t invalid=0;auto status=flag.value()->copy_from_host(&invalid,sizeof(invalid));if(!status.ok())return status;
      auto pipeline=runtime().pipeline("embedding_validate");if(!pipeline)return pipeline.status();
      const std::uint64_t count=p.indices;
      status=run_threads(*pipeline.value(),indices_result.value(),[&](MTL::ComputeCommandEncoder& encoder){
        encoder.setBuffer(inputs[0].buffer()->native(),0,0);encoder.setBuffer(flag.value()->native(),0,1);
        encoder.setBytes(&count,sizeof(count),2);encoder.setBytes(&p.vocabulary,sizeof(p.vocabulary),3);
      });if(!status.ok())return status;
      status=flag.value()->copy_to_host(&invalid,sizeof(invalid));if(!status.ok())return status;
      if(invalid)return Status(StatusCode::kInvalidArgument,"embedding index out of range");
    }
    const auto count=count_result.value();auto buffer=MetalBuffer::create(DType::kFloat32,count);if(!buffer)return buffer.status();
    if(count) {
      auto pipeline=runtime().pipeline("embedding_gather");if(!pipeline)return pipeline.status();const std::uint64_t width=p.width;
      auto status=run_threads(*pipeline.value(),count,[&](MTL::ComputeCommandEncoder& encoder){
        encoder.setBuffer(inputs[0].buffer()->native(),0,0);encoder.setBuffer(inputs[1].buffer()->native(),0,1);encoder.setBuffer(buffer.value()->native(),0,2);
        encoder.setBytes(&count,sizeof(count),3);encoder.setBytes(&width,sizeof(width),4);
      });if(!status.ok())return status;
    }
    return MetalTensor(DType::kFloat32,p.shape,buffer.move_value());
  }
  const auto p=make_attention_softmax_plan(op,descriptors);
  auto checked=checked_thread_count(numel(p.shape));if(!checked)return checked.status();
  auto buffer=MetalBuffer::create(DType::kFloat32,checked.value());if(!buffer)return buffer.status();
  if(p.rows) {
    auto metadata=index_metadata_buffer(p.metadata);if(!metadata)return metadata.status();
    auto pipeline=runtime().pipeline("attention_softmax");if(!pipeline)return pipeline.status();
    const std::uint32_t rows=p.rows,causal=op.causal;const std::uint64_t columns=p.columns,rank=p.shape.size();
    auto status=run_threads(*pipeline.value(),rows,[&](MTL::ComputeCommandEncoder& encoder){
      encoder.setBuffer(inputs[0].buffer()->native(),0,0);encoder.setBuffer(inputs[inputs.size()==2?1:0].buffer()->native(),0,1);
      encoder.setBuffer(buffer.value()->native(),0,2);encoder.setBuffer(metadata.value()->native(),0,3);
      encoder.setBytes(&rows,sizeof(rows),4);encoder.setBytes(&columns,sizeof(columns),5);encoder.setBytes(&rank,sizeof(rank),6);
      encoder.setBytes(&p.scale,sizeof(p.scale),7);encoder.setBytes(&p.mask_kind,sizeof(p.mask_kind),8);encoder.setBytes(&causal,sizeof(causal),9);
    });if(!status.ok())return status;
  }
  return MetalTensor(DType::kFloat32,p.shape,buffer.move_value());
}

Expected<std::vector<MetalTensor>> execute_math(const OpDesc& op,const std::vector<MetalTensor>& inputs) {
  std::vector<Tensor> descriptors;for(const auto& t:inputs)descriptors.push_back(to_core_tensor(t));
  const auto p=make_math_plan(op,descriptors);
  auto checked=checked_thread_count(numel(p.shape));if(!checked)return checked.status();
  auto output=MetalBuffer::create(p.dtype,checked.value());if(!output)return output.status();
  std::vector<MetalTensor> results;results.emplace_back(p.dtype,p.shape,output.value());
  std::shared_ptr<MetalBuffer> indices=output.value();
  if(p.code==6){auto buffer=MetalBuffer::create(DType::kInt32,checked.value());if(!buffer)return buffer.status();indices=buffer.move_value();results.emplace_back(DType::kInt32,p.shape,indices);}
  if(!p.groups)return results;
  Shape metadata{p.groups,p.reduce,p.inner,p.k,static_cast<Dim>(p.rank),p.code,
                 inputs[0].dtype()==DType::kFloat32,op.largest,op.sorted};
  metadata.insert(metadata.end(),p.metadata.begin(),p.metadata.end());
  auto gpu=index_metadata_buffer(std::move(metadata));if(!gpu)return gpu.status();
  auto pipeline=runtime().pipeline("math_values");if(!pipeline)return pipeline.status();
  auto status=run_threads(*pipeline.value(),static_cast<std::uint32_t>(p.groups),[&](MTL::ComputeCommandEncoder& encoder){
    for(std::size_t i=0;i<3;++i)encoder.setBuffer(inputs[i<inputs.size()?i:0].buffer()->native(),0,i);
    encoder.setBuffer(output.value()->native(),0,3);encoder.setBuffer(indices->native(),0,4);encoder.setBuffer(gpu.value()->native(),0,5);
  });
  if(!status.ok())return status;
  return results;
}

Expected<MetalTensor> execute_predicate(const OpDesc& op, const std::vector<MetalTensor>& inputs) {
  if (is_predicate_elementwise(op.kind)) {
    std::vector<Tensor> descriptors;
    for (const auto& input : inputs) descriptors.push_back(to_core_tensor(input));
    const auto plan = make_predicate_plan(op, descriptors);
    auto count_result = checked_thread_count(numel(plan.output_shape));
    if (!count_result) return count_result.status();
    const auto count = count_result.value();
    auto buffer = MetalBuffer::create(plan.output_dtype, count);
    if (!buffer) return buffer.status();
    MetalTensor result(plan.output_dtype, plan.output_shape, buffer.value());
    if (!count) return result;
    Shape metadata = plan.output_shape;
    for (const auto& strides : plan.input_strides) metadata.insert(metadata.end(), strides.begin(), strides.end());
    auto gpu_metadata = index_metadata_buffer(std::move(metadata));
    if (!gpu_metadata) return gpu_metadata.status();
    auto pipeline = runtime().pipeline("predicate_values");
    if (!pipeline) return pipeline.status();
    const std::uint64_t rank = plan.output_shape.size();
    const auto code = predicate_operation_code(op.kind);
    const auto dtype = predicate_dtype_code(inputs[op.kind == OpKind::kWhere ? 1 : 0].dtype());
    const auto status = run_threads(*pipeline.value(), count, [&](MTL::ComputeCommandEncoder& encoder) {
      for (std::size_t i = 0; i < 3; ++i) encoder.setBuffer(inputs[i < inputs.size() ? i : 0].buffer()->native(), 0, i);
      encoder.setBuffer(buffer.value()->native(), 0, 3);
      encoder.setBytes(&count, sizeof(count), 4); encoder.setBuffer(gpu_metadata.value()->native(), 0, 5);
      encoder.setBytes(&rank, sizeof(rank), 6); encoder.setBytes(&code, sizeof(code), 7);
      encoder.setBytes(&dtype, sizeof(dtype), 8);
    });
    if (!status.ok()) return status;
    return result;
  }
  const auto& input = inputs[0];
  if (op.kind == OpKind::kAny || op.kind == OpKind::kAll) {
    if (input.dtype() != DType::kBool) return Status(StatusCode::kInvalidArgument, "any/all require bool tensors");
    const auto axes = op.reduction_axes.value_or(input.shape().empty() ? Shape{} : Shape{op.axis});
    const auto plan = make_reduction_plan(input.shape(), axes);
    if (!plan.reduction_size)
      return fill(OpDesc{OpKind::kFill}, plan.output_shape, DType::kBool, op.kind == OpKind::kAll);
    auto count_result = checked_thread_count(numel(plan.output_shape));
    if (!count_result) return count_result.status();
    const auto count = count_result.value();
    auto buffer = MetalBuffer::create(DType::kBool, count);
    if (!buffer) return buffer.status();
    MetalTensor result(DType::kBool, plan.output_shape, buffer.value());
    if (!count) return result;
    auto metadata = index_metadata_buffer(plan.index_metadata);
    if (!metadata) return metadata.status();
    auto pipeline = runtime().pipeline("reduce_bool");
    if (!pipeline) return pipeline.status();
    const std::uint64_t rank = input.shape().size();
    const std::uint32_t all = op.kind == OpKind::kAll;
    const auto status = run_threads(*pipeline.value(), count, [&](MTL::ComputeCommandEncoder& encoder) {
      encoder.setBuffer(input.buffer()->native(), 0, 0); encoder.setBuffer(buffer.value()->native(), 0, 1);
      encoder.setBytes(&count, sizeof(count), 2); encoder.setBytes(&plan.reduction_size, sizeof(plan.reduction_size), 3);
      encoder.setBuffer(metadata.value()->native(), 0, 4); encoder.setBytes(&rank, sizeof(rank), 5);
      encoder.setBytes(&all, sizeof(all), 6);
    });
    if (!status.ok()) return status;
    return result;
  }
  const auto& mask = inputs[1];
  const auto plan = make_masked_select_plan(to_core_tensor(input), to_core_tensor(mask));
  auto mask_count_result = checked_thread_count(mask.size());
  auto input_count_result = checked_thread_count(input.size());
  if (!mask_count_result) return mask_count_result.status();
  if (!input_count_result) return input_count_result.status();
  const auto count = mask_count_result.value(), input_count = input_count_result.value();
  Dim selected = 0;
  std::shared_ptr<MetalBuffer> prefix;
  if (count) {
    auto first = MetalBuffer::create(DType::kInt32, static_cast<std::size_t>(count) * 2);
    auto second = MetalBuffer::create(DType::kInt32, static_cast<std::size_t>(count) * 2);
    if (!first) return first.status();
    if (!second) return second.status();
    prefix = first.move_value(); auto temporary = second.move_value();
    auto initialize = runtime().pipeline("mask_prefix"), scan = runtime().pipeline("mask_scan");
    if (!initialize) return initialize.status();
    if (!scan) return scan.status();
    auto status = run_threads(*initialize.value(), count, [&](MTL::ComputeCommandEncoder& encoder) {
      encoder.setBuffer(mask.buffer()->native(), 0, 0); encoder.setBuffer(prefix->native(), 0, 1);
      encoder.setBytes(&count, sizeof(count), 2);
    });
    if (!status.ok()) return status;
    for (std::uint64_t step = 1; step < count; step *= 2) {
      status = run_threads(*scan.value(), count, [&](MTL::ComputeCommandEncoder& encoder) {
        encoder.setBuffer(prefix->native(), 0, 0); encoder.setBuffer(temporary->native(), 0, 1);
        encoder.setBytes(&count, sizeof(count), 2); encoder.setBytes(&step, sizeof(step), 3);
      });
      if (!status.ok()) return status;
      std::swap(prefix, temporary);
    }
    // All commands completed; shared storage permits reading only the final
    // eight-byte count. The tensor and mask contents stay on the device.
    std::memcpy(&selected, static_cast<const Dim*>(prefix->native()->contents()) + count - 1, sizeof(selected));
  }
  Shape shape{selected}; shape.insert(shape.end(), plan.tail_shape.begin(), plan.tail_shape.end());
  const auto output_count = numel(shape);
  auto buffer = MetalBuffer::create(input.dtype(), static_cast<std::size_t>(output_count));
  if (!buffer) return buffer.status();
  MetalTensor result(input.dtype(), shape, buffer.value());
  if (output_count) {
    auto pipeline = runtime().pipeline("mask_copy");
    if (!pipeline) return pipeline.status();
    const std::uint64_t block = plan.block_size;
    const auto dtype = predicate_dtype_code(input.dtype());
    const auto status = run_threads(*pipeline.value(), input_count, [&](MTL::ComputeCommandEncoder& encoder) {
      encoder.setBuffer(input.buffer()->native(), 0, 0); encoder.setBuffer(mask.buffer()->native(), 0, 1);
      encoder.setBuffer(prefix->native(), 0, 2); encoder.setBuffer(buffer.value()->native(), 0, 3);
      encoder.setBytes(&input_count, sizeof(input_count), 4); encoder.setBytes(&block, sizeof(block), 5);
      encoder.setBytes(&dtype, sizeof(dtype), 6);
    });
    if (!status.ok()) return status;
  }
  return result;
}

Expected<MetalTensor> execute_unary(const OpDesc& op, const MetalTensor& input) {
  if (op.kind == OpKind::kTranspose || op.kind == OpKind::kSlice) {
    return execute_transpose(op, input);
  }
  if (op.kind == OpKind::kCast) {
    return execute_cast(op, input);
  }
  if (input.dtype() == DType::kBool)
    return Status(StatusCode::kInvalidArgument, "numeric operations do not support bool tensors");
  if (op.kind == OpKind::kAddScalar || op.kind == OpKind::kSubtractScalar ||
      op.kind == OpKind::kMultiplyScalar || op.kind == OpKind::kDivideScalar) {
    return execute_scalar(op, input);
  }
  if (op.kind == OpKind::kSoftmax) {
    return execute_softmax(op, input);
  }
  if (op.kind == OpKind::kRmsNorm) {
    return execute_rmsnorm(op, input);
  }
  if (op.kind == OpKind::kLayerNorm) {
    return execute_layernorm(op, input);
  }

  auto thread_count_result = checked_thread_count(input.size());
  if (!thread_count_result) {
    return thread_count_result.status();
  }
  const auto thread_count = thread_count_result.move_value();
  auto kernel_name_result = unary_kernel_name(op.kind, input.dtype());
  if (!kernel_name_result) {
    return kernel_name_result.status();
  }
  auto output_buffer_result =
      MetalBuffer::create(input.dtype(), static_cast<std::size_t>(input.size()));
  if (!output_buffer_result) {
    return output_buffer_result.status();
  }
  auto output_buffer = output_buffer_result.move_value();
  MetalTensor output(input.dtype(), input.shape(), output_buffer);
  if (thread_count == 0) {
    return output;
  }
  auto pipeline_result = runtime().pipeline(kernel_name_result.move_value());
  if (!pipeline_result) {
    return pipeline_result.status();
  }
  const Status run_status =
      run_threads(*pipeline_result.move_value(), thread_count, [&](MTL::ComputeCommandEncoder& encoder) {
        encoder.setBuffer(input.buffer()->native(), 0, 0);
        encoder.setBuffer(output_buffer->native(), 0, 1);
        encoder.setBytes(&thread_count, sizeof(thread_count), 2);
      });
  if (!run_status.ok()) {
    return run_status;
  }
  return output;
}

Expected<MetalTensor> execute_binary(const OpDesc& op, const MetalTensor& lhs, const MetalTensor& rhs) {
  if (lhs.dtype() == DType::kBool || rhs.dtype() == DType::kBool)
    return Status(StatusCode::kInvalidArgument, "arithmetic does not support bool tensors");
  if (lhs.dtype() != rhs.dtype()) {
    return Status(StatusCode::kInvalidArgument, "dtype mismatch for Metal binary operation");
  }

  auto kernel_name_result = binary_kernel_name(op.kind, lhs.dtype());
  if (!kernel_name_result) {
    return kernel_name_result.status();
  }

  BroadcastPlan plan;
  try {
    plan = make_broadcast_plan(lhs.shape(), rhs.shape());
  } catch (const std::invalid_argument& error) {
    return Status(StatusCode::kInvalidArgument, error.what());
  }
  const auto output_elements = numel(plan.output_shape);
  auto thread_count_result = checked_thread_count(output_elements);
  if (!thread_count_result) {
    return thread_count_result.status();
  }
  const auto thread_count = thread_count_result.move_value();
  auto output_buffer_result =
      MetalBuffer::create(lhs.dtype(), static_cast<std::size_t>(output_elements));
  if (!output_buffer_result) {
    return output_buffer_result.status();
  }
  auto output_buffer = output_buffer_result.move_value();
  MetalTensor output(lhs.dtype(), plan.output_shape, output_buffer);
  if (thread_count == 0) {
    return output;
  }
  const bool broadcasting = lhs.shape() != rhs.shape();
  std::shared_ptr<MetalBuffer> metadata_buffer;
  const std::uint64_t rank = plan.output_shape.size();
  std::uint32_t operation = 0;
  if (broadcasting) {
    // Only shape/stride metadata is uploaded. A device buffer avoids setBytes'
    // small inline-data limit for high-rank tensors.
    if (rank > std::numeric_limits<std::size_t>::max() / (3 * sizeof(std::uint64_t))) {
      return Status(StatusCode::kInvalidArgument, "broadcast metadata byte size overflow");
    }
    std::vector<std::uint64_t> metadata;
    metadata.reserve(static_cast<std::size_t>(rank) * 3);
    for (std::size_t axis = 0; axis < rank; ++axis) {
      metadata.push_back(static_cast<std::uint64_t>(plan.output_shape[axis]));
      metadata.push_back(static_cast<std::uint64_t>(plan.lhs_strides[axis]));
      metadata.push_back(static_cast<std::uint64_t>(plan.rhs_strides[axis]));
    }
    auto allocated = MetalBuffer::create(DType::kInt32, metadata.size() * 2);
    if (!allocated) return allocated.status();
    metadata_buffer = allocated.move_value();
    const auto copied = metadata_buffer->copy_from_host(metadata.data(), metadata_buffer->nbytes());
    if (!copied.ok()) return copied;
    auto code = arithmetic_operation_code(op.kind);
    if (!code) return code.status();
    operation = code.move_value();
  }
  auto pipeline_result = runtime().pipeline(broadcasting
      ? (lhs.dtype() == DType::kFloat32 ? "broadcast_f32" : "broadcast_i32")
      : kernel_name_result.move_value());
  if (!pipeline_result) {
    return pipeline_result.status();
  }
  const Status run_status = run_threads(*pipeline_result.move_value(), thread_count, [&](MTL::ComputeCommandEncoder& encoder) {
    encoder.setBuffer(lhs.buffer()->native(), 0, 0);
    encoder.setBuffer(rhs.buffer()->native(), 0, 1);
    encoder.setBuffer(output_buffer->native(), 0, 2);
    encoder.setBytes(&thread_count, sizeof(thread_count), 3);
    if (metadata_buffer) {
      encoder.setBuffer(metadata_buffer->native(), 0, 4);
      encoder.setBytes(&rank, sizeof(rank), 5);
      encoder.setBytes(&operation, sizeof(operation), 6);
    }
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
  // Reject non-representable int32 fill values before allocating, so the host
  // double->int32 cast below is never UB and matches the CPU reference.
  if (dtype == DType::kInt32 && !is_int32_representable(value)) {
    return Status(StatusCode::kInvalidArgument, "fill value is out of range for int32");
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
      case DType::kBool: {
        const std::uint8_t fill_value = value != 0;
        encoder.setBytes(&fill_value, sizeof(fill_value), 1);
        break;
      }
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

static Expected<MetalTensor> reduce_selected_axes(const OpDesc& op, const MetalTensor& input) {
  if (op.kind != OpKind::kSum && op.kind != OpKind::kMax && op.kind != OpKind::kMean)
    return Status(StatusCode::kInvalidArgument, "unsupported Metal reduction operation");
  ReductionPlan plan;
  try {
    plan = make_reduction_plan(input.shape(), *op.reduction_axes);
  } catch (const std::invalid_argument& error) {
    return Status(StatusCode::kInvalidArgument, error.what());
  }
  if (op.kind == OpKind::kMean && input.dtype() != DType::kFloat32)
    return Status(StatusCode::kInvalidArgument, "mean only supports float32 tensors");
  if (op.kind == OpKind::kMax && plan.reduction_size == 0)
    return Status(StatusCode::kInvalidArgument, "max reduction requires non-empty axes");
  auto input_count = checked_thread_count(input.size());
  if (!input_count) return input_count.status();
  if (op.reduction_axes->empty()) {
    OpDesc copy{OpKind::kCast}; copy.target_dtype = input.dtype();
    return execute_cast(copy, input);
  }
  auto count_result = checked_thread_count(numel(plan.output_shape));
  if (!count_result) return count_result.status();
  const auto count = count_result.move_value();
  auto reduce_result = checked_reduction_extent_for_metal(plan.reduction_size);
  if (!reduce_result) return reduce_result.status();
  const auto reduce_count = reduce_result.move_value();
  auto allocated = MetalBuffer::create(input.dtype(), count);
  if (!allocated) return allocated.status();
  auto buffer = allocated.move_value();
  MetalTensor result(input.dtype(), plan.output_shape, buffer);
  if (count == 0) return result;
  if (reduce_count == 0) {
    const auto status = fill_empty_reduction_output(op.kind, input.dtype(), count, buffer);
    if (!status.ok()) return status;
    return result;
  }
  if (plan.index_metadata.size() > std::numeric_limits<std::size_t>::max() / sizeof(Dim))
    return Status(StatusCode::kInvalidArgument, "Metal reduction metadata size overflow");
  auto metadata_result = MetalBuffer::create(DType::kInt32, plan.index_metadata.size() * 2);
  if (!metadata_result) return metadata_result.status();
  auto metadata = metadata_result.move_value();
  auto status = metadata->copy_from_host(plan.index_metadata.data(), metadata->nbytes());
  if (!status.ok()) return status;
  auto pipeline = runtime().pipeline(input.dtype() == DType::kFloat32 ? "reduce_axes_f32" : "reduce_axes_i32");
  if (!pipeline) return pipeline.status();
  const std::uint64_t output_rank = plan.output_shape.size(), rank = input.shape().size();
  const std::uint32_t operation = op.kind == OpKind::kSum ? 0 : op.kind == OpKind::kMax ? 1 : 2;
  status = run_threads(*pipeline.move_value(), count, [&](MTL::ComputeCommandEncoder& encoder) {
    encoder.setBuffer(input.buffer()->native(), 0, 0);
    encoder.setBuffer(buffer->native(), 0, 1);
    encoder.setBytes(&count, sizeof(count), 2);
    encoder.setBytes(&reduce_count, sizeof(reduce_count), 3);
    encoder.setBuffer(metadata->native(), 0, 4);
    encoder.setBytes(&output_rank, sizeof(output_rank), 5);
    encoder.setBytes(&rank, sizeof(rank), 6);
    encoder.setBytes(&operation, sizeof(operation), 7);
  });
  if (!status.ok()) return status;
  return result;
}

Expected<MetalTensor> reduce(const OpDesc& op, const MetalTensor& input) {
  if (input.dtype() == DType::kBool)
    return Status(StatusCode::kInvalidArgument, "numeric reductions do not support bool tensors");
  if (op.reduction_axes) return reduce_selected_axes(op, input);
  auto dims_result = checked_reduction_dims(input, op);
  if (!dims_result) {
    return dims_result.status();
  }
  auto dims = dims_result.move_value();
  if (op.kind == OpKind::kMean && input.dtype() != DType::kFloat32) {
    return Status(StatusCode::kInvalidArgument, "mean only supports float32 tensors");
  }

  auto output_buffer_result =
      MetalBuffer::create(input.dtype(), static_cast<std::size_t>(dims.output_elements));
  if (!output_buffer_result) {
    return output_buffer_result.status();
  }
  auto output_buffer = output_buffer_result.move_value();
  MetalTensor output(input.dtype(), dims.output_shape, output_buffer);
  if (dims.output_elements == 0) {
    return output;
  }
  if (dims.reduce_elements == 0) {
    const Status fill_status = fill_empty_reduction_output(
        op.kind, input.dtype(), dims.output_elements, output_buffer);
    if (!fill_status.ok()) {
      return fill_status;
    }
    return output;
  }

  auto kernel_name_result = reduction_kernel_name(op.kind, input.dtype());
  if (!kernel_name_result) {
    return kernel_name_result.status();
  }
  auto pipeline_result = runtime().pipeline(kernel_name_result.move_value());
  if (!pipeline_result) {
    return pipeline_result.status();
  }
  const Status run_status =
      run_threads(*pipeline_result.move_value(), dims.output_elements, [&](MTL::ComputeCommandEncoder& encoder) {
        encoder.setBuffer(input.buffer()->native(), 0, 0);
        encoder.setBuffer(output_buffer->native(), 0, 1);
        encoder.setBytes(&dims.output_elements, sizeof(dims.output_elements), 2);
        encoder.setBytes(&dims.reduce_elements, sizeof(dims.reduce_elements), 3);
        encoder.setBytes(&dims.inner_elements, sizeof(dims.inner_elements), 4);
      });
  if (!run_status.ok()) {
    return run_status;
  }
  return output;
}

Expected<MatmulPlan> checked_metal_matmul_plan(const MetalTensor& lhs, const MetalTensor& rhs) {
  if (lhs.dtype() != DType::kFloat32 || rhs.dtype() != DType::kFloat32)
    return Status(StatusCode::kInvalidArgument, "Metal matmul only supports float32 tensors");
  try {
    auto plan = make_matmul_plan(lhs.shape(), rhs.shape());
    auto count = checked_thread_count(numel(plan.output_shape));
    if (!count) return count.status();
    return plan;
  } catch (const std::invalid_argument& error) {
    return Status(StatusCode::kInvalidArgument, error.what());
  }
}

Expected<MetalTensor> matmul_custom(const MetalTensor& lhs, const MetalTensor& rhs) {
  auto planned = checked_metal_matmul_plan(lhs, rhs);
  if (!planned) return planned.status();
  const auto plan = planned.move_value();
  const auto count = static_cast<std::uint32_t>(numel(plan.output_shape));
  auto allocated = MetalBuffer::create(DType::kFloat32, count);
  if (!allocated) return allocated.status();
  auto buffer = allocated.move_value();
  MetalTensor output(DType::kFloat32, plan.output_shape, buffer);
  if (!count) return output;
  Shape metadata = plan.batch_shape;
  metadata.insert(metadata.end(), plan.lhs_batch_strides.begin(), plan.lhs_batch_strides.end());
  metadata.insert(metadata.end(), plan.rhs_batch_strides.begin(), plan.rhs_batch_strides.end());
  auto gpu_metadata = index_metadata_buffer(std::move(metadata));
  if (!gpu_metadata) return gpu_metadata.status();
  auto pipeline = runtime().pipeline("matmul_f32");
  if (!pipeline) return pipeline.status();
  const std::uint64_t rank = plan.batch_shape.size();
  auto status = run_threads(*pipeline.move_value(), count, [&](MTL::ComputeCommandEncoder& encoder) {
    encoder.setBuffer(lhs.buffer()->native(), 0, 0);
    encoder.setBuffer(rhs.buffer()->native(), 0, 1);
    encoder.setBuffer(buffer->native(), 0, 2);
    encoder.setBytes(&plan.m, sizeof(plan.m), 3);
    encoder.setBytes(&plan.k, sizeof(plan.k), 4);
    encoder.setBytes(&plan.n, sizeof(plan.n), 5);
    encoder.setBytes(&count, sizeof(count), 6);
    encoder.setBuffer(gpu_metadata.value()->native(), 0, 7);
    encoder.setBytes(&rank, sizeof(rank), 8);
  });
  if (!status.ok()) return status;
  return output;
}

}  // namespace tensorcx::metal
