#include "tensorcx/backends/cuda/cuda_predicate.h"
#include "tensorcx/backends/cuda/cuda_backend.h"
#include "tensorcx/backends/cuda/cuda_buffer.h"
#include "tensorcx/core/predicate.h"
#include <limits>
#include <stdexcept>

namespace tensorcx::cuda {
namespace {
Expected<std::shared_ptr<CudaBuffer>> dim_buffer(Dim count) {
  if (count < 0 || count > std::numeric_limits<Dim>::max() / 2)
    return Status(StatusCode::kInvalidArgument, "predicate metadata size overflow");
  return CudaBuffer::create(DType::kInt32, {count * 2});
}
Expected<std::shared_ptr<CudaBuffer>> upload_metadata(const Shape& metadata) {
  if (metadata.size() > static_cast<std::size_t>(std::numeric_limits<Dim>::max() / 2))
    return Status(StatusCode::kInvalidArgument, "predicate metadata size overflow");
  auto result = dim_buffer(static_cast<Dim>(metadata.size()));
  if (!result) return result.status();
  if (!metadata.empty()) {
    auto status = runtime_status(cudaMemcpy(result.value()->data(), metadata.data(),
        metadata.size() * sizeof(Dim), cudaMemcpyHostToDevice), "predicate metadata upload");
    if (!status.ok()) return status;
  }
  return result;
}
const void* data(const Tensor& tensor) {
  return std::static_pointer_cast<CudaBuffer>(tensor.buffer)->data();
}
}

Status execute_predicate(const BackendExecution& execution) {
  const auto kind = execution.op.kind;
  ContextScope device;
  if (!device.ready()) return device.status();
  if (is_predicate_elementwise(kind)) {
    const auto plan = make_predicate_plan(execution.op, execution.inputs);
    auto output = CudaBuffer::create(plan.output_dtype, plan.output_shape);
    if (!output) return output.status();
    const auto count = numel(plan.output_shape);
    if (count) {
      Shape metadata = plan.output_shape;
      for (const auto& strides : plan.input_strides) metadata.insert(metadata.end(), strides.begin(), strides.end());
      auto gpu_metadata = upload_metadata(metadata);
      if (!gpu_metadata) return gpu_metadata.status();
      const auto status = runtime_status(launch_predicate(data(execution.inputs[0]),
          execution.inputs.size() > 1 ? data(execution.inputs[1]) : nullptr,
          execution.inputs.size() > 2 ? data(execution.inputs[2]) : nullptr,
          output.value()->data(), count, static_cast<const Dim*>(gpu_metadata.value()->data()),
          plan.output_shape.size(), kind, execution.inputs[kind == OpKind::kWhere ? 1 : 0].dtype), "CUDA predicate");
      if (!status.ok()) return status;
    }
    execution.outputs[0] = to_core_tensor(CudaTensor(plan.output_dtype, plan.output_shape, output.move_value()));
    return Status::Ok();
  }
  const auto& input = execution.inputs[0];
  if (kind == OpKind::kAny || kind == OpKind::kAll) {
    if (input.dtype != DType::kBool) return {StatusCode::kInvalidArgument, "any/all require bool tensors"};
    const auto axes = execution.op.reduction_axes.value_or(input.shape.empty() ? Shape{} : Shape{execution.op.axis});
    const auto plan = make_reduction_plan(input.shape, axes);
    auto output = CudaBuffer::create(DType::kBool, plan.output_shape);
    if (!output) return output.status();
    if (numel(plan.output_shape)) {
      auto metadata = upload_metadata(plan.index_metadata);
      if (!metadata) return metadata.status();
      const auto status = runtime_status(launch_bool_reduce(static_cast<const std::uint8_t*>(data(input)),
          static_cast<std::uint8_t*>(output.value()->data()), numel(plan.output_shape), plan.reduction_size,
          static_cast<const Dim*>(metadata.value()->data()), input.shape.size(), kind == OpKind::kAll), "CUDA bool reduction");
      if (!status.ok()) return status;
    }
    execution.outputs[0] = to_core_tensor(CudaTensor(DType::kBool, plan.output_shape, output.move_value()));
    return Status::Ok();
  }
  const auto& mask = execution.inputs[1];
  const auto plan = make_masked_select_plan(input, mask);
  const auto count = numel(mask.shape);
  Dim selected = 0;
  std::shared_ptr<CudaBuffer> prefix;
  if (count) {
    auto first = dim_buffer(count), second = dim_buffer(count);
    if (!first) return first.status();
    if (!second) return second.status();
    prefix = first.move_value(); auto temporary = second.move_value();
    auto status = runtime_status(launch_mask_prefix(static_cast<const std::uint8_t*>(data(mask)),
        static_cast<Dim*>(prefix->data()), count), "CUDA mask prefix initialization");
    if (!status.ok()) return status;
    for (Dim step = 1; step < count;) {
      status = runtime_status(launch_prefix_step(static_cast<const Dim*>(prefix->data()),
          static_cast<Dim*>(temporary->data()), count, step), "CUDA mask prefix scan");
      if (!status.ok()) return status;
      std::swap(prefix, temporary);
      if (step > count / 2) break;
      step *= 2;
    }
    status = runtime_status(cudaMemcpy(&selected, static_cast<const Dim*>(prefix->data()) + count - 1,
        sizeof(Dim), cudaMemcpyDeviceToHost), "CUDA mask result count");
    if (!status.ok()) return status;
  }
  Shape shape{selected}; shape.insert(shape.end(), plan.tail_shape.begin(), plan.tail_shape.end());
  auto output = CudaBuffer::create(input.dtype, shape);
  if (!output) return output.status();
  if (selected && numel(input.shape)) {
    const auto status = runtime_status(launch_mask_copy(data(input), static_cast<const std::uint8_t*>(data(mask)),
        static_cast<const Dim*>(prefix->data()), output.value()->data(), numel(input.shape), plan.block_size, input.dtype),
        "CUDA masked selection");
    if (!status.ok()) return status;
  }
  execution.outputs[0] = to_core_tensor(CudaTensor(input.dtype, shape, output.move_value()));
  return Status::Ok();
}
}  // namespace tensorcx::cuda
