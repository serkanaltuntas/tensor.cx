#include "cortex/core/backend.h"

#include <exception>
#include <string>
#include <string_view>
#include <utility>

#include "cortex/core/shape.h"

namespace cortex {
namespace {

Status invalid_argument_status(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Status validate_concrete_tensor_metadata(const Tensor& tensor, std::string_view context) {
  if (tensor.device.type.empty()) {
    return invalid_argument_status(std::string(context) + " requires a device type");
  }
  if (tensor.device.index != 0) {
    return invalid_argument_status(std::string(context) + " requires device index 0");
  }
  if (tensor.offset != 0) {
    return invalid_argument_status(std::string(context) + " requires offset 0");
  }
  try {
    static_cast<void>(numel(tensor.shape));
    if (tensor.strides != contiguous_strides(tensor.shape)) {
      return invalid_argument_status(std::string(context) + " must be contiguous");
    }
  } catch (const std::exception& error) {
    return invalid_argument_status(error.what());
  }
  if (!tensor.buffer) {
    return invalid_argument_status(std::string(context) + " requires a buffer");
  }
  return Status::Ok();
}

}  // namespace

Status validate_fill_output_descriptor(
    const Tensor& output,
    std::string_view expected_device_type) {
  if (expected_device_type.empty()) {
    if (output.device.type.empty()) {
      return invalid_argument_status("fill output descriptor requires a device type");
    }
  } else if (output.device.type != expected_device_type) {
    return invalid_argument_status(
        "fill output descriptor device type does not match backend");
  }
  if (output.device.index != 0) {
    return invalid_argument_status("fill output descriptor requires device index 0");
  }
  if (output.offset != 0) {
    return invalid_argument_status("fill output descriptor requires offset 0");
  }
  try {
    static_cast<void>(numel(output.shape));
    if (output.strides != contiguous_strides(output.shape)) {
      return invalid_argument_status("fill output descriptor must be contiguous");
    }
  } catch (const std::exception& error) {
    return invalid_argument_status(error.what());
  }
  if (output.buffer) {
    return invalid_argument_status("fill output descriptor must not include a buffer");
  }
  return Status::Ok();
}

Status validate_primitive_execution_contract(
    const BackendExecution& execution,
    std::string_view fill_device_type) {
  if (execution.op_class != BackendOpClass::kPrimitive) {
    return invalid_argument_status("primitive backend execution requires primitive op class");
  }
  if (execution.launch.has_value() || execution.compilation_target.has_value()) {
    return invalid_argument_status(
        "primitive backend execution cannot include kernel launch metadata");
  }
  if (!execution.kernel_arguments.empty()) {
    return invalid_argument_status(
        "primitive backend execution cannot include kernel arguments");
  }
  const auto schema = primitive_op_schema(execution.op.kind);
  if (!schema.has_value()) {
    return invalid_argument_status("primitive operation is not expressible by BackendExecution");
  }
  if (execution.inputs.size() != schema->input_count) {
    return invalid_argument_status("primitive backend execution input count mismatch");
  }
  if (execution.outputs.size() != schema->output_count) {
    return invalid_argument_status("primitive backend execution output count mismatch");
  }
  if (execution.op.kind == OpKind::kFill) {
    return validate_fill_output_descriptor(execution.outputs[0], fill_device_type);
  }
  return Status::Ok();
}

Status validate_kernel_execution_contract(const BackendExecution& execution) {
  if (execution.op_class != BackendOpClass::kKernel) {
    return invalid_argument_status("kernel backend execution requires kernel op class");
  }
  if (!execution.inputs.empty()) {
    return invalid_argument_status(
        "kernel backend execution uses kernel arguments instead of primitive inputs");
  }
  if (execution.outputs.empty()) {
    return invalid_argument_status("kernel backend execution requires output metadata");
  }
  for (const Tensor& output : execution.outputs) {
    const Status status = validate_concrete_tensor_metadata(output, "kernel output metadata");
    if (!status.ok()) {
      return status;
    }
  }
  if (!execution.launch.has_value()) {
    return invalid_argument_status("kernel backend execution requires launch metadata");
  }
  const LaunchConfig& launch = *execution.launch;
  if (launch.grid_x == 0 || launch.grid_y == 0 || launch.grid_z == 0 ||
      launch.threads_per_group_x == 0 || launch.threads_per_group_y == 0 ||
      launch.threads_per_group_z == 0) {
    return invalid_argument_status("kernel launch dimensions must be non-zero");
  }
  if (!execution.compilation_target.has_value()) {
    return invalid_argument_status("kernel backend execution requires a compilation target");
  }
  if (execution.compilation_target->entry_point.empty()) {
    return invalid_argument_status("kernel compilation target requires an entry point");
  }
  if (execution.compilation_target->artifact_kind == KernelArtifactKind::kNone) {
    return invalid_argument_status("kernel compilation target requires an artifact kind");
  }
  if (execution.compilation_target->artifact.empty()) {
    return invalid_argument_status("kernel compilation target requires an artifact");
  }
  for (const KernelArgument& argument : execution.kernel_arguments) {
    switch (argument.kind) {
      case KernelArgumentKind::kTensor:
        if (!argument.tensor) {
          return invalid_argument_status("kernel tensor argument requires tensor metadata");
        }
        if (Status status =
                validate_concrete_tensor_metadata(*argument.tensor, "kernel tensor argument");
            !status.ok()) {
          return status;
        }
        break;
      case KernelArgumentKind::kUInt32:
        break;
    }
  }
  return Status::Ok();
}

}  // namespace cortex
