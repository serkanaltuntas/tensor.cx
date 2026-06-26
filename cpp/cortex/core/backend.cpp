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

}  // namespace cortex
