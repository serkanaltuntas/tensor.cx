#include "cortex/backends/null/null_backend.h"

#include <array>
#include <optional>
#include <span>
#include <string>

#include "cortex/core/dtype.h"
#include "cortex/core/shape.h"

namespace cortex::null_backend {
namespace {

Status validate_execution_contract(const BackendExecution& execution) {
  if (execution.op_class == BackendOpClass::kPrimitive) {
    return validate_primitive_execution_contract(execution);
  }

  if (execution.outputs.empty()) {
    return Status(
        StatusCode::kInvalidArgument,
        "kernel backend execution requires output metadata");
  }
  if (!execution.launch.has_value()) {
    return Status(StatusCode::kInvalidArgument, "kernel backend execution requires launch metadata");
  }
  if (!execution.compilation_target.has_value()) {
    return Status(
        StatusCode::kInvalidArgument,
        "kernel backend execution requires a compilation target");
  }
  if (execution.compilation_target->entry_point.empty()) {
    return Status(StatusCode::kInvalidArgument, "kernel compilation target requires an entry point");
  }
  if (execution.compilation_target->artifact_kind == KernelArtifactKind::kNone) {
    return Status(
        StatusCode::kInvalidArgument,
        "kernel compilation target requires an artifact kind");
  }
  if (execution.compilation_target->artifact.empty()) {
    return Status(
        StatusCode::kInvalidArgument,
        "kernel compilation target requires an artifact");
  }
  return Status::Ok();
}

Status expect_status_code(const char* scenario, const Status& status, StatusCode expected) {
  if (status.code() == expected) {
    return Status::Ok();
  }
  return Status(
      StatusCode::kInternal,
      std::string("null backend contract smoke test failed: ") + scenario);
}

}  // namespace

std::string NullBackend::name() const { return "null"; }

Status NullBackend::execute(const BackendExecution& execution) {
  const Status contract = validate_execution_contract(execution);
  if (!contract.ok()) {
    return contract;
  }
  return Status(StatusCode::kUnavailable, "null backend does not execute operations");
}

Status contract_smoke_test() {
  NullBackend backend;
  if (backend.name() != "null") {
    return Status(StatusCode::kInternal, "null backend contract smoke test failed: backend name");
  }

  std::array<Tensor, 2> binary_inputs{};
  const std::span<const Tensor> binary_input_span(binary_inputs.data(), binary_inputs.size());
  const Shape fill_shape{2};
  const Tensor fill_descriptor{
      DType::kFloat32,
      fill_shape,
      contiguous_strides(fill_shape),
      Device{"null", 0},
      nullptr,
      0,
  };
  std::array<Tensor, 1> outputs{};
  const std::span<Tensor> output_span(outputs.data(), outputs.size());
  const BackendExecution valid_primitive{
      BackendOpClass::kPrimitive,
      OpDesc{OpKind::kAdd},
      binary_input_span,
      output_span,
      std::nullopt,
      std::nullopt,
  };
  if (Status status = expect_status_code(
          "valid primitive",
          backend.execute(valid_primitive),
          StatusCode::kUnavailable);
      !status.ok()) {
    return status;
  }

  BackendExecution missing_input = valid_primitive;
  missing_input.inputs = {};
  if (Status status = expect_status_code(
          "missing primitive input",
          backend.execute(missing_input),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  BackendExecution empty_outputs = valid_primitive;
  empty_outputs.outputs = {};
  if (Status status = expect_status_code(
          "empty primitive outputs",
          backend.execute(empty_outputs),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  std::array<Tensor, 2> two_outputs{};
  BackendExecution too_many_outputs = valid_primitive;
  too_many_outputs.outputs = std::span<Tensor>(two_outputs.data(), two_outputs.size());
  if (Status status = expect_status_code(
          "too many primitive outputs",
          backend.execute(too_many_outputs),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  std::array<Tensor, 1> unary_inputs{};
  BackendExecution valid_unary = valid_primitive;
  valid_unary.op = OpDesc{OpKind::kSoftmax, 0};
  valid_unary.inputs = std::span<const Tensor>(unary_inputs.data(), unary_inputs.size());
  if (Status status = expect_status_code(
          "valid unary primitive",
          backend.execute(valid_unary),
          StatusCode::kUnavailable);
      !status.ok()) {
    return status;
  }

  BackendExecution valid_fill = valid_primitive;
  valid_fill.op = OpDesc{OpKind::kFill};
  valid_fill.inputs = {};
  outputs[0] = fill_descriptor;
  if (Status status = expect_status_code(
          "valid fill primitive",
          backend.execute(valid_fill),
          StatusCode::kUnavailable);
      !status.ok()) {
    return status;
  }

  BackendExecution fill_default_output = valid_fill;
  outputs[0] = Tensor{};
  if (Status status = expect_status_code(
          "fill default output descriptor",
          backend.execute(fill_default_output),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  outputs[0] = fill_descriptor;
  BackendExecution fill_with_input = valid_fill;
  fill_with_input.inputs = binary_input_span;
  if (Status status = expect_status_code(
          "fill primitive with input",
          backend.execute(fill_with_input),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  BackendExecution primitive_with_launch = valid_primitive;
  primitive_with_launch.launch = LaunchConfig{1, 1, 1, 1, 1, 1};
  if (Status status = expect_status_code(
          "primitive with launch metadata",
          backend.execute(primitive_with_launch),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  BackendExecution primitive_with_target = valid_primitive;
  primitive_with_target.compilation_target =
      CompilationTarget{KernelArtifactKind::kStaticLibrary, "noop_library", "noop"};
  if (Status status = expect_status_code(
          "primitive with compilation target",
          backend.execute(primitive_with_target),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  BackendExecution kernel_missing_launch = valid_primitive;
  kernel_missing_launch.op_class = BackendOpClass::kKernel;
  kernel_missing_launch.compilation_target =
      CompilationTarget{KernelArtifactKind::kStaticLibrary, "noop_library", "noop"};
  if (Status status = expect_status_code(
          "kernel missing launch",
          backend.execute(kernel_missing_launch),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  BackendExecution kernel_missing_target = valid_primitive;
  kernel_missing_target.op_class = BackendOpClass::kKernel;
  kernel_missing_target.launch = LaunchConfig{1, 1, 1, 1, 1, 1};
  if (Status status = expect_status_code(
          "kernel missing compilation target",
          backend.execute(kernel_missing_target),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  BackendExecution kernel_empty_entry = kernel_missing_target;
  kernel_empty_entry.compilation_target =
      CompilationTarget{KernelArtifactKind::kStaticLibrary, "noop_library", ""};
  if (Status status = expect_status_code(
          "kernel empty entry point",
          backend.execute(kernel_empty_entry),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  BackendExecution kernel_no_artifact_kind = kernel_missing_target;
  kernel_no_artifact_kind.compilation_target =
      CompilationTarget{KernelArtifactKind::kNone, "noop_library", "noop"};
  if (Status status = expect_status_code(
          "kernel missing artifact kind",
          backend.execute(kernel_no_artifact_kind),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  BackendExecution kernel_empty_artifact = kernel_missing_target;
  kernel_empty_artifact.compilation_target =
      CompilationTarget{KernelArtifactKind::kStaticLibrary, "", "noop"};
  if (Status status = expect_status_code(
          "kernel empty artifact",
          backend.execute(kernel_empty_artifact),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  BackendExecution valid_kernel = kernel_missing_target;
  valid_kernel.compilation_target =
      CompilationTarget{KernelArtifactKind::kStaticLibrary, "noop_library", "noop"};
  if (Status status =
          expect_status_code("valid kernel", backend.execute(valid_kernel), StatusCode::kUnavailable);
      !status.ok()) {
    return status;
  }
  return Status::Ok();
}

}  // namespace cortex::null_backend
