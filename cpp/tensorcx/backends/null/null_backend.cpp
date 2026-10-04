#include "tensorcx/backends/null/null_backend.h"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include "tensorcx/core/buffer.h"
#include "tensorcx/core/dtype.h"
#include "tensorcx/core/shape.h"

namespace tensorcx::null_backend {
namespace {

class SmokeBuffer final : public Buffer {
 public:
  explicit SmokeBuffer(std::size_t nbytes) : nbytes_(nbytes) {}

  std::size_t nbytes() const override { return nbytes_; }

 private:
  std::size_t nbytes_;
};

Status validate_execution_contract(const BackendExecution& execution) {
  if (execution.op_class == BackendOpClass::kPrimitive) {
    return validate_primitive_execution_contract(execution);
  }
  return validate_kernel_execution_contract(execution);
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
      std::span<const KernelArgument>(),
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

  const Shape kernel_output_shape{2};
  std::array<Tensor, 1> kernel_outputs{Tensor{
      DType::kFloat32,
      kernel_output_shape,
      contiguous_strides(kernel_output_shape),
      Device{"null", 0},
      std::make_shared<SmokeBuffer>(2 * dtype_size(DType::kFloat32)),
      0,
  }};
  const std::span<Tensor> kernel_output_span(kernel_outputs.data(), kernel_outputs.size());
  std::array<KernelArgument, 2> kernel_arguments{{
      KernelArgument{KernelArgumentKind::kTensor, &kernel_outputs[0], 0},
      KernelArgument{KernelArgumentKind::kUInt32, nullptr, 2},
  }};
  const std::span<const KernelArgument> kernel_argument_span(
      kernel_arguments.data(),
      kernel_arguments.size());

  const BackendExecution valid_kernel{
      BackendOpClass::kKernel,
      OpDesc{OpKind::kAdd},
      std::span<const Tensor>(),
      kernel_output_span,
      LaunchConfig{1, 1, 1, 1, 1, 1},
      CompilationTarget{KernelArtifactKind::kStaticLibrary, "noop_library", "noop"},
      kernel_argument_span,
  };
  if (Status status =
          expect_status_code("valid kernel", backend.execute(valid_kernel), StatusCode::kUnavailable);
      !status.ok()) {
    return status;
  }

  BackendExecution kernel_with_primitive_inputs = valid_kernel;
  kernel_with_primitive_inputs.inputs = binary_input_span;
  if (Status status = expect_status_code(
          "kernel with primitive inputs",
          backend.execute(kernel_with_primitive_inputs),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  BackendExecution kernel_default_output = valid_kernel;
  outputs[0] = Tensor{};
  kernel_default_output.outputs = output_span;
  if (Status status = expect_status_code(
          "kernel default output metadata",
          backend.execute(kernel_default_output),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  std::array<KernelArgument, 1> invalid_kernel_arguments{{
      KernelArgument{KernelArgumentKind::kTensor, nullptr, 0},
  }};
  BackendExecution kernel_null_tensor_argument = valid_kernel;
  kernel_null_tensor_argument.kernel_arguments =
      std::span<const KernelArgument>(invalid_kernel_arguments.data(), invalid_kernel_arguments.size());
  if (Status status = expect_status_code(
          "kernel null tensor argument",
          backend.execute(kernel_null_tensor_argument),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  BackendExecution kernel_missing_launch = valid_kernel;
  kernel_missing_launch.launch = std::nullopt;
  kernel_missing_launch.compilation_target =
      CompilationTarget{KernelArtifactKind::kStaticLibrary, "noop_library", "noop"};
  if (Status status = expect_status_code(
          "kernel missing launch",
          backend.execute(kernel_missing_launch),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  BackendExecution kernel_missing_output = kernel_missing_launch;
  kernel_missing_output.launch = LaunchConfig{1, 1, 1, 1, 1, 1};
  kernel_missing_output.outputs = {};
  if (Status status = expect_status_code(
          "kernel missing output metadata",
          backend.execute(kernel_missing_output),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  BackendExecution kernel_missing_target = valid_kernel;
  kernel_missing_target.compilation_target = std::nullopt;
  if (Status status = expect_status_code(
          "kernel missing compilation target",
          backend.execute(kernel_missing_target),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  BackendExecution kernel_zero_launch_dimension = kernel_missing_target;
  kernel_zero_launch_dimension.launch = LaunchConfig{};
  kernel_zero_launch_dimension.compilation_target =
      CompilationTarget{KernelArtifactKind::kStaticLibrary, "noop_library", "noop"};
  if (Status status = expect_status_code(
          "kernel zero launch dimension",
          backend.execute(kernel_zero_launch_dimension),
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
  return Status::Ok();
}

}  // namespace tensorcx::null_backend
