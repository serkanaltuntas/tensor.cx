#include "cortex/backends/null/null_backend.h"

#include <array>
#include <optional>
#include <span>
#include <string>

namespace cortex::null_backend {
namespace {

Status validate_execution_contract(const BackendExecution& execution) {
  if (execution.outputs.empty()) {
    return Status(StatusCode::kInvalidArgument, "backend execution requires at least one output");
  }
  if (execution.op_class == BackendOpClass::kPrimitive) {
    if (execution.launch.has_value() || execution.compilation_target.has_value()) {
      return Status(
          StatusCode::kInvalidArgument,
          "primitive backend execution cannot include kernel launch metadata");
    }
    return Status::Ok();
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

  std::array<Tensor, 1> outputs{};
  const std::span<Tensor> output_span(outputs.data(), outputs.size());
  const BackendExecution valid_primitive{
      BackendOpClass::kPrimitive,
      OpDesc{OpKind::kAdd},
      {},
      output_span,
      std::nullopt,
      std::nullopt,
  };
  if (Status status =
          expect_status_code("valid primitive", backend.execute(valid_primitive), StatusCode::kUnavailable);
      !status.ok()) {
    return status;
  }

  BackendExecution empty_outputs = valid_primitive;
  empty_outputs.outputs = {};
  if (Status status =
          expect_status_code("empty outputs", backend.execute(empty_outputs), StatusCode::kInvalidArgument);
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
