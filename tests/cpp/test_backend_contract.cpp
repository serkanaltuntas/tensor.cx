#include <array>
#include <iostream>
#include <optional>
#include <span>
#include <string>

#include "cortex/backends/cpu/cpu_backend.h"
#include "cortex/backends/null/null_backend.h"
#include "cortex/core/backend.h"
#include "cortex/core/dtype.h"
#include "cortex/core/operation.h"
#include "cortex/core/shape.h"
#include "cortex/core/status.h"
#include "cortex/core/tensor.h"

namespace {

int failures = 0;

void fail(const std::string& scenario, const cortex::Status& status) {
  std::cerr << scenario << " failed with status " << static_cast<int>(status.code())
            << ": " << status.message() << '\n';
  ++failures;
}

void expect_ok(const std::string& scenario, const cortex::Status& status) {
  if (!status.ok()) {
    fail(scenario, status);
  }
}

void expect_status(
    const std::string& scenario,
    const cortex::Status& status,
    cortex::StatusCode expected) {
  if (status.code() != expected) {
    fail(scenario, status);
  }
}

cortex::Tensor fill_descriptor(cortex::Device device = cortex::Device{"cpu", 0}) {
  const cortex::Shape shape{2};
  return cortex::Tensor{
      cortex::DType::kFloat32,
      shape,
      cortex::contiguous_strides(shape),
      std::move(device),
      nullptr,
      0,
  };
}

}  // namespace

int main() {
  expect_ok("null backend contract smoke", cortex::null_backend::contract_smoke_test());
  expect_ok("cpu backend contract smoke", cortex::cpu::contract_smoke_test());

  std::array<cortex::Tensor, 1> fill_outputs{fill_descriptor()};
  const cortex::BackendExecution valid_fill{
      cortex::BackendOpClass::kPrimitive,
      cortex::OpDesc{cortex::OpKind::kFill},
      std::span<const cortex::Tensor>(),
      std::span<cortex::Tensor>(fill_outputs.data(), fill_outputs.size()),
      std::nullopt,
      std::nullopt,
  };
  expect_ok(
      "valid fill primitive contract",
      cortex::validate_primitive_execution_contract(valid_fill, "cpu"));

  cortex::BackendExecution primitive_with_launch = valid_fill;
  primitive_with_launch.launch = cortex::LaunchConfig{1, 1, 1, 1, 1, 1};
  expect_status(
      "primitive rejects launch metadata",
      cortex::validate_primitive_execution_contract(primitive_with_launch, "cpu"),
      cortex::StatusCode::kInvalidArgument);

  std::array<cortex::Tensor, 1> wrong_device_outputs{fill_descriptor(cortex::Device{"metal", 0})};
  cortex::BackendExecution fill_wrong_device = valid_fill;
  fill_wrong_device.outputs =
      std::span<cortex::Tensor>(wrong_device_outputs.data(), wrong_device_outputs.size());
  expect_status(
      "fill descriptor rejects wrong device",
      cortex::validate_primitive_execution_contract(fill_wrong_device, "cpu"),
      cortex::StatusCode::kInvalidArgument);

  cortex::BackendExecution kernel_missing_target{
      cortex::BackendOpClass::kKernel,
      cortex::OpDesc{cortex::OpKind::kAdd},
      std::span<const cortex::Tensor>(),
      std::span<cortex::Tensor>(fill_outputs.data(), fill_outputs.size()),
      cortex::LaunchConfig{1, 1, 1, 1, 1, 1},
      std::nullopt,
  };
  expect_status(
      "kernel rejects missing compilation target",
      cortex::validate_kernel_execution_contract(kernel_missing_target),
      cortex::StatusCode::kInvalidArgument);

  cortex::BackendExecution kernel_zero_launch = kernel_missing_target;
  kernel_zero_launch.launch = cortex::LaunchConfig{};
  kernel_zero_launch.compilation_target =
      cortex::CompilationTarget{cortex::KernelArtifactKind::kStaticLibrary, "module", "entry"};
  expect_status(
      "kernel rejects zero launch dimension",
      cortex::validate_kernel_execution_contract(kernel_zero_launch),
      cortex::StatusCode::kInvalidArgument);

  cortex::BackendExecution valid_kernel = kernel_missing_target;
  valid_kernel.compilation_target =
      cortex::CompilationTarget{cortex::KernelArtifactKind::kStaticLibrary, "module", "entry"};
  expect_ok("valid kernel contract", cortex::validate_kernel_execution_contract(valid_kernel));

  return failures == 0 ? 0 : 1;
}
