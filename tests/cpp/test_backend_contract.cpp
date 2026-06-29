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
      std::span<const cortex::KernelArgument>(),
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

  std::array<cortex::KernelArgument, 1> primitive_kernel_arguments{{
      cortex::KernelArgument{cortex::KernelArgumentKind::kUInt32, nullptr, 1},
  }};
  cortex::BackendExecution primitive_with_kernel_arguments = valid_fill;
  primitive_with_kernel_arguments.kernel_arguments = std::span<const cortex::KernelArgument>(
      primitive_kernel_arguments.data(),
      primitive_kernel_arguments.size());
  expect_status(
      "primitive rejects kernel arguments",
      cortex::validate_primitive_execution_contract(primitive_with_kernel_arguments, "cpu"),
      cortex::StatusCode::kInvalidArgument);

  std::array<cortex::Tensor, 1> wrong_device_outputs{fill_descriptor(cortex::Device{"metal", 0})};
  cortex::BackendExecution fill_wrong_device = valid_fill;
  fill_wrong_device.outputs =
      std::span<cortex::Tensor>(wrong_device_outputs.data(), wrong_device_outputs.size());
  expect_status(
      "fill descriptor rejects wrong device",
      cortex::validate_primitive_execution_contract(fill_wrong_device, "cpu"),
      cortex::StatusCode::kInvalidArgument);

  const cortex::cpu::CpuTensor kernel_output_cpu =
      cortex::cpu::empty(cortex::Shape{2}, cortex::DType::kFloat32);
  std::array<cortex::Tensor, 1> kernel_outputs{
      cortex::cpu::to_core_tensor(kernel_output_cpu)};
  std::array<cortex::KernelArgument, 2> kernel_arguments{{
      cortex::KernelArgument{cortex::KernelArgumentKind::kTensor, &kernel_outputs[0], 0},
      cortex::KernelArgument{cortex::KernelArgumentKind::kUInt32, nullptr, 2},
  }};
  cortex::BackendExecution kernel_missing_target{
      cortex::BackendOpClass::kKernel,
      cortex::OpDesc{cortex::OpKind::kAdd},
      std::span<const cortex::Tensor>(),
      std::span<cortex::Tensor>(kernel_outputs.data(), kernel_outputs.size()),
      cortex::LaunchConfig{1, 1, 1, 1, 1, 1},
      std::nullopt,
      std::span<const cortex::KernelArgument>(kernel_arguments.data(), kernel_arguments.size()),
  };
  expect_status(
      "kernel rejects missing compilation target",
      cortex::validate_kernel_execution_contract(kernel_missing_target),
      cortex::StatusCode::kInvalidArgument);

  cortex::BackendExecution kernel_with_inputs = kernel_missing_target;
  std::array<cortex::Tensor, 1> primitive_inputs{kernel_outputs[0]};
  kernel_with_inputs.inputs =
      std::span<const cortex::Tensor>(primitive_inputs.data(), primitive_inputs.size());
  expect_status(
      "kernel rejects primitive inputs",
      cortex::validate_kernel_execution_contract(kernel_with_inputs),
      cortex::StatusCode::kInvalidArgument);

  cortex::BackendExecution kernel_default_output = kernel_missing_target;
  kernel_default_output.compilation_target =
      cortex::CompilationTarget{cortex::KernelArtifactKind::kStaticLibrary, "module", "entry"};
  kernel_default_output.outputs = std::span<cortex::Tensor>(fill_outputs.data(), fill_outputs.size());
  expect_status(
      "kernel rejects default output metadata",
      cortex::validate_kernel_execution_contract(kernel_default_output),
      cortex::StatusCode::kInvalidArgument);

  std::array<cortex::KernelArgument, 1> null_tensor_argument{{
      cortex::KernelArgument{cortex::KernelArgumentKind::kTensor, nullptr, 0},
  }};
  cortex::BackendExecution kernel_null_tensor_argument = kernel_default_output;
  kernel_null_tensor_argument.outputs =
      std::span<cortex::Tensor>(kernel_outputs.data(), kernel_outputs.size());
  kernel_null_tensor_argument.kernel_arguments =
      std::span<const cortex::KernelArgument>(null_tensor_argument.data(), null_tensor_argument.size());
  expect_status(
      "kernel rejects null tensor argument",
      cortex::validate_kernel_execution_contract(kernel_null_tensor_argument),
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
