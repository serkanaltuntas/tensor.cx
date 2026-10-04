#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensorcx/backends/cpu/cpu_backend.h"
#include "tensorcx/backends/null/null_backend.h"
#include "tensorcx/core/backend.h"
#include "tensorcx/core/dtype.h"
#include "tensorcx/core/operation.h"
#include "tensorcx/core/shape.h"
#include "tensorcx/core/status.h"
#include "tensorcx/core/tensor.h"

namespace {

int failures = 0;

void fail(const std::string& scenario, const tensorcx::Status& status) {
  std::cerr << scenario << " failed with status " << static_cast<int>(status.code())
            << ": " << status.message() << '\n';
  ++failures;
}

void expect_ok(const std::string& scenario, const tensorcx::Status& status) {
  if (!status.ok()) {
    fail(scenario, status);
  }
}

void expect_status(
    const std::string& scenario,
    const tensorcx::Status& status,
    tensorcx::StatusCode expected) {
  if (status.code() != expected) {
    fail(scenario, status);
  }
}

tensorcx::Tensor fill_descriptor(tensorcx::Device device = tensorcx::Device{"cpu", 0}) {
  const tensorcx::Shape shape{2};
  return tensorcx::Tensor{
      tensorcx::DType::kFloat32,
      shape,
      tensorcx::contiguous_strides(shape),
      std::move(device),
      nullptr,
      0,
  };
}

void arithmetic_contract_tests() {
  using namespace tensorcx;
  using cpu::CpuTensor;
  cpu::CpuBackend backend;
  const auto check = [](bool passed, const char* scenario) {
    if (!passed) { std::cerr << scenario << '\n'; ++failures; }
  };
  const CpuTensor integers({3}, std::vector<std::int32_t>{INT32_MIN, 7, INT32_MAX});
  const CpuTensor ones({3}, std::vector<std::int32_t>{1, 1, 1});
  const auto subtract = cpu::execute_binary(OpDesc{OpKind::kSubtract}, integers, ones);
  check(subtract.int32_data() == std::vector<std::int32_t>{INT32_MAX, 6, INT32_MAX - 1},
        "int32 subtraction must wrap");
  const auto negate = cpu::execute_unary(OpDesc{OpKind::kNegate}, integers);
  check(negate.int32_data() == std::vector<std::int32_t>{INT32_MIN, -7, -INT32_MAX},
        "int32 negation must wrap INT_MIN");
  const CpuTensor zeros({2}, std::vector<float>{0.0F, -0.0F});
  const auto negated_zero = cpu::execute_unary(OpDesc{OpKind::kNegate}, zeros);
  check(std::signbit(negated_zero.float_data()[0]) && !std::signbit(negated_zero.float_data()[1]),
        "float negation must flip signed zero");
  const CpuTensor numerators({2}, std::vector<float>{1.0F, 0.0F});
  const auto division = cpu::execute_binary(OpDesc{OpKind::kDivide}, numerators, zeros);
  check(std::isinf(division.float_data()[0]) && std::isnan(division.float_data()[1]),
        "float division must retain IEEE zero behavior");
  OpDesc zero_divisor{OpKind::kDivideScalar}; zero_divisor.scalar_value = -0.0;
  const auto scalar_division = cpu::execute_unary(zero_divisor, numerators);
  check(std::isinf(scalar_division.float_data()[0]) && std::signbit(scalar_division.float_data()[0]) &&
            std::isnan(scalar_division.float_data()[1]),
        "scalar division must retain IEEE signed zero behavior");
  zero_divisor.scalar_value = 1; zero_divisor.scalar_left = true;
  const auto reversed_division = cpu::execute_unary(zero_divisor, zeros);
  check(std::isinf(reversed_division.float_data()[0]) && !std::signbit(reversed_division.float_data()[0]) &&
            std::isinf(reversed_division.float_data()[1]) && std::signbit(reversed_division.float_data()[1]),
        "reverse scalar division must retain tensor signed zeros");
  for (const double scalar : {std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::quiet_NaN()}) {
    OpDesc op{OpKind::kAddScalar}; op.scalar_value = scalar;
    const auto result = cpu::execute_unary(op, numerators);
    check(std::isnan(scalar) ? std::isnan(result.float_data()[0]) : std::isinf(result.float_data()[0]),
          "float scalar arithmetic must allow nonfinite values");
  }
  const auto empty_float = cpu::empty({0}, DType::kFloat32);
  for (const auto kind : {OpKind::kNegate, OpKind::kAddScalar, OpKind::kSubtractScalar,
                          OpKind::kMultiplyScalar, OpKind::kDivideScalar}) {
    const auto result = cpu::execute_unary(OpDesc{kind}, empty_float);
    check(result.shape() == Shape{0} && result.size() == 0,
          "empty float arithmetic must preserve metadata");
  }
  const CpuTensor value({1}, std::vector<std::int32_t>{7});
  const CpuTensor float_value({1}, std::vector<float>{8.0F});
  for (const auto kind : {OpKind::kAddScalar, OpKind::kSubtractScalar,
                          OpKind::kMultiplyScalar, OpKind::kDivideScalar}) {
    for (const bool left : {false, true}) {
      OpDesc op{kind}; op.scalar_value = 2; op.scalar_left = left;
      std::array<Tensor, 1> inputs{cpu::to_core_tensor(float_value)}, outputs{};
      BackendExecution execution{BackendOpClass::kPrimitive, op, inputs, outputs,
                                 std::nullopt, std::nullopt, {}};
      expect_ok("scalar execution schema", backend.execute(execution));
      const float expected = kind == OpKind::kAddScalar ? 10.0F :
                             kind == OpKind::kSubtractScalar ? (left ? -6.0F : 6.0F) :
                             kind == OpKind::kMultiplyScalar ? 16.0F : (left ? 0.25F : 4.0F);
      if (outputs[0].buffer)
        check(cpu::from_core_tensor(outputs[0]).float_data()[0] == expected,
              "float scalar arithmetic/order mismatch");
      if (kind != OpKind::kDivideScalar) {
        const std::int32_t expected_int = kind == OpKind::kAddScalar ? 9 :
                                        kind == OpKind::kSubtractScalar ? (left ? -5 : 5) : 14;
        check(cpu::execute_unary(op, value).int32_data()[0] == expected_int,
              "int32 scalar arithmetic/order mismatch");
      }
    }
  }
  OpDesc boundary{OpKind::kAddScalar}; boundary.scalar_value = INT32_MAX;
  check(cpu::execute_unary(boundary, ones).int32_data()[0] == INT32_MIN,
        "int32 scalar addition must wrap");
  boundary.kind = OpKind::kMultiplyScalar; boundary.scalar_value = -1;
  check(cpu::execute_unary(boundary, integers).int32_data()[0] == INT32_MIN,
        "int32 scalar multiplication must wrap");
  boundary.kind = OpKind::kSubtractScalar; boundary.scalar_value = INT32_MIN; boundary.scalar_left = true;
  check(cpu::execute_unary(boundary, ones).int32_data()[0] == INT32_MAX,
        "reverse int32 scalar subtraction must wrap");

  // Reject invalid scalar descriptors even when an empty result bypasses loops,
  // and do not publish a new output on failure.
  for (const Shape& shape : {Shape{0}, Shape{1}}) {
    const auto input = cpu::empty(shape, DType::kInt32);
    std::array<Tensor, 1> inputs{cpu::to_core_tensor(input)}, outputs{cpu::to_core_tensor(value)};
    const auto original = outputs[0].buffer;
    for (const auto kind : {OpKind::kAddScalar, OpKind::kSubtractScalar, OpKind::kMultiplyScalar}) {
      for (const double scalar : {0.5, 2147483648.0, -2147483649.0,
                                 std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::quiet_NaN()}) {
        OpDesc op{kind}; op.scalar_value = scalar;
        const BackendExecution execution{BackendOpClass::kPrimitive, op, inputs, outputs,
                                         std::nullopt, std::nullopt, {}};
        expect_status("invalid native int scalar", backend.execute(execution), StatusCode::kInvalidArgument);
        check(outputs[0].buffer == original, "invalid arithmetic must preserve output slot");
      }
    }
    BackendExecution unary{BackendOpClass::kPrimitive, OpDesc{OpKind::kDivideScalar}, inputs, outputs,
                           std::nullopt, std::nullopt, {}};
    expect_status("int scalar division", backend.execute(unary), StatusCode::kInvalidArgument);
    std::array<Tensor, 2> pair{inputs[0], inputs[0]};
    BackendExecution binary{BackendOpClass::kPrimitive, OpDesc{OpKind::kDivide}, pair, outputs,
                            std::nullopt, std::nullopt, {}};
    expect_status("int tensor division", backend.execute(binary), StatusCode::kInvalidArgument);
    unary.op.kind = OpKind::kNegate;
    expect_ok("integer negate schema/empty", backend.execute(unary));
    unary.op.kind = OpKind::kAddScalar; unary.inputs = pair;
    expect_status("scalar wrong input count", backend.execute(unary), StatusCode::kInvalidArgument);
    binary.op.kind = OpKind::kSubtract; binary.inputs = inputs;
    expect_status("subtract wrong input count", backend.execute(binary), StatusCode::kInvalidArgument);
  }
}

}  // namespace

int main() {
  arithmetic_contract_tests();
  // Malformed native metadata must not wrap its byte count to a small buffer.
  for (auto dtype : {tensorcx::DType::kFloat32, tensorcx::DType::kInt32}) {
    for (std::size_t elements : {0U, 2U}) {
      const auto count = std::numeric_limits<std::size_t>::max() /
                             tensorcx::dtype_size(dtype) + 1 + elements;
      const auto buffer = std::make_shared<tensorcx::cpu::CpuBuffer>(dtype, elements);
      try {
        tensorcx::cpu::CpuTensor invalid(dtype, {static_cast<tensorcx::Dim>(count)}, buffer);
        std::cerr << "CPU tensor accepted overflowing byte size\n";
        ++failures;
      } catch (const std::invalid_argument&) {
        // Expected before any allocation or buffer access.
      }
      tensorcx::Tensor metadata{dtype, {static_cast<tensorcx::Dim>(count)}, {1},
                              {"cpu", 0}, buffer, 0};
      try {
        tensorcx::cpu::from_core_tensor(metadata);
        std::cerr << "CPU conversion accepted overflowing byte size\n";
        ++failures;
      } catch (const std::invalid_argument&) {
      }
    }
  }
  expect_ok("null backend contract smoke", tensorcx::null_backend::contract_smoke_test());
  expect_ok("cpu backend contract smoke", tensorcx::cpu::contract_smoke_test());

  // A left-to-right suffix product overflows before the trailing zero. The
  // tensor and its contiguous strides are nevertheless valid and allocate zero
  // bytes. Run this path in the native sanitizer job as well as Python tests.
  const tensorcx::Shape empty_shape{0, 1, tensorcx::Dim{1} << 62, 4, 0};
  const auto empty_input = tensorcx::cpu::empty(empty_shape, tensorcx::DType::kFloat32);
  for (const auto kind : {tensorcx::OpKind::kSum, tensorcx::OpKind::kMax,
                         tensorcx::OpKind::kMean, tensorcx::OpKind::kSoftmax,
                         tensorcx::OpKind::kRmsNorm, tensorcx::OpKind::kLayerNorm}) {
    const bool reduction = kind == tensorcx::OpKind::kSum ||
                           kind == tensorcx::OpKind::kMax || kind == tensorcx::OpKind::kMean;
    tensorcx::OpDesc op{kind};
    op.axis = 1;
    const auto result = reduction ? tensorcx::cpu::reduce(op, empty_input)
                                  : tensorcx::cpu::execute_unary(op, empty_input);
    auto expected_shape = empty_shape;
    if (reduction) expected_shape.erase(expected_shape.begin() + 1);
    if (result.size() != 0 || result.shape() != expected_shape) {
      std::cerr << "CPU empty axis operation returned incorrect metadata\n";
      ++failures;
    }
  }
  // Empty inner dimensions must also bypass the outer loop in debug builds.
  const tensorcx::Shape empty_inner_shape{1'000'000'000'000, 1, 0};
  const auto empty_inner = tensorcx::cpu::empty(empty_inner_shape, tensorcx::DType::kFloat32);
  for (const auto kind : {tensorcx::OpKind::kRmsNorm, tensorcx::OpKind::kLayerNorm}) {
    tensorcx::OpDesc op{kind};
    op.axis = 1;
    const auto result = tensorcx::cpu::execute_unary(op, empty_inner);
    if (result.size() != 0 || result.shape() != empty_inner_shape) {
      std::cerr << "CPU empty normalization returned incorrect metadata\n";
      ++failures;
    }
  }

  std::array<tensorcx::Tensor, 1> fill_outputs{fill_descriptor()};
  const tensorcx::BackendExecution valid_fill{
      tensorcx::BackendOpClass::kPrimitive,
      tensorcx::OpDesc{tensorcx::OpKind::kFill},
      std::span<const tensorcx::Tensor>(),
      std::span<tensorcx::Tensor>(fill_outputs.data(), fill_outputs.size()),
      std::nullopt,
      std::nullopt,
      std::span<const tensorcx::KernelArgument>(),
  };
  expect_ok(
      "valid fill primitive contract",
      tensorcx::validate_primitive_execution_contract(valid_fill, "cpu"));

  tensorcx::BackendExecution primitive_with_launch = valid_fill;
  primitive_with_launch.launch = tensorcx::LaunchConfig{1, 1, 1, 1, 1, 1};
  expect_status(
      "primitive rejects launch metadata",
      tensorcx::validate_primitive_execution_contract(primitive_with_launch, "cpu"),
      tensorcx::StatusCode::kInvalidArgument);

  std::array<tensorcx::KernelArgument, 1> primitive_kernel_arguments{{
      tensorcx::KernelArgument{tensorcx::KernelArgumentKind::kUInt32, nullptr, 1},
  }};
  tensorcx::BackendExecution primitive_with_kernel_arguments = valid_fill;
  primitive_with_kernel_arguments.kernel_arguments = std::span<const tensorcx::KernelArgument>(
      primitive_kernel_arguments.data(),
      primitive_kernel_arguments.size());
  expect_status(
      "primitive rejects kernel arguments",
      tensorcx::validate_primitive_execution_contract(primitive_with_kernel_arguments, "cpu"),
      tensorcx::StatusCode::kInvalidArgument);

  std::array<tensorcx::Tensor, 1> wrong_device_outputs{fill_descriptor(tensorcx::Device{"metal", 0})};
  tensorcx::BackendExecution fill_wrong_device = valid_fill;
  fill_wrong_device.outputs =
      std::span<tensorcx::Tensor>(wrong_device_outputs.data(), wrong_device_outputs.size());
  expect_status(
      "fill descriptor rejects wrong device",
      tensorcx::validate_primitive_execution_contract(fill_wrong_device, "cpu"),
      tensorcx::StatusCode::kInvalidArgument);

  const tensorcx::cpu::CpuTensor kernel_output_cpu =
      tensorcx::cpu::empty(tensorcx::Shape{2}, tensorcx::DType::kFloat32);
  std::array<tensorcx::Tensor, 1> kernel_outputs{
      tensorcx::cpu::to_core_tensor(kernel_output_cpu)};
  std::array<tensorcx::KernelArgument, 2> kernel_arguments{{
      tensorcx::KernelArgument{tensorcx::KernelArgumentKind::kTensor, &kernel_outputs[0], 0},
      tensorcx::KernelArgument{tensorcx::KernelArgumentKind::kUInt32, nullptr, 2},
  }};
  tensorcx::BackendExecution kernel_missing_target{
      tensorcx::BackendOpClass::kKernel,
      tensorcx::OpDesc{tensorcx::OpKind::kAdd},
      std::span<const tensorcx::Tensor>(),
      std::span<tensorcx::Tensor>(kernel_outputs.data(), kernel_outputs.size()),
      tensorcx::LaunchConfig{1, 1, 1, 1, 1, 1},
      std::nullopt,
      std::span<const tensorcx::KernelArgument>(kernel_arguments.data(), kernel_arguments.size()),
  };
  expect_status(
      "kernel rejects missing compilation target",
      tensorcx::validate_kernel_execution_contract(kernel_missing_target),
      tensorcx::StatusCode::kInvalidArgument);

  tensorcx::BackendExecution kernel_with_inputs = kernel_missing_target;
  std::array<tensorcx::Tensor, 1> primitive_inputs{kernel_outputs[0]};
  kernel_with_inputs.inputs =
      std::span<const tensorcx::Tensor>(primitive_inputs.data(), primitive_inputs.size());
  expect_status(
      "kernel rejects primitive inputs",
      tensorcx::validate_kernel_execution_contract(kernel_with_inputs),
      tensorcx::StatusCode::kInvalidArgument);

  tensorcx::BackendExecution kernel_default_output = kernel_missing_target;
  kernel_default_output.compilation_target =
      tensorcx::CompilationTarget{tensorcx::KernelArtifactKind::kStaticLibrary, "module", "entry"};
  kernel_default_output.outputs = std::span<tensorcx::Tensor>(fill_outputs.data(), fill_outputs.size());
  expect_status(
      "kernel rejects default output metadata",
      tensorcx::validate_kernel_execution_contract(kernel_default_output),
      tensorcx::StatusCode::kInvalidArgument);

  std::array<tensorcx::KernelArgument, 1> null_tensor_argument{{
      tensorcx::KernelArgument{tensorcx::KernelArgumentKind::kTensor, nullptr, 0},
  }};
  tensorcx::BackendExecution kernel_null_tensor_argument = kernel_default_output;
  kernel_null_tensor_argument.outputs =
      std::span<tensorcx::Tensor>(kernel_outputs.data(), kernel_outputs.size());
  kernel_null_tensor_argument.kernel_arguments =
      std::span<const tensorcx::KernelArgument>(null_tensor_argument.data(), null_tensor_argument.size());
  expect_status(
      "kernel rejects null tensor argument",
      tensorcx::validate_kernel_execution_contract(kernel_null_tensor_argument),
      tensorcx::StatusCode::kInvalidArgument);

  tensorcx::BackendExecution kernel_zero_launch = kernel_missing_target;
  kernel_zero_launch.launch = tensorcx::LaunchConfig{};
  kernel_zero_launch.compilation_target =
      tensorcx::CompilationTarget{tensorcx::KernelArtifactKind::kStaticLibrary, "module", "entry"};
  expect_status(
      "kernel rejects zero launch dimension",
      tensorcx::validate_kernel_execution_contract(kernel_zero_launch),
      tensorcx::StatusCode::kInvalidArgument);

  tensorcx::BackendExecution valid_kernel = kernel_missing_target;
  valid_kernel.compilation_target =
      tensorcx::CompilationTarget{tensorcx::KernelArtifactKind::kStaticLibrary, "module", "entry"};
  expect_ok("valid kernel contract", tensorcx::validate_kernel_execution_contract(valid_kernel));

  return failures == 0 ? 0 : 1;
}
