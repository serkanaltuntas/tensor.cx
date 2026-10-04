#include <array>
#include <bit>
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

void cast_broadcast_contract_tests() {
  using namespace tensorcx;
  using cpu::CpuTensor;
  cpu::CpuBackend backend;
  const auto check = [](bool passed, const char* scenario) {
    if (!passed) { std::cerr << scenario << '\n'; ++failures; }
  };
  const auto invalid_plan = [&](const Shape& lhs, const Shape& rhs) {
    try { (void)make_broadcast_plan(lhs, rhs); check(false, "invalid broadcast plan accepted"); }
    catch (const std::invalid_argument&) {}
  };
  auto plan = make_broadcast_plan({2, 1, 3}, {4, 1});
  check(plan.output_shape == Shape{2, 4, 3} && plan.lhs_strides == Shape{3, 0, 1} &&
            plan.rhs_strides == Shape{0, 1, 0}, "broadcast stride mapping incorrect");
  check(make_broadcast_plan({}, {}).output_shape.empty(), "rank-zero broadcast changed rank");
  check(make_broadcast_plan({0, 3}, {1, 3}).output_shape == Shape{0, 3},
        "zero/one broadcast must stay empty");
  check(make_broadcast_plan(Shape(128, 1), {}).output_shape == Shape(128, 1),
        "broadcast imposed an arbitrary rank limit");
  invalid_plan({2}, {3}); invalid_plan({0}, {2}); invalid_plan({0, -1}, {1});
  invalid_plan({INT64_MAX, 2}, {}); invalid_plan({0, INT64_MAX, 2}, {});
  invalid_plan({INT64_MAX, 1}, {1, 2});
  // Individually valid empty inputs may produce overflowing output strides.
  invalid_plan({0, INT64_MAX, 1}, {0, 1, 2});

  const CpuTensor lhs({2, 1}, std::vector<float>{2, 8});
  const CpuTensor rhs({3}, std::vector<float>{1, 2, 4});
  for (const auto kind : {OpKind::kAdd, OpKind::kSubtract, OpKind::kMultiply, OpKind::kDivide}) {
    std::array<Tensor, 2> inputs{cpu::to_core_tensor(lhs), cpu::to_core_tensor(rhs)};
    std::array<Tensor, 1> outputs{};
    const BackendExecution e{BackendOpClass::kPrimitive, OpDesc{kind}, inputs, outputs,
                             std::nullopt, std::nullopt, {}};
    expect_ok("broadcast backend execution", backend.execute(e));
    if (!outputs[0].buffer) continue;
    const auto result = cpu::from_core_tensor(outputs[0]);
    check(result.shape() == Shape{2, 3}, "broadcast output shape incorrect");
    for (std::size_t i = 0; i < 6; ++i) {
      const float a = lhs.float_data()[i / 3], b = rhs.float_data()[i % 3];
      const float expected = kind == OpKind::kAdd ? a+b : kind == OpKind::kSubtract ? a-b :
                             kind == OpKind::kMultiply ? a*b : a/b;
      check(result.float_data()[i] == expected, "broadcast mapped incorrect input values");
    }
  }
  const CpuTensor scalar({}, std::vector<std::int32_t>{INT32_MAX});
  const CpuTensor integer_rhs({2}, std::vector<std::int32_t>{1, 2});
  check(cpu::execute_binary(OpDesc{OpKind::kAdd}, scalar, integer_rhs).int32_data() ==
            std::vector<std::int32_t>{INT32_MIN, INT32_MIN + 1}, "broadcast int32 wrap mismatch");
  const auto empty = cpu::empty({0, 3}, DType::kFloat32);
  check(cpu::execute_binary(OpDesc{OpKind::kAdd}, empty, rhs).shape() == Shape{0, 3},
        "empty broadcast output incorrect");

  const CpuTensor floats({6}, std::vector<float>{-2147483648.0F, 2147483520.0F,
                                               -1.9F, 1.9F, -0.0F, 0.0F});
  OpDesc cast{OpKind::kCast}; cast.target_dtype = DType::kInt32;
  std::array<Tensor, 1> inputs{cpu::to_core_tensor(floats)}, outputs{};
  BackendExecution e{BackendOpClass::kPrimitive, cast, inputs, outputs, std::nullopt, std::nullopt, {}};
  expect_ok("checked float-to-int cast", backend.execute(e));
  if (outputs[0].buffer)
    check(cpu::from_core_tensor(outputs[0]).int32_data() ==
              std::vector<std::int32_t>{INT32_MIN, 2147483520, -1, 1, 0, 0},
          "cast truncation/boundaries incorrect");
  const auto original_output = outputs[0].buffer;
  for (const float invalid : {2147483648.0F, -2147483904.0F,
                              std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::quiet_NaN()}) {
    const CpuTensor bad({2}, std::vector<float>{1, invalid});
    inputs[0] = cpu::to_core_tensor(bad);
    expect_status("invalid float-to-int cast", backend.execute(e), StatusCode::kInvalidArgument);
    check(outputs[0].buffer == original_output, "failed cast replaced output slot");
    check(bad.float_data()[0] == 1, "failed cast mutated input");
  }
  e.op.target_dtype = static_cast<DType>(99);
  expect_status("invalid cast dtype", backend.execute(e), StatusCode::kInvalidArgument);
  e.op = cast; e.inputs = {};
  expect_status("missing cast input", backend.execute(e), StatusCode::kInvalidArgument);
  for (const auto dtype : {DType::kFloat32, DType::kInt32}) {
    const auto source = dtype == DType::kFloat32 ? floats : cpu::execute_unary(cast, floats);
    OpDesc copy{OpKind::kCast}; copy.target_dtype = dtype;
    const auto result = cpu::execute_unary(copy, source);
    check(result.buffer() != source.buffer() && result.shape() == source.shape() &&
              result.dtype() == dtype, "same-dtype cast must copy independently");
    const auto empty_source = cpu::empty({2, 0}, dtype);
    copy.target_dtype = dtype == DType::kFloat32 ? DType::kInt32 : DType::kFloat32;
    const auto empty_cast = cpu::execute_unary(copy, empty_source);
    check(empty_cast.shape() == Shape{2, 0} && empty_cast.size() == 0 &&
              empty_cast.dtype() == copy.target_dtype, "empty cast metadata mismatch");
  }
  OpDesc to_float{OpKind::kCast};
  const CpuTensor exact({}, std::vector<std::int32_t>{16777217});
  check(cpu::execute_unary(to_float, exact).float_data()[0] == 16777216.0F,
        "integer cast must round to float32 nearest");
}

void transpose_contract_tests() {
  using namespace tensorcx;
  cpu::CpuBackend backend;
  const auto check = [](bool passed, const char* scenario) {
    if (!passed) { std::cerr << scenario << '\n'; ++failures; }
  };
  const auto plan = make_transpose_plan({2, 3, 4}, {-1, 0, 1});
  check(plan.output_shape == Shape{4, 2, 3} && plan.input_strides == Shape{1, 12, 4},
        "transpose stride mapping incorrect");
  check(make_transpose_plan({}, {}).output_shape.empty(), "scalar transpose changed rank");
  const std::array<std::uint32_t, 6> bits{0, 0x80000000, 0x7fc12345, 1, 0xff800000, 0x7f812345};
  cpu::CpuTensor special(DType::kFloat32, {2, 3});
  for (std::size_t i = 0; i < bits.size(); ++i)
    special.mutable_float_data()[i] = std::bit_cast<float>(bits[i]);
  OpDesc permute{OpKind::kTranspose}; permute.axes = {1, 0};
  const auto copied = cpu::execute_unary(permute, special);
  for (std::size_t i = 0; i < bits.size(); ++i)
    check(std::bit_cast<std::uint32_t>(copied.float_data()[i]) == bits[(i % 2) * 3 + i / 2],
          "CPU transpose changed stored bits");
  for (const Shape axes : {Shape{}, Shape{0}, Shape{0, 0}, Shape{0, 2}, Shape{-3, 1}}) {
    try { (void)make_transpose_plan({2, 3}, axes); check(false, "invalid permutation accepted"); }
    catch (const std::invalid_argument&) {}
  }
  // Valid zero-sized input may have an invalid output stride/product order.
  try {
    (void)make_transpose_plan({INT64_MAX, 0, 2}, {1, 0, 2});
    check(false, "transpose accepted overflowing output strides");
  } catch (const std::invalid_argument&) {}
  const cpu::CpuTensor input({2, 3}, std::vector<std::int32_t>{1, 2, 3, 4, 5, 6});
  std::array<Tensor, 1> inputs{cpu::to_core_tensor(input)}, outputs{};
  OpDesc op{OpKind::kTranspose}; op.axes = {1, 0};
  BackendExecution execution{BackendOpClass::kPrimitive, op, inputs, outputs,
                             std::nullopt, std::nullopt, {}};
  expect_ok("native transpose", backend.execute(execution));
  if (outputs[0].buffer) {
    check(outputs[0].shape == Shape{3, 2} && outputs[0].strides == Shape{2, 1} &&
              outputs[0].buffer != inputs[0].buffer &&
              cpu::from_core_tensor(outputs[0]).int32_data() ==
                  std::vector<std::int32_t>{1, 4, 2, 5, 3, 6},
          "native transpose data/ownership mismatch");
  }
  const auto original = outputs[0].buffer;
  for (const Shape axes : {Shape{}, Shape{0, 0}, Shape{2, 0}}) {
    execution.op.axes = axes;
    expect_status("invalid native transpose", backend.execute(execution), StatusCode::kInvalidArgument);
    check(outputs[0].buffer == original, "failed transpose replaced output slot");
  }
  execution.inputs = {};
  expect_status("missing transpose input", backend.execute(execution), StatusCode::kInvalidArgument);
  execution.inputs = inputs; execution.outputs = {};
  expect_status("missing transpose output", backend.execute(execution), StatusCode::kInvalidArgument);
  for (const auto dtype : {DType::kFloat32, DType::kInt32}) {
    const auto empty = cpu::empty({0, 3}, dtype);
    const auto result = cpu::execute_unary(op, empty);
    check(result.shape() == Shape{3, 0} && result.dtype() == dtype && result.size() == 0,
          "empty transpose metadata mismatch");
  }
}

}  // namespace

int main() {
  transpose_contract_tests();
  cast_broadcast_contract_tests();
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
