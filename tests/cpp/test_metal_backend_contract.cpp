#include <array>
#include <bit>
#include <cmath>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tensorcx/backends/cpu/cpu_backend.h"
#include "tensorcx/backends/metal/metal_backend.h"
#include "tensorcx/backends/metal/metal_buffer.h"
#include "tensorcx/backends/metal/metal_kernels.h"
#include "tensorcx/backends/metal/metal_tensor.h"
#include "tensorcx/core/status.h"

namespace {

bool arithmetic_matches_cpu(
    const tensorcx::OpDesc& op,
    const tensorcx::cpu::CpuTensor& lhs,
    const tensorcx::cpu::CpuTensor* rhs = nullptr) {
  using namespace tensorcx;
  auto left = metal::from_cpu(lhs);
  if (!left) return false;
  std::array<Tensor, 2> inputs{metal::to_core_tensor(left.value()), {}};
  if (rhs) {
    auto right = metal::from_cpu(*rhs);
    if (!right) return false;
    inputs[1] = metal::to_core_tensor(right.value());
  }
  std::array<Tensor, 1> outputs{};
  BackendExecution execution{
      BackendOpClass::kPrimitive, op,
      std::span<const Tensor>(inputs.data(), rhs ? 2 : 1),
      std::span<Tensor>(outputs), std::nullopt, std::nullopt, {}};
  metal::MetalBackend backend;
  const auto status = backend.execute(execution);
  if (!status.ok()) {
    std::cerr << "Metal arithmetic dispatch failed: " << status.message() << '\n';
    return false;
  }
  auto actual = metal::to_cpu(metal::from_core_tensor(outputs[0]));
  if (!actual) return false;
  const auto expected = rhs ? cpu::execute_binary(op, lhs, *rhs)
                            : cpu::execute_unary(op, lhs);
  if (actual.value().shape() != expected.shape() ||
      actual.value().dtype() != expected.dtype()) return false;
  if (expected.dtype() == DType::kInt32) {
    return actual.value().int32_data() == expected.int32_data();
  }
  const auto& values = actual.value().float_data();
  const auto& reference = expected.float_data();
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (std::isnan(reference[i])) {
      if (!std::isnan(values[i])) return false;
    } else if (std::isinf(reference[i]) || reference[i] == 0.0F) {
      if (values[i] != reference[i] || std::signbit(values[i]) != std::signbit(reference[i])) {
        return false;
      }
    } else if (!std::isfinite(values[i]) ||
               std::fabs(values[i] - reference[i]) > 1.0e-6F + 1.0e-6F * std::fabs(reference[i])) {
      return false;
    }
  }
  return true;
}

bool arithmetic_contract() {
  using namespace tensorcx;
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const cpu::CpuTensor floats(Shape{8}, std::vector<float>{0.0F, -0.0F, 1.0F, -2.0F, inf, -inf, nan, 6.0F});
  const cpu::CpuTensor divisors(Shape{8}, std::vector<float>{-0.0F, 0.0F, 0.0F, -0.0F, inf, 2.0F, 1.0F, 3.0F});
  const cpu::CpuTensor integers(Shape{4}, std::vector<std::int32_t>{
      std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max(), 0, -1});
  const cpu::CpuTensor integer_rhs(Shape{4}, std::vector<std::int32_t>{1, -1, 1, 2});
  if (!arithmetic_matches_cpu(OpDesc{OpKind::kSubtract}, floats, &divisors) ||
      !arithmetic_matches_cpu(OpDesc{OpKind::kDivide}, floats, &divisors) ||
      !arithmetic_matches_cpu(OpDesc{OpKind::kNegate}, floats) ||
      !arithmetic_matches_cpu(OpDesc{OpKind::kSubtract}, integers, &integer_rhs) ||
      !arithmetic_matches_cpu(OpDesc{OpKind::kNegate}, integers)) return false;
  for (const auto kind : {OpKind::kAddScalar, OpKind::kSubtractScalar,
                          OpKind::kMultiplyScalar, OpKind::kDivideScalar}) {
    for (const bool scalar_left : {false, true}) {
      OpDesc op{kind};
      op.scalar_left = scalar_left;
      for (const double scalar : {-2.0, 0.0, -0.0}) {
        op.scalar_value = scalar;
        if (!arithmetic_matches_cpu(op, floats)) return false;
        if (kind != OpKind::kDivideScalar && !arithmetic_matches_cpu(op, integers)) return false;
      }
    }
  }
  // Direct native dispatch must reject invalid int32 scalar values and integer
  // division even for empty tensors, before publishing a new output buffer.
  auto empty_buffer = metal::MetalBuffer::create(DType::kInt32, 0);
  if (!empty_buffer) return false;
  metal::MetalTensor empty(DType::kInt32, {0}, empty_buffer.move_value());
  std::array<Tensor, 2> inputs{metal::to_core_tensor(empty), metal::to_core_tensor(empty)};
  std::array<Tensor, 1> outputs{inputs[0]};
  const auto original_buffer = outputs[0].buffer;
  metal::MetalBackend backend;
  BackendExecution execution{
      BackendOpClass::kPrimitive, OpDesc{OpKind::kAddScalar},
      std::span<const Tensor>(inputs.data(), 1), std::span<Tensor>(outputs),
      std::nullopt, std::nullopt, {}};
  for (const auto kind : {OpKind::kAddScalar, OpKind::kSubtractScalar, OpKind::kMultiplyScalar}) {
    execution.op.kind = kind;
    for (const double invalid : {1.5, 2147483648.0, -2147483649.0,
                                std::numeric_limits<double>::infinity(),
                                std::numeric_limits<double>::quiet_NaN()}) {
      execution.op.scalar_value = invalid;
      if (backend.execute(execution).code() != StatusCode::kInvalidArgument ||
          outputs[0].buffer != original_buffer) return false;
    }
  }
  execution.op = OpDesc{OpKind::kDivideScalar};
  execution.op.scalar_value = 2.0;
  if (backend.execute(execution).code() != StatusCode::kInvalidArgument ||
      outputs[0].buffer != original_buffer) return false;
  execution.op = OpDesc{OpKind::kDivide};
  execution.inputs = std::span<const Tensor>(inputs);
  return backend.execute(execution).code() == StatusCode::kInvalidArgument &&
         outputs[0].buffer == original_buffer;
}

bool broadcast_contract() {
  using namespace tensorcx;
  const std::vector<std::pair<Shape, Shape>> shapes{
      {{2, 1, 3}, {1, 4, 1}}, {{2, 3}, {3}}, {{}, {2, 3}},
      {{2, 1, 1}, {1, 1, 5}}, {{0, 3}, {1, 3}}, {{1, 3}, {0, 1}},
      // Metadata is larger than Metal setBytes' inline-data limit.
      {Shape(180, 1), {}}, {{2, 3}, {2, 3}}};
  for (const auto& [left_shape, right_shape] : shapes) {
    for (const auto dtype : {DType::kFloat32, DType::kInt32}) {
      cpu::CpuTensor left(dtype, left_shape), right(dtype, right_shape);
      for (auto* tensor : {&left, &right}) {
        for (std::size_t i = 0; i < static_cast<std::size_t>(tensor->size()); ++i) {
          if (dtype == DType::kFloat32) {
            tensor->mutable_float_data()[i] = 0.5F + static_cast<float>(i % 7);
          } else {
            tensor->mutable_int32_data()[i] = i % 2 ? std::numeric_limits<std::int32_t>::min()
                                                    : std::numeric_limits<std::int32_t>::max();
          }
        }
      }
      for (const auto kind : {OpKind::kAdd, OpKind::kSubtract, OpKind::kMultiply, OpKind::kDivide}) {
        if (kind == OpKind::kDivide && dtype == DType::kInt32) continue;
        if (!arithmetic_matches_cpu(OpDesc{kind}, left, &right) ||
            !arithmetic_matches_cpu(OpDesc{kind}, right, &left)) return false;
      }
    }
  }
  const cpu::CpuTensor special_left(Shape{2, 1}, std::vector<float>{1.0F, -0.0F});
  const cpu::CpuTensor special_right(Shape{1, 4}, std::vector<float>{
      0.0F, -0.0F, std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN()});
  for (const auto kind : {OpKind::kAdd, OpKind::kSubtract, OpKind::kMultiply, OpKind::kDivide}) {
    if (!arithmetic_matches_cpu(OpDesc{kind}, special_left, &special_right) ||
        !arithmetic_matches_cpu(OpDesc{kind}, special_right, &special_left)) return false;
  }
  metal::MetalBackend backend;
  auto left = metal::from_cpu(cpu::CpuTensor(DType::kFloat32, {2, 3}));
  auto right = metal::from_cpu(cpu::CpuTensor(DType::kFloat32, {2, 4}));
  if (!left || !right) return false;
  std::array<Tensor, 2> inputs{metal::to_core_tensor(left.value()), metal::to_core_tensor(right.value())};
  std::array<Tensor, 1> outputs{inputs[0]};
  const auto previous = outputs[0].buffer;
  for (const auto kind : {OpKind::kAdd, OpKind::kSubtract, OpKind::kMultiply, OpKind::kDivide}) {
    BackendExecution execution{BackendOpClass::kPrimitive, OpDesc{kind}, inputs, outputs};
    if (backend.execute(execution).code() != StatusCode::kInvalidArgument ||
        outputs[0].buffer != previous) return false;
  }
  return true;
}

bool cast_contract() {
  using namespace tensorcx;
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const cpu::CpuTensor integers(Shape{7}, std::vector<std::int32_t>{
      std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max(),
      16777217, 16777219, -16777217, 0, -1});
  const cpu::CpuTensor floats(Shape{8}, std::vector<float>{
      -2147483648.0F, 2147483520.0F, 1.9F, -1.9F, 0.0F, -0.0F, 0.99F, -0.99F});
  metal::MetalBackend backend;
  for (const auto dtype : {DType::kFloat32, DType::kInt32}) {
    for (const auto& input : {integers, floats, cpu::CpuTensor(DType::kFloat32, {}),
                              cpu::CpuTensor(DType::kInt32, {}),
                              cpu::CpuTensor(DType::kFloat32, {2, 0}),
                              cpu::CpuTensor(DType::kInt32, {0, 3})}) {
      OpDesc op{OpKind::kCast}; op.target_dtype = dtype;
      if (!arithmetic_matches_cpu(op, input)) return false;
      auto native = metal::from_cpu(input);
      if (!native) return false;
      std::array<Tensor, 1> inputs{metal::to_core_tensor(native.value())}, outputs{};
      BackendExecution execution{BackendOpClass::kPrimitive, op, inputs, outputs};
      if (!backend.execute(execution).ok() || outputs[0].buffer == inputs[0].buffer) return false;
      auto actual = metal::to_cpu(metal::from_core_tensor(outputs[0]));
      if (!actual) return false;
      const auto expected = cpu::execute_unary(op, input);
      if (dtype == DType::kFloat32 && actual.value().float_data() != expected.float_data()) return false;
    }
  }
  // Same-dtype float casts must preserve exact bits, including NaN payloads
  // and signed zero, while allocating independent storage.
  const cpu::CpuTensor special(Shape{5}, std::vector<float>{
      -0.0F, inf, -inf, std::bit_cast<float>(std::uint32_t{0x7fc01234}),
      std::bit_cast<float>(std::uint32_t{0xffc02345})});
  auto native_special = metal::from_cpu(special);
  if (!native_special) return false;
  auto copied = metal::execute_unary(OpDesc{OpKind::kCast}, native_special.value());
  if (!copied || copied.value().buffer() == native_special.value().buffer()) return false;
  auto copied_cpu = metal::to_cpu(copied.value());
  if (!copied_cpu) return false;
  for (std::size_t i = 0; i < special.float_data().size(); ++i) {
    if (std::bit_cast<std::uint32_t>(special.float_data()[i]) !=
        std::bit_cast<std::uint32_t>(copied_cpu.value().float_data()[i])) return false;
  }
  // Invalid device values may touch a temporary result, never caller storage.
  for (const float invalid : {nan, inf, -inf, 2147483648.0F,
                              std::nextafter(-2147483648.0F, -inf)}) {
    std::vector<float> values(257, 3.5F);
    values[0] = 1.25F;
    values.back() = invalid;
    const cpu::CpuTensor original(Shape{257}, std::move(values));
    auto native = metal::from_cpu(original);
    if (!native) return false;
    std::array<Tensor, 1> inputs{metal::to_core_tensor(native.value())}, outputs{inputs[0]};
    const auto previous = outputs[0].buffer;
    OpDesc op{OpKind::kCast}; op.target_dtype = DType::kInt32;
    BackendExecution execution{BackendOpClass::kPrimitive, op, inputs, outputs};
    if (backend.execute(execution).code() != StatusCode::kInvalidArgument ||
        outputs[0].buffer != previous) return false;
    auto unchanged = metal::to_cpu(native.value());
    if (!unchanged || unchanged.value().float_data()[0] != 1.25F ||
        unchanged.value().float_data()[2] != 3.5F) return false;
    if (std::bit_cast<std::uint32_t>(unchanged.value().float_data().back()) !=
        std::bit_cast<std::uint32_t>(invalid)) return false;
  }
  for (const Shape shape : {Shape{0}, Shape{1}}) {
    auto native = metal::from_cpu(cpu::CpuTensor(DType::kFloat32, shape));
    if (!native) return false;
    std::array<Tensor, 1> inputs{metal::to_core_tensor(native.value())}, outputs{inputs[0]};
    const auto previous = outputs[0].buffer;
    OpDesc op{OpKind::kCast}; op.target_dtype = static_cast<DType>(99);
    BackendExecution execution{BackendOpClass::kPrimitive, op, inputs, outputs};
    if (backend.execute(execution).code() != StatusCode::kInvalidArgument ||
        outputs[0].buffer != previous) return false;
    execution.op.target_dtype = DType::kInt32;
    execution.inputs = {};
    if (backend.execute(execution).code() != StatusCode::kInvalidArgument ||
        outputs[0].buffer != previous) return false;
  }
  return true;
}

}  // namespace

int main() {
  // This guard must work before device discovery, including GPU-free CI.
  try {
    tensorcx::metal::MetalTensor invalid(tensorcx::DType::kFloat32, {0}, nullptr);
    std::cerr << "Metal tensor accepted a null buffer\n";
    return 1;
  } catch (const std::invalid_argument&) {
  }
  std::cout << "Metal null-buffer metadata guard passed\n";

  const auto status = tensorcx::metal::contract_smoke_test();
  if (status.code() == tensorcx::StatusCode::kUnavailable) {
    std::cout << "metal backend contract smoke skipped: " << status.message() << '\n';
    return 77;
  }
  if (!status.ok()) {
    std::cerr << "metal backend contract smoke failed with status "
              << static_cast<int>(status.code()) << ": " << status.message() << '\n';
    return 1;
  }

  if (!arithmetic_contract()) {
    std::cerr << "Metal arithmetic CPU parity or validation contract failed\n";
    return 1;
  }
  if (!broadcast_contract() || !cast_contract()) {
    std::cerr << "Metal broadcast/cast CPU parity or validation contract failed\n";
    return 1;
  }

  for (const auto dtype : {tensorcx::DType::kFloat32, tensorcx::DType::kInt32}) {
    auto allocated = tensorcx::metal::MetalBuffer::create(dtype, 2);
    if (!allocated) {
      std::cerr << allocated.status().message() << '\n';
      return 1;
    }
    const auto buffer = allocated.move_value();
    const auto overflow = static_cast<tensorcx::Dim>(
        std::numeric_limits<std::size_t>::max() / tensorcx::dtype_size(dtype) + 3);
    // Under/oversized shapes and a byte count that wraps to the real 8 bytes.
    for (const auto count : {tensorcx::Dim{1}, tensorcx::Dim{3}, overflow}) {
      try {
        tensorcx::metal::MetalTensor invalid(dtype, {count}, buffer);
        std::cerr << "Metal tensor accepted mismatched buffer size\n";
        return 1;
      } catch (const std::invalid_argument&) {
      }
      auto metadata = tensorcx::metal::to_core_tensor(
          tensorcx::metal::MetalTensor(dtype, {2}, buffer));
      metadata.shape = {count};
      try {
        tensorcx::metal::from_core_tensor(metadata);
        std::cerr << "Metal conversion accepted mismatched buffer size\n";
        return 1;
      } catch (const std::invalid_argument&) {
      }
    }
    const auto other = dtype == tensorcx::DType::kFloat32
                           ? tensorcx::DType::kInt32 : tensorcx::DType::kFloat32;
    try {
      tensorcx::metal::MetalTensor invalid(other, {2}, buffer);
      std::cerr << "Metal tensor accepted mismatched buffer dtype\n";
      return 1;
    } catch (const std::invalid_argument&) {
    }
    auto empty = tensorcx::metal::MetalBuffer::create(dtype, 0);
    if (!empty) return 1;
    tensorcx::metal::MetalTensor valid_empty(dtype, {2, 0}, empty.move_value());
    if (valid_empty.size() != 0 || valid_empty.nbytes() != 0) return 1;
  }
  const tensorcx::Shape empty_shape{0, 1, tensorcx::Dim{1} << 62, 4, 0};
  auto buffer = tensorcx::metal::MetalBuffer::create(tensorcx::DType::kFloat32, 0);
  if (!buffer) return 1;
  const tensorcx::metal::MetalTensor input(
      tensorcx::DType::kFloat32, empty_shape, buffer.move_value());
  for (const auto kind : {tensorcx::OpKind::kSum, tensorcx::OpKind::kMax,
                         tensorcx::OpKind::kMean, tensorcx::OpKind::kSoftmax,
                         tensorcx::OpKind::kRmsNorm, tensorcx::OpKind::kLayerNorm}) {
    const bool reduction = kind == tensorcx::OpKind::kSum ||
                           kind == tensorcx::OpKind::kMax || kind == tensorcx::OpKind::kMean;
    tensorcx::OpDesc op{kind};
    op.axis = 1;
    auto result = reduction ? tensorcx::metal::reduce(op, input)
                            : tensorcx::metal::execute_unary(op, input);
    auto expected_shape = empty_shape;
    if (reduction) expected_shape.erase(expected_shape.begin() + 1);
    if (!result || result.value().size() != 0 || result.value().shape() != expected_shape) {
      std::cerr << "Metal empty axis operation failed\n";
      return 1;
    }
  }
  return 0;
}
