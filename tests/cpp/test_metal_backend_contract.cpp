#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
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
