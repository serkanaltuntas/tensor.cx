#include <iostream>
#include <limits>
#include <stdexcept>

#include "tensorcx/backends/metal/metal_backend.h"
#include "tensorcx/backends/metal/metal_buffer.h"
#include "tensorcx/backends/metal/metal_kernels.h"
#include "tensorcx/backends/metal/metal_tensor.h"
#include "tensorcx/core/status.h"

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
