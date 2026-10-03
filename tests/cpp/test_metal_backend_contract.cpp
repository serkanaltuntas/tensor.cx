#include <iostream>
#include <limits>
#include <stdexcept>

#include "cortex/backends/metal/metal_backend.h"
#include "cortex/backends/metal/metal_buffer.h"
#include "cortex/backends/metal/metal_tensor.h"
#include "cortex/core/status.h"

int main() {
  // This guard must work before device discovery, including GPU-free CI.
  try {
    cortex::metal::MetalTensor invalid(cortex::DType::kFloat32, {0}, nullptr);
    std::cerr << "Metal tensor accepted a null buffer\n";
    return 1;
  } catch (const std::invalid_argument&) {
  }
  std::cout << "Metal null-buffer metadata guard passed\n";

  const auto status = cortex::metal::contract_smoke_test();
  if (status.code() == cortex::StatusCode::kUnavailable) {
    std::cout << "metal backend contract smoke skipped: " << status.message() << '\n';
    return 77;
  }
  if (!status.ok()) {
    std::cerr << "metal backend contract smoke failed with status "
              << static_cast<int>(status.code()) << ": " << status.message() << '\n';
    return 1;
  }

  for (const auto dtype : {cortex::DType::kFloat32, cortex::DType::kInt32}) {
    auto allocated = cortex::metal::MetalBuffer::create(dtype, 2);
    if (!allocated) {
      std::cerr << allocated.status().message() << '\n';
      return 1;
    }
    const auto buffer = allocated.move_value();
    const auto overflow = static_cast<cortex::Dim>(
        std::numeric_limits<std::size_t>::max() / cortex::dtype_size(dtype) + 3);
    // Under/oversized shapes and a byte count that wraps to the real 8 bytes.
    for (const auto count : {cortex::Dim{1}, cortex::Dim{3}, overflow}) {
      try {
        cortex::metal::MetalTensor invalid(dtype, {count}, buffer);
        std::cerr << "Metal tensor accepted mismatched buffer size\n";
        return 1;
      } catch (const std::invalid_argument&) {
      }
      auto metadata = cortex::metal::to_core_tensor(
          cortex::metal::MetalTensor(dtype, {2}, buffer));
      metadata.shape = {count};
      try {
        cortex::metal::from_core_tensor(metadata);
        std::cerr << "Metal conversion accepted mismatched buffer size\n";
        return 1;
      } catch (const std::invalid_argument&) {
      }
    }
    const auto other = dtype == cortex::DType::kFloat32
                           ? cortex::DType::kInt32 : cortex::DType::kFloat32;
    try {
      cortex::metal::MetalTensor invalid(other, {2}, buffer);
      std::cerr << "Metal tensor accepted mismatched buffer dtype\n";
      return 1;
    } catch (const std::invalid_argument&) {
    }
    auto empty = cortex::metal::MetalBuffer::create(dtype, 0);
    if (!empty) return 1;
    cortex::metal::MetalTensor valid_empty(dtype, {2, 0}, empty.move_value());
    if (valid_empty.size() != 0 || valid_empty.nbytes() != 0) return 1;
  }
  return 0;
}
