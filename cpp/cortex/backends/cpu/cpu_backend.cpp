#include "cortex/backends/cpu/cpu_backend.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace cortex::cpu {

namespace {

void validate_binary_inputs(const CpuTensor& lhs, const CpuTensor& rhs) {
  if (lhs.device().type != "cpu" || rhs.device().type != "cpu") {
    throw std::invalid_argument("CPU operations require CPU tensors");
  }
  if (lhs.dtype() != rhs.dtype()) {
    throw std::invalid_argument("dtype mismatch for binary operation");
  }
  if (lhs.shape() != rhs.shape()) {
    throw std::invalid_argument("shape mismatch for binary operation");
  }
}

}  // namespace

CpuTensor empty(Shape shape, DType dtype) {
  return CpuTensor(dtype, std::move(shape));
}

CpuTensor fill(Shape shape, DType dtype, double value) {
  CpuTensor result(dtype, std::move(shape));
  switch (dtype) {
    case DType::kFloat32: {
      auto& data = result.mutable_float_data();
      std::fill(data.begin(), data.end(), static_cast<float>(value));
      break;
    }
    case DType::kInt32: {
      auto& data = result.mutable_int32_data();
      std::fill(data.begin(), data.end(), static_cast<std::int32_t>(value));
      break;
    }
  }
  return result;
}

CpuTensor execute_binary(const OpDesc& op, const CpuTensor& lhs, const CpuTensor& rhs) {
  validate_binary_inputs(lhs, rhs);

  CpuTensor result(lhs.dtype(), lhs.shape());
  switch (lhs.dtype()) {
    case DType::kFloat32: {
      const auto& lhs_data = lhs.float_data();
      const auto& rhs_data = rhs.float_data();
      auto& out = result.mutable_float_data();
      for (std::size_t i = 0; i < out.size(); ++i) {
        switch (op.kind) {
          case OpKind::kAdd:
            out[i] = lhs_data[i] + rhs_data[i];
            break;
          case OpKind::kMultiply:
            out[i] = lhs_data[i] * rhs_data[i];
            break;
          default:
            throw std::invalid_argument("unsupported binary float32 operation");
        }
      }
      break;
    }
    case DType::kInt32: {
      const auto& lhs_data = lhs.int32_data();
      const auto& rhs_data = rhs.int32_data();
      auto& out = result.mutable_int32_data();
      for (std::size_t i = 0; i < out.size(); ++i) {
        switch (op.kind) {
          case OpKind::kAdd:
            out[i] = lhs_data[i] + rhs_data[i];
            break;
          case OpKind::kMultiply:
            out[i] = lhs_data[i] * rhs_data[i];
            break;
          default:
            throw std::invalid_argument("unsupported binary int32 operation");
        }
      }
      break;
    }
  }
  return result;
}

CpuTensor matmul(const CpuTensor& lhs, const CpuTensor& rhs) {
  if (lhs.device().type != "cpu" || rhs.device().type != "cpu") {
    throw std::invalid_argument("CPU matmul requires CPU tensors");
  }
  if (lhs.dtype() != DType::kFloat32 || rhs.dtype() != DType::kFloat32) {
    throw std::invalid_argument("matmul only supports float32 tensors");
  }
  if (lhs.shape().size() != 2 || rhs.shape().size() != 2) {
    throw std::invalid_argument("matmul requires rank-2 tensors");
  }

  const auto m = lhs.shape()[0];
  const auto k = lhs.shape()[1];
  const auto rhs_k = rhs.shape()[0];
  const auto n = rhs.shape()[1];
  if (k != rhs_k) {
    throw std::invalid_argument("matmul shape mismatch");
  }

  CpuTensor result(DType::kFloat32, Shape{m, n});
  const auto& lhs_data = lhs.float_data();
  const auto& rhs_data = rhs.float_data();
  auto& out = result.mutable_float_data();

  for (std::int64_t row = 0; row < m; ++row) {
    for (std::int64_t col = 0; col < n; ++col) {
      float sum = 0.0F;
      for (std::int64_t inner = 0; inner < k; ++inner) {
        sum += lhs_data[static_cast<std::size_t>(row * k + inner)] *
               rhs_data[static_cast<std::size_t>(inner * n + col)];
      }
      out[static_cast<std::size_t>(row * n + col)] = sum;
    }
  }

  return result;
}

}  // namespace cortex::cpu
