#include "cortex/backends/cpu/cpu_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include "cortex/core/dtype.h"

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

struct ReductionDims {
  Shape output_shape;
  std::int64_t outer;
  std::int64_t reduce;
  std::int64_t inner;
};

std::int64_t normalize_axis(std::int64_t axis, std::size_t rank) {
  if (rank == 0) {
    if (axis == 0 || axis == -1) {
      return 0;
    }
    throw std::invalid_argument("reduction axis is out of range");
  }
  const auto signed_rank = static_cast<std::int64_t>(rank);
  if (axis < 0) {
    axis += signed_rank;
  }
  if (axis < 0 || axis >= signed_rank) {
    throw std::invalid_argument("reduction axis is out of range");
  }
  return axis;
}

ReductionDims reduction_dims(const CpuTensor& input, std::int64_t axis) {
  if (input.device().type != "cpu") {
    throw std::invalid_argument("CPU reductions require CPU tensors");
  }

  const auto normalized_axis = normalize_axis(axis, input.shape().size());
  if (input.shape().empty()) {
    return ReductionDims{Shape{}, 1, 1, 1};
  }
  Shape output_shape;
  output_shape.reserve(input.shape().size() - 1);
  for (std::size_t index = 0; index < input.shape().size(); ++index) {
    if (index != static_cast<std::size_t>(normalized_axis)) {
      output_shape.push_back(input.shape()[index]);
    }
  }

  std::int64_t outer = 1;
  for (std::size_t index = 0; index < static_cast<std::size_t>(normalized_axis); ++index) {
    outer *= input.shape()[index];
  }
  std::int64_t inner = 1;
  for (std::size_t index = static_cast<std::size_t>(normalized_axis) + 1;
       index < input.shape().size();
       ++index) {
    inner *= input.shape()[index];
  }

  return ReductionDims{
      std::move(output_shape),
      outer,
      input.shape()[static_cast<std::size_t>(normalized_axis)],
      inner};
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
      if (!is_int32_representable(value)) {
        throw std::invalid_argument("fill value is out of range for int32");
      }
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
        // Compute in uint32 and cast back so int32 overflow is defined
        // two's-complement wraparound (matching NumPy), not signed-overflow UB.
        // This keeps the §12.3 exact-equality contract identical to the Metal
        // add_i32/mul_i32 kernels, which use the same uint round-trip.
        const auto a = static_cast<std::uint32_t>(lhs_data[i]);
        const auto b = static_cast<std::uint32_t>(rhs_data[i]);
        switch (op.kind) {
          case OpKind::kAdd:
            out[i] = static_cast<std::int32_t>(a + b);
            break;
          case OpKind::kMultiply:
            out[i] = static_cast<std::int32_t>(a * b);
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

CpuTensor reduce(const OpDesc& op, const CpuTensor& input) {
  const ReductionDims dims = reduction_dims(input, op.axis);
  if (op.kind == OpKind::kMax && dims.reduce == 0) {
    throw std::invalid_argument("max reduction requires a non-empty axis");
  }

  CpuTensor result(input.dtype(), dims.output_shape);
  switch (input.dtype()) {
    case DType::kFloat32: {
      const auto& input_data = input.float_data();
      auto& out = result.mutable_float_data();
      for (std::int64_t id = 0; id < result.size(); ++id) {
        const auto outer_index = id / dims.inner;
        const auto inner_index = id % dims.inner;
        const auto base = outer_index * dims.reduce * dims.inner + inner_index;
        switch (op.kind) {
          case OpKind::kSum: {
            float sum = 0.0F;
            for (std::int64_t reduce_index = 0; reduce_index < dims.reduce; ++reduce_index) {
              sum += input_data[static_cast<std::size_t>(
                  base + reduce_index * dims.inner)];
            }
            out[static_cast<std::size_t>(id)] = sum;
            break;
          }
          case OpKind::kMax: {
            float max_value = -std::numeric_limits<float>::infinity();
            for (std::int64_t reduce_index = 0; reduce_index < dims.reduce; ++reduce_index) {
              const float value =
                  input_data[static_cast<std::size_t>(base + reduce_index * dims.inner)];
              if (std::isnan(value)) {
                max_value = value;
                break;
              }
              max_value = std::max(max_value, value);
            }
            out[static_cast<std::size_t>(id)] = max_value;
            break;
          }
          default:
            throw std::invalid_argument("unsupported float32 reduction operation");
        }
      }
      break;
    }
    case DType::kInt32: {
      const auto& input_data = input.int32_data();
      auto& out = result.mutable_int32_data();
      for (std::int64_t id = 0; id < result.size(); ++id) {
        const auto outer_index = id / dims.inner;
        const auto inner_index = id % dims.inner;
        const auto base = outer_index * dims.reduce * dims.inner + inner_index;
        switch (op.kind) {
          case OpKind::kSum: {
            std::uint32_t sum = 0;
            for (std::int64_t reduce_index = 0; reduce_index < dims.reduce; ++reduce_index) {
              sum += static_cast<std::uint32_t>(
                  input_data[static_cast<std::size_t>(base + reduce_index * dims.inner)]);
            }
            out[static_cast<std::size_t>(id)] = static_cast<std::int32_t>(sum);
            break;
          }
          case OpKind::kMax: {
            std::int32_t max_value = std::numeric_limits<std::int32_t>::min();
            for (std::int64_t reduce_index = 0; reduce_index < dims.reduce; ++reduce_index) {
              max_value = std::max(
                  max_value,
                  input_data[static_cast<std::size_t>(base + reduce_index * dims.inner)]);
            }
            out[static_cast<std::size_t>(id)] = max_value;
            break;
          }
          default:
            throw std::invalid_argument("unsupported int32 reduction operation");
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
