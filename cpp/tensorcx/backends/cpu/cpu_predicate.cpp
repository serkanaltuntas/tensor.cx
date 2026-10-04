#include "tensorcx/backends/cpu/cpu_backend.h"
#include "tensorcx/core/predicate.h"
#include <cstring>
#include <stdexcept>

namespace tensorcx::cpu {
namespace {
template <typename T>
bool compare(T a, T b, OpKind op) {
  switch (op) {
    case OpKind::kEqual: return a == b;
    case OpKind::kNotEqual: return a != b;
    case OpKind::kLess: return a < b;
    case OpKind::kLessEqual: return a <= b;
    case OpKind::kGreater: return a > b;
    case OpKind::kGreaterEqual: return a >= b;
    default: throw std::invalid_argument("unknown comparison");
  }
}
}

CpuTensor predicate(const OpDesc& op, std::span<const CpuTensor> inputs) {
  std::vector<Tensor> descriptors;
  for (const auto& input : inputs) descriptors.push_back(to_core_tensor(input));
  const auto plan = make_predicate_plan(op, descriptors);
  CpuTensor output(plan.output_dtype, plan.output_shape);
  const auto offset = [&](Dim i, std::size_t arg) {
    return broadcast_offset(i, plan.output_shape, plan.input_strides[arg]);
  };
  for (Dim i = 0; i < output.size(); ++i) {
    const auto a = offset(i, 0);
    if (op.kind == OpKind::kWhere) {
      const auto arg = inputs[0].bool_data()[a] ? 1U : 2U;
      const auto bytes = dtype_size(output.dtype());
      std::memcpy(static_cast<std::uint8_t*>(output.mutable_data()) + i * bytes,
                  static_cast<const std::uint8_t*>(inputs[arg].data()) + offset(i, arg) * bytes, bytes);
    } else if (is_logical(op.kind)) {
      const bool left = inputs[0].bool_data()[a] != 0;
      const bool right = inputs.size() == 2 && inputs[1].bool_data()[offset(i, 1)] != 0;
      output.mutable_bool_data()[i] = op.kind == OpKind::kLogicalNot ? !left :
          op.kind == OpKind::kLogicalAnd ? left && right :
          op.kind == OpKind::kLogicalOr ? left || right : left != right;
    } else {
      const auto b = offset(i, 1);
      if (inputs[0].dtype() == DType::kFloat32)
        output.mutable_bool_data()[i] = compare(inputs[0].float_data()[a], inputs[1].float_data()[b], op.kind);
      else if (inputs[0].dtype() == DType::kInt32)
        output.mutable_bool_data()[i] = compare(inputs[0].int32_data()[a], inputs[1].int32_data()[b], op.kind);
      else
        output.mutable_bool_data()[i] = compare(inputs[0].bool_data()[a] != 0, inputs[1].bool_data()[b] != 0, op.kind);
    }
  }
  return output;
}

CpuTensor reduce_boolean(const OpDesc& op, const CpuTensor& input) {
  if (input.dtype() != DType::kBool) throw std::invalid_argument("any/all require bool tensors");
  const auto axes = op.reduction_axes.value_or(input.shape().empty() ? Shape{} : Shape{op.axis});
  const auto plan = make_reduction_plan(input.shape(), axes);
  CpuTensor output(DType::kBool, plan.output_shape);
  for (Dim i = 0; i < output.size(); ++i) {
    bool value = op.kind == OpKind::kAll;
    for (Dim j = 0; j < plan.reduction_size; ++j) {
      Dim remaining = i * plan.reduction_size + j, source = 0;
      for (std::size_t axis = input.shape().size(); axis-- > 0;) {
        const auto extent = plan.index_metadata[2 * axis];
        source += (remaining % extent) * plan.index_metadata[2 * axis + 1];
        remaining /= extent;
      }
      if (op.kind == OpKind::kAll) value = value && input.bool_data()[source] != 0;
      else value = value || input.bool_data()[source] != 0;
    }
    output.mutable_bool_data()[i] = value;
  }
  return output;
}

CpuTensor masked_select(const CpuTensor& input, const CpuTensor& mask) {
  const auto plan = make_masked_select_plan(to_core_tensor(input), to_core_tensor(mask));
  Dim selected = 0;
  for (auto value : mask.bool_data()) selected += value != 0;
  Shape shape{selected}; shape.insert(shape.end(), plan.tail_shape.begin(), plan.tail_shape.end());
  CpuTensor output(input.dtype(), shape);
  const auto bytes = static_cast<std::size_t>(plan.block_size) * dtype_size(input.dtype());
  if (!bytes) return output;
  Dim target = 0;
  for (Dim i = 0; i < mask.size(); ++i) {
    if (mask.bool_data()[i]) {
      std::memcpy(static_cast<std::uint8_t*>(output.mutable_data()) + target++ * bytes,
                  static_cast<const std::uint8_t*>(input.data()) + i * bytes, bytes);
    }
  }
  return output;
}

}  // namespace tensorcx::cpu
