#include "tensorcx/backends/vulkan/vulkan_backend.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "tensorcx/backends/vulkan/vulkan_context.h"
#include "tensorcx/backends/vulkan/vulkan_transform.h"
#include "tensorcx/backends/vulkan/vulkan_predicate.h"
#include "tensorcx/backends/vulkan/vulkan_reduction.h"
#include "tensorcx/core/predicate.h"

namespace tensorcx::vulkan {
namespace {
Status invalid(const std::string& text) { return Status(StatusCode::kInvalidArgument, text); }
std::uint32_t narrow(std::int64_t value) {
  if (value < 0 || static_cast<std::uint64_t>(value) > UINT32_MAX)
    throw std::invalid_argument("Vulkan tensor metadata exceeds uint32 limits");
  return static_cast<std::uint32_t>(value);
}
std::uint32_t scalar_bits(DType dtype, double value) {
  if (dtype == DType::kInt32) {
    if (!is_int32_representable(value)) throw std::invalid_argument("int32 scalar must be exactly representable");
    return std::bit_cast<std::uint32_t>(static_cast<std::int32_t>(value));
  }
  if (dtype == DType::kBool) return value != 0.0 ? 0x01010101u : 0u;
  return std::bit_cast<std::uint32_t>(static_cast<float>(value));
}
std::uint32_t shader_operation(OpKind kind) {
  switch (kind) {
    case OpKind::kFill: return 0;
    case OpKind::kAdd: return 1;
    case OpKind::kSubtract: return 2;
    case OpKind::kMultiply: return 3;
    case OpKind::kDivide: return 4;
    case OpKind::kAddScalar: return 5;
    case OpKind::kSubtractScalar: return 6;
    case OpKind::kMultiplyScalar: return 7;
    case OpKind::kDivideScalar: return 8;
    case OpKind::kNegate: return 9;
    default: throw std::invalid_argument("operation is not supported by the Vulkan backend");
  }
}
}  // namespace

Status VulkanBackend::execute(const BackendExecution& execution) {
  if (auto status = validate_primitive_execution_contract(execution, "vulkan"); !status.ok()) return status;
  if (execution.op.kind == OpKind::kSum || execution.op.kind == OpKind::kMax ||
      execution.op.kind == OpKind::kMean || execution.op.kind == OpKind::kMin ||
      execution.op.kind == OpKind::kArgmax)
    return execute_reduction(execution);
  if (is_predicate_elementwise(execution.op.kind) || execution.op.kind == OpKind::kAny ||
      execution.op.kind == OpKind::kAll || execution.op.kind == OpKind::kMaskedSelect)
    return execute_predicate(execution);
  if (execution.op.kind == OpKind::kCast || execution.op.kind == OpKind::kTranspose ||
      execution.op.kind == OpKind::kSlice || execution.op.kind == OpKind::kConcat)
    return execute_transform(execution);
  try {
    const auto operation = shader_operation(execution.op.kind);
    std::optional<VulkanTensor> lhs, rhs;
    Shape shape;
    DType dtype;
    int index;
    BroadcastPlan plan;
    if (operation == 0) {
      const auto& descriptor = execution.outputs[0];
      shape = descriptor.shape; dtype = descriptor.dtype; index = descriptor.device.index;
    } else {
      auto result = from_core_tensor(execution.inputs[0]);
      if (!result) return result.status();
      lhs = result.move_value();
      shape = lhs->shape(); dtype = lhs->dtype(); index = lhs->device().index;
      if (dtype == DType::kBool) return invalid("Vulkan arithmetic requires float32 or int32");
      if ((operation == 4 || operation == 8) && dtype != DType::kFloat32)
        return invalid("division requires float32 tensors");
      if (operation <= 4) {
        auto right = from_core_tensor(execution.inputs[1]);
        if (!right) return right.status();
        rhs = right.move_value();
        if (rhs->dtype() != dtype) return invalid("Vulkan arithmetic dtype mismatch");
        if (rhs->buffer()->owner() != lhs->buffer()->owner())
          return invalid("Vulkan arithmetic requires the same device owner");
        plan = make_broadcast_plan(lhs->shape(), rhs->shape());
        shape = plan.output_shape;
      } else {
        plan = make_broadcast_plan(shape, shape);
      }
    }
    // Validate scalar narrowing and the complete broadcast before allocating.
    const auto bits = operation == 0 || (operation >= 5 && operation <= 8)
        ? scalar_bits(dtype, execution.op.scalar_value) : 0u;
    auto allocated = empty(shape, dtype, index);
    if (!allocated) return allocated.status();
    auto output = allocated.move_value();
    auto count = narrow(output.size());
    if (dtype == DType::kBool) count = static_cast<std::uint32_t>((output.nbytes() + 3) / 4);
    if (count) {
      auto context = output.buffer()->owner();
      std::vector<std::uint32_t> metadata;
      if (operation != 0) {
        metadata.reserve(shape.size() * 3);
        for (const auto dim : shape) metadata.push_back(narrow(dim));
        for (const auto stride : plan.lhs_strides) metadata.push_back(narrow(stride));
        for (const auto stride : plan.rhs_strides) metadata.push_back(narrow(stride));
      }
      if (metadata.empty()) metadata.push_back(0);
      auto meta = context->allocate(metadata.size() * sizeof(std::uint32_t));
      context->upload(meta, metadata.data(), metadata.size() * sizeof(std::uint32_t));
      const auto& out = output.buffer()->allocation();
      const auto& left = lhs ? lhs->buffer()->allocation() : out;
      const auto& right = rhs ? rhs->buffer()->allocation() : left;
      context->compute({left, right, out, meta}, PushConstants{
          count, operation, bits, static_cast<std::uint32_t>(execution.op.scalar_left),
          0, operation == 0 ? 0u : narrow(shape.size()),
          static_cast<std::uint32_t>(dtype == DType::kInt32)});
    }
    execution.outputs[0] = to_core_tensor(output);
    return Status::Ok();
  } catch (const std::invalid_argument& e) { return invalid(e.what()); }
    catch (const std::exception& e) { return Status(StatusCode::kInternal, e.what()); }
}

}  // namespace tensorcx::vulkan
