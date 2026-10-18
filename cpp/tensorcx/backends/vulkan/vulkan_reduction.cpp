#include "tensorcx/backends/vulkan/vulkan_reduction.h"

#include <bit>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "tensorcx/backends/vulkan/vulkan_context.h"
#include "tensorcx/backends/vulkan/vulkan_tensor.h"
#include "tensorcx/core/math.h"

namespace tensorcx::vulkan {
namespace {
std::uint32_t narrow(Dim value) {
  if (value < 0 || static_cast<std::uint64_t>(value) > UINT32_MAX)
    throw std::invalid_argument("Vulkan reduction metadata exceeds uint32 limits");
  return static_cast<std::uint32_t>(value);
}
std::uint32_t shader_operation(OpKind kind) {
  switch (kind) {
    case OpKind::kSum: return 40;
    case OpKind::kMax: return 41;
    case OpKind::kMean: return 42;
    case OpKind::kMin: return 43;
    case OpKind::kArgmax: return 44;
    default: throw std::invalid_argument("unsupported Vulkan reduction operation");
  }
}
}  // namespace

Status execute_reduction(const BackendExecution& execution) {
  try {
    auto converted = from_core_tensor(execution.inputs[0]);
    if (!converted) return converted.status();
    const auto input = converted.move_value();
    if (input.dtype() == DType::kBool)
      throw std::invalid_argument("numeric reductions require float32 or int32 tensors");
    const auto kind = execution.op.kind;
    auto operation = shader_operation(kind);
    if (kind == OpKind::kMean && input.dtype() != DType::kFloat32)
      throw std::invalid_argument("mean only supports float32 tensors");
    ReductionPlan plan;
    auto output_dtype = input.dtype();
    if (kind == OpKind::kMin || kind == OpKind::kArgmax) {
      const auto math = make_math_plan(execution.op, execution.inputs);
      plan = {math.shape, math.metadata, math.reduce};
      output_dtype = math.dtype;
    } else {
      if (input.shape().empty() && !execution.op.reduction_axes &&
          execution.op.axis != 0 && execution.op.axis != -1)
        throw std::invalid_argument("reduction axis is out of range");
      const auto axes = execution.op.reduction_axes.value_or(
          input.shape().empty() ? Shape{} : Shape{execution.op.axis});
      plan = make_reduction_plan(input.shape(), axes);
      if (kind == OpKind::kMax && !plan.reduction_size)
        throw std::invalid_argument("max reduction requires non-empty axes");
      // Explicit axis=() copies storage, including signed zero and NaN bits;
      // a scalar single-axis sum/mean still performs arithmetic as on CPU.
      if (execution.op.reduction_axes && execution.op.reduction_axes->empty()) operation = 45;
    }
    const auto count = narrow(numel(plan.output_shape));
    const auto reduction = narrow(plan.reduction_size);
    (void)narrow(input.size());
    std::vector<std::uint32_t> words;
    if (count && reduction && operation != 45)
      for (auto value : plan.index_metadata) words.push_back(narrow(value));
    if (words.empty()) words.push_back(0);
    auto allocated = empty(plan.output_shape, output_dtype, input.device().index);
    if (!allocated) return allocated.status();
    auto output = allocated.move_value();
    if (count) {
      auto context = input.buffer()->owner();
      auto meta = context->allocate(words.size() * sizeof(std::uint32_t));
      context->upload(meta, words.data(), words.size() * sizeof(std::uint32_t));
      const auto& target = output.buffer()->allocation();
      const auto& source = input.size() ? input.buffer()->allocation() : target;
      // Only shape-derived metadata is prepared on the host. Tensor values
      // stay on the GPU; the divisor matches CPU's float32 extent conversion.
      const auto divisor = std::bit_cast<std::uint32_t>(static_cast<float>(reduction));
      context->compute({source, source, target, meta},
                       {count, operation, reduction, narrow(plan.output_shape.size()), 0,
                        narrow(input.shape().size()), input.dtype() == DType::kInt32 ? 1u : 0u,
                        divisor});
    }
    execution.outputs[0] = to_core_tensor(output);
    return Status::Ok();
  } catch (const std::invalid_argument& e) { return Status(StatusCode::kInvalidArgument, e.what()); }
    catch (const std::exception& e) { return Status(StatusCode::kInternal, e.what()); }
}

}  // namespace tensorcx::vulkan
