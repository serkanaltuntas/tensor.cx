#include "tensorcx/backends/vulkan/vulkan_math.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tensorcx/backends/vulkan/vulkan_context.h"
#include "tensorcx/backends/vulkan/vulkan_tensor.h"
#include "tensorcx/core/math.h"

namespace tensorcx::vulkan {
namespace {
std::uint32_t narrow(Dim value) {
  if (value < 0 || static_cast<std::uint64_t>(value) > UINT32_MAX)
    throw std::invalid_argument("Vulkan math metadata exceeds uint32 limits");
  return static_cast<std::uint32_t>(value);
}
std::uint32_t shader_operation(OpKind kind) {
  switch (kind) {
    case OpKind::kLog: return 60;
    case OpKind::kSqrt: return 61;
    case OpKind::kAbs: return 62;
    case OpKind::kClip: return 63; // Upper bound is a second GPU pass (64).
    case OpKind::kTopK: return 65;
    default: throw std::invalid_argument("unsupported Vulkan math operation");
  }
}
}  // namespace

Status execute_math(const BackendExecution& execution) {
  try {
    const auto operation = shader_operation(execution.op.kind);
    std::vector<VulkanTensor> inputs;
    for (const auto& descriptor : execution.inputs) {
      auto converted = from_core_tensor(descriptor);
      if (!converted) return converted.status();
      inputs.push_back(converted.move_value());
      if (inputs.back().buffer()->owner() != inputs.front().buffer()->owner())
        throw std::invalid_argument("Vulkan math requires the same device owner");
    }
    const auto plan = make_math_plan(execution.op, execution.inputs);
    const auto count = narrow(numel(plan.shape)), groups = narrow(plan.groups);
    const bool topk = execution.op.kind == OpKind::kTopK;
    std::uint32_t rank = 0, reduce = 0, inner = 0, k = 0;
    std::vector<std::uint32_t> words;
    if (count) {
      for (const auto& input : inputs) (void)narrow(input.size());
      if (topk) {
        reduce = narrow(plan.reduce); inner = narrow(plan.inner); k = narrow(plan.k);
      } else if (execution.op.kind == OpKind::kClip) {
        rank = narrow(plan.rank);
        for (auto word : plan.metadata) words.push_back(narrow(word));
      }
    }
    if (words.empty()) words.push_back(0);
    auto allocated = empty(plan.shape, plan.dtype, inputs[0].device().index);
    if (!allocated) return allocated.status();
    auto output = allocated.move_value();
    std::optional<VulkanTensor> indices;
    if (topk) {
      auto created = empty(plan.shape, DType::kInt32, inputs[0].device().index);
      if (!created) return created.status();
      indices = created.move_value();
    }
    if (groups) {
      auto context = inputs[0].buffer()->owner();
      auto meta = context->allocate(words.size() * sizeof(std::uint32_t));
      context->upload(meta, words.data(), words.size() * sizeof(std::uint32_t));
      const auto& source = inputs[0].buffer()->allocation();
      const auto& target = output.buffer()->allocation();
      const auto& right = topk ? indices->buffer()->allocation()
          : execution.op.kind == OpKind::kClip ? inputs[1].buffer()->allocation() : source;
      const auto flags = static_cast<std::uint32_t>(execution.op.largest) |
                         (static_cast<std::uint32_t>(execution.op.sorted) << 1u);
      context->compute({source, right, target, meta},
          {groups, operation, reduce, inner, 0, topk ? k : rank,
           inputs[0].dtype() == DType::kInt32 ? 1u : 0u, flags});
      if (execution.op.kind == OpKind::kClip) {
        // Expanded intermediate data stays private. One writer per element;
        // the second pass reads/writes that same element after the first wait.
        context->compute({target, inputs[2].buffer()->allocation(), target, meta},
                         {groups, 64u, 0, 0, 0, rank,
                          inputs[0].dtype() == DType::kInt32 ? 1u : 0u});
      }
    }
    auto result = to_core_tensor(output);
    if (indices) {
      auto index_result = to_core_tensor(*indices);
      execution.outputs[0] = std::move(result);
      execution.outputs[1] = std::move(index_result);
    } else execution.outputs[0] = std::move(result);
    return Status::Ok();
  } catch (const std::invalid_argument& e) { return Status(StatusCode::kInvalidArgument, e.what()); }
    catch (const std::exception& e) { return Status(StatusCode::kInternal, e.what()); }
}

}  // namespace tensorcx::vulkan
