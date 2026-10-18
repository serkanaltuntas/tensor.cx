#include "tensorcx/backends/vulkan/vulkan_matmul.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "tensorcx/backends/vulkan/vulkan_context.h"
#include "tensorcx/backends/vulkan/vulkan_tensor.h"

namespace tensorcx::vulkan {
namespace {
std::uint32_t narrow(Dim value) {
  if (value < 0 || static_cast<std::uint64_t>(value) > UINT32_MAX)
    throw std::invalid_argument("Vulkan matmul metadata exceeds uint32 limits");
  return static_cast<std::uint32_t>(value);
}
}  // namespace

Status execute_matmul(const BackendExecution& execution) {
  try {
    if (execution.op.matmul_preference != MatmulPreference::kAuto &&
        execution.op.matmul_preference != MatmulPreference::kCustom)
      throw std::invalid_argument("Vulkan matmul supports auto or custom");
    auto left = from_core_tensor(execution.inputs[0]);
    if (!left) return left.status();
    auto right = from_core_tensor(execution.inputs[1]);
    if (!right) return right.status();
    const auto lhs = left.move_value(), rhs = right.move_value();
    if (lhs.dtype() != DType::kFloat32 || rhs.dtype() != DType::kFloat32)
      throw std::invalid_argument("matmul only supports float32 tensors");
    if (lhs.buffer()->owner() != rhs.buffer()->owner())
      throw std::invalid_argument("Vulkan matmul requires the same device owner");
    const auto plan = make_matmul_plan(lhs.shape(), rhs.shape());
    const auto count = narrow(numel(plan.output_shape));
    std::uint32_t m = 0, k = 0, n = 0, rank = 0;
    std::vector<std::uint32_t> words;
    // Empty outputs never traverse metadata; K=0 writes +0 without reading
    // either source. Do not narrow unused, potentially huge empty dimensions.
    if (count && plan.k) {
      m = narrow(plan.m); k = narrow(plan.k); n = narrow(plan.n);
      rank = narrow(plan.batch_shape.size());
      (void)narrow(lhs.size()); (void)narrow(rhs.size());
      for (const auto& values : {plan.batch_shape, plan.lhs_batch_strides, plan.rhs_batch_strides})
        for (auto value : values) words.push_back(narrow(value));
    }
    if (words.empty()) words.push_back(0);
    auto allocated = empty(plan.output_shape, DType::kFloat32, lhs.device().index);
    if (!allocated) return allocated.status();
    auto output = allocated.move_value();
    if (count) {
      auto context = lhs.buffer()->owner();
      auto meta = context->allocate(words.size() * sizeof(std::uint32_t));
      context->upload(meta, words.data(), words.size() * sizeof(std::uint32_t));
      const auto& target = output.buffer()->allocation();
      const auto& a = k ? lhs.buffer()->allocation() : target;
      const auto& b = k ? rhs.buffer()->allocation() : target;
      context->compute({a, b, target, meta}, {count, 50u, k, n, 0u, rank, m});
    }
    execution.outputs[0] = to_core_tensor(output);
    return Status::Ok();
  } catch (const std::invalid_argument& e) { return Status(StatusCode::kInvalidArgument, e.what()); }
    catch (const std::exception& e) { return Status(StatusCode::kInternal, e.what()); }
}

}  // namespace tensorcx::vulkan
