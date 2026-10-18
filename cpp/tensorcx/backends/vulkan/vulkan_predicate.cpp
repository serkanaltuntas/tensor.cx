#include "tensorcx/backends/vulkan/vulkan_predicate.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "tensorcx/backends/vulkan/vulkan_context.h"
#include "tensorcx/backends/vulkan/vulkan_tensor.h"
#include "tensorcx/core/predicate.h"

namespace tensorcx::vulkan {
namespace {
std::uint32_t narrow(Dim value) {
  if (value < 0 || static_cast<std::uint64_t>(value) > UINT32_MAX)
    throw std::invalid_argument("Vulkan predicate metadata exceeds uint32 limits");
  return static_cast<std::uint32_t>(value);
}
std::uint32_t dtype_code(DType dtype) {
  return dtype == DType::kBool ? 2u : dtype == DType::kInt32 ? 1u : 0u;
}
std::uint32_t word_count(const VulkanTensor& tensor) {
  const auto count = narrow(tensor.size());
  return tensor.dtype() == DType::kBool ? count / 4 + (count % 4 != 0) : count;
}
std::uint32_t shader_operation(OpKind kind) {
  switch (kind) {
    case OpKind::kEqual: return 20;
    case OpKind::kNotEqual: return 21;
    case OpKind::kLess: return 22;
    case OpKind::kLessEqual: return 23;
    case OpKind::kGreater: return 24;
    case OpKind::kGreaterEqual: return 25;
    case OpKind::kLogicalAnd: return 26;
    case OpKind::kLogicalOr: return 27;
    case OpKind::kLogicalXor: return 28;
    case OpKind::kLogicalNot: return 29;
    case OpKind::kAny: return 32;
    case OpKind::kAll: return 33;
    default: throw std::invalid_argument("unsupported Vulkan predicate operation");
  }
}
std::shared_ptr<Allocation> upload_metadata(const std::shared_ptr<Context>& context,
                                          const std::vector<std::uint32_t>& words) {
  auto allocation = context->allocate(words.size() * sizeof(std::uint32_t));
  context->upload(allocation, words.data(), words.size() * sizeof(std::uint32_t));
  return allocation;
}
std::shared_ptr<Allocation> storage_or(const VulkanTensor& tensor, const VulkanTensor& output) {
  return tensor.size() ? tensor.buffer()->allocation() : output.buffer()->allocation();
}
}  // namespace

Status execute_predicate(const BackendExecution& execution) {
  try {
    std::vector<VulkanTensor> inputs;
    for (const auto& tensor : execution.inputs) {
      auto value = from_core_tensor(tensor);
      if (!value) return value.status();
      inputs.push_back(value.move_value());
    }
    const auto& input = inputs.front();
    auto context = input.buffer()->owner();
    for (const auto& tensor : inputs)
      if (tensor.buffer()->owner() != context)
        throw std::invalid_argument("Vulkan predicates require the same device owner");
    const auto kind = execution.op.kind;
    if (is_predicate_elementwise(kind)) {
      const auto plan = make_predicate_plan(execution.op, execution.inputs);
      const auto logical_count = narrow(numel(plan.output_shape));
      auto allocated = empty(plan.output_shape, plan.output_dtype, input.device().index);
      if (!allocated) return allocated.status();
      auto output = allocated.move_value();
      if (logical_count) {
        const auto pass = [&](std::size_t right, std::uint32_t operation) {
          std::vector<std::uint32_t> words{logical_count};
          for (auto extent : plan.output_shape) words.push_back(narrow(extent));
          for (auto stride : plan.input_strides[0]) words.push_back(narrow(stride));
          for (auto stride : plan.input_strides[right]) words.push_back(narrow(stride));
          auto meta = upload_metadata(context, words);
          context->compute({input.buffer()->allocation(), inputs[right].buffer()->allocation(),
                            output.buffer()->allocation(), meta},
                           {word_count(output), operation, 0, 0, 0, narrow(plan.output_shape.size()),
                            dtype_code(kind == OpKind::kWhere ? inputs[right].dtype() : input.dtype()),
                            dtype_code(plan.output_dtype)});
        };
        if (kind == OpKind::kWhere) {
          // Keep the Vulkan 1.1 four-storage-buffer minimum: initialize from
          // the false branch, then replace true lanes. Each pass owns full
          // packed bool words; Context supplies barriers between submissions.
          pass(2, 30); pass(1, 31);
        } else {
          pass(inputs.size() == 1 ? 0 : 1, shader_operation(kind));
        }
      }
      execution.outputs[0] = to_core_tensor(output);
      return Status::Ok();
    }
    if (kind == OpKind::kAny || kind == OpKind::kAll) {
      if (input.dtype() != DType::kBool) throw std::invalid_argument("any/all require bool tensors");
      const auto axes = execution.op.reduction_axes.value_or(input.shape().empty() ? Shape{} : Shape{execution.op.axis});
      const auto plan = make_reduction_plan(input.shape(), axes);
      const auto logical_count = narrow(numel(plan.output_shape));
      const auto reduction = narrow(plan.reduction_size);
      auto allocated = empty(plan.output_shape, DType::kBool, input.device().index);
      if (!allocated) return allocated.status();
      auto output = allocated.move_value();
      if (logical_count) {
        std::vector<std::uint32_t> words{logical_count};
        if (reduction)
          for (auto value : plan.index_metadata) words.push_back(narrow(value));
        auto meta = upload_metadata(context, words);
        auto source = storage_or(input, output);
        context->compute({source, source, output.buffer()->allocation(), meta},
                         {word_count(output), shader_operation(kind), reduction, 0, 0,
                          narrow(input.shape().size()), 2, 2});
      }
      execution.outputs[0] = to_core_tensor(output);
      return Status::Ok();
    }
    if (kind != OpKind::kMaskedSelect) throw std::invalid_argument("unsupported Vulkan predicate operation");
    const auto& mask = inputs[1];
    const auto plan = make_masked_select_plan(execution.inputs[0], execution.inputs[1]);
    const auto mask_count = narrow(mask.size());
    std::uint32_t selected = 0;
    std::shared_ptr<Allocation> prefix;
    if (mask_count) {
      const auto bytes = static_cast<std::size_t>(mask_count) * sizeof(std::uint32_t);
      prefix = context->allocate(bytes);
      auto temporary = context->allocate(bytes);
      auto meta = upload_metadata(context, {0});
      context->compute({mask.buffer()->allocation(), prefix, prefix, meta},
                       {mask_count, 34, 0, 0, 0, 0, 2, 0});
      for (std::uint32_t step = 1; step < mask_count;) {
        context->compute({prefix, prefix, temporary, meta}, {mask_count, 35, step, 0, 0, 0, 0, 0});
        std::swap(prefix, temporary);
        if (step > mask_count / 2) break;
        step *= 2;
      }
      // The mask and prefix remain device-resident; only the dynamic result
      // extent crosses back to the host, not tensor values or selected indices.
      context->download(prefix, &selected, sizeof(selected), bytes - sizeof(selected));
    }
    Shape shape{selected}; shape.insert(shape.end(), plan.tail_shape.begin(), plan.tail_shape.end());
    const auto logical_count = narrow(numel(shape));
    auto allocated = empty(shape, input.dtype(), input.device().index);
    if (!allocated) return allocated.status();
    auto output = allocated.move_value();
    if (logical_count) {
      auto meta = upload_metadata(context, {logical_count});
      context->compute({input.buffer()->allocation(), prefix, output.buffer()->allocation(), meta},
                       {word_count(output), 36, mask_count, narrow(plan.block_size), 0, 0,
                        dtype_code(input.dtype()), dtype_code(input.dtype())});
    }
    execution.outputs[0] = to_core_tensor(output);
    return Status::Ok();
  } catch (const std::invalid_argument& e) { return Status(StatusCode::kInvalidArgument, e.what()); }
    catch (const std::exception& e) { return Status(StatusCode::kInternal, e.what()); }
}

}  // namespace tensorcx::vulkan
