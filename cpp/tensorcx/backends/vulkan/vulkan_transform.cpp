#include "tensorcx/backends/vulkan/vulkan_transform.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "tensorcx/backends/vulkan/vulkan_context.h"
#include "tensorcx/backends/vulkan/vulkan_tensor.h"

namespace tensorcx::vulkan {
namespace {
std::uint32_t narrow(Dim value) {
  if (value < 0 || static_cast<std::uint64_t>(value) > UINT32_MAX)
    throw std::invalid_argument("Vulkan transform metadata exceeds uint32 limits");
  return static_cast<std::uint32_t>(value);
}
std::uint32_t dtype_code(DType dtype) {
  switch (dtype) {
    case DType::kFloat32: return 0;
    case DType::kInt32: return 1;
    case DType::kBool: return 2;
  }
  throw std::invalid_argument("unsupported Vulkan cast dtype");
}
std::uint32_t word_count(const VulkanTensor& output) {
  const auto count = narrow(output.size());
  return output.dtype() == DType::kBool ? count / 4 + (count % 4 != 0) : count;
}
void dispatch(const VulkanTensor& input, const VulkanTensor& output,
              const std::shared_ptr<Allocation>& auxiliary,
              const std::vector<std::uint32_t>& metadata, PushConstants push) {
  auto context = output.buffer()->owner();
  auto meta = context->allocate(metadata.size() * sizeof(std::uint32_t));
  context->upload(meta, metadata.data(), metadata.size() * sizeof(std::uint32_t));
  const auto& out = output.buffer()->allocation();
  // An empty concat operand cannot be dereferenced by the shader. Bind a real
  // allocation anyway: Vulkan descriptors must not contain a null buffer.
  const auto& in = input.size() ? input.buffer()->allocation() : out;
  context->compute({in, auxiliary ? auxiliary : out, out, meta}, push);
}
}  // namespace

Status execute_transform(const BackendExecution& execution) {
  try {
    std::vector<VulkanTensor> inputs;
    std::vector<Shape> shapes;
    for (const auto& tensor : execution.inputs) {
      auto native = from_core_tensor(tensor);
      if (!native) return native.status();
      inputs.push_back(native.move_value());
      shapes.push_back(inputs.back().shape());
    }
    const auto& input = inputs.front();
    const auto& op = execution.op;
    Shape output_shape = input.shape();
    auto target = input.dtype();
    std::vector<std::uint32_t> metadata;
    std::vector<std::uint32_t> blocks;
    std::uint32_t output_block = 0, offset = 0, rank = 0, operation = 0;
    if (op.kind == OpKind::kConcat) {
      for (const auto& item : inputs) {
        if (item.dtype() != input.dtype()) throw std::invalid_argument("concat dtypes must match");
        if (item.buffer()->owner() != input.buffer()->owner())
          throw std::invalid_argument("Vulkan concat requires the same device owner");
      }
      const auto plan = make_concat_plan(shapes, op.axis);
      output_shape = plan.output_shape;
      if (numel(output_shape)) {
        output_block = narrow(output_shape[plan.axis] * plan.inner);
        for (const auto& item : inputs) blocks.push_back(narrow(item.shape()[plan.axis] * plan.inner));
      }
      operation = 12;
    } else if (op.kind == OpKind::kCast) {
      target = op.target_dtype;
      (void)dtype_code(target);
      operation = 10;
    } else {
      const auto plan = op.kind == OpKind::kTranspose
          ? make_transpose_plan(input.shape(), op.axes)
          : make_slice_plan(input.shape(), op.slice_starts, op.slice_steps, op.slice_shape);
      output_shape = plan.output_shape;
      offset = narrow(plan.offset);
      rank = narrow(output_shape.size());
      if (numel(output_shape)) {
        for (auto extent : output_shape) metadata.push_back(narrow(extent));
        // The shared planner proves every final index lies in the source.
        // Signed strides are encoded modulo 2^32, so negative-step slices need
        // neither shaderInt64 nor signed-overflow arithmetic in the shader.
        for (auto stride : plan.input_strides) metadata.push_back(static_cast<std::uint32_t>(stride));
      }
      operation = 11;
    }
    const auto logical_count = narrow(numel(output_shape));
    auto allocated = empty(output_shape, target, input.device().index);
    if (!allocated) return allocated.status();
    auto output = allocated.move_value();
    if (logical_count) {
      const auto count = word_count(output);
      const auto source_dtype = dtype_code(input.dtype());
      const auto target_dtype = dtype_code(target);
      if (op.kind == OpKind::kConcat) {
        // Each pass owns complete output words. This also handles bool segments
        // sharing a word without byte writes, atomics, or write/write races.
        // Context::compute supplies the visibility barrier between passes.
        std::uint32_t start = 0;
        for (std::size_t i = 0; i < inputs.size(); ++i) {
          metadata = {logical_count, blocks[i], output_block};
          dispatch(inputs[i], output, nullptr, metadata,
                   {count, operation, start, static_cast<std::uint32_t>(i == 0),
                    0, 0, source_dtype, target_dtype});
          start += blocks[i];
        }
      } else {
        metadata.insert(metadata.begin(), logical_count);
        std::shared_ptr<Allocation> error;
        if (input.dtype() == DType::kFloat32 && target == DType::kInt32) {
          error = input.buffer()->owner()->allocate(sizeof(std::uint32_t));
          const std::uint32_t zero = 0;
          input.buffer()->owner()->upload(error, &zero, sizeof(zero));
        }
        dispatch(input, output, error, metadata,
                 {count, operation, offset, 0, 0, rank, source_dtype, target_dtype});
        if (error) {
          std::uint32_t invalid = 0;
          input.buffer()->owner()->download(error, &invalid, sizeof(invalid));
          if (invalid) throw std::invalid_argument("float32 value is out of range for int32 cast");
        }
      }
    }
    // Failed conversions never publish partial output or modify input storage.
    execution.outputs[0] = to_core_tensor(output);
    return Status::Ok();
  } catch (const std::invalid_argument& e) { return Status(StatusCode::kInvalidArgument, e.what()); }
    catch (const std::exception& e) { return Status(StatusCode::kInternal, e.what()); }
}

}  // namespace tensorcx::vulkan
