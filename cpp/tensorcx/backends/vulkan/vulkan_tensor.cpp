#include "tensorcx/backends/vulkan/vulkan_tensor.h"

#include <limits>
#include <stdexcept>
#include <utility>

#include "tensorcx/backends/vulkan/vulkan_context.h"

namespace tensorcx::vulkan {
namespace {
std::size_t checked_bytes(DType dtype, const Shape& shape) {
  const auto count = numel(shape);
  const auto width = dtype_size(dtype);
  if (static_cast<std::uint64_t>(count) > std::numeric_limits<std::size_t>::max() / width)
    throw std::invalid_argument("Vulkan tensor byte size overflows size_t");
  return static_cast<std::size_t>(count) * width;
}
}  // namespace

VulkanBuffer::VulkanBuffer(DType dtype, std::size_t bytes, std::shared_ptr<Context> owner,
                           std::shared_ptr<Allocation> allocation)
    : dtype_(dtype), bytes_(bytes), owner_(std::move(owner)), allocation_(std::move(allocation)) {
  if (!owner_ || (bytes_ && (!allocation_ || allocation_->owner != owner_ ||
                            allocation_->capacity < bytes_)))
    throw std::invalid_argument("Vulkan buffer requires matching owned storage");
}
int VulkanBuffer::index() const { return owner_->index; }

VulkanTensor::VulkanTensor(DType dtype, Shape shape, std::shared_ptr<VulkanBuffer> buffer)
    : dtype_(dtype), shape_(std::move(shape)), strides_(contiguous_strides(shape_)),
      device_{"vulkan", buffer ? buffer->index() : 0}, buffer_(std::move(buffer)) {
  if (!buffer_ || buffer_->dtype() != dtype_ || checked_bytes(dtype_, shape_) > buffer_->nbytes())
    throw std::invalid_argument("Vulkan tensor dtype/shape does not match its buffer");
}
std::size_t VulkanTensor::nbytes() const { return checked_bytes(dtype_, shape_); }
std::string device_string(int index) { return index == 0 ? "vulkan" : "vulkan:" + std::to_string(index); }

Expected<VulkanTensor> empty(Shape shape, DType dtype, int index) {
  try {
    const auto bytes = checked_bytes(dtype, shape);
    // A zero element count does not prove the row-major stride products fit.
    // Reject all invalid metadata before entering allocation or pool routing.
    (void)contiguous_strides(shape);
    auto context = get_context(index);
    if (!context) return context.status();
    auto allocation = context.value()->allocate(bytes, false, AllocationKind::kTensor);
    auto buffer = std::make_shared<VulkanBuffer>(dtype, bytes, context.value(), std::move(allocation));
    return VulkanTensor(dtype, std::move(shape), std::move(buffer));
  } catch (const std::invalid_argument& e) { return Status(StatusCode::kInvalidArgument, e.what()); }
    catch (const std::exception& e) { return Status(StatusCode::kInternal, e.what()); }
}

Expected<VulkanTensor> from_cpu(const cpu::CpuTensor& input, int index) {
  auto result = empty(input.shape(), input.dtype(), index);
  if (!result) return result.status();
  try {
    auto& buffer = *result.value().buffer();
    buffer.owner()->upload(buffer.allocation(), input.data(), result.value().nbytes());
    return result.move_value();
  } catch (const std::exception& e) { return Status(StatusCode::kInternal, e.what()); }
}
Expected<cpu::CpuTensor> to_cpu(const VulkanTensor& input) {
  try {
    cpu::CpuTensor result(input.dtype(), input.shape());
    const auto& buffer = *input.buffer();
    buffer.owner()->download(buffer.allocation(), result.mutable_data(), input.nbytes());
    return result;
  } catch (const std::exception& e) { return Status(StatusCode::kInternal, e.what()); }
}
Expected<VulkanTensor> clone(const VulkanTensor& input) {
  auto validated = from_core_tensor(to_core_tensor(input));
  if (!validated) return validated.status();
  auto result = empty(input.shape(), input.dtype(), input.device().index);
  if (!result) return result.status();
  try {
    // Vulkan copies require four-byte alignment, including packed bool tails.
    // Padding belongs to both allocations and is never exposed as tensor data.
    const auto bytes = (input.nbytes() + 3) & ~std::size_t{3};
    input.buffer()->owner()->copy(result.value().buffer()->allocation(),
                                 input.buffer()->allocation(), bytes);
    return result.move_value();
  } catch (const std::invalid_argument& e) { return Status(StatusCode::kInvalidArgument, e.what()); }
    catch (const std::exception& e) { return Status(StatusCode::kInternal, e.what()); }
}
Tensor to_core_tensor(const VulkanTensor& input) {
  return Tensor{input.dtype(), input.shape(), input.strides(), input.device(), input.buffer(), 0};
}
Expected<VulkanTensor> from_core_tensor(const Tensor& input) {
  try {
    if (input.device.type != "vulkan" || input.device.index < 0 || input.offset != 0 ||
        input.strides != contiguous_strides(input.shape))
      return Status(StatusCode::kInvalidArgument, "expected a contiguous zero-offset Vulkan tensor");
    auto buffer = std::dynamic_pointer_cast<VulkanBuffer>(input.buffer);
    if (!buffer || buffer->index() != input.device.index)
      return Status(StatusCode::kInvalidArgument, "Vulkan tensor buffer/device owner mismatch");
    auto context = get_context(input.device.index);
    if (!context) return context.status();
    if (context.value() != buffer->owner())
      return Status(StatusCode::kInvalidArgument, "Vulkan tensor has a foreign device context");
    return VulkanTensor(input.dtype, input.shape, std::move(buffer));
  } catch (const std::exception& e) { return Status(StatusCode::kInvalidArgument, e.what()); }
}

}  // namespace tensorcx::vulkan
