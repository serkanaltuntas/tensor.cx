#pragma once

#include <memory>
#include <string>

#include "tensorcx/backends/cpu/cpu_tensor.h"
#include "tensorcx/core/buffer.h"
#include "tensorcx/core/expected.h"
#include "tensorcx/core/tensor.h"

namespace tensorcx::vulkan {

struct Allocation;
struct Context;

class VulkanBuffer final : public Buffer {
 public:
  VulkanBuffer(DType dtype, std::size_t bytes, std::shared_ptr<Context> owner,
               std::shared_ptr<Allocation> allocation);
  std::size_t nbytes() const override { return bytes_; }
  DType dtype() const { return dtype_; }
  int index() const;
  const std::shared_ptr<Context>& owner() const { return owner_; }
  const std::shared_ptr<Allocation>& allocation() const { return allocation_; }

 private:
  DType dtype_;
  std::size_t bytes_;
  std::shared_ptr<Context> owner_;
  std::shared_ptr<Allocation> allocation_;
};

class VulkanTensor {
 public:
  VulkanTensor(DType dtype, Shape shape, std::shared_ptr<VulkanBuffer> buffer);
  DType dtype() const { return dtype_; }
  const Shape& shape() const { return shape_; }
  const Shape& strides() const { return strides_; }
  const Device& device() const { return device_; }
  std::int64_t size() const { return numel(shape_); }
  std::size_t nbytes() const;
  const std::shared_ptr<VulkanBuffer>& buffer() const { return buffer_; }

 private:
  DType dtype_;
  Shape shape_;
  Shape strides_;
  Device device_;
  std::shared_ptr<VulkanBuffer> buffer_;
};

std::string device_string(int index);
Expected<VulkanTensor> empty(Shape shape, DType dtype, int index = 0);
Expected<VulkanTensor> from_cpu(const cpu::CpuTensor& input, int index = 0);
Expected<cpu::CpuTensor> to_cpu(const VulkanTensor& input);
Tensor to_core_tensor(const VulkanTensor& input);
Expected<VulkanTensor> from_core_tensor(const Tensor& input);

}  // namespace tensorcx::vulkan
