#include "tensorcx/backends/cpu/cpu_buffer.h"

#include <stdexcept>
#include <limits>
#include <utility>

#include "tensorcx/core/dtype.h"

namespace tensorcx::cpu {

CpuBuffer::CpuBuffer(DType dtype, std::size_t elements) : dtype_(dtype) {
  switch (dtype_) {
    case DType::kFloat32:
      data_ = std::vector<float>(elements);
      break;
    case DType::kInt32:
      data_ = std::vector<std::int32_t>(elements);
      break;
    case DType::kBool:
      data_ = std::vector<std::uint8_t>(elements);
      break;
    default:
      throw std::invalid_argument("unknown CPU buffer dtype");
  }
}

CpuBuffer::CpuBuffer(std::vector<float> values)
    : dtype_(DType::kFloat32), data_(std::move(values)) {}

CpuBuffer::CpuBuffer(std::vector<std::int32_t> values)
    : dtype_(DType::kInt32), data_(std::move(values)) {}

CpuBuffer::CpuBuffer(std::vector<std::uint8_t> values)
    : dtype_(DType::kBool), data_(std::move(values)) {}


CpuBuffer::CpuBuffer(DType dtype, std::size_t elements, void* data, std::shared_ptr<void> owner)
    : dtype_(dtype), external_data_(data), external_elements_(elements), external_owner_(std::move(owner)) {
  if (!external_owner_ || (elements && !data))
    throw std::invalid_argument("borrowed CPU storage requires an owner and data");
  const auto bytes = dtype_size(dtype);
  if (elements > std::numeric_limits<std::size_t>::max() / bytes)
    throw std::invalid_argument("borrowed CPU byte size overflow");
  if (elements && reinterpret_cast<std::uintptr_t>(data) % bytes)
    throw std::invalid_argument("borrowed CPU data is misaligned");
}

const void* CpuBuffer::data() const {
  if (external_owner_) return external_data_;
  return std::visit([](const auto& values) -> const void* { return values.data(); }, data_);
}
StorageRelation CpuBuffer::storage_relation(const Buffer& other) const {
  const auto* rhs = dynamic_cast<const CpuBuffer*>(&other);
  if (!rhs) return StorageRelation::kDisjoint;
  if (this == rhs) return StorageRelation::kSameRange;
  if (!nbytes() || !rhs->nbytes()) return StorageRelation::kDisjoint;
  const auto a = reinterpret_cast<std::uintptr_t>(data());
  const auto b = reinterpret_cast<std::uintptr_t>(rhs->data());
  if (a == b && nbytes() == rhs->nbytes()) return StorageRelation::kSameRange;
  const bool overlap = a <= b ? b - a < nbytes() : a - b < rhs->nbytes();
  return overlap ? StorageRelation::kPartialOverlap : StorageRelation::kDisjoint;
}
void* CpuBuffer::mutable_data() {
  if (external_owner_) return external_data_;
  return std::visit([](auto& values) -> void* { return values.data(); }, data_);
}
std::size_t CpuBuffer::nbytes() const {
  if (external_owner_) return external_elements_ * dtype_size(dtype_);
  return std::visit([](const auto& values) { return values.size() * sizeof(values[0]); }, data_);
}

std::span<const float> CpuBuffer::float_data() const {
  if (dtype_ != DType::kFloat32) throw std::invalid_argument("CPU dtype mismatch");
  return {static_cast<const float*>(data()), nbytes() / sizeof(float)};
}
std::span<float> CpuBuffer::mutable_float_data() {
  if (dtype_ != DType::kFloat32) throw std::invalid_argument("CPU dtype mismatch");
  return {static_cast<float*>(mutable_data()), nbytes() / sizeof(float)};
}

std::span<const std::int32_t> CpuBuffer::int32_data() const {
  if (dtype_ != DType::kInt32) throw std::invalid_argument("CPU dtype mismatch");
  return {static_cast<const std::int32_t*>(data()), nbytes() / sizeof(std::int32_t)};
}
std::span<std::int32_t> CpuBuffer::mutable_int32_data() {
  if (dtype_ != DType::kInt32) throw std::invalid_argument("CPU dtype mismatch");
  return {static_cast<std::int32_t*>(mutable_data()), nbytes() / sizeof(std::int32_t)};
}

std::span<const std::uint8_t> CpuBuffer::bool_data() const {
  if (dtype_ != DType::kBool) throw std::invalid_argument("CPU dtype mismatch");
  return {static_cast<const std::uint8_t*>(data()), nbytes() / sizeof(std::uint8_t)};
}
std::span<std::uint8_t> CpuBuffer::mutable_bool_data() {
  if (dtype_ != DType::kBool) throw std::invalid_argument("CPU dtype mismatch");
  return {static_cast<std::uint8_t*>(mutable_data()), nbytes() / sizeof(std::uint8_t)};
}

}  // namespace tensorcx::cpu
