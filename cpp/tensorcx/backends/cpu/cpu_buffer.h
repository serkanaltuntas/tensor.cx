#pragma once

#include <cstdint>
#include <span>
#include <memory>
#include <variant>
#include <vector>

#include "tensorcx/core/buffer.h"
#include "tensorcx/core/dtype.h"

namespace tensorcx::cpu {

class CpuBuffer final : public Buffer {
 public:
  CpuBuffer(DType dtype, std::size_t elements);
  explicit CpuBuffer(std::vector<float> values);
  explicit CpuBuffer(std::vector<std::int32_t> values);
  explicit CpuBuffer(std::vector<std::uint8_t> values);

  // Borrowed storage keeps its external owner alive; no allocation or copy.
  CpuBuffer(DType dtype, std::size_t elements, void* data, std::shared_ptr<void> owner);

  DType dtype() const { return dtype_; }
  std::size_t nbytes() const override;
  StorageRelation storage_relation(const Buffer& other) const override;

  std::span<const std::uint8_t> bool_data() const;
  std::span<std::uint8_t> mutable_bool_data();
  const void* data() const;
  void* mutable_data();
  std::span<const float> float_data() const;
  std::span<const std::int32_t> int32_data() const;

  std::span<float> mutable_float_data();
  std::span<std::int32_t> mutable_int32_data();

 private:
  DType dtype_;
  void* external_data_{nullptr};
  std::size_t external_elements_{0};
  std::shared_ptr<void> external_owner_;
  std::variant<std::vector<float>, std::vector<std::int32_t>, std::vector<std::uint8_t>> data_;
};

}  // namespace tensorcx::cpu
