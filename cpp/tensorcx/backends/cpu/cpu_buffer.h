#pragma once

#include <cstdint>
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

  DType dtype() const { return dtype_; }
  std::size_t nbytes() const override;

  const std::vector<std::uint8_t>& bool_data() const;
  std::vector<std::uint8_t>& mutable_bool_data();
  const void* data() const;
  void* mutable_data();
  const std::vector<float>& float_data() const;
  const std::vector<std::int32_t>& int32_data() const;

  std::vector<float>& mutable_float_data();
  std::vector<std::int32_t>& mutable_int32_data();

 private:
  DType dtype_;
  std::variant<std::vector<float>, std::vector<std::int32_t>, std::vector<std::uint8_t>> data_;
};

}  // namespace tensorcx::cpu
