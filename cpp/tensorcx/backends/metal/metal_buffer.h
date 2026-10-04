#pragma once

#include <cstddef>
#include <memory>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "tensorcx/core/buffer.h"
#include "tensorcx/core/dtype.h"
#include "tensorcx/core/expected.h"
#include "tensorcx/core/status.h"

namespace tensorcx::metal {

class MetalBuffer final : public Buffer {
 public:
  static Expected<std::shared_ptr<MetalBuffer>> create(DType dtype, std::size_t elements);

  DType dtype() const { return dtype_; }
  std::size_t nbytes() const override { return nbytes_; }
  MTL::Buffer* native() const { return buffer_.get(); }

  Status copy_from_host(const void* src, std::size_t nbytes);
  Status copy_to_host(void* dst, std::size_t nbytes) const;

 private:
  MetalBuffer(DType dtype, std::size_t elements);
  bool valid() const { return nbytes_ == 0 || static_cast<bool>(buffer_); }

  DType dtype_;
  std::size_t nbytes_;
  NS::SharedPtr<MTL::Buffer> buffer_;
};

}  // namespace tensorcx::metal
