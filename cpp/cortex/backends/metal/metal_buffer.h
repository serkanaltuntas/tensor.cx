#pragma once

#include <cstddef>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "cortex/core/buffer.h"
#include "cortex/core/dtype.h"

namespace cortex::metal {

class MetalBuffer final : public Buffer {
 public:
  MetalBuffer(DType dtype, std::size_t elements);

  DType dtype() const { return dtype_; }
  std::size_t nbytes() const override { return nbytes_; }

  void copy_from_host(const void* src, std::size_t nbytes);
  void copy_to_host(void* dst, std::size_t nbytes) const;

 private:
  DType dtype_;
  std::size_t nbytes_;
  NS::SharedPtr<MTL::Buffer> buffer_;
};

}  // namespace cortex::metal
