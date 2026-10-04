#pragma once

#include <cstddef>

namespace tensorcx {

class Buffer {
 public:
  virtual ~Buffer() = default;
  virtual std::size_t nbytes() const = 0;
};

}  // namespace tensorcx
