#pragma once

#include <span>
#include <string>

#include "cortex/core/operation.h"
#include "cortex/core/status.h"
#include "cortex/core/tensor.h"

namespace cortex {

class Backend {
 public:
  virtual ~Backend() = default;

  virtual std::string name() const = 0;
  virtual Status execute(const OpDesc& op,
                         std::span<const Tensor> inputs,
                         std::span<Tensor> outputs) = 0;
};

}  // namespace cortex
