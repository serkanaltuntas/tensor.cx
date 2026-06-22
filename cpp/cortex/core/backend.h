#pragma once

#include <span>
#include <string>

#include "cortex/core/operation.h"
#include "cortex/core/status.h"
#include "cortex/core/tensor.h"

namespace cortex {

// PHASE 8 TARGET — NOT YET WIRED.
//
// This is the single data-driven dispatch entry point §5.6 mandates: one
// execute() per backend, switching on OpDesc, instead of one virtual method per
// op. It is defined now to fix the shape of the contract, but no backend
// implements it and nothing calls it yet. Through Phase 5 the live dispatch is
// the per-op typed entry points in each backend (cortex::cpu::execute_binary,
// cortex::metal::execute_binary, matmul_custom, fill, ...), routed by the
// nanobind layer. Backends are migrated onto this interface in Phase 8
// ("Backend interface hardening"), where the stub-backend Definition of Done
// also requires OpDesc to carry op attributes (see operation.h). Until then,
// treat this header as a design placeholder, not the running code path.
class Backend {
 public:
  virtual ~Backend() = default;

  virtual std::string name() const = 0;
  virtual Status execute(const OpDesc& op,
                         std::span<const Tensor> inputs,
                         std::span<Tensor> outputs) = 0;
};

}  // namespace cortex
