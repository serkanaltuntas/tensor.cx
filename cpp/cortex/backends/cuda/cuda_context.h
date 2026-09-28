#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <memory>

#include "cortex/core/expected.h"

namespace cortex::cuda {
Status driver_status(CUresult error, const char* operation);
Status runtime_status(cudaError_t error, const char* operation);

// Private backend ownership: never reset a primary context owned by other users.
class PrimaryContext {
 public:
  static Expected<std::shared_ptr<PrimaryContext>> acquire();
  ~PrimaryContext();
  PrimaryContext(const PrimaryContext&) = delete;
  PrimaryContext& operator=(const PrimaryContext&) = delete;
  CUcontext get() const { return context_; }
 private:
  PrimaryContext() = default;
  CUcontext context_{};
};

// Push/pop also preserves a foreign Driver context on the same device.
class ContextScope {
 public:
  ContextScope();
  explicit ContextScope(std::shared_ptr<PrimaryContext> owner);
  ~ContextScope();
  ContextScope(const ContextScope&) = delete;
  ContextScope& operator=(const ContextScope&) = delete;
  const Status& status() const { return status_; }
  bool ready() const { return status_.ok(); }
  const std::shared_ptr<PrimaryContext>& owner() const { return owner_; }
 private:
  std::shared_ptr<PrimaryContext> owner_;
  Status status_;
  bool pushed_{false};
  void push();
};
}  // namespace cortex::cuda
