#pragma once

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "tensorcx/core/buffer.h"
#include "tensorcx/core/backend.h"

namespace tensorcx {

inline std::vector<std::shared_ptr<StorageAccess>> storage_accesses(const BackendExecution& execution) {
  std::vector<std::shared_ptr<StorageAccess>> result;
  const auto append = [&](const Tensor& tensor) {
    if (tensor.buffer) if (auto owner = tensor.buffer->access()) result.push_back(std::move(owner));
  };
  for (const auto& tensor : execution.inputs) append(tensor);
  for (const auto& tensor : execution.outputs) append(tensor);
  for (const auto& argument : execution.kernel_arguments)
    if (argument.kind == KernelArgumentKind::kTensor && argument.tensor) append(*argument.tensor);
  return result;
}

// Keep all owners and locks alive through the caller's native completion.
// Recursive leases allow a composite operation to dispatch validated sub-ops.
class StorageLease {
 public:
  explicit StorageLease(std::vector<std::shared_ptr<StorageAccess>> owners)
      : owners_(std::move(owners)) {
    std::erase(owners_, nullptr);
    std::sort(owners_.begin(), owners_.end(), [](const auto& a, const auto& b) {
      return std::less<StorageAccess*>{}(a.get(), b.get());
    });
    owners_.erase(std::unique(owners_.begin(), owners_.end()), owners_.end());
    // Validate before touching any possibly inherited mutex, and again after
    // locking because another operation may have poisoned an owner meanwhile.
    for (const auto& owner : owners_)
      if (auto s = owner->validate(); !s.ok()) { status_ = std::move(s); return; }
    locks_.reserve(owners_.size());
    for (const auto& owner : owners_) {
      locks_.emplace_back(owner->mutex);
      if (auto s = owner->validate(); !s.ok()) { status_ = std::move(s); return; }
    }
  }
  StorageLease(const StorageLease&) = delete;
  StorageLease& operator=(const StorageLease&) = delete;
  const Status& status() const { return status_; }

 private:
  std::vector<std::shared_ptr<StorageAccess>> owners_;
  std::vector<std::unique_lock<std::recursive_mutex>> locks_;
  Status status_;
};

}  // namespace tensorcx
