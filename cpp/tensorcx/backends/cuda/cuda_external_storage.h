#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "tensorcx/core/backend.h"
#include "tensorcx/core/dtype.h"
#include "tensorcx/core/expected.h"
#include "tensorcx/core/storage_access.h"

namespace tensorcx::cuda {

// A typed provider, never a cast of an arbitrary DLPack owner's shared<void>.
class ExternalStorage {
 public:
  virtual ~ExternalStorage() = default;
  virtual std::shared_ptr<StorageAccess> access() = 0;
  virtual std::uintptr_t begin() const = 0;
  virtual std::size_t bytes() const = 0;
  virtual std::size_t mapped_bytes() const = 0;
  virtual Status validate_range(void* data, DType dtype, std::size_t bytes) const = 0;
  virtual StorageIdentity identity(void* data, std::size_t bytes) const = 0;
  virtual Status complete_cuda() = 0;
  virtual void poison() noexcept = 0;
};

void register_external_storage(const std::shared_ptr<ExternalStorage>& owner);
void unregister_external_storage(std::uintptr_t begin, const ExternalStorage* owner) noexcept;
Expected<std::shared_ptr<ExternalStorage>> find_external_storage(void* data, std::size_t bytes);

// CUDA operations finish before releasing an external storage ownership lease,
// including their failure paths. Ordinary CUDA buffers add no new GPU waits.
class ExternalLease {
 public:
  explicit ExternalLease(std::vector<std::shared_ptr<ExternalStorage>> owners);
  ~ExternalLease();
  const Status& status() const { return lease_.status(); }
  Status finish();
 private:
  std::vector<std::shared_ptr<ExternalStorage>> owners_;
  StorageLease lease_;
  bool finished_ = false;
  Status completion_;
};

std::vector<std::shared_ptr<ExternalStorage>> external_owners(const BackendExecution& execution);

}  // namespace tensorcx::cuda
