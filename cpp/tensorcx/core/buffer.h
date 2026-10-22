#pragma once

#include <cstddef>
#include <memory>
#include <mutex>

#include "tensorcx/core/status.h"

namespace tensorcx {

enum class StorageRelation { kDisjoint, kSameRange, kPartialOverlap };

// Opaque host identity only; no device handle or native pointer reaches core.
struct StorageIdentity {
  const void* token;
  std::size_t offset, bytes;
};

class StorageAccess {
 public:
  virtual ~StorageAccess() = default;
  // Must reject inherited/poisoned owners without taking a native driver lock.
  virtual Status validate() const = 0;
  std::recursive_mutex mutex;
};

class Buffer {
 public:
  virtual ~Buffer() = default;
  virtual std::size_t nbytes() const = 0;
  virtual std::shared_ptr<StorageAccess> access() const { return {}; }
  virtual StorageIdentity storage_identity() const { return {this, 0, nbytes()}; }
  // Backends with external storage override this without exposing their handles.
  virtual StorageRelation storage_relation(const Buffer& other) const {
    if (this == &other) return StorageRelation::kSameRange;
    const auto a = storage_identity(), b = other.storage_identity();
    if (!a.token || a.token != b.token || !a.bytes || !b.bytes) return StorageRelation::kDisjoint;
    if (a.offset == b.offset && a.bytes == b.bytes) return StorageRelation::kSameRange;
    const bool overlap = a.offset <= b.offset ? b.offset - a.offset < a.bytes
                                              : a.offset - b.offset < b.bytes;
    return overlap ? StorageRelation::kPartialOverlap : StorageRelation::kDisjoint;
  }
};

}  // namespace tensorcx
