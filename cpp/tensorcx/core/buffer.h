#pragma once

#include <cstddef>

namespace tensorcx {

enum class StorageRelation { kDisjoint, kSameRange, kPartialOverlap };

class Buffer {
 public:
  virtual ~Buffer() = default;
  virtual std::size_t nbytes() const = 0;
  // Backends with external storage override this without exposing their handles.
  virtual StorageRelation storage_relation(const Buffer& other) const {
    return this == &other ? StorageRelation::kSameRange : StorageRelation::kDisjoint;
  }
};

}  // namespace tensorcx
