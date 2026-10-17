#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>
#include "tensorcx/core/operation.h"

namespace tensorcx {

enum class ProfileCategory { kDispatch, kAllocation, kDeallocation, kCopy, kCopyAllocate, kCompute, kMemset, kSynchronize };

struct ProfileEvent {
  std::uint64_t id{}, parent_id{}, start_ns{}, duration_ns{}, self_ns{};
  ProfileCategory category{};
  // Instrumentation supplies static literals, never user text or tensor data.
  const char* name{};
  const char* backend{};
  std::optional<std::uint64_t> bytes;
  std::optional<OpKind> operation;
};

struct ProfileResult {
  std::uint64_t elapsed_ns{}, dropped_events{};
  std::vector<ProfileEvent> events;
};

std::uint64_t begin_profile(std::size_t max_events);
ProfileResult end_profile(std::uint64_t token);
const char* profile_category_name(ProfileCategory category);
const char* profile_operation_name(OpKind operation);

struct ProfileSession;

// Thread-local, host steady-clock spans. No clock reads or allocations while
// disabled. Recording uses capacity reserved by begin_profile; never throws.
class ProfileScope {
 public:
  ProfileScope(ProfileCategory category, const char* name, const char* backend,
               std::optional<std::uint64_t> bytes = std::nullopt,
               std::optional<OpKind> operation = std::nullopt) noexcept;
  ~ProfileScope() noexcept;
  bool recording() const noexcept { return recorded_; }
  void set_bytes(std::uint64_t bytes) noexcept;
  ProfileScope(const ProfileScope&) = delete;
  ProfileScope& operator=(const ProfileScope&) = delete;
 private:
  ProfileSession* session_{};
  ProfileScope* parent_{};
  std::chrono::steady_clock::time_point started_{};
  std::uint64_t children_ns_{};
  std::size_t index_{};
  bool recorded_{};
};

}  // namespace tensorcx
