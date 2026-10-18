#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>
#include "tensorcx/core/operation.h"

namespace tensorcx {

enum class ProfileCategory { kDispatch, kAllocation, kDeallocation, kCopy, kCopyAllocate, kCompute, kMemset, kSynchronize, kMemoryPool };

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
  struct DeviceEvent {
    std::uint64_t id{}, host_event_id{};
    const char* name{};
    const char* backend{};
    std::optional<OpKind> operation;
    std::optional<float> elapsed_ms;
    const char* error{"incomplete"};
  };
  bool device_timing{};
  std::uint64_t dropped_device_events{};
  std::vector<DeviceEvent> device_events;
};

std::uint64_t begin_profile(std::size_t max_events, bool device_timing = false);
ProfileResult end_profile(std::uint64_t token);
const char* profile_category_name(ProfileCategory category);
const char* profile_operation_name(OpKind operation);

struct ProfileSession;

// Backend-neutral bounded result slots; no device handle, clock or pending work
// enters the core. A backend fills each slot before its synchronous call ends.
struct DeviceProfileToken {
  std::uint64_t session{};
  std::size_t index{};
  explicit operator bool() const noexcept { return session != 0; }
};
DeviceProfileToken reserve_device_profile(const char* name, const char* backend) noexcept;
void finish_device_profile(DeviceProfileToken token, std::optional<float> elapsed_ms,
                           const char* error) noexcept;

// Thread-local, host steady-clock spans. No clock reads or allocations while
// disabled. Recording uses capacity reserved by begin_profile; never throws.
class ProfileScope {
 public:
  ProfileScope(ProfileCategory category, const char* name, const char* backend,
               std::optional<std::uint64_t> bytes = std::nullopt,
               std::optional<OpKind> operation = std::nullopt) noexcept;
  ~ProfileScope() noexcept;
  bool recording() const noexcept { return recorded_; }
  std::uint64_t event_id() const noexcept { return recorded_ ? index_ + 1 : 0; }
  std::optional<OpKind> operation() const noexcept { return operation_; }
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
  std::optional<OpKind> operation_;
};

}  // namespace tensorcx
