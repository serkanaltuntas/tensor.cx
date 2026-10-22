#pragma once

#include "tensorcx/backends/vulkan/vulkan_resources.h"

namespace tensorcx::vulkan {
struct PoolStats {
  std::size_t max_cached_bytes{}, max_cached_blocks{}, cached_bytes{}, cached_blocks{};
  std::size_t in_use_bytes{}, peak_in_use_bytes{}, peak_cached_bytes{};
  std::uint64_t hits{}, misses{}, allocation_calls{}, free_calls{}, allocation_failures{},
      oom_retries{}, uncached_returns{}, frame_hits{}, frame_misses{}, frame_returns{},
      frame_reset_failures{}, cached_frames{};
  bool active{};
};

// Opt-in thread-local allocation routing. Live storage keeps its originating
// pool alive; closing never invalidates tensors that escaped the scope.
class MemoryPool : public std::enable_shared_from_this<MemoryPool> {
 public:
  MemoryPool(int index, std::size_t bytes, std::size_t blocks);
  ~MemoryPool();
  Status enter();
  Status exit();
  Status trim();
  PoolStats stats() const;
  int index() const { return index_; }
  std::unique_ptr<Allocation> acquire(std::size_t capacity, bool host_visible);
  void release(std::unique_ptr<Allocation> allocation) noexcept;
  // Called with the context mutex held. Only completed, reset frames return.
  std::unique_ptr<Frame> take_frame();
  void return_frame(std::unique_ptr<Frame> frame, bool reset_ok);
  void close_noexcept() noexcept;
 private:
  Status close();
  void trim_locked();
  int index_;
  std::shared_ptr<Context> owner_;
  mutable std::mutex mutex_;
  PoolStats stats_;
  std::vector<std::unique_ptr<Allocation>> cached_;
  std::unique_ptr<Frame> frame_;
  bool used_ = false;
};

std::shared_ptr<MemoryPool> active_memory_pool(int index);
}  // namespace tensorcx::vulkan
