#include "tensorcx/backends/vulkan/vulkan_memory_pool.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "tensorcx/core/profiling.h"

namespace tensorcx::vulkan {
namespace {
thread_local MemoryPool* active = nullptr;
struct Holder {
  std::shared_ptr<MemoryPool> pool;
  ~Holder() { active = nullptr; if (pool) pool->close_noexcept(); }
};
Holder& holder() { static thread_local Holder value; return value; }
Status invalid(const char* text) { return {StatusCode::kInvalidArgument, text}; }
}  // namespace

MemoryPool::MemoryPool(int index, std::size_t bytes, std::size_t blocks) : index_(index) {
  if (index < 0 || !blocks || blocks > 4096)
    throw std::invalid_argument("invalid Vulkan memory pool device or block limit");
  stats_.max_cached_bytes = bytes; stats_.max_cached_blocks = blocks;
  cached_.reserve(blocks);
}
MemoryPool::~MemoryPool() { close_noexcept(); }
Status MemoryPool::enter() {
  std::lock_guard lock(mutex_);
  if (active) return invalid("Vulkan memory pools cannot be nested on the same thread");
  if (used_) return invalid("memory pool contexts are single-use");
  auto context = get_context(index_);
  if (!context) return context.status();
  owner_ = context.move_value();
  holder().pool = shared_from_this();
  stats_.active = used_ = true; active = this;
  return Status::Ok();
}
Status MemoryPool::exit() {
  if (active != this) return invalid("memory pool must exit on its owning thread");
  active = nullptr;
  auto keep_alive = std::move(holder().pool);
  return close();
}
std::shared_ptr<MemoryPool> active_memory_pool(int index) {
  return active && active->index() == index ? active->shared_from_this() : nullptr;
}

std::unique_ptr<Allocation> MemoryPool::acquire(std::size_t capacity, bool host_visible) {
  ProfileScope span(ProfileCategory::kMemoryPool, "vulkan_pool_acquire", "vulkan", capacity);
  std::lock_guard lock(mutex_);
  if (active != this || !stats_.active || owner_->poisoned)
    throw std::runtime_error("inactive Vulkan memory pool or poisoned device");
  auto found = std::find_if(cached_.begin(), cached_.end(), [=](const auto& item) {
    return item->capacity == capacity && item->host_visible == host_visible;
  });
  std::unique_ptr<Allocation> result;
  if (found != cached_.end()) {
    result = std::move(*found); cached_.erase(found);
    stats_.cached_bytes -= result->memory_bytes;
    owner_->cache_change(*result, false);
    ++stats_.hits;
  } else {
    ++stats_.misses;
    // Cached blocks occupy real Vulkan allocation slots, even when idle.
    if (owner_->allocation_count >= owner_->limits.maxMemoryAllocationCount) trim_locked();
    try {
      ++stats_.allocation_calls;
      try { result = owner_->allocate_raw(capacity, host_visible); }
      catch (const VulkanError& error) {
        if (error.result != VK_ERROR_OUT_OF_DEVICE_MEMORY || cached_.empty()) throw;
        trim_locked(); ++stats_.oom_retries; ++stats_.allocation_calls;
        result = owner_->allocate_raw(capacity, host_visible);
      }
    } catch (...) { ++stats_.allocation_failures; throw; }
  }
  stats_.cached_blocks = cached_.size();
  stats_.in_use_bytes += result->memory_bytes;
  stats_.peak_in_use_bytes = std::max(stats_.peak_in_use_bytes, stats_.in_use_bytes);
  return result;
}

void MemoryPool::release(std::unique_ptr<Allocation> allocation) noexcept {
  try {
    ProfileScope span(ProfileCategory::kMemoryPool, "vulkan_pool_release", "vulkan", allocation->memory_bytes);
    std::lock_guard lock(mutex_);
    stats_.in_use_bytes -= allocation->memory_bytes;
    // Every submitted use retains a strong reference in its Frame. Last
    // release therefore proves no use is pending; unproven waits quarantine
    // that Frame and its references. Poisoned devices never refill the cache.
    if (stats_.active && !owner_->poisoned && cached_.size() < stats_.max_cached_blocks &&
        allocation->memory_bytes <= stats_.max_cached_bytes - stats_.cached_bytes) {
      owner_->cache_change(*allocation, true);
      stats_.cached_bytes += allocation->memory_bytes;
      stats_.peak_cached_bytes = std::max(stats_.peak_cached_bytes, stats_.cached_bytes);
      cached_.push_back(std::move(allocation));
      stats_.cached_blocks = cached_.size();
      return;
    }
    ++stats_.uncached_returns; ++stats_.free_calls;
  } catch (...) {}  // Unique ownership still frees on bookkeeping failure.
}

std::unique_ptr<Frame> MemoryPool::take_frame() {
  std::lock_guard lock(mutex_);
  if (frame_) {
    ++stats_.frame_hits; owner_->count(&ContextStats::frame_reuses);
    stats_.cached_frames = 0;
    return std::move(frame_);
  }
  ++stats_.frame_misses;
  return nullptr;
}
void MemoryPool::return_frame(std::unique_ptr<Frame> frame, bool reset_ok) {
  std::lock_guard lock(mutex_);
  if (!reset_ok) { ++stats_.frame_reset_failures; return; }
  if (stats_.active && !owner_->poisoned && !frame_) {
    frame_ = std::move(frame); stats_.cached_frames = 1; ++stats_.frame_returns;
  }
}
void MemoryPool::trim_locked() {
  while (!cached_.empty()) {
    auto item = std::move(cached_.back()); cached_.pop_back();
    owner_->cache_change(*item, false);
    stats_.cached_bytes -= item->memory_bytes; ++stats_.free_calls;
  }
  stats_.cached_blocks = 0;
}
Status MemoryPool::trim() {
  if (active != this) return invalid("memory pool trim requires its active owning thread");
  std::lock_guard queue_lock(owner_->mutex);
  std::lock_guard lock(mutex_);
  trim_locked(); frame_.reset(); stats_.cached_frames = 0;
  return Status::Ok();
}
Status MemoryPool::close() {
  {
    std::lock_guard lock(mutex_);
    // Escaping storage may release the last pool reference under Context's
    // queue lock, after exit already drained everything. Avoid locking it again.
    if (!stats_.active && cached_.empty() && !frame_) return Status::Ok();
  }
  std::lock_guard queue_lock(owner_->mutex);
  std::lock_guard lock(mutex_);
  stats_.active = false;
  trim_locked(); frame_.reset(); stats_.cached_frames = 0;
  return Status::Ok();
}
void MemoryPool::close_noexcept() noexcept { try { (void)close(); } catch (...) {} }
PoolStats MemoryPool::stats() const { std::lock_guard lock(mutex_); return stats_; }
}  // namespace tensorcx::vulkan
