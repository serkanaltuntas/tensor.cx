#include "tensorcx/backends/cuda/cuda_memory_pool.h"
#include "tensorcx/backends/cuda/cuda_profiling.h"
#include <algorithm>
#include <stdexcept>

namespace tensorcx::cuda {
namespace {
// Trivial TLS is safe for late resource destructors; initialize ownership only
// on explicit entry, never on the default allocation path.
thread_local MemoryPool* active{};
struct Holder {
  std::shared_ptr<MemoryPool> pool;
  ~Holder() { active = nullptr; if (pool) pool->close_noexcept(); }
};
Holder& holder() { static thread_local Holder value; return value; }
Status invalid(const char* message) { return {StatusCode::kInvalidArgument, message}; }
}

PoolRuntime default_pool_runtime() { return {profiled_malloc, profiled_free, cudaGetLastError}; }

MemoryPool::MemoryPool(std::size_t bytes, std::size_t blocks, PoolRuntime runtime) : runtime_(runtime) {
  if (!blocks || blocks > 4096) throw std::invalid_argument("max_cached_blocks must be in [1, 4096]");
  if (!runtime.allocate || !runtime.free || !runtime.clear_error) throw std::invalid_argument("invalid pool runtime");
  stats_.max_cached_bytes = bytes;
  stats_.max_cached_blocks = blocks;
  cached_.reserve(blocks);  // Return-to-cache never allocates host storage.
}
MemoryPool::~MemoryPool() { close_noexcept(); }

Status MemoryPool::enter() {
  std::lock_guard lock(mutex_);
  if (active) return invalid("memory pools cannot be nested on the same thread");
  if (used_) return invalid("memory pool contexts are single-use");
  ContextScope context;
  if (!context.ready()) return context.status();
  owner_ = context.owner();
  holder().pool = shared_from_this();
  used_ = true;
  stats_.active = true;
  active = this;
  return Status::Ok();
}
Status MemoryPool::exit() {
  if (active != this) return invalid("memory pool must exit on its owning thread");
  // Detach before closing, even when freeing cached blocks fails.
  active = nullptr;
  auto keep_alive = std::move(holder().pool);
  return close();
}
std::shared_ptr<MemoryPool> active_memory_pool() {
  return active ? active->shared_from_this() : nullptr;
}

Expected<void*> MemoryPool::allocate(std::size_t bytes) {
  ProfileScope span(ProfileCategory::kMemoryPool, "pool_acquire", "cuda", bytes);
  std::lock_guard lock(mutex_);
  if (active != this || !stats_.active || !bytes) return invalid("inactive pool or zero-byte allocation");
  const auto found = std::find_if(cached_.begin(), cached_.end(), [=](const Block& b) { return b.bytes == bytes; });
  void* pointer{};
  if (found != cached_.end()) {
    pointer = found->pointer;
    stats_.cached_bytes -= bytes;
    cached_.erase(found);
    ++stats_.hits;
  } else {
    ++stats_.misses;
    ++stats_.allocation_calls;
    auto error = runtime_.allocate(&pointer, bytes);
    if (error == cudaErrorMemoryAllocation && !cached_.empty()) {
      // Clear only the recoverable OOM. Do not turn unrelated CUDA failures
      // into successful allocation retries.
      const auto pending = runtime_.clear_error();
      if (pending != cudaSuccess && pending != cudaErrorMemoryAllocation) {
        ++stats_.allocation_failures;
        return runtime_status(pending, "CUDA pool allocation prior error");
      }
      const auto drained = trim_locked();
      if (!drained.ok()) { ++stats_.allocation_failures; return drained; }
      ++stats_.oom_retries;
      ++stats_.allocation_calls;
      error = runtime_.allocate(&pointer, bytes);
    }
    if (error != cudaSuccess) {
      ++stats_.allocation_failures;
      return runtime_status(error, "CUDA pool allocation");
    }
  }
  stats_.cached_blocks = cached_.size();
  stats_.in_use_bytes += bytes;
  stats_.peak_in_use_bytes = std::max(stats_.peak_in_use_bytes, stats_.in_use_bytes);
  return pointer;
}

cudaError_t MemoryPool::free_locked(void* pointer, std::size_t bytes) {
  ++stats_.free_calls;
  const auto error = runtime_.free(pointer);
  if (error != cudaSuccess) { ++stats_.free_failures; stats_.failed_free_bytes += bytes; }
  return error;
}

void MemoryPool::release(void* pointer, std::size_t bytes, bool reusable, bool context_ready) noexcept {
  try {
    ProfileScope span(ProfileCategory::kMemoryPool, "pool_release", "cuda", bytes);
    std::lock_guard lock(mutex_);
    stats_.in_use_bytes -= bytes;
    if (!context_ready) {
      ++stats_.free_failures;
      stats_.failed_free_bytes += bytes;
      return;  // Same best-effort context failure boundary as direct deletion.
    }
    if (stats_.active && reusable && cached_.size() < stats_.max_cached_blocks &&
        bytes <= stats_.max_cached_bytes - stats_.cached_bytes) {
      // Internal kernels/copies use the legacy stream and are synchronous.
      // Incomplete/error paths fall back to cudaFree, never cache pending work.
      // Driver query does not consume the Runtime's thread-local last error.
      if (cuStreamQuery(CU_STREAM_LEGACY) == CUDA_SUCCESS) {
        cached_.push_back({pointer, bytes});
        stats_.cached_bytes += bytes;
        stats_.cached_blocks = cached_.size();
        stats_.peak_cached_bytes = std::max(stats_.peak_cached_bytes, stats_.cached_bytes);
        return;
      }
      ++stats_.readiness_failures;
    }
    ++stats_.uncached_returns;
    (void)free_locked(pointer, bytes);
  } catch (...) {
    // Destructors cannot throw, including on host bookkeeping failure.
    if (context_ready) (void)runtime_.free(pointer);
  }
}

Status MemoryPool::trim_locked() {
  Status first = Status::Ok();
  while (!cached_.empty()) {
    const auto block = cached_.back();
    cached_.pop_back();
    stats_.cached_bytes -= block.bytes;
    stats_.cached_blocks = cached_.size();
    const auto error = free_locked(block.pointer, block.bytes);
    // Ownership is relinquished once free is attempted; an error may describe
    // earlier work. Never retry an ambiguous free and risk double deletion.
    if (first.ok() && error != cudaSuccess) first = runtime_status(error, "CUDA pool trim");
  }
  return first;
}
Status MemoryPool::trim() {
  if (active != this) return invalid("memory pool trim requires its active owning thread");
  std::lock_guard lock(mutex_);
  ContextScope context(owner_);
  if (!context.ready()) return context.status();
  return trim_locked();
}
Status MemoryPool::close() {
  std::lock_guard lock(mutex_);
  stats_.active = false;
  if (cached_.empty()) return Status::Ok();
  ContextScope context(owner_);
  if (!context.ready()) return context.status();
  return trim_locked();
}
void MemoryPool::close_noexcept() noexcept { try { (void)close(); } catch (...) {} }
PoolStats MemoryPool::stats() const { std::lock_guard lock(mutex_); return stats_; }
}  // namespace tensorcx::cuda
