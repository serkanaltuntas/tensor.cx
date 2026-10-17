#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include "tensorcx/backends/cuda/cuda_context.h"

namespace tensorcx::cuda {

struct PoolStats {
  std::size_t max_cached_bytes{}, max_cached_blocks{}, cached_bytes{}, cached_blocks{};
  std::size_t in_use_bytes{}, peak_in_use_bytes{}, peak_cached_bytes{};
  std::uint64_t hits{}, misses{}, allocation_calls{}, free_calls{}, allocation_failures{},
      oom_retries{}, uncached_returns{}, readiness_failures{}, free_failures{}, failed_free_bytes{};
  bool active{};
};

// Injectable runtime calls permit deterministic OOM/free-error tests without
// exhausting a shared GPU. The Python API always uses the production functions.
struct PoolRuntime {
  cudaError_t (*allocate)(void**, std::size_t);
  cudaError_t (*free)(void*);
  cudaError_t (*clear_error)();
};
PoolRuntime default_pool_runtime();

class MemoryPool : public std::enable_shared_from_this<MemoryPool> {
 public:
  MemoryPool(std::size_t bytes, std::size_t blocks, PoolRuntime runtime = default_pool_runtime());
  ~MemoryPool();
  Status enter();
  Status exit();
  Status trim();
  PoolStats stats() const;
  // Private backend use: the caller holds the pool's primary context current.
  Expected<void*> allocate(std::size_t bytes);
  void release(void* pointer, std::size_t bytes, bool reusable, bool context_ready) noexcept;
  void close_noexcept() noexcept;
 private:
  struct Block { void* pointer; std::size_t bytes; };
  Status close();
  Status trim_locked();
  cudaError_t free_locked(void* pointer, std::size_t bytes);
  mutable std::mutex mutex_;
  std::shared_ptr<PrimaryContext> owner_;
  PoolRuntime runtime_;
  PoolStats stats_;
  std::vector<Block> cached_;
  bool used_{};
};

std::shared_ptr<MemoryPool> active_memory_pool();
}  // namespace tensorcx::cuda
