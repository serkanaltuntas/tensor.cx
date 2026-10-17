#pragma once
#include <atomic>
#include <thread>
#include "tensorcx/backends/cuda/cuda_memory_pool.h"
#include "tensorcx/backends/cuda/cuda_buffer.h"

namespace pool_contract {
inline cudaError_t next_allocation_error = cudaSuccess;
inline unsigned repeat_allocation_error = 0;
inline cudaError_t pending_error = cudaSuccess;
inline bool fail_free = false;
inline std::atomic<bool>* pending_release = nullptr;
inline cudaError_t allocate(void** pointer, std::size_t bytes) {
  const auto error = next_allocation_error;
  if (repeat_allocation_error) --repeat_allocation_error;
  else next_allocation_error = cudaSuccess;
  return error == cudaSuccess ? cudaMalloc(pointer, bytes) : error;
}
inline cudaError_t free(void* pointer) {
  // Reaching this hook proves the readiness check selected direct deletion.
  if (pending_release) pending_release->store(true);
  const auto actual = cudaFree(pointer);
  return std::exchange(fail_free, false) ? cudaErrorInvalidValue : actual;
}
inline cudaError_t clear() { return std::exchange(pending_error, cudaSuccess); }

template <typename Check>
void run(Check check) {
  using namespace tensorcx;
  using namespace tensorcx::cuda;
  const auto make = [] { return std::make_shared<MemoryPool>(1024, 4, PoolRuntime{allocate, free, clear}); };
  const auto buffer = [](std::size_t bytes) { return CudaBuffer::create(DType::kBool, {static_cast<Dim>(bytes)}).value(); };
  auto pool = make();
  check(pool->enter().ok(), "pool enter");
  auto first = buffer(64);
  auto* address = first->data();
  first.reset();
  auto second = buffer(64);
  check(second->data() == address && pool->stats().hits == 1, "exact-size pool reuse");
  second.reset();
  next_allocation_error = cudaErrorMemoryAllocation;
  auto recovered = buffer(128);
  check(pool->stats().oom_retries == 1 && pool->stats().free_calls == 1,
        "OOM did not drain cache and retry exactly once");
  recovered.reset();
  next_allocation_error = cudaErrorInvalidValue;
  check(!CudaBuffer::create(DType::kBool, {129}), "non-OOM allocation error swallowed");
  check(pool->stats().oom_retries == 1 && pool->stats().cached_bytes == 128, "non-OOM drained cache");
  next_allocation_error = cudaErrorMemoryAllocation;
  pending_error = cudaErrorLaunchFailure;
  check(!CudaBuffer::create(DType::kBool, {130}), "prior launch error hidden by retry");
  check(pool->stats().oom_retries == 1 && pool->stats().cached_bytes == 128, "prior error retried allocation");
  fail_free = true;
  check(!pool->trim().ok(), "trim free error not reported");
  check(pool->stats().cached_bytes == 0 && pool->stats().free_failures == 1,
        "ambiguous free retained for double-free retry");
  check(pool->exit().ok(), "pool exit after trim failure");
  check(pool->stats().free_calls == 2, "ambiguous free retried at exit");

  pool = make();
  check(pool->enter().ok(), "failed retry pool enter");
  auto retained = buffer(64);
  retained.reset();
  next_allocation_error = cudaErrorMemoryAllocation;
  repeat_allocation_error = 1;  // Both original allocation and its retry fail.
  check(!CudaBuffer::create(DType::kBool, {128}), "failed OOM retry accepted");
  check(pool->stats().allocation_calls == 3 && pool->stats().oom_retries == 1 &&
        pool->stats().allocation_failures == 1 && pool->stats().in_use_bytes == 0 &&
        pool->stats().cached_bytes == 0, "failed OOM retry accounting");
  check(pool->exit().ok(), "failed retry pool exit");

  pool = make();
  check(pool->enter().ok(), "no-cache OOM pool enter");
  next_allocation_error = cudaErrorMemoryAllocation;
  check(!CudaBuffer::create(DType::kBool, {128}), "no-cache OOM accepted");
  check(pool->stats().allocation_calls == 1 && pool->stats().oom_retries == 0 &&
        pool->stats().allocation_failures == 1, "no-cache OOM retried");
  check(pool->exit().ok(), "no-cache OOM pool exit");

  pool = make();
  check(pool->enter().ok(), "OOM drain error pool enter");
  retained = buffer(64);
  retained.reset();
  next_allocation_error = cudaErrorMemoryAllocation;
  fail_free = true;
  check(!CudaBuffer::create(DType::kBool, {128}), "OOM drain error swallowed");
  check(pool->stats().allocation_calls == 2 && pool->stats().oom_retries == 0 &&
        pool->stats().cached_bytes == 0 && pool->stats().free_failures == 1,
        "OOM drain error retried allocation or retained ambiguous free");
  check(pool->exit().ok() && pool->stats().free_calls == 1, "OOM drain retried ambiguous free at exit");

  pool = make();
  check(pool->enter().ok(), "exit error pool enter");
  retained = buffer(64);
  retained.reset();
  fail_free = true;
  check(!pool->exit().ok(), "exit free error swallowed");
  check(!pool->stats().active && pool->stats().cached_bytes == 0, "exit error did not detach pool");

  CUcontext foreign{};
  check(cuCtxCreate(&foreign, 0, 0) == CUDA_SUCCESS, "foreign context create");
  pool = make();
  check(pool->enter().ok(), "pool foreign enter");
  auto value = buffer(64);
  value.reset();
  check(pool->trim().ok() && pool->exit().ok(), "pool foreign trim/exit");
  CUcontext current{};
  check(cuCtxGetCurrent(&current) == CUDA_SUCCESS && current == foreign, "pool failed to restore foreign context");
  check(cuCtxDestroy(foreign) == CUDA_SUCCESS, "foreign context destroy");

  // Pending work on an error/early-return path must retain cudaFree behavior.
  pool = make();
  check(pool->enter().ok(), "pending pool enter");
  value = buffer(64);
  std::atomic<bool> release{false};
  {
    ContextScope context;
    check(cuLaunchHostFunc(CU_STREAM_LEGACY, [](void* state) {
      auto& done = *static_cast<std::atomic<bool>*>(state);
      while (!done.load()) std::this_thread::yield();
    }, &release) == CUDA_SUCCESS, "pending-work fixture launch");
  }
  pending_release = &release;
  value.reset();
  // Also clean up a regressed cache path, where the free hook did not run;
  // the assertions below must fail promptly instead of hanging on the fixture.
  release.store(true);
  pending_release = nullptr;
  {
    ContextScope context;
    check(cuStreamSynchronize(CU_STREAM_LEGACY) == CUDA_SUCCESS, "pending-work fixture completion");
  }
  check(pool->stats().readiness_failures == 1 && pool->stats().cached_blocks == 0,
        "pending legacy work cached without readiness");
  check(pool->exit().ok(), "pending pool exit");

  // Holder is constructed after the buffer TLS object, hence closed first.
  pool = make();
  std::thread worker([pool] {
    static thread_local std::shared_ptr<CudaBuffer> late;
    (void)&late;
    if (!pool->enter().ok()) std::terminate();
    late = CudaBuffer::create(DType::kBool, {64}).value();
    // Deliberately abandon entry; native TLS closes before late storage dies.
  });
  worker.join();
  check(!pool->stats().active && pool->stats().in_use_bytes == 0 && pool->stats().free_calls == 1,
        "pool late TLS buffer teardown");
}
}  // namespace pool_contract
