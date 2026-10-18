#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include "tensorcx/core/profiling.h"

namespace tensorcx::cuda {

struct DriverEventApi {
  using Event = CUevent;
  static bool create(Event& event) noexcept { return cuEventCreate(&event, CU_EVENT_DEFAULT) == CUDA_SUCCESS; }
  static bool record(Event event) noexcept { return cuEventRecord(event, nullptr) == CUDA_SUCCESS; }
  static bool elapsed(float& ms, Event start, Event stop) noexcept {
    return cuEventElapsedTime(&ms, start, stop) == CUDA_SUCCESS;
  }
  static bool destroy(Event event) noexcept { return cuEventDestroy(event) == CUDA_SUCCESS; }
};

// Scoped inside the backend's existing current-context guard. Driver events do
// not touch the Runtime's last-error slot. Never wait or throw for profiling.
template <class Api = DriverEventApi>
class DeviceTimer {
 public:
  explicit DeviceTimer(const char* name) noexcept : token_(reserve_device_profile(name, "cuda")) {
    if (!token_) return;
    if (!Api::create(start_)) error_ = "create_start_failed";
    else if (!Api::create(stop_)) error_ = "create_stop_failed";
    else if (!Api::record(start_)) error_ = "record_start_failed";
  }
  ~DeviceTimer() noexcept { finish(false); }
  DeviceTimer(const DeviceTimer&) = delete;
  DeviceTimer& operator=(const DeviceTimer&) = delete;

  // Call after successful launch, before the existing completion wait.
  void stop() noexcept {
    if (!token_ || error_ || stopped_) return;
    stopped_ = true;
    if (!Api::record(stop_)) error_ = "record_stop_failed";
  }
  // Call after the original launch/wait result is known. Return it unchanged.
  void finish(bool succeeded) noexcept {
    if (!token_) return;
    if (!succeeded && !error_) error_ = "operation_failed";
    std::optional<float> elapsed;
    if (!error_) {
      float ms{};
      if (!stopped_) error_ = "incomplete";
      else if (!Api::elapsed(ms, start_, stop_) || !std::isfinite(ms) || ms < 0)
        error_ = "elapsed_failed";
      else elapsed = ms;
    }
    if (stop_ && !Api::destroy(stop_) && !error_) error_ = "destroy_stop_failed";
    if (start_ && !Api::destroy(start_) && !error_) error_ = "destroy_start_failed";
    start_ = {}; stop_ = {};
    finish_device_profile(token_, elapsed, error_);
    token_ = {};
  }
 private:
  DeviceProfileToken token_;
  typename Api::Event start_{}, stop_{};
  const char* error_{};
  bool stopped_{};
};

inline cudaError_t finish_profiled_launch(DeviceTimer<>& timer) {
  const auto error = cudaGetLastError();
  if (error != cudaSuccess) { timer.finish(false); return error; }
  timer.stop();
  cudaError_t status;
  {
    ProfileScope scope(ProfileCategory::kSynchronize, "cudaDeviceSynchronize", "cuda");
    status = cudaDeviceSynchronize();
  }
  timer.finish(status == cudaSuccess);
  return status;
}

inline cudaError_t profiled_malloc(void** pointer, std::size_t bytes) {
  ProfileScope scope(ProfileCategory::kAllocation, "cudaMalloc", "cuda", bytes);
  return cudaMalloc(pointer, bytes);
}
inline cudaError_t profiled_free(void* pointer) {
  ProfileScope scope(ProfileCategory::kDeallocation, "cudaFree", "cuda");
  return cudaFree(pointer);
}
inline cudaError_t profiled_memcpy(void* target, const void* source, std::size_t bytes, cudaMemcpyKind kind) {
  const char* name = kind == cudaMemcpyHostToDevice ? "host_to_device" :
                     kind == cudaMemcpyDeviceToHost ? "device_to_host" : "device_to_device";
  ProfileScope scope(ProfileCategory::kCopy, name, "cuda", bytes);
  return cudaMemcpy(target, source, bytes, kind);
}
inline cudaError_t profiled_memset(void* target, int value, std::size_t bytes) {
  ProfileScope scope(ProfileCategory::kMemset, "cudaMemset", "cuda", bytes);
  return cudaMemset(target, value, bytes);
}
inline cudaError_t profiled_synchronize() {
  ProfileScope scope(ProfileCategory::kSynchronize, "cudaDeviceSynchronize", "cuda");
  return cudaDeviceSynchronize();
}

}  // namespace tensorcx::cuda
