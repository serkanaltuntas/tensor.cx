#include "tensorcx/backends/cuda/cuda_context.h"
#include <mutex>

namespace tensorcx::cuda {
namespace {
std::mutex owner_mutex;
std::weak_ptr<PrimaryContext> shared_owner;
}
Status driver_status(CUresult error, const char* operation) {
  if (error == CUDA_SUCCESS) return Status::Ok();
  const char* text = nullptr;
  (void)cuGetErrorString(error, &text);
  const auto code = error == CUDA_ERROR_NO_DEVICE || error == CUDA_ERROR_NOT_INITIALIZED ||
                            error == CUDA_ERROR_SYSTEM_DRIVER_MISMATCH
                        ? StatusCode::kUnavailable : StatusCode::kInternal;
  return {code, std::string(operation) + ": " + (text ? text : "CUDA Driver failure")};
}
Status runtime_status(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) return Status::Ok();
  const auto code = error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
                            error == cudaErrorInitializationError
                        ? StatusCode::kUnavailable : StatusCode::kInternal;
  return {code, std::string(operation) + ": " + cudaGetErrorString(error)};
}
Expected<std::shared_ptr<PrimaryContext>> PrimaryContext::acquire() {
  std::lock_guard lock(owner_mutex);
  if (auto live = shared_owner.lock()) return live;
  if (auto status = driver_status(cuInit(0), "initialize CUDA Driver"); !status.ok()) return status;
  auto owner = std::shared_ptr<PrimaryContext>(new PrimaryContext);
  if (auto status = driver_status(cuDevicePrimaryCtxRetain(&owner->context_, 0),
                                  "retain CUDA primary context"); !status.ok()) return status;
  shared_owner = owner;
  return owner;
}
PrimaryContext::~PrimaryContext() {
  if (context_) (void)cuDevicePrimaryCtxRelease(0);
}
ContextScope::ContextScope() {
  auto owner = PrimaryContext::acquire();
  if (!owner) { status_ = owner.status(); return; }
  owner_ = owner.move_value();
  push();
}
ContextScope::ContextScope(std::shared_ptr<PrimaryContext> owner) : owner_(std::move(owner)) {
  push();
}
void ContextScope::push() {
  if (!owner_) { status_ = {StatusCode::kInvalidArgument, "missing CUDA context owner"}; return; }
  status_ = driver_status(cuCtxPushCurrent(owner_->get()), "push CUDA primary context");
  pushed_ = status_.ok();
}
ContextScope::~ContextScope() {
  if (pushed_) {
    CUcontext popped{};
    (void)cuCtxPopCurrent(&popped);
  }
}
}  // namespace tensorcx::cuda
