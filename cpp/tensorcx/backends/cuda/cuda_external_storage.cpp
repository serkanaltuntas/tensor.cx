#include "tensorcx/backends/cuda/cuda_external_storage.h"

#include <map>
#include <mutex>
#include <stdexcept>

#include "tensorcx/backends/cuda/cuda_buffer.h"
#include "tensorcx/backends/cuda/cuda_context.h"

namespace tensorcx::cuda {
namespace {
std::mutex registry_mutex;
struct RegistryEntry { std::weak_ptr<ExternalStorage> weak; const ExternalStorage* identity; };
std::map<std::uintptr_t, RegistryEntry> registry;
std::vector<std::shared_ptr<StorageAccess>> accesses(
    const std::vector<std::shared_ptr<ExternalStorage>>& owners) {
  std::vector<std::shared_ptr<StorageAccess>> result;
  result.reserve(owners.size());
  for (const auto& owner : owners) result.push_back(owner->access());
  return result;
}
}  // namespace

void register_external_storage(const std::shared_ptr<ExternalStorage>& owner) {
  if (!current_process()) throw std::invalid_argument("shared CUDA storage requires a fresh process after fork");
  if (!owner || !owner->bytes()) return;
  std::lock_guard lock(registry_mutex);
  registry[owner->begin()] = {owner, owner.get()};
}
void unregister_external_storage(std::uintptr_t begin, const ExternalStorage* owner) noexcept {
  if (!current_process() || !begin) return;
  try {
    std::lock_guard lock(registry_mutex);
    const auto it = registry.find(begin);
    if (it != registry.end() && it->second.identity == owner) registry.erase(it);
  } catch (...) { /* A stale weak entry never authorizes native access. */ }
}

Expected<std::shared_ptr<ExternalStorage>> find_external_storage(void* data, std::size_t bytes) {
  if (!current_process()) return Status(StatusCode::kUnavailable, "shared CUDA storage requires a fresh process after fork");
  if (!data) return std::shared_ptr<ExternalStorage>{};
  const auto pointer = reinterpret_cast<std::uintptr_t>(data);
  std::weak_ptr<ExternalStorage> candidate;
  std::weak_ptr<ExternalStorage> next;
  {
    std::lock_guard lock(registry_mutex);
    auto it = registry.upper_bound(pointer);
    if (it != registry.end()) next = it->second.weak;
    if (it != registry.begin()) { --it; candidate = it->second.weak; }
  }
  // Never drop a last strong owner under the registry mutex: its final release
  // unregisters the range and may need to synchronize external native work.
  auto owner = candidate.lock();
  if (owner && pointer >= owner->begin() &&
      (pointer - owner->begin() < owner->mapped_bytes() ||
       (!bytes && pointer - owner->begin() == owner->mapped_bytes()))) {
    // Retain provenance even for logical overruns/padding: borrow() must reject
    // them through validate_range(), never accept them as ordinary CUDA memory.
    return owner;
  }
  auto successor = next.lock();
  if (successor && pointer < successor->begin() && bytes > successor->begin() - pointer)
    return Status(StatusCode::kInvalidArgument, "CUDA view overlaps registered shared storage");
  return std::shared_ptr<ExternalStorage>{};
}

ExternalLease::ExternalLease(std::vector<std::shared_ptr<ExternalStorage>> owners)
    : owners_(std::move(owners)), lease_(accesses(owners_)) {
  std::sort(owners_.begin(), owners_.end(), [](const auto& a, const auto& b) {
    return std::less<ExternalStorage*>{}(a.get(), b.get());
  });
  owners_.erase(std::unique(owners_.begin(), owners_.end()), owners_.end());
}
ExternalLease::~ExternalLease() {
  if (!finished_ && status().ok()) {
    try { (void)finish(); }
    catch (...) { for (const auto& owner : owners_) owner->poison(); }
  }
}
Status ExternalLease::finish() {
  if (finished_) return completion_;
  if (!status().ok()) return status();
  try {
    for (const auto& owner : owners_) {
      auto completed = owner->complete_cuda();
      if (!completed.ok()) { owner->poison(); if (completion_.ok()) completion_ = std::move(completed); }
    }
  } catch (...) {
    for (const auto& owner : owners_) owner->poison();
    finished_ = true;
    throw;
  }
  finished_ = true;
  return completion_;
}

std::vector<std::shared_ptr<ExternalStorage>> external_owners(const BackendExecution& execution) {
  std::vector<std::shared_ptr<ExternalStorage>> result;
  auto append = [&](const Tensor& tensor) {
    if (auto buffer = std::dynamic_pointer_cast<CudaBuffer>(tensor.buffer))
      if (buffer->external_storage()) result.push_back(buffer->external_storage());
  };
  for (const auto& tensor : execution.inputs) append(tensor);
  for (const auto& tensor : execution.outputs) append(tensor);
  for (const auto& argument : execution.kernel_arguments)
    if (argument.kind == KernelArgumentKind::kTensor && argument.tensor) append(*argument.tensor);
  return result;
}

}  // namespace tensorcx::cuda
