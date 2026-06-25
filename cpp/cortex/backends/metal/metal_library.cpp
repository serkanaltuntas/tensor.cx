#include "cortex/backends/metal/metal_library.h"

#include <dispatch/dispatch.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "cortex/backends/metal/metal_context.h"
#include "cortex/core/status.h"

namespace cortex::metal {
namespace {

class DispatchData final {
 public:
  explicit DispatchData(dispatch_data_t data) : data_(data) {}
  DispatchData(const DispatchData&) = delete;
  DispatchData& operator=(const DispatchData&) = delete;

  ~DispatchData() {
    if (data_) {
      dispatch_release(data_);
    }
  }

  dispatch_data_t get() const { return data_; }

 private:
  dispatch_data_t data_{nullptr};
};

std::string error_message(const char* prefix, NS::Error* error) {
  if (error && error->localizedDescription()) {
    return std::string(prefix) + ": " + error->localizedDescription()->utf8String();
  }
  return prefix;
}

}  // namespace

Expected<std::string> validate_library_function(
    const std::vector<std::uint8_t>& metallib,
    const std::string& function_name) {
  if (metallib.empty()) {
    return Status(StatusCode::kInvalidArgument, "Metal library data is empty");
  }
  if (function_name.empty()) {
    return Status(StatusCode::kInvalidArgument, "Metal function name is empty");
  }
  if (function_name.find('\0') != std::string::npos) {
    return Status(StatusCode::kInvalidArgument, "Metal function name cannot contain null bytes");
  }

  NS::SharedPtr<NS::AutoreleasePool> pool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  auto& context = default_context();
  if (!context.ready()) {
    return context.status();
  }

  DispatchData library_data(dispatch_data_create(
      metallib.data(),
      metallib.size(),
      nullptr,
      DISPATCH_DATA_DESTRUCTOR_DEFAULT));
  if (!library_data.get()) {
    return Status(StatusCode::kInternal, "failed to create Metal library data");
  }

  NS::Error* error = nullptr;
  NS::SharedPtr<MTL::Library> library =
      NS::TransferPtr(context.device().newLibrary(library_data.get(), &error));
  if (!library) {
    return Status(
        StatusCode::kInvalidArgument,
        error_message("failed to load Metal library", error));
  }

  NS::SharedPtr<NS::String> name =
      NS::TransferPtr(NS::String::alloc()->init(function_name.c_str(), NS::UTF8StringEncoding));
  if (!name) {
    return Status(StatusCode::kInternal, "failed to allocate Metal function name");
  }

  NS::SharedPtr<MTL::Function> function =
      NS::TransferPtr(library->newFunction(name.get()));
  if (!function) {
    return Status(
        StatusCode::kInvalidArgument,
        "Metal library does not contain function: " + function_name);
  }

  return function_name;
}

}  // namespace cortex::metal
