#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tensorcx/core/expected.h"

namespace tensorcx::metal {

class MetalTensor;

struct KernelArgument {
  enum class Kind {
    kTensor,
    kUInt32,
  };

  Kind kind;
  const MetalTensor* tensor;
  std::uint32_t uint32_value;
};

Expected<std::string> validate_library_function(
    const std::vector<std::uint8_t>& metallib,
    const std::string& function_name);

Expected<std::string> launch_library_function(
    const std::vector<std::uint8_t>& metallib,
    const std::string& function_name,
    const std::vector<KernelArgument>& arguments,
    std::uint32_t thread_count,
    std::uint32_t threads_per_threadgroup);

}  // namespace tensorcx::metal
