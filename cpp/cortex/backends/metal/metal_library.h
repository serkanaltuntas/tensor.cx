#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "cortex/core/expected.h"

namespace cortex::metal {

Expected<std::string> validate_library_function(
    const std::vector<std::uint8_t>& metallib,
    const std::string& function_name);

}  // namespace cortex::metal
