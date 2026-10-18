#pragma once

#include "tensorcx/core/backend.h"

namespace tensorcx::vulkan {

// Called after validating the common primitive execution contract.
Status execute_predicate(const BackendExecution& execution);

}  // namespace tensorcx::vulkan
