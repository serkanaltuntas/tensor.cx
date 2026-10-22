#pragma once

#include "tensorcx/core/backend.h"

namespace tensorcx::vulkan {
bool is_nn_operation(OpKind kind);
Status execute_nn(Backend& backend, const BackendExecution& execution);
}  // namespace tensorcx::vulkan
