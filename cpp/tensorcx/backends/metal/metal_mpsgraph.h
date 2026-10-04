#pragma once

#include "tensorcx/backends/metal/metal_tensor.h"
#include "tensorcx/core/expected.h"

namespace tensorcx::metal {

Expected<MetalTensor> matmul_mpsgraph(const MetalTensor& lhs, const MetalTensor& rhs);

}  // namespace tensorcx::metal
