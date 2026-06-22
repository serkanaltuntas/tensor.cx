#pragma once

#include "cortex/backends/metal/metal_tensor.h"
#include "cortex/core/expected.h"

namespace cortex::metal {

Expected<MetalTensor> matmul_mpsgraph(const MetalTensor& lhs, const MetalTensor& rhs);

}  // namespace cortex::metal
