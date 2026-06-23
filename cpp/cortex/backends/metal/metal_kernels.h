#pragma once

#include "cortex/backends/metal/metal_tensor.h"
#include "cortex/core/dtype.h"
#include "cortex/core/expected.h"
#include "cortex/core/operation.h"
#include "cortex/core/shape.h"

namespace cortex::metal {

Expected<MetalTensor> execute_unary(const OpDesc& op, const MetalTensor& input);
Expected<MetalTensor> execute_binary(const OpDesc& op, const MetalTensor& lhs, const MetalTensor& rhs);
Expected<MetalTensor> fill(const OpDesc& op, Shape shape, DType dtype, double value);
Expected<MetalTensor> reduce(const OpDesc& op, const MetalTensor& input);
Expected<MetalTensor> matmul_custom(const MetalTensor& lhs, const MetalTensor& rhs);

}  // namespace cortex::metal
