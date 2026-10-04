#pragma once

#include "tensorcx/backends/metal/metal_tensor.h"
#include "tensorcx/core/dtype.h"
#include "tensorcx/core/expected.h"
#include "tensorcx/core/operation.h"
#include "tensorcx/core/shape.h"

namespace tensorcx::metal {

Expected<MetalTensor> execute_concat(const OpDesc& op, const std::vector<MetalTensor>& inputs);
Expected<MetalTensor> execute_unary(const OpDesc& op, const MetalTensor& input);
Expected<MetalTensor> execute_binary(const OpDesc& op, const MetalTensor& lhs, const MetalTensor& rhs);
Expected<MetalTensor> fill(const OpDesc& op, Shape shape, DType dtype, double value);
Expected<MetalTensor> reduce(const OpDesc& op, const MetalTensor& input);
Expected<MetalTensor> matmul_custom(const MetalTensor& lhs, const MetalTensor& rhs);

}  // namespace tensorcx::metal
