#pragma once

#include "cortex/backends/cpu/cpu_tensor.h"
#include "cortex/core/operation.h"

namespace cortex::cpu {

CpuTensor empty(Shape shape, DType dtype);
CpuTensor fill(Shape shape, DType dtype, double value);
CpuTensor execute_binary(const OpDesc& op, const CpuTensor& lhs, const CpuTensor& rhs);
CpuTensor reduce(const OpDesc& op, const CpuTensor& input);
CpuTensor matmul(const CpuTensor& lhs, const CpuTensor& rhs);

}  // namespace cortex::cpu
