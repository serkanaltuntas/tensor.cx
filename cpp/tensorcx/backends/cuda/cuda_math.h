#pragma once
#include "tensorcx/core/backend.h"
#include <cuda_runtime_api.h>
namespace tensorcx::cuda {
Status execute_math(const BackendExecution& execution);
cudaError_t launch_math(const void* a,const void* b,const void* c,void* output,void* indices,
                        const Dim* metadata,std::size_t groups);
}
