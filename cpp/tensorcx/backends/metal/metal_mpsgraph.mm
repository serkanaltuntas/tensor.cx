#include "tensorcx/backends/metal/metal_mpsgraph.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

#include "tensorcx/backends/metal/metal_buffer.h"
#include "tensorcx/backends/metal/metal_context.h"
#include "tensorcx/backends/metal/metal_kernels.h"
#include "tensorcx/core/dtype.h"
#include "tensorcx/core/status.h"

namespace tensorcx::metal {
namespace {

struct MatmulDims {
  std::uint32_t m;
  std::uint32_t k;
  std::uint32_t n;
  std::int64_t output_elements;
};

Expected<std::uint32_t> checked_dim(Dim dim, const char* name) {
  if (dim < 0) {
    return Status(StatusCode::kInvalidArgument, "shape dimensions must be non-negative");
  }
  if (dim > std::numeric_limits<std::uint32_t>::max()) {
    return Status(
        StatusCode::kInvalidArgument,
        std::string("Metal matmul dimension exceeds 2^32 - 1: ") + name);
  }
  return static_cast<std::uint32_t>(dim);
}

// NOTE: This intentionally mirrors metal_kernels.cpp's checked_matmul_dims but
// omits its per-operand (M*K, K*N) 2^32 guards. Those exist only because the
// custom MSL kernel indexes operands with 32-bit uint arithmetic; MPSGraph uses
// 64-bit MPSShape indexing, so the guards do not apply here. The k==0 path below
// delegates to matmul_custom, which re-validates with the full guard set, so no
// unguarded shape can reach the custom kernel through this function.
Expected<MatmulDims> checked_matmul_dims(const MetalTensor& lhs, const MetalTensor& rhs) {
  if (lhs.dtype() != DType::kFloat32 || rhs.dtype() != DType::kFloat32) {
    return Status(StatusCode::kInvalidArgument, "MPSGraph matmul only supports float32 tensors");
  }
  if (lhs.shape().size() != 2 || rhs.shape().size() != 2) {
    return Status(StatusCode::kInvalidArgument, "matmul requires rank-2 tensors");
  }
  if (lhs.shape()[1] != rhs.shape()[0]) {
    return Status(StatusCode::kInvalidArgument, "matmul shape mismatch");
  }

  auto m_result = checked_dim(lhs.shape()[0], "M");
  if (!m_result) {
    return m_result.status();
  }
  auto k_result = checked_dim(lhs.shape()[1], "K");
  if (!k_result) {
    return k_result.status();
  }
  auto n_result = checked_dim(rhs.shape()[1], "N");
  if (!n_result) {
    return n_result.status();
  }

  const auto m = m_result.move_value();
  const auto k = k_result.move_value();
  const auto n = n_result.move_value();
  if (m != 0 && n > std::numeric_limits<std::uint32_t>::max() / m) {
    return Status(
        StatusCode::kInvalidArgument,
        "Metal kernels currently support at most 2^32 - 1 elements");
  }
  const std::int64_t output_elements =
      static_cast<std::int64_t>(m) * static_cast<std::int64_t>(n);
  if (output_elements > std::numeric_limits<std::uint32_t>::max()) {
    return Status(
        StatusCode::kInvalidArgument,
        "Metal kernels currently support at most 2^32 - 1 elements");
  }
  return MatmulDims{m, k, n, output_elements};
}

id<MTLBuffer> as_objc_buffer(MTL::Buffer* buffer) {
  return (__bridge id<MTLBuffer>)reinterpret_cast<void*>(buffer);
}

id<MTLCommandQueue> as_objc_command_queue(MTL::CommandQueue& command_queue) {
  return (__bridge id<MTLCommandQueue>)reinterpret_cast<void*>(&command_queue);
}

MPSShape* shape2(std::uint32_t rows, std::uint32_t cols) {
  return @[ @(rows), @(cols) ];
}

Status exception_status(NSException* exception) {
  std::ostringstream message;
  message << "MPSGraph matmul failed";
  if (exception.name) {
    message << ": " << exception.name.UTF8String;
  }
  if (exception.reason) {
    message << ": " << exception.reason.UTF8String;
  }
  return Status(StatusCode::kInternal, message.str());
}

}  // namespace

Expected<MetalTensor> matmul_mpsgraph(const MetalTensor& lhs, const MetalTensor& rhs) {
  auto dims_result = checked_matmul_dims(lhs, rhs);
  if (!dims_result) {
    return dims_result.status();
  }
  const auto dims = dims_result.move_value();
  if (dims.k == 0) {
    return matmul_custom(lhs, rhs);
  }

  auto output_buffer_result =
      MetalBuffer::create(DType::kFloat32, static_cast<std::size_t>(dims.output_elements));
  if (!output_buffer_result) {
    return output_buffer_result.status();
  }
  auto output_buffer = output_buffer_result.move_value();
  MetalTensor output(DType::kFloat32, Shape{dims.m, dims.n}, output_buffer);
  if (dims.output_elements == 0) {
    return output;
  }

  auto& context = default_context();
  if (!context.ready()) {
    return context.status();
  }

  @try {
    @autoreleasepool {
      MPSGraph* graph = [MPSGraph new];
      MPSShape* lhs_shape = shape2(dims.m, dims.k);
      MPSShape* rhs_shape = shape2(dims.k, dims.n);
      MPSShape* output_shape = shape2(dims.m, dims.n);

      MPSGraphTensor* lhs_tensor =
          [graph placeholderWithShape:lhs_shape dataType:MPSDataTypeFloat32 name:@"lhs"];
      MPSGraphTensor* rhs_tensor =
          [graph placeholderWithShape:rhs_shape dataType:MPSDataTypeFloat32 name:@"rhs"];
      MPSGraphTensor* output_tensor =
          [graph matrixMultiplicationWithPrimaryTensor:lhs_tensor
                                       secondaryTensor:rhs_tensor
                                                  name:@"matmul"];

      MPSGraphTensorData* lhs_data =
          [[MPSGraphTensorData alloc] initWithMTLBuffer:as_objc_buffer(lhs.buffer()->native())
                                                  shape:lhs_shape
                                               dataType:MPSDataTypeFloat32];
      MPSGraphTensorData* rhs_data =
          [[MPSGraphTensorData alloc] initWithMTLBuffer:as_objc_buffer(rhs.buffer()->native())
                                                  shape:rhs_shape
                                               dataType:MPSDataTypeFloat32];
      MPSGraphTensorData* output_data =
          [[MPSGraphTensorData alloc] initWithMTLBuffer:as_objc_buffer(output_buffer->native())
                                                  shape:output_shape
                                               dataType:MPSDataTypeFloat32];

      if (!lhs_data || !rhs_data || !output_data) {
        return Status(StatusCode::kInternal, "failed to create MPSGraph tensor data");
      }

      MPSGraphTensorDataDictionary* feeds = @{ lhs_tensor: lhs_data, rhs_tensor: rhs_data };
      MPSGraphTensorDataDictionary* results = @{ output_tensor: output_data };
      [graph runWithMTLCommandQueue:as_objc_command_queue(context.command_queue())
                               feeds:feeds
                    targetOperations:nil
                   resultsDictionary:results];
    }
  } @catch (NSException* exception) {
    return exception_status(exception);
  }

  return output;
}

}  // namespace tensorcx::metal
