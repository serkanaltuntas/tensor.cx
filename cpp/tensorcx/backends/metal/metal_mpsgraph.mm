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

id<MTLBuffer> as_objc_buffer(MTL::Buffer* buffer) {
  return (__bridge id<MTLBuffer>)reinterpret_cast<void*>(buffer);
}

id<MTLCommandQueue> as_objc_command_queue(MTL::CommandQueue& command_queue) {
  return (__bridge id<MTLCommandQueue>)reinterpret_cast<void*>(&command_queue);
}

MPSShape* graph_shape(const Shape& shape) {
  NSMutableArray<NSNumber*>* values = [NSMutableArray arrayWithCapacity:shape.size()];
  for (Dim dim : shape) [values addObject:@(dim)];
  return values;
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
  if (lhs.dtype() != DType::kFloat32 || rhs.dtype() != DType::kFloat32)
    return Status(StatusCode::kInvalidArgument, "MPSGraph matmul only supports float32 tensors");
  auto planned = checked_metal_matmul_plan(lhs, rhs);
  if (!planned) return planned.status();
  const auto plan = planned.move_value();
  if (plan.k == 0) return matmul_custom(lhs, rhs);
  const auto count = numel(plan.output_shape);
  auto allocated = MetalBuffer::create(DType::kFloat32, static_cast<std::size_t>(count));
  if (!allocated) return allocated.status();
  auto output_buffer = allocated.move_value();
  MetalTensor output(DType::kFloat32, plan.output_shape, output_buffer);
  if (!count) return output;
  // Align batch axes, omit common singleton axes, and promote vector inputs.
  // These are metadata-only shapes over the original contiguous buffers.
  Shape left, right, result;
  for (std::size_t axis = 0; axis < plan.batch_shape.size(); ++axis) {
    if (plan.batch_shape[axis] == 1) continue;
    left.push_back(plan.lhs_batch_strides[axis] ? plan.batch_shape[axis] : 1);
    right.push_back(plan.rhs_batch_strides[axis] ? plan.batch_shape[axis] : 1);
    result.push_back(plan.batch_shape[axis]);
  }
  left.insert(left.end(), {plan.m, plan.k});
  right.insert(right.end(), {plan.k, plan.n});
  result.insert(result.end(), {plan.m, plan.n});
  if (result.size() > 16)
    return Status(StatusCode::kInvalidArgument, "MPSGraph matmul supports at most 16 non-singleton-batch plus matrix dimensions; use custom");

  auto& context = default_context();
  if (!context.ready()) {
    return context.status();
  }

  @try {
    @autoreleasepool {
      MPSGraph* graph = [MPSGraph new];
      MPSShape* lhs_shape = graph_shape(left);
      MPSShape* rhs_shape = graph_shape(right);
      MPSShape* output_shape = graph_shape(result);

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
