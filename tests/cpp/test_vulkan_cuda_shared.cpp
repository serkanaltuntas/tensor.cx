#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "tensorcx/backends/cuda/cuda_buffer.h"
#include "tensorcx/backends/cuda/cuda_external_storage.h"
#include "tensorcx/backends/vulkan/vulkan_backend.h"
#include "tensorcx/backends/vulkan/vulkan_context.h"
#include "tensorcx/backends/vulkan/vulkan_cuda_shared.h"

using namespace tensorcx;
namespace vk = tensorcx::vulkan;
namespace {
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void check(const Status& status) { require(status.ok(), status.message().c_str()); }
void equal(const cpu::CpuTensor& actual, const cpu::CpuTensor& expected) {
  require(actual.shape() == expected.shape() && actual.dtype() == expected.dtype(), "shared metadata changed");
  const auto bytes = static_cast<std::size_t>(actual.size()) * dtype_size(actual.dtype());
  require(!bytes || !std::memcmp(actual.data(), expected.data(), bytes), "shared bytes differ");
}
void cpu_values(cpu::CpuTensor& tensor, int turn) {
  if (tensor.dtype() == DType::kFloat32) {
    std::size_t i = 0;
    for (auto& value : tensor.mutable_float_data()) value = static_cast<float>(++i) * 0.25F + turn;
  } else if (tensor.dtype() == DType::kInt32) {
    std::int32_t i = 0;
    for (auto& value : tensor.mutable_int32_data()) value = ++i * 17 - turn;
  } else {
    std::size_t i = 0;
    for (auto& value : tensor.mutable_bool_data()) value = static_cast<std::uint8_t>((i++ + turn) % 2);
  }
}
void contract(vk::SharedOrigin origin) {
  auto context = vk::get_context(0).value();
  const auto before = context->statistics();
  const auto allocation_count = context->allocation_count.load();
  for (auto dtype : {DType::kFloat32, DType::kInt32, DType::kBool}) {
    for (const Shape shape : {Shape{}, Shape{0}, Shape{1}, Shape{3}, Shape{5}, Shape{257}, Shape{8, 128}}) {
      cpu::CpuTensor expected(dtype, shape); cpu_values(expected, 0);
      auto created = vk::shared_tensors(shape, dtype, 0, origin, &expected);
      if (!created) check(created.status());
      auto pair = created.move_value();
      auto vk_core = vk::to_core_tensor(pair.vulkan);
      auto cuda_core = cuda::to_core_tensor(pair.cuda);
      if (pair.vulkan.nbytes()) {
        require(vk_core.buffer->storage_relation(*cuda_core.buffer) == StorageRelation::kSameRange,
                "cross-backend aliases lost storage identity");
        require(cuda_core.buffer->storage_relation(*vk_core.buffer) == StorageRelation::kSameRange,
                "cross-backend storage relation is asymmetric");
      }
      equal(vk::to_cpu(pair.vulkan).value(), expected);
      equal(cuda::to_cpu(pair.cuda).value(), expected);
      for (int turn = 1; turn <= 12; ++turn) {
        cpu_values(expected, turn);
        const auto bytes = pair.vulkan.nbytes();
        if (turn % 2) {
          // A foreign CUDA writer must complete before an explicit handoff.
          cuda::ExternalLease lease({pair.cuda.buffer()->external_storage()}); check(lease.status());
          cuda::ContextScope scope; check(scope.status());
          if (bytes) check(cuda::runtime_status(cudaMemcpy(pair.cuda.buffer()->data(), expected.data(),
              bytes, cudaMemcpyHostToDevice), "shared contract CUDA write"));
          check(lease.finish());
        } else context->upload(pair.vulkan.buffer()->allocation(), expected.data(), bytes);
        equal(vk::to_cpu(pair.vulkan).value(), expected);
        equal(cuda::to_cpu(pair.cuda).value(), expected);
        if (dtype == DType::kFloat32) {
          for (bool cuda_backend : {false, true}) {
            const std::array<Tensor, 1> inputs{cuda_backend ? cuda_core : vk_core};
            std::array<Tensor, 1> outputs{};
            BackendExecution execution; execution.op.kind = OpKind::kAddScalar;
            execution.op.scalar_value = 2; execution.inputs = inputs; execution.outputs = outputs;
            if (cuda_backend) { cuda::CudaBackend backend; check(backend.execute(execution)); }
            else { vk::VulkanBackend backend; check(backend.execute(execution)); }
            auto result = cuda_backend ? cuda::to_cpu(cuda::from_core_tensor(outputs[0]).value()).move_value()
                                       : vk::to_cpu(vk::from_core_tensor(outputs[0]).value()).move_value();
            for (std::size_t i = 0; i < result.float_data().size(); ++i)
              require(result.float_data()[i] == expected.float_data()[i] + 2, "shared primitive result differs");
          }
        }
      }
      // Keep only the CUDA alias alive, then only the Vulkan alias in a second allocation.
      vk_core = {}; cuda_core = {};
      pair.vulkan = vk::empty({0}, dtype).move_value();
      equal(cuda::to_cpu(pair.cuda).value(), expected);
      auto second = vk::shared_tensors(shape, dtype, 0, origin, &expected).move_value();
      second.cuda = cuda::empty({0}, dtype).move_value();
      equal(vk::to_cpu(second.vulkan).value(), expected);
    }
  }
  const auto after = context->statistics();
  require(after.physical_blocks == before.physical_blocks && after.physical_bytes == before.physical_bytes,
          "shared physical storage leaked after both aliases died");
  require(context->allocation_count.load() == allocation_count, "shared allocation count leaked");
}
}  // namespace

int main() {
  try {
    for (auto origin : {vk::SharedOrigin::kVulkan, vk::SharedOrigin::kCuda}) {
      const auto support = vk::shared_support(0, origin);
      if (!support.ok()) { std::cerr << support.message() << '\n'; return 77; }
      contract(origin);
      require(vk::validation_errors() == 0, "shared storage triggered Vulkan validation errors");
    }
    std::cout << "shared Vulkan/CUDA storage: both origins, dtype/empty/tail, handoff, primitives and lifetime passed\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
