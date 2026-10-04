#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <cmath>

#include "tensorcx/backends/cuda/cuda_backend.h"
#include "tensorcx/backends/cpu/cpu_backend.h"

using namespace tensorcx;
namespace {
void require(bool result, const char* message) {
  if (!result) throw std::runtime_error(message);
}
void invalid(const Status& status) {
  require(status.code() == StatusCode::kInvalidArgument, status.message().c_str());
}
Tensor descriptor(Shape shape = {3}) {
  const auto strides = contiguous_strides(shape);
  return {DType::kFloat32, std::move(shape), strides, {"cuda", 0}, nullptr, 0};
}
}  // namespace

int main() {
  try {
    cuda::CudaBackend backend;
    std::array<Tensor, 1> outputs{descriptor()};
    BackendExecution fill{BackendOpClass::kPrimitive, OpDesc{OpKind::kFill}, {}, outputs};
    fill.op.scalar_value = 2.5;
    auto malformed = fill;
    malformed.outputs = {};
    invalid(backend.execute(malformed));
    malformed = fill;
    malformed.launch = LaunchConfig{1, 1, 1, 1, 1, 1};
    invalid(backend.execute(malformed));
    for (int scenario = 0; scenario < 6; ++scenario) {
      outputs[0] = descriptor();
      switch (scenario) {
        case 0: outputs[0].device.type = "cpu"; break;
        case 1: outputs[0].device.index = 1; break;
        case 2: outputs[0].offset = 1; break;
        case 3: outputs[0].strides = {2}; break;
        case 4: outputs[0].dtype = DType::kInt32; break;
        case 5: outputs[0].buffer = cpu::CpuTensor(DType::kFloat32, {3}).buffer(); break;
      }
      invalid(backend.execute(fill));
    }
    malformed = fill;
    malformed.op_class = BackendOpClass::kKernel;
    invalid(backend.execute(malformed));
    outputs[0] = descriptor();
    if (!cuda::available()) {
      if (std::getenv("TENSORCX_REQUIRE_CUDA")) {
        throw std::runtime_error("required CUDA device unavailable");
      }
      std::cout << "CUDA device unavailable; metadata validation passed\n";
      return 77;
    }
    require(backend.execute(fill).ok(), "fill failed");
    auto filled = cuda::from_core_tensor(outputs[0]);
    require(bool(filled), "fill metadata invalid");
    auto host = cuda::to_cpu(filled.value());
    require(bool(host) && host.value().float_data() == std::vector<float>({2.5, 2.5, 2.5}),
            "fill value mismatch");
    const Tensor original = outputs[0];
    std::array<Tensor, 2> inputs{original, original};
    BackendExecution binary{BackendOpClass::kPrimitive, OpDesc{OpKind::kAdd}, inputs, outputs};
    for (const auto kind : {OpKind::kAdd, OpKind::kMultiply}) {
      binary.op.kind = kind;
      require(backend.execute(binary).ok(), "binary failed");
      auto result = cuda::to_cpu(cuda::from_core_tensor(outputs[0]).value());
      const float expected = kind == OpKind::kAdd ? 5.0f : 6.25f;
      require(bool(result) && result.value().float_data() == std::vector<float>(3, expected),
              "binary value mismatch");
    }
    {
      cpu::CpuTensor source({2,2},std::vector<float>{1,2,3,4});
      auto device=cuda::from_cpu(source);
      require(bool(device),"primitive input copy failed");
      std::array<Tensor,2> gpu_inputs{cuda::to_core_tensor(device.value()),cuda::to_core_tensor(device.value())};
      std::array<Tensor,2> cpu_inputs{cpu::to_core_tensor(source),cpu::to_core_tensor(source)};
      std::array<Tensor,1> gpu_outputs{},cpu_outputs{};
      cpu::CpuBackend reference;
      for(const auto kind:{OpKind::kMatmul,OpKind::kSum,OpKind::kMax,OpKind::kMean,
          OpKind::kExp,OpKind::kGelu,OpKind::kSilu,OpKind::kSoftmax,OpKind::kRmsNorm,OpKind::kLayerNorm}) {
        OpDesc op{kind};op.axis=1;
        const std::size_t count=kind==OpKind::kMatmul?2:1;
        BackendExecution gpu{BackendOpClass::kPrimitive,op,std::span(gpu_inputs).first(count),gpu_outputs};
        BackendExecution cpu{BackendOpClass::kPrimitive,op,std::span(cpu_inputs).first(count),cpu_outputs};
        require(reference.execute(cpu).ok(),"CPU primitive failed");
        require(backend.execute(gpu).ok(),"CUDA primitive failed");
        auto actual=cuda::to_cpu(cuda::from_core_tensor(gpu_outputs[0]).value()).value();
        auto expected=cpu::from_core_tensor(cpu_outputs[0]);
        require(actual.shape()==expected.shape(),"primitive shape mismatch");
        for(std::size_t i=0;i<actual.float_data().size();++i)
          require(std::abs(actual.float_data()[i]-expected.float_data()[i])<1e-4F,"primitive value mismatch");
        auto previous=gpu_outputs[0].buffer;
        gpu_inputs[0].offset=1;
        invalid(backend.execute(gpu));
        require(gpu_outputs[0].buffer==previous,"primitive failure published output");
        gpu_inputs[0].offset=0;
        if(kind==OpKind::kRmsNorm || kind==OpKind::kLayerNorm) {
          gpu.op.epsilon=-1;
          invalid(backend.execute(gpu));
          require(gpu_outputs[0].buffer==previous,"epsilon failure published output");
        }
      }
    }
    // Long contiguous rows exercise the cooperative path under device
    // memcheck/racecheck as well as ordinary native acceptance.
    for (const std::int64_t width : {256, 257, 4097}) {
      std::vector<float> values(3 * width);
      for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = i < static_cast<std::size_t>(2 * width) ? static_cast<float>(i % 17) / 17 : 0.3F;
      cpu::CpuTensor source({3, width}, values);
      auto device = cuda::from_cpu(source).value();
      std::array<Tensor, 1> gpu_inputs{cuda::to_core_tensor(device)}, cpu_inputs{cpu::to_core_tensor(source)};
      std::array<Tensor, 1> gpu_outputs{}, cpu_outputs{};
      cpu::CpuBackend reference;
      for (auto kind : {OpKind::kSum, OpKind::kMax, OpKind::kMean,
                        OpKind::kSoftmax, OpKind::kRmsNorm, OpKind::kLayerNorm}) {
        OpDesc op{kind}; op.axis = 1;
        BackendExecution gpu{BackendOpClass::kPrimitive, op, gpu_inputs, gpu_outputs};
        BackendExecution cpu{BackendOpClass::kPrimitive, op, cpu_inputs, cpu_outputs};
        require(reference.execute(cpu).ok(), "staged reference failed");
        require(backend.execute(gpu).ok(), "staged CUDA failed");
        auto actual = cuda::to_cpu(cuda::from_core_tensor(gpu_outputs[0]).value()).value();
        auto expected = cpu::from_core_tensor(cpu_outputs[0]);
        require(actual.shape() == expected.shape(), "staged shape mismatch");
        for (std::size_t i = 0; i < actual.float_data().size(); ++i)
          require(std::abs(actual.float_data()[i] - expected.float_data()[i]) < 1e-4F, "staged value mismatch");
      }
    }
    const Tensor previous_output = outputs[0];
    for (int scenario = 0; scenario < 9; ++scenario) {
      inputs[0] = original;
      switch (scenario) {
        case 0: inputs[0].device.type = "cpu"; break;
        case 1: inputs[0].device.index = 1; break;
        case 2: inputs[0].offset = 1; break;
        case 3: inputs[0].strides = {2}; break;
        case 4: inputs[0].buffer.reset(); break;
        case 5: inputs[0].buffer = cpu::CpuTensor(DType::kFloat32, {3}).buffer(); break;
        case 6: inputs[0].shape = {4}; break;
        case 7: inputs[0].dtype = DType::kInt32; break;
        case 8: inputs[0].shape = {-1}; break;
      }
      invalid(backend.execute(binary));
      require(outputs[0].buffer == previous_output.buffer, "failure changed output");
    }
    inputs[0] = original;
    auto unsupported = binary;
    unsupported.op.kind = OpKind::kMatmul;
    invalid(backend.execute(unsupported));
    // A kernel contract with an unknown artifact must still be rejected.
    KernelArgument arg{KernelArgumentKind::kTensor, &original, 0};
    BackendExecution kernel{BackendOpClass::kKernel, {}, {}, outputs,
      LaunchConfig{3, 1, 1, 32, 1, 1},
      CompilationTarget{KernelArtifactKind::kBinary, "unused", "entry"},
      std::span<const KernelArgument>(&arg, 1)};
    invalid(backend.execute(kernel));
    outputs[0] = descriptor({std::numeric_limits<std::int64_t>::max() / 2 + 1});
    invalid(backend.execute(fill));
    // Empty outputs retain ownership but perform no invalid zero-block launch.
    outputs[0] = descriptor({2, 0});
    require(backend.execute(fill).ok(), "empty fill failed");
    require(outputs[0].buffer->nbytes() == 0, "empty allocation size mismatch");
    std::cout << "CUDA backend contract passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CUDA backend contract failed: " << error.what() << '\n';
    return 1;
  }
}
