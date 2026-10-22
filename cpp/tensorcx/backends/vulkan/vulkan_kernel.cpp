#include "tensorcx/backends/vulkan/vulkan_kernel.h"
#include "tensorcx/backends/vulkan/vulkan_resources.h"
#include "tensorcx/core/profiling.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <map>
#include <set>
#include <unordered_map>
#include <unistd.h>

namespace tensorcx::vulkan {
namespace {
constexpr auto entry_name = "tensorcx_kernel_v1";
const auto registry_process = ::getpid();
std::mutex registry_mutex;
std::unordered_map<std::string, std::weak_ptr<VulkanKernelModule>> registry;
std::atomic<std::uint64_t> next_id{1};
Status invalid(std::string text) { return {StatusCode::kInvalidArgument, std::move(text)}; }
void require(bool value, const char* text) { if (!value) throw std::invalid_argument(text); }
void check(VkResult code, const char* operation) { if (code != VK_SUCCESS) throw VulkanError(code, operation); }
bool valid_signature(const KernelSignature& s) {
  const auto tensors = std::count(s.kinds.begin(), s.kinds.end(), 't');
  return s.kinds.size() >= 2 && s.kinds.size() <= 4 && tensors >= 1 && tensors <= 3 &&
      std::count(s.kinds.begin(), s.kinds.end(), 'u') == 1 && tensors + 1 == static_cast<int>(s.kinds.size()) &&
      s.output_index < s.kinds.size() && s.guard_index < s.kinds.size() &&
      s.kinds[s.output_index] == 't' && s.kinds[s.guard_index] == 'u';
}
Expected<std::shared_ptr<VulkanKernelModule>> resolve(const std::string& id) {
  if (::getpid() != registry_process) return invalid("Vulkan kernel modules are process-local");
  std::lock_guard lock(registry_mutex);
  auto found = registry.find(id);
  if (found != registry.end()) if (auto module = found->second.lock()) return module;
  return invalid("unknown or expired Vulkan kernel artifact");
}

// This validates the declared resource/entry ABI, not arbitrary shader safety.
// The producer proves guarded canonical accesses; SPIR-V Tools validates the
// complete shader. Like other native artifacts, this is not an untrusted-code API.
std::size_t validate_spirv(const std::vector<std::uint32_t>& w, std::size_t tensors) {
  require(w.size() >= 5 && w[0] == 0x07230203 && w[1] == 0x00010300 && w[3] &&
          w[3] <= 1000000 && !w[4], "unsupported Vulkan SPIR-V header");
  std::map<std::uint32_t, std::vector<std::uint32_t>> types, variables;
  std::map<std::uint32_t, std::uint32_t> sets, bindings, strides, builtins;
  std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> offsets;
  std::set<std::uint32_t> interfaces;
  std::uint32_t entry = 0, function_type = 0, return_type = 0;
  std::size_t geometry = 0, entries = 0, modes = 0, models = 0, capabilities = 0;
  for (std::size_t i = 5; i < w.size();) {
    const auto n = w[i] >> 16, op = w[i] & 65535;
    require(n && n <= w.size() - i, "invalid Vulkan SPIR-V instruction size");
    const auto* p = w.data() + i;
    if (op == 17) { require(n == 2 && p[1] == 1, "Vulkan kernel requires core Shader capability only"); ++capabilities; }
    if (op == 14) { require(n == 3 && p[1] == 0 && p[2] == 1, "Vulkan kernel requires Logical GLSL450 memory"); ++models; }
    if (op == 15) {
      require(n >= 4 && p[1] == 5 && !entries++, "Vulkan kernel requires one compute entry");
      entry = p[2];
      const auto* name = reinterpret_cast<const char*>(p + 3);
      const auto size = static_cast<std::size_t>(n - 3) * 4;
      const auto* end = static_cast<const char*>(std::memchr(name, 0, size));
      require(end && std::string_view(name, end - name) == entry_name, "Vulkan kernel entry mismatch");
      const auto words = (static_cast<std::size_t>(end - name) + 1 + 3) / 4;
      for (std::size_t j = 3 + words; j < n; ++j) interfaces.insert(p[j]);
    }
    if (op == 16) {
      require(n == 6 && p[1] == entry && p[2] == 17 && p[3] == 1 && p[4] == 1 && p[5] == 1 &&
              !modes++, "Vulkan kernel requires one unit LocalSize template");
      geometry = i + 3;
    }
    require(op != 331, "Vulkan kernel does not support execution-mode specialization IDs");
    if (op >= 19 && op <= 39) {
      require(n >= 2 && p[1] < w[3] && !types.contains(p[1]), "invalid Vulkan type declaration");
      types[p[1]] = {p, p + n};
    }
    if (op == 54 && n == 5 && p[2] == entry) { return_type = p[1]; function_type = p[4]; }
    if (op == 59) {
      require(n >= 4 && p[2] < w[3] && !variables.contains(p[2]), "invalid Vulkan variable declaration");
      variables[p[2]] = {p, p + n};
    }
    if (op == 71) {
      require(n >= 3, "invalid Vulkan resource decoration");
      if (p[2] == 33 || p[2] == 34 || p[2] == 6 || p[2] == 11) {
        require(n == 4, "invalid Vulkan resource decoration width");
        auto& map = p[2] == 33 ? bindings : p[2] == 34 ? sets : p[2] == 6 ? strides : builtins;
        require(map.emplace(p[1], p[3]).second, "duplicate Vulkan resource decoration");
      }
      require(p[2] != 41, "unresolved Vulkan shader linkage");
    }
    if (op == 72) {
      require(n >= 4, "invalid Vulkan member decoration");
      if (p[3] == 35) {
        require(n == 5 && offsets.emplace(std::pair{p[1], p[2]}, p[4]).second,
                "invalid Vulkan member offset");
      }
    }
    i += n;
  }
  require(capabilities == 1 && models == 1 && entries == 1 && modes == 1 && geometry,
          "missing Vulkan shader capability, model, entry or geometry");
  auto type = [&](std::uint32_t id, std::uint32_t op, std::size_t size) -> const std::vector<std::uint32_t>& {
    const auto found = types.find(id);
    require(found != types.end() && (found->second[0] & 65535) == op && found->second.size() == size,
            "Vulkan kernel resource type mismatch");
    return found->second;
  };
  auto uint32 = [&](std::uint32_t id) { const auto& t = type(id, 21, 4); require(t[2] == 32 && !t[3], "Vulkan kernel requires uint32 storage"); };
  type(return_type, 19, 2);
  require(type(function_type, 33, 3)[2] == return_type, "Vulkan entry must have no arguments");
  require(bindings.size() == tensors && sets.size() == tensors && builtins.size() == 2,
          "Vulkan kernel resource count mismatch");
  std::set<std::uint32_t> slots, input_ids;
  std::size_t push_count = 0;
  for (const auto& [id, variable] : variables) {
    const auto storage = variable[3];
    if (storage == 7) continue; // Function-local scratch is not a resource.
    const auto& pointer = type(variable[1], 32, 4);
    require(pointer[2] == storage, "Vulkan resource storage-class mismatch");
    if (storage == 12) {
      require(bindings.contains(id) && sets.contains(id) && !sets[id] && bindings[id] < tensors &&
              slots.insert(bindings[id]).second, "Vulkan descriptor set or binding mismatch");
      const auto& structure = type(pointer[3], 30, 3);
      const auto& array = type(structure[2], 29, 3);
      uint32(array[2]);
      require(strides.contains(structure[2]) && strides[structure[2]] == 4 &&
              offsets.contains({pointer[3], 0}) && offsets[{pointer[3], 0}] == 0,
              "Vulkan buffer stride/offset mismatch");
    } else if (storage == 9) {
      ++push_count;
      const auto& structure = type(pointer[3], 30, 7);
      for (std::uint32_t field = 0; field < 5; ++field) {
        uint32(structure[field + 2]);
        require(offsets.contains({pointer[3], field}) && offsets[{pointer[3], field}] == 4 * field,
                "Vulkan launch parameter offset mismatch");
      }
    } else if (storage == 1) {
      const auto& vector = type(pointer[3], 23, 4);
      uint32(vector[2]);
      require(vector[3] == 3 && builtins.contains(id) && (builtins[id] == 26 || builtins[id] == 27),
              "Vulkan workgroup builtin mismatch");
      input_ids.insert(id);
    } else require(false, "unsupported Vulkan kernel storage class");
  }
  require(slots.size() == tensors && push_count == 1 && input_ids == interfaces && input_ids.size() == 2 &&
          builtins[*input_ids.begin()] != builtins[*input_ids.rbegin()], "Vulkan kernel interface mismatch");
  return geometry;
}
}  // namespace

std::string kernel_manifest(const KernelSignature& s) {
  return "tensorcx.vulkan.v1|spirv13|vulkan11|" + s.kinds + "|" + std::to_string(s.output_index) +
      "|" + std::to_string(s.guard_index) + "|canonical-prefix-v1";
}
Status compiled_kernel_support(int index) {
  if (::getpid() != registry_process) return invalid("Vulkan kernel compilation requires a fresh process after fork");
  auto context = get_context(index);
  return context ? Status::Ok() : context.status();
}
KernelPipeline::~KernelPipeline() {
  if (handle && process == static_cast<std::uint64_t>(::getpid())) owner->DestroyPipeline(owner->device, handle, nullptr);
}
struct VulkanKernelModule::Impl {
  std::shared_ptr<Context> owner;
  std::uint64_t process = static_cast<std::uint64_t>(::getpid());
  std::vector<std::uint32_t> words;
  std::size_t geometry = 0;
  mutable std::mutex mutex;
  std::shared_ptr<KernelPipeline> cached;
  std::uint32_t cached_block = 0;
  std::shared_ptr<KernelPipeline> pipeline(std::uint32_t block) {
    std::lock_guard lock(mutex);
    if (cached && cached_block == block) return cached;
    auto code = words; code[geometry] = block; // Only the validated LocalSize X literal varies.
    auto result = std::make_shared<KernelPipeline>(); result->owner = owner; result->process = process;
    std::lock_guard queue_lock(owner->mutex);
    if (owner->poisoned) throw std::runtime_error("Vulkan context is poisoned after a device error");
    VkShaderModuleCreateInfo module{}; module.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module.codeSize = code.size() * 4; module.pCode = code.data();
    VkShaderModule shader = VK_NULL_HANDLE;
    check(owner->CreateShaderModule(owner->device, &module, nullptr, &shader), "vkCreateShaderModule");
    VkComputePipelineCreateInfo info{}; info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; info.stage.module = shader;
    info.stage.pName = entry_name; info.layout = owner->pipeline_layout;
    const auto status = owner->CreateComputePipelines(owner->device, VK_NULL_HANDLE, 1, &info, nullptr, &result->handle);
    owner->DestroyShaderModule(owner->device, shader, nullptr);
    check(status, "vkCreateComputePipelines");
    cached = result; cached_block = block;
    return result;
  }
};
VulkanKernelModule::VulkanKernelModule() : impl_(std::make_unique<Impl>()) {}
VulkanKernelModule::~VulkanKernelModule() {
  if (impl_->process != static_cast<std::uint64_t>(::getpid())) {
    (void)impl_.release(); // Never touch inherited driver handles or mutexes.
    return;
  }
  std::lock_guard lock(registry_mutex);
  if (!id_.empty()) registry.erase(id_);
}
Expected<std::shared_ptr<VulkanKernelModule>> VulkanKernelModule::load(
    const std::string& artifact, KernelSignature signature, int index) {
  if (::getpid() != registry_process) return invalid("Vulkan kernel compilation requires a fresh process after fork");
  try {
    require(valid_signature(signature), "invalid Vulkan kernel signature");
    const auto header = kernel_manifest(signature) + "\n";
    require(artifact.size() <= 32 * 1024 * 1024 && artifact.starts_with(header) &&
            artifact.size() >= header.size() + 20 && (artifact.size() - header.size()) % 4 == 0,
            "Vulkan kernel manifest or byte-size mismatch");
    auto module = std::shared_ptr<VulkanKernelModule>(new VulkanKernelModule());
    module->signature_ = std::move(signature);
    module->impl_->words.resize((artifact.size() - header.size()) / 4);
    std::memcpy(module->impl_->words.data(), artifact.data() + header.size(), artifact.size() - header.size());
    module->impl_->geometry = validate_spirv(module->impl_->words,
        std::count(module->signature_.kinds.begin(), module->signature_.kinds.end(), 't'));
    auto context = get_context(index); if (!context) return context.status();
    module->impl_->owner = context.move_value();
    (void)module->impl_->pipeline(1); // Every cache hit gets a fresh native validation/JIT.
    module->id_ = "vulkan_kernel_" + std::to_string(next_id++);
    { std::lock_guard lock(registry_mutex); registry[module->id_] = module; }
    return module;
  } catch (const std::invalid_argument& error) { return invalid(error.what()); }
    catch (const std::exception& error) { return Status(StatusCode::kInternal, error.what()); }
}

bool VulkanKernelModule::current_process() const {
  return impl_->process == static_cast<std::uint64_t>(::getpid());
}
Expected<VulkanTensor> VulkanKernelModule::launch(std::span<const KernelArgument> arguments,
    const Tensor& output, std::uint32_t threads, std::uint32_t block) const {
  if (!current_process()) return invalid("Vulkan kernel modules are process-local");
  ProfileScope profile(ProfileCategory::kDispatch, "vulkan_generated_launch", "vulkan");
  try {
    const auto& context = impl_->owner;
    require(block && block <= context->limits.maxComputeWorkGroupSize[0] &&
            block <= context->limits.maxComputeWorkGroupInvocations, "Vulkan kernel block size exceeds device limits");
    require(arguments.size() == signature_.kinds.size(), "Vulkan kernel argument count mismatch");
    auto target = from_core_tensor(output); if (!target) return target.status();
    require(target.value().buffer()->owner() == context && output.dtype == DType::kFloat32 &&
            threads <= static_cast<std::uint64_t>(numel(output.shape)), "Vulkan output owner/dtype/size mismatch");
    std::array<std::shared_ptr<Allocation>, 4> buffers{};
    std::size_t slot = 0, output_slot = 0;
    for (std::size_t i = 0; i < arguments.size(); ++i) {
      const auto& argument = arguments[i];
      if (signature_.kinds[i] == 'u') {
        require(argument.kind == KernelArgumentKind::kUInt32 && !argument.tensor && argument.uint32_value == threads,
                "Vulkan kernel guard argument must equal thread_count");
      } else {
        require(argument.kind == KernelArgumentKind::kTensor && argument.tensor, "Vulkan kernel tensor argument kind mismatch");
        auto value = from_core_tensor(*argument.tensor); if (!value) return value.status();
        require(value.value().dtype() == DType::kFloat32 && value.value().shape() == output.shape &&
                value.value().buffer()->owner() == context, "Vulkan kernel tensor dtype/shape/owner mismatch");
        if (i == signature_.output_index) {
          require(value.value().buffer() == target.value().buffer(), "Vulkan output argument mismatch");
          output_slot = slot;
        }
        buffers[slot++] = value.value().buffer()->allocation();
      }
    }
    auto pipeline = threads ? impl_->pipeline(block) : nullptr;
    auto result = empty(output.shape, DType::kFloat32, context->index); if (!result) return result.status();
    // Preserve the supplied template, zero-work values, suffixes and all input
    // aliases. Every result owns private GPU storage; no CPU data fallback.
    context->copy(result.value().buffer()->allocation(), target.value().buffer()->allocation(), result.value().nbytes());
    buffers[output_slot] = result.value().buffer()->allocation();
    while (slot < buffers.size()) buffers[slot++] = result.value().buffer()->allocation();
    if (threads) {
      PushConstants push{}; push.count = threads; push.scalar_bits = block;
      context->compute(buffers, push, std::move(pipeline), block);
    }
    return result.move_value();
  } catch (const std::invalid_argument& error) { return invalid(error.what()); }
    catch (const std::exception& error) { return Status(StatusCode::kInternal, error.what()); }
}

Status execute_compiled_kernel(const BackendExecution& e) {
  if (auto status = validate_kernel_execution_contract(e); !status.ok()) return status;
  if (e.outputs.size() != 1 || e.compilation_target->artifact_kind != KernelArtifactKind::kBinary ||
      e.compilation_target->entry_point != entry_name) return invalid("unsupported Vulkan kernel artifact/output/entry");
  const auto& l = *e.launch;
  if (l.grid_y != 1 || l.grid_z != 1 || l.threads_per_group_y != 1 || l.threads_per_group_z != 1)
    return invalid("Vulkan kernel requires a one-dimensional logical launch");
  auto module = resolve(e.compilation_target->artifact); if (!module) return module.status();
  auto result = module.value()->launch(e.kernel_arguments, e.outputs[0], l.grid_x, l.threads_per_group_x);
  if (!result) return result.status();
  e.outputs[0] = to_core_tensor(result.value());
  return Status::Ok();
}
Expected<VulkanTensor> launch_compiled_kernel(const std::shared_ptr<VulkanKernelModule>& module,
    std::span<const KernelArgument> arguments, std::uint32_t threads, std::uint32_t block) {
  if (!module) return invalid("missing Vulkan kernel module");
  if (!module->current_process()) return invalid("Vulkan kernel modules are process-local");
  const auto out = module->signature().output_index;
  if (out >= arguments.size() || arguments[out].kind != KernelArgumentKind::kTensor || !arguments[out].tensor)
    return invalid("missing Vulkan kernel output");
  if (!threads) return module->launch(arguments, *arguments[out].tensor, threads, block);
  std::array<Tensor, 1> outputs{*arguments[out].tensor};
  BackendExecution e{BackendOpClass::kKernel, {}, {}, outputs, LaunchConfig{threads, 1, 1, block, 1, 1},
      CompilationTarget{KernelArtifactKind::kBinary, module->artifact_id(), entry_name}, arguments};
  VulkanBackend backend;
  if (auto status = backend.execute(e); !status.ok()) return status;
  return from_core_tensor(outputs[0]);
}
}  // namespace tensorcx::vulkan
