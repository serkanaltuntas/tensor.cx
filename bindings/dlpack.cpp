#include <Python.h>
#include <nanobind/nanobind.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include "../third_party/dlpack/dlpack.h"
#include "tensorcx/backends/cpu/cpu_backend.h"
#if TENSORCX_ENABLE_CUDA
#include "tensorcx/backends/cuda/cuda_backend.h"
#include "tensorcx/backends/cuda/cuda_buffer.h"
#endif

namespace nb = nanobind;
namespace {
using namespace tensorcx;
[[noreturn]] void reject(const char* message) {
  PyErr_SetString(PyExc_BufferError, message);
  throw nb::python_error();
}

Tensor clone(Backend& backend, const Tensor& input) {
  OpDesc op{OpKind::kTranspose};
  op.axes.resize(input.shape.size());
  std::iota(op.axes.begin(), op.axes.end(), 0);
  std::array<Tensor, 1> inputs{input}, outputs;
  BackendExecution execution{BackendOpClass::kPrimitive, op, inputs, outputs, std::nullopt, std::nullopt, {}};
  Status status;
  { nb::gil_scoped_release release; status = backend.execute(execution); }
  if (!status.ok()) throw std::runtime_error(status.message());
  return std::move(outputs[0]);
}

struct Export {
  DLManagedTensor legacy{};
  DLManagedTensorVersioned versioned{};
  Tensor owner;
};
void capsule_delete(PyObject* capsule) noexcept {
  if (PyCapsule_IsValid(capsule, "dltensor")) {
    auto* tensor = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule, "dltensor"));
    if (tensor->deleter) tensor->deleter(tensor);
  } else if (PyCapsule_IsValid(capsule, "dltensor_versioned")) {
    auto* tensor = static_cast<DLManagedTensorVersioned*>(PyCapsule_GetPointer(capsule, "dltensor_versioned"));
    if (tensor->deleter) tensor->deleter(tensor);
  }
}
nb::object export_tensor(Tensor tensor, void* data, DLDevice device, bool versioned, bool copied) {
  auto state = std::make_unique<Export>();
  state->owner = std::move(tensor);
  const auto dtype = state->owner.dtype;
  const DLDataType type{static_cast<std::uint8_t>(dtype == DType::kFloat32 ? kDLFloat : dtype == DType::kInt32 ? kDLInt : kDLBool),
                        static_cast<std::uint8_t>(dtype_size(dtype) * 8), 1};
  if (state->owner.shape.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
    reject("DLPack rank is too large");
  DLTensor description{numel(state->owner.shape) ? data : nullptr, device,
                       static_cast<std::int32_t>(state->owner.shape.size()), type,
                       state->owner.shape.data(), state->owner.strides.data(), 0};
  void* pointer;
  const char* name;
  if (versioned) {
    state->versioned.version = {1, 0};
    state->versioned.manager_ctx = state.get();
    state->versioned.deleter = [](DLManagedTensorVersioned* t) { delete static_cast<Export*>(t->manager_ctx); };
    state->versioned.flags = copied ? DLPACK_FLAG_BITMASK_IS_COPIED : 0;
    state->versioned.dl_tensor = description;
    pointer = &state->versioned; name = "dltensor_versioned";
  } else {
    state->legacy.manager_ctx = state.get();
    state->legacy.deleter = [](DLManagedTensor* t) { delete static_cast<Export*>(t->manager_ctx); };
    state->legacy.dl_tensor = description;
    pointer = &state->legacy; name = "dltensor";
  }
  auto* capsule = PyCapsule_New(pointer, name, capsule_delete);
  if (!capsule) throw nb::python_error();
  state.release();
  return nb::steal<nb::object>(capsule);
}

struct Imported {
  DLTensor description;
  std::uint64_t flags;
  std::shared_ptr<void> owner;
};
Imported consume(nb::handle capsule) {
  if (PyCapsule_IsValid(capsule.ptr(), "dltensor_versioned")) {
    auto* t = static_cast<DLManagedTensorVersioned*>(PyCapsule_GetPointer(capsule.ptr(), "dltensor_versioned"));
    if (PyCapsule_SetName(capsule.ptr(), "used_dltensor_versioned")) throw nb::python_error();
    std::shared_ptr<void> owner(t, [](void* ptr) {
      auto* tensor = static_cast<DLManagedTensorVersioned*>(ptr);
      if (tensor->deleter) tensor->deleter(tensor);
    });
    // Only the prefix through deleter is stable across major ABI versions.
    if (t->version.major != 1) reject("unsupported DLPack major version");
    if (t->flags & ~(DLPACK_FLAG_BITMASK_READ_ONLY | DLPACK_FLAG_BITMASK_IS_COPIED))
      reject("unsupported DLPack flags");
    return {t->dl_tensor, t->flags, std::move(owner)};
  }
  if (PyCapsule_IsValid(capsule.ptr(), "dltensor")) {
    auto* t = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule.ptr(), "dltensor"));
    if (PyCapsule_SetName(capsule.ptr(), "used_dltensor")) throw nb::python_error();
    std::shared_ptr<void> owner(t, [](void* ptr) {
      auto* tensor = static_cast<DLManagedTensor*>(ptr);
      if (tensor->deleter) tensor->deleter(tensor);
    });
    return {t->dl_tensor, 0, std::move(owner)};
  }
  reject("expected an unconsumed DLPack capsule");
}

nb::object import_tensor(nb::handle capsule, bool copy, bool forbid_copy, int expected_type, int expected_id) {
  auto imported = consume(capsule);
  const auto& t = imported.description;
  if (forbid_copy && (imported.flags & DLPACK_FLAG_BITMASK_IS_COPIED)) reject("producer copied despite copy=False");
  if (!copy && (imported.flags & DLPACK_FLAG_BITMASK_READ_ONLY)) reject("read-only DLPack input requires copy=True");
  if (expected_type >= 0 && (t.device.device_type != expected_type || t.device.device_id != expected_id))
    reject("DLPack capsule device differs from producer device");
  if ((t.device.device_type != kDLCPU && t.device.device_type != kDLCUDA) || t.device.device_id != 0)
    reject("DLPack supports CPU and CUDA device 0 only");
  DType dtype;
  if (t.dtype.lanes != 1) reject("DLPack vector lanes are unsupported");
  if (t.dtype.code == kDLFloat && t.dtype.bits == 32) dtype = DType::kFloat32;
  else if (t.dtype.code == kDLInt && t.dtype.bits == 32) dtype = DType::kInt32;
  else if (t.dtype.code == kDLBool && t.dtype.bits == 8) dtype = DType::kBool;
  else reject("DLPack requires float32, int32 or bool");
  if (t.ndim < 0 || (t.ndim && !t.shape)) reject("invalid DLPack shape");
  Shape shape;
  if (t.ndim) shape.assign(t.shape, t.shape + t.ndim);
  const auto count = numel(shape);
  const auto canonical = contiguous_strides(shape);
  // Strides on singleton/empty axes do not change the accessible memory.
  if (count && t.strides) {
    for (int a = 0; a < t.ndim; ++a)
      if (shape[a] > 1 && t.strides[a] != canonical[a]) reject("DLPack requires C-contiguous input");
  }
  const auto element_bytes = dtype_size(dtype);
  if (static_cast<std::uint64_t>(count) > std::numeric_limits<std::size_t>::max() / element_bytes)
    reject("DLPack byte size overflow");
  const auto bytes = static_cast<std::size_t>(count) * element_bytes;
  auto address = reinterpret_cast<std::uintptr_t>(t.data);
  if (t.byte_offset > std::numeric_limits<std::uintptr_t>::max() - address) reject("DLPack pointer offset overflow");
  address += t.byte_offset;
  if (bytes && (!t.data || address % element_bytes || bytes > std::numeric_limits<std::uintptr_t>::max() - address))
    reject("DLPack requires an aligned, non-null data range");
  void* pointer = count ? reinterpret_cast<void*>(address) : nullptr;
  if (t.device.device_type == kDLCPU) {
    auto buffer = std::make_shared<cpu::CpuBuffer>(dtype, static_cast<std::size_t>(count), pointer, std::move(imported.owner));
    cpu::CpuTensor result(dtype, shape, std::move(buffer));
    if (copy) { cpu::CpuBackend backend; result = cpu::from_core_tensor(clone(backend, cpu::to_core_tensor(result))); }
    return nb::cast(std::move(result));
  }
#if TENSORCX_ENABLE_CUDA
  auto buffer = [&] {
    nb::gil_scoped_release release;
    return cuda::CudaBuffer::borrow(dtype, shape, pointer, imported.owner);
  }();
  if (!buffer) reject(buffer.status().message().c_str());
  cuda::CudaTensor result(dtype, shape, buffer.move_value());
  if (copy) {
    cuda::CudaBackend backend;
    auto converted = cuda::from_core_tensor(clone(backend, cuda::to_core_tensor(result)));
    if (!converted) reject(converted.status().message().c_str());
    result = converted.move_value();
  }
  return nb::cast(std::move(result));
#else
  reject("CUDA DLPack requires a CUDA-enabled tensor.cx build");
#endif
}
}  // namespace

void bind_dlpack(nb::module_& module) {
  module.def("_to_dlpack", [](const tensorcx::cpu::CpuTensor& input, bool versioned, bool copy) {
    tensorcx::cpu::CpuBackend backend;
    auto tensor = tensorcx::cpu::to_core_tensor(input);
    if (copy) tensor = clone(backend, tensor);
    auto native = tensorcx::cpu::from_core_tensor(tensor);
    return export_tensor(tensor, native.mutable_data(), DLDevice{kDLCPU, 0}, versioned, copy);
  }, nb::arg("input"), nb::arg("versioned"), nb::arg("copy"));
#if TENSORCX_ENABLE_CUDA
  module.def("_to_dlpack", [](const tensorcx::cuda::CudaTensor& input, bool versioned, bool copy) {
    tensorcx::cuda::CudaBackend backend;
    auto tensor = tensorcx::cuda::to_core_tensor(input);
    if (copy) tensor = clone(backend, tensor);
    auto native = tensorcx::cuda::from_core_tensor(tensor);
    if (!native) reject(native.status().message().c_str());
    return export_tensor(tensor, native.value().buffer()->data(), DLDevice{kDLCUDA, 0}, versioned, copy);
  }, nb::arg("input"), nb::arg("versioned"), nb::arg("copy"));
#endif
  module.def("_from_dlpack", import_tensor, nb::arg("capsule"), nb::arg("copy"), nb::arg("forbid_copy"), nb::arg("expected_type"), nb::arg("expected_id"));
}
