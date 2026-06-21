#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "cortex/backends/cpu/cpu_backend.h"
#include "cortex/backends/cpu/cpu_tensor.h"
#include "cortex/core/dtype.h"
#include "cortex/core/shape.h"

#ifndef CORTEX_RUNTIME_VERSION
#define CORTEX_RUNTIME_VERSION "0.0.0"
#endif

namespace nb = nanobind;

namespace {

using cortex::DType;
using cortex::OpDesc;
using cortex::OpKind;
using cortex::Shape;
using cortex::cpu::CpuTensor;

DType parse_dtype(nb::handle dtype, DType inferred) {
  if (dtype.is_none()) {
    return inferred;
  }

  const std::string name = nb::cast<std::string>(dtype);
  if (name == "float32") {
    return DType::kFloat32;
  }
  if (name == "int32") {
    return DType::kInt32;
  }
  throw std::invalid_argument("unsupported dtype: expected float32 or int32");
}

void validate_cpu_device(const std::string& device) {
  if (device != "cpu") {
    throw std::invalid_argument("only device='cpu' is available in Phase 1");
  }
}

Shape parse_shape(nb::handle shape) {
  if (nb::isinstance<nb::int_>(shape)) {
    const auto dim = nb::cast<cortex::Dim>(shape);
    return Shape{dim};
  }

  Shape result;
  for (nb::handle item : nb::iter(shape)) {
    result.push_back(nb::cast<cortex::Dim>(item));
  }
  return result;
}

CpuTensor tensor_from_sequence(nb::handle data, nb::handle dtype, const std::string& device) {
  validate_cpu_device(device);

  std::vector<nb::object> items;
  bool saw_float = false;
  for (nb::handle item : nb::iter(data)) {
    items.emplace_back(nb::borrow<nb::object>(item));
    if (nb::isinstance<nb::float_>(item)) {
      saw_float = true;
    }
  }

  const DType inferred = saw_float ? DType::kFloat32 : DType::kInt32;
  const DType actual_dtype = parse_dtype(dtype, inferred);
  Shape shape{static_cast<cortex::Dim>(items.size())};

  switch (actual_dtype) {
    case DType::kFloat32: {
      std::vector<float> values;
      values.reserve(items.size());
      for (const nb::object& item : items) {
        values.push_back(nb::cast<float>(item));
      }
      return CpuTensor(std::move(shape), std::move(values));
    }
    case DType::kInt32: {
      std::vector<std::int32_t> values;
      values.reserve(items.size());
      for (const nb::object& item : items) {
        values.push_back(nb::cast<std::int32_t>(item));
      }
      return CpuTensor(std::move(shape), std::move(values));
    }
  }
  throw std::invalid_argument("unsupported dtype");
}

nb::tuple shape_tuple(const Shape& shape) {
  nb::tuple result = nb::steal<nb::tuple>(PyTuple_New(static_cast<Py_ssize_t>(shape.size())));
  for (std::size_t i = 0; i < shape.size(); ++i) {
    PyTuple_SET_ITEM(result.ptr(), static_cast<Py_ssize_t>(i), PyLong_FromLongLong(shape[i]));
  }
  return result;
}

nb::object tensor_to_numpy(const CpuTensor& tensor) {
  nb::object np = nb::module_::import_("numpy");
  nb::list values;

  switch (tensor.dtype()) {
    case DType::kFloat32:
      for (float value : tensor.float_data()) {
        values.append(nb::float_(value));
      }
      break;
    case DType::kInt32:
      for (std::int32_t value : tensor.int32_data()) {
        values.append(nb::int_(value));
      }
      break;
  }

  return np.attr("array")(values, nb::arg("dtype") = std::string(cortex::dtype_name(tensor.dtype())));
}

CpuTensor binary_op(const CpuTensor& lhs, const CpuTensor& rhs, OpKind kind) {
  return cortex::cpu::execute_binary(OpDesc{kind}, lhs, rhs);
}

}  // namespace

NB_MODULE(_core, module) {
  module.doc() = "Native extension module for Cortex Runtime.";
  module.def("version", []() { return CORTEX_RUNTIME_VERSION; });
  module.attr("float32") = "float32";
  module.attr("int32") = "int32";

  nb::class_<CpuTensor>(module, "Tensor")
      .def_prop_ro("shape", [](const CpuTensor& tensor) { return shape_tuple(tensor.shape()); })
      .def_prop_ro("strides", [](const CpuTensor& tensor) { return shape_tuple(tensor.strides()); })
      .def_prop_ro("dtype",
                   [](const CpuTensor& tensor) {
                     return std::string(cortex::dtype_name(tensor.dtype()));
                   })
      .def_prop_ro("device", [](const CpuTensor&) { return "cpu"; })
      .def_prop_ro("nbytes", [](const CpuTensor& tensor) { return tensor.buffer()->nbytes(); })
      .def("numpy", &tensor_to_numpy)
      .def("cpu", [](const CpuTensor& tensor) { return tensor; })
      .def("to",
           [](const CpuTensor& tensor, const std::string& device) {
             validate_cpu_device(device);
             return tensor;
           },
           nb::arg("device"))
      .def("__add__", [](const CpuTensor& lhs, const CpuTensor& rhs) {
        return binary_op(lhs, rhs, OpKind::kAdd);
      })
      .def("__mul__", [](const CpuTensor& lhs, const CpuTensor& rhs) {
        return binary_op(lhs, rhs, OpKind::kMultiply);
      });

  module.def("tensor",
             &tensor_from_sequence,
             nb::arg("data"),
             nb::arg("dtype").none() = nb::none(),
             nb::arg("device") = "cpu");
  module.def("empty",
             [](nb::handle shape, nb::handle dtype, const std::string& device) {
               validate_cpu_device(device);
               return cortex::cpu::empty(parse_shape(shape), parse_dtype(dtype, DType::kFloat32));
             },
             nb::arg("shape"),
             nb::arg("dtype") = "float32",
             nb::arg("device") = "cpu");
  module.def("zeros",
             [](nb::handle shape, nb::handle dtype, const std::string& device) {
               validate_cpu_device(device);
               return cortex::cpu::fill(parse_shape(shape), parse_dtype(dtype, DType::kFloat32), 0.0);
             },
             nb::arg("shape"),
             nb::arg("dtype") = "float32",
             nb::arg("device") = "cpu");
  module.def("ones",
             [](nb::handle shape, nb::handle dtype, const std::string& device) {
               validate_cpu_device(device);
               return cortex::cpu::fill(parse_shape(shape), parse_dtype(dtype, DType::kFloat32), 1.0);
             },
             nb::arg("shape"),
             nb::arg("dtype") = "float32",
             nb::arg("device") = "cpu");
}
