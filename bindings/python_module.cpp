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
#include "cortex/core/status.h"

#if CORTEX_ENABLE_METAL
#include "cortex/backends/metal/metal_backend.h"
#include "cortex/backends/metal/metal_tensor.h"
#endif

#ifndef CORTEX_RUNTIME_VERSION
#define CORTEX_RUNTIME_VERSION "0+unknown"
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
    throw std::invalid_argument("native CPU factory only accepts device='cpu'");
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

  nb::object array =
      np.attr("array")(values, nb::arg("dtype") = std::string(cortex::dtype_name(tensor.dtype())));
  return array.attr("reshape")(shape_tuple(tensor.shape()));
}

CpuTensor binary_op(const CpuTensor& lhs, const CpuTensor& rhs, OpKind kind) {
  return cortex::cpu::execute_binary(OpDesc{kind}, lhs, rhs);
}

#if CORTEX_ENABLE_METAL
void throw_status(const cortex::Status& status) {
  switch (status.code()) {
    case cortex::StatusCode::kInvalidArgument:
      throw std::invalid_argument(status.message());
    case cortex::StatusCode::kUnavailable:
    case cortex::StatusCode::kInternal:
      throw std::runtime_error(status.message());
    case cortex::StatusCode::kOk:
      break;
  }
  throw std::runtime_error(status.message());
}

template <typename T>
T unwrap(cortex::Expected<T> result) {
  if (!result) {
    throw_status(result.status());
  }
  return result.move_value();
}
#endif

}  // namespace

NB_MODULE(_core, module) {
  module.doc() = "Native extension module for Cortex Runtime.";
  module.def("version", []() { return CORTEX_RUNTIME_VERSION; });
  module.attr("float32") = "float32";
  module.attr("int32") = "int32";

  nb::class_<CpuTensor>(module, "CpuTensor")
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
  module.def("fill",
             [](nb::handle shape, nb::handle dtype, double value, const std::string& device) -> nb::object {
               const Shape parsed_shape = parse_shape(shape);
               const DType parsed_dtype = parse_dtype(dtype, DType::kFloat32);
               if (device == "cpu") {
                 return nb::cast(cortex::cpu::fill(parsed_shape, parsed_dtype, value));
               }
#if CORTEX_ENABLE_METAL
               if (device == "metal") {
                 return nb::cast(unwrap(cortex::metal::fill(
                     OpDesc{OpKind::kFill}, parsed_shape, parsed_dtype, value)));
               }
#endif
               throw std::invalid_argument("device is not available: " + device);
             },
             nb::arg("shape"),
             nb::arg("dtype") = "float32",
             nb::arg("value") = 0.0,
             nb::arg("device") = "cpu");

  module.def("add_cpu",
             [](const CpuTensor& lhs, const CpuTensor& rhs) {
               return binary_op(lhs, rhs, OpKind::kAdd);
             },
             nb::arg("lhs"),
             nb::arg("rhs"));
  module.def("multiply_cpu",
             [](const CpuTensor& lhs, const CpuTensor& rhs) {
               return binary_op(lhs, rhs, OpKind::kMultiply);
             },
             nb::arg("lhs"),
             nb::arg("rhs"));
  module.def("add",
             [](const CpuTensor& lhs, const CpuTensor& rhs) {
               return binary_op(lhs, rhs, OpKind::kAdd);
             },
             nb::arg("lhs"),
             nb::arg("rhs"));
  module.def("multiply",
             [](const CpuTensor& lhs, const CpuTensor& rhs) {
               return binary_op(lhs, rhs, OpKind::kMultiply);
             },
             nb::arg("lhs"),
             nb::arg("rhs"));
  module.def("is_available",
             [](const std::string& device) {
               if (device == "cpu") {
                 return true;
               }
#if CORTEX_ENABLE_METAL
               if (device == "metal") {
                 return cortex::metal::available();
               }
#endif
               return false;
             },
             nb::arg("device"));
  module.def("devices", []() {
    nb::list result;
    result.append("cpu");
#if CORTEX_ENABLE_METAL
    if (cortex::metal::available()) {
      result.append("metal");
    }
#endif
    return result;
  });
  module.def("device_name",
             [](const std::string& device) {
               if (device == "cpu") {
                 return std::string("CPU");
               }
#if CORTEX_ENABLE_METAL
               if (device == "metal") {
                 const auto names = cortex::metal::devices();
                 if (!names.empty()) {
                   return names.front();
                 }
               }
#endif
               throw std::invalid_argument("device is not available: " + device);
             },
             nb::arg("device"));

#if CORTEX_ENABLE_METAL
  nb::class_<cortex::metal::MetalTensor>(module, "MetalTensor")
      .def_prop_ro("shape", [](const cortex::metal::MetalTensor& tensor) {
        return shape_tuple(tensor.shape());
      })
      .def_prop_ro("strides", [](const cortex::metal::MetalTensor& tensor) {
        return shape_tuple(tensor.strides());
      })
      .def_prop_ro("dtype",
                   [](const cortex::metal::MetalTensor& tensor) {
                     return std::string(cortex::dtype_name(tensor.dtype()));
                   })
      .def_prop_ro("device", [](const cortex::metal::MetalTensor&) { return "metal"; })
      .def_prop_ro("nbytes",
                   [](const cortex::metal::MetalTensor& tensor) {
                     return tensor.nbytes();
                   })
      .def("__add__", [](const cortex::metal::MetalTensor& lhs,
                         const cortex::metal::MetalTensor& rhs) {
        return unwrap(cortex::metal::execute_binary(OpDesc{OpKind::kAdd}, lhs, rhs));
      })
      .def("__mul__", [](const cortex::metal::MetalTensor& lhs,
                         const cortex::metal::MetalTensor& rhs) {
        return unwrap(cortex::metal::execute_binary(OpDesc{OpKind::kMultiply}, lhs, rhs));
      });

  module.def("cpu_to_metal",
             [](const CpuTensor& tensor) { return unwrap(cortex::metal::from_cpu(tensor)); },
             nb::arg("tensor"));
  module.def("metal_to_cpu",
             [](const cortex::metal::MetalTensor& tensor) {
               return unwrap(cortex::metal::to_cpu(tensor));
             },
             nb::arg("tensor"));
  module.def("add",
             [](const cortex::metal::MetalTensor& lhs, const cortex::metal::MetalTensor& rhs) {
               return unwrap(cortex::metal::execute_binary(OpDesc{OpKind::kAdd}, lhs, rhs));
             },
             nb::arg("lhs"),
             nb::arg("rhs"));
  module.def("multiply",
             [](const cortex::metal::MetalTensor& lhs, const cortex::metal::MetalTensor& rhs) {
               return unwrap(cortex::metal::execute_binary(OpDesc{OpKind::kMultiply}, lhs, rhs));
             },
             nb::arg("lhs"),
             nb::arg("rhs"));
#endif
}
