#include <nanobind/nanobind.h>

#ifndef CORTEX_RUNTIME_VERSION
#define CORTEX_RUNTIME_VERSION "0.0.0"
#endif

namespace nb = nanobind;

NB_MODULE(_core, module) {
  module.doc() = "Native extension module for Cortex Runtime.";
  module.def("version", []() { return CORTEX_RUNTIME_VERSION; });
}
