//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_PYTHON_CONFIG_H_
#define NURI_PYTHON_CONFIG_H_

#include <utility>
#include <vector>

#include <pybind11/pybind11.h>

#include "nuri/core/element.h"

#ifndef NURI_PYTHON_MODULE_NAME
#error "NURI_PYTHON_MODULE_NAME is not defined"
#endif

#define NURI_PYTHON_MODULE(m) PYBIND11_MODULE(NURI_PYTHON_MODULE_NAME, m)

namespace nuri {
namespace python_internal {
// NOLINTNEXTLINE(misc-unused-alias-decls)
namespace py = pybind11;
using rvp = py::return_value_policy;

using PropertyMap = std::vector<std::pair<std::string, std::string>>;
}  // namespace python_internal
}  // namespace nuri

PYBIND11_MAKE_OPAQUE(std::vector<nuri::Isotope>)
PYBIND11_MAKE_OPAQUE(nuri::python_internal::PropertyMap)

#endif /* NURI_PYTHON_CONFIG_H_ */
