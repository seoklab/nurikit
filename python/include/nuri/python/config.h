//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_PYTHON_CONFIG_H_
#define NURI_PYTHON_CONFIG_H_

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/typing.h>

#include "nuri/core/element.h"
#include "nuri/fmt/cif.h"

#ifndef NURI_PYTHON_MODULE_NAME
#error "NURI_PYTHON_MODULE_NAME is not defined"
#endif

#define NURI_PYTHON_MODULE(m) PYBIND11_MODULE(NURI_PYTHON_MODULE_NAME, m)

namespace nuri {
namespace python_internal {
// NOLINTNEXTLINE(misc-unused-alias-decls)
namespace py = pybind11;
// NOLINTNEXTLINE(misc-unused-alias-decls)
namespace pyt = pybind11::typing;

using rvp = py::return_value_policy;
}  // namespace python_internal
}  // namespace nuri

PYBIND11_MAKE_OPAQUE(std::vector<nuri::Isotope>)
PYBIND11_MAKE_OPAQUE(std::vector<nuri::internal::CifFrame>)

#endif /* NURI_PYTHON_CONFIG_H_ */
