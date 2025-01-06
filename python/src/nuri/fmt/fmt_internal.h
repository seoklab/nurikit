//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//
#ifndef NURI_PYTHON_FMT_FMT_INTERNAL_H_
#define NURI_PYTHON_FMT_FMT_INTERNAL_H_

#include <pybind11/pybind11.h>

namespace nuri {
namespace python_internal {
extern void bind_cif(py::module &m);
}  // namespace python_internal
}  // namespace nuri

#endif /* NURI_PYTHON_FMT_FMT_INTERNAL_H_ */
