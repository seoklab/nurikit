//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_PYTHON_UTILS_H_
#define NURI_PYTHON_UTILS_H_

#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include "nuri/core/element.h"

namespace nuri_py {
template <class CppType, class... Args>
using PyProxyCls =
  pybind11::class_<CppType, std::unique_ptr<CppType, pybind11::nodelete>,
                   Args...>;
}  // namespace nuri_py

PYBIND11_MAKE_OPAQUE(std::vector<nuri::Isotope>)

#endif /* NURI_PYTHON_UTILS_H_ */
