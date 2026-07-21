//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_PYTHON_CORE_CONTAINERS_H_
#define NURI_PYTHON_CORE_CONTAINERS_H_

#include <pybind11/pybind11.h>

#include "nuri/core/container/property_map.h"
#include "nuri/python/utils.h"

namespace nuri {
namespace python_internal {
using ProxyPropertyMap = TypeErasedProxyWrapper<internal::PropertyMap *>;

inline internal::PropertyMap to_property_map(const py::object &obj) {
  try {
    return obj.cast<internal::PropertyMap>();
  } catch (const py::cast_error &) {
    // .cast() alone would bury the real "keys/values must be strings" error.
    return py::type::of<internal::PropertyMap>()(py::dict(obj))
        .cast<internal::PropertyMap>();
  }
}

extern void bind_containers(py::module &m);
}  // namespace python_internal
}  // namespace nuri

#endif /* NURI_PYTHON_CORE_CONTAINERS_H_ */
