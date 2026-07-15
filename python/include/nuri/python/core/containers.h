//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_PYTHON_CORE_CONTAINERS_H_
#define NURI_PYTHON_CORE_CONTAINERS_H_

#include <string_view>

#include <pybind11/pybind11.h>

#include "nuri/core/container/property_map.h"
#include "nuri/python/utils.h"

namespace nuri {
namespace python_internal {
using ProxyPropertyMap = TypeErasedProxyWrapper<internal::PropertyMap *>;

// Coerces an arbitrary Python mapping into a PropertyMap for property setters.
// A _PropertyMap/_ProxyPropertyMap is used directly; any other mapping is
// materialized through ``dict(obj)`` (which raises for non-mapping inputs),
// then validated key/value-wise so non-string entries raise a clean TypeError.
inline internal::PropertyMap to_property_map(const py::object &obj) {
  if (py::isinstance<internal::PropertyMap>(obj)
      || py::isinstance<ProxyPropertyMap>(obj))
    return obj.cast<internal::PropertyMap>();

  py::dict dict(obj);
  internal::PropertyMap map;
  map.reserve(dict.size());
  for (auto item: dict) {
    if (!py::isinstance<py::str>(item.first))
      throw py::type_error("keys must be strings");
    if (!py::isinstance<py::str>(item.second))
      throw py::type_error("values must be strings");

    internal::set_key(map, item.first.cast<std::string_view>(),
                      item.second.cast<std::string_view>());
  }
  return map;
}

extern void bind_containers(py::module &m);
}  // namespace python_internal
}  // namespace nuri

#endif /* NURI_PYTHON_CORE_CONTAINERS_H_ */
