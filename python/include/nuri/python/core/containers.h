//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_PYTHON_CORE_CONTAINERS_H_
#define NURI_PYTHON_CORE_CONTAINERS_H_

#include <pybind11/pybind11.h>

#include "nuri/core/property_map.h"
#include "nuri/python/utils.h"

namespace nuri {
namespace python_internal {
using ProxyPropertyMap = TypeErasedProxyWrapper<internal::PropertyMap *>;

extern void bind_containers(py::module &m);
}  // namespace python_internal
}  // namespace nuri

#endif /* NURI_PYTHON_CORE_CONTAINERS_H_ */
