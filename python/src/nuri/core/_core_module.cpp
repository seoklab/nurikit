//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <string_view>

#include "nuri/python/core/containers.h"
#include "nuri/python/core/core_module.h"
#include "nuri/python/utils.h"

namespace nuri {
namespace python_internal {
namespace {
NURI_PYTHON_MODULE(m) {
  bind_containers(m);
  bind_element(m);
  bind_molecule(m);

  m.def(
      "_py_array_cast_test_helper",
      [](py::handle obj, std::string_view kind) {
        if (kind == "matrix") {
          auto arr = py_array_cast<3, 3>(obj);
          return arr.numpy();
        }

        if (kind == "col_vector") {
          auto arr = py_array_cast<3, 1>(obj);
          return arr.numpy();
        }

        if (kind == "row_vector") {
          auto arr = py_array_cast<1, 3>(obj);
          return arr.numpy();
        }

        // dynamic
        auto arr = py_array_cast(obj);
        return arr.numpy();
      },
      py::arg("obj"), py::arg("kind"));
}
}  // namespace
}  // namespace python_internal
}  // namespace nuri
