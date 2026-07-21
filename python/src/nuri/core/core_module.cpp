//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/python/core/core_module.h"

#include <cstdint>
#include <optional>
#include <string_view>

#include "nuri/eigen_config.h"
#include "core_internal.h"
#include "nuri/python/core/containers.h"
#include "nuri/python/utils.h"
#include "nuri/random.h"

namespace nuri {
namespace python_internal {
namespace {
struct EigenViewTestData {
  MatrixXd dynamic;
  Matrix3Xd semi_dynamic;
  Matrix4d fixed = Matrix4d::Zero();
  VectorXd vector;
};

template <class T>
py::array wrap_view(T &d, py::handle owner, bool readonly, bool transpose) {
  if (transpose) {
    auto t = d.transpose();
    return eigen_as_numpy_view(t, owner, readonly);
  }

  return eigen_as_numpy_view(d, owner, readonly);
}

NURI_PYTHON_MODULE(m) {
  m.doc() = R"doc(
The core module of NuriKit.

This module contains the core classes of NuriKit. The core module is not
very useful by itself, but is a dependency of many other modules. Chemical
data structures, such as elements, isotopes, and molecules, and also the
graph structure and algorithms, are defined in this module.
)doc";

  bind_containers(m);
  bind_element_impl(m);
  bind_molecule_impl(m);

  py::module_ geometry = m.def_submodule("geometry");
  bind_geometry(geometry);

  m.def(
      "seed_thread",
      [](std::optional<uint64_t> seed) {
        int sv = -1;
        if (seed)
          sv = static_cast<int>(*seed % (1ULL << 31));

        internal::seed_thread(sv);
      },
      py::arg("seed") = py::none(),
      R"doc(
Set the seed of random number generator for the current thread.

:param seed: The seed to set. If not specified, a random seed is chosen.
)doc");

  m.def("_random_test_helper", [](int max) { return internal::draw_uid(max); });

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

  py::class_<EigenViewTestData>(m, "_EigenViewTestData")
      .def(py::init([](py::handle dynamic, py::handle semi_dynamic,
                       py::handle fixed, py::handle vector) {
             EigenViewTestData d;
             if (!dynamic.is_none()) {
               auto w = py_array_cast<>(dynamic);
               d.dynamic = w.eigen();
             }
             if (!semi_dynamic.is_none()) {
               auto w = py_array_cast<3>(semi_dynamic);
               d.semi_dynamic = w.eigen();
             }
             if (!fixed.is_none()) {
               auto w = py_array_cast<4, 4>(fixed);
               d.fixed = w.eigen();
             }
             if (!vector.is_none()) {
               auto w = py_array_cast<Eigen::Dynamic, 1>(vector);
               d.vector = w.eigen();
             }
             return d;
           }),
           py::kw_only(), py::arg("dynamic") = py::none(),
           py::arg("semi_dynamic") = py::none(), py::arg("fixed") = py::none(),
           py::arg("vector") = py::none())
      .def(
          "dynamic",
          [](py::handle self, bool readonly, bool transpose) {
            auto &d = self.cast<EigenViewTestData &>();
            return wrap_view(d.dynamic, self, readonly, transpose);
          },
          py::arg("readonly"), py::arg("transpose"))
      .def(
          "semi_dynamic",
          [](py::handle self, bool readonly, bool transpose) {
            auto &d = self.cast<EigenViewTestData &>();
            return wrap_view(d.semi_dynamic, self, readonly, transpose);
          },
          py::arg("readonly"), py::arg("transpose"))
      .def(
          "fixed",
          [](py::handle self, bool readonly, bool transpose) {
            auto &d = self.cast<EigenViewTestData &>();
            return wrap_view(d.fixed, self, readonly, transpose);
          },
          py::arg("readonly"), py::arg("transpose"))
      .def(
          "vector",
          [](py::handle self, bool readonly, bool transpose) {
            auto &d = self.cast<EigenViewTestData &>();
            return wrap_view(d.vector, self, readonly, transpose);
          },
          py::arg("readonly"), py::arg("transpose"));

  internal::seed_thread(-1);
}
}  // namespace
}  // namespace python_internal
}  // namespace nuri
