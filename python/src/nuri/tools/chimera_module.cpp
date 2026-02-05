//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <absl/strings/str_cat.h>
#include <pybind11/pytypes.h>

#include "nuri/eigen_config.h"
#include "nuri/python/utils.h"
#include "nuri/tools/chimera.h"

namespace nuri {
namespace python_internal {
namespace {
NURI_PYTHON_MODULE(m) {
  // For types
  py::module_::import("nuri.core");

  py::class_<MmResult>(m, "MmResult")
      .def_property_readonly(
          "transform",
          [](const MmResult &r) {
            return eigen_as_numpy(r.xform.matrix().transpose());
          },
          R"doc(
A copy of 4x4 best-fit rigid-body transformation tensor, to align ``query`` to
``template``.
)doc")
      .def_property_readonly(
          "selected", [](MmResult &r) -> ArrayXi & { return r.sel; },
          R"doc(
The final indices of the selected (inlier) points used in the alignment.
)doc")
      .def_property_readonly(
          "aligned_rmsd", [](const MmResult &r) { return std::sqrt(r.msd); },
          R"doc(
The root-mean-square deviation of the selected (inlier) points after alignment.
)doc");

  m.def(
      "match_maker",
      [](py::handle py_query, py::handle py_templ, double cutoff,
         double global_ratio, double viol_ratio) {
        auto query = py_array_cast<3>(py_query);
        auto templ = py_array_cast<3>(py_templ);
        if (query.eigen().cols() != templ.eigen().cols()) {
          throw py::value_error(
              absl::StrCat("Query and template structures have different "
                           "number of points, got ",
                           query.eigen().cols(), " vs ", templ.eigen().cols()));
        }

        if (cutoff <= 0.0) {
          throw py::value_error(
              absl::StrCat("distance cutoff must be positive, got ", cutoff));
        }
        if (global_ratio <= 0.0 || global_ratio > 1.0) {
          throw py::value_error(absl::StrCat(
              "global_ratio must be between 0 and 1, got ", global_ratio));
        }
        if (viol_ratio <= 0.0 || viol_ratio > 1.0) {
          throw py::value_error(absl::StrCat(
              "viol_ratio must be between 0 and 1, got ", viol_ratio));
        }

        auto ret = match_maker(query.eigen(), templ.eigen(), cutoff,
                               global_ratio, viol_ratio);
        if (ret.msd < 0)
          throw std::runtime_error("Match-Maker alignment failed");
        return ret;
      },
      py::arg("query"), py::arg("templ"), py::arg("cutoff") = 2.0,
      py::arg("global_ratio") = 0.1, py::arg("viol_ratio") = 0.5,
      R"doc(
Perform the Match-Maker algorithm to find a rigid-body alignment between two
sets of points. This is based on the Match-Maker algorithm used in UCSF Chimera.

:param query: The query structure. Must be representable as a 2D numpy array of
  shape ``(N, 3)``.
:param templ: The template structure. Must be representable as a 2D numpy array
  of shape ``(N, 3)``.
:param cutoff: Distance cutoff in angstroms. A point pair is counted as a
  violation if its post-fit distance exceeds this threshold.
:param global_ratio: Maximum fraction of currently considered aligned points
  that may be excluded as outliers in a single iteration.
:param viol_ratio: Maximum fraction of currently violating points that may be
  excluded in a single iteration.
:returns: The result of the Match-Maker alignment.

.. note::
  The two point clouds must have the same number of points, and are assumed to
  be in correspondence by index. Each point should represent a single residue
  (e.g., ``CA`` atom) in a biomolecular structure.
)doc");
}
}  // namespace
}  // namespace python_internal
}  // namespace nuri
