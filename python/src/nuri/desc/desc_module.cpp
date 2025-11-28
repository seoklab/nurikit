//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <absl/strings/str_cat.h>
#include <Eigen/Dense>
#include <pybind11/gil.h>

#include "nuri/eigen_config.h"
#include "nuri/desc/surface.h"
#include "nuri/python/core/core_module.h"
#include "nuri/python/utils.h"

namespace nuri {
namespace python_internal {
namespace {
void sr_sasa_validate_common_args(int nprobe, double rprobe) {
  if (nprobe <= 0) {
    throw py::value_error(
        absl::StrCat("number of probes must be positive, got ", nprobe));
  }

  if (!(rprobe > 0)) {
    throw py::value_error(
        absl::StrCat("radius of probes must be positive, got ", rprobe));
  }
}

NURI_PYTHON_MODULE(m) {
  m.def(
       "shrake_rupley_sasa",
       [&](const PyMol &mol, int ci, int nprobe, double rprobe) {
         int conf = check_conf(*mol, ci);
         sr_sasa_validate_common_args(nprobe, rprobe);

         ArrayXd sasa;
         {
           py::gil_scoped_release rel;
           sasa = shrake_rupley_sasa(*mol, mol->confs()[conf], nprobe, rprobe);
         }
         return eigen_as_numpy(sasa);
       },
       py::arg("mol"), py::arg("conf") = 0, py::arg("nprobe") = 92,
       py::arg("rprobe") = 1.4, R"doc(
Calculate the Solvent-Accessible Surface Area (SASA) of a molecule conformation
using the Shrake-Rupley algorithm.

:param mol: The input molecule.
:param conf: The conformation index. If not specified, uses the first
  conformation.
:param nprobe: The number of probe spheres. Default is 92.
:param rprobe: The radius of the probe spheres. Default is 1.4 angstroms.
:returns: The calculated SASA values per atom (in angstroms squared).
:raises IndexError: If the conformation index is out of range.
:raises ValueError: If `nprobe` or `rprobe` is not positive.

.. note::
  This function does not automatically handle implicit hydrogens. If the
  molecule contains implicit hydrogens, consider revealing them before calling
  this function for accurate results
  (see :func:`nuri.core.Molecule.reveal_hydrogens`).
)doc")
      .def(
          "shrake_rupley_sasa",
          [&](py::handle py_pts, py::handle py_radii, int nprobe,
              double rprobe) {
            auto pts = py_array_cast<3>(py_pts);
            auto radii = py_array_cast<E::Dynamic, 1>(py_radii);

            if (pts.eigen().cols() != radii.eigen().size()) {
              throw py::value_error(
                  absl::StrCat("number of points (", pts.eigen().cols(),
                               ") does not match number of radii (",
                               radii.eigen().size(), ")"));
            }

            if (!(radii.eigen().array() > 0).all()) {
              throw py::value_error("all radii must be positive values");
            }

            sr_sasa_validate_common_args(nprobe, rprobe);

            ArrayXd sasa;
            {
              py::gil_scoped_release rel;
              ArrayXd rcuts = radii.eigen().array() + rprobe;
              sasa = internal::sr_sasa_impl(pts.eigen(), rcuts, nprobe,
                                            internal::SrSasaMethod::kAuto);
            }
            return eigen_as_numpy(sasa);
          },
          py::arg("pts"), py::arg("radii"), py::arg("nprobe") = 92,
          py::arg("rprobe") = 1.4,
          R"doc(
Calculate the Solvent-Accessible Surface Area (SASA) of a molecule conformation
using the Shrake-Rupley algorithm.

:param pts: The coordinates of the atoms, as a 2D array of shape ``(N, 3)``.
:param radii: The radii of the atoms, as a 1D array of shape ``(N,)``.
:param nprobe: The number of probe spheres. Default is 92.
:param rprobe: The radius of the probe spheres. Default is 1.4 angstroms.
:returns: The calculated SASA values per atom (in angstroms squared).
:raises ValueError: If the number of `pts` and `radii` do not match, any `radii`
  are not positive, `nprobe` is not positive, or `rprobe` is not positive.
)doc");
}
}  // namespace
}  // namespace python_internal
}  // namespace nuri
