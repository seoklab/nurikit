//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/strings/str_cat.h>
#include <Eigen/Dense>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>

#include "nuri/eigen_config.h"
#include "nuri/desc/surface.h"
#include "nuri/python/core/core_module.h"
#include "nuri/python/utils.h"

namespace nuri {
namespace python_internal {
namespace {
void sr_sasa_validate_common_args(int nprobe, double rprobe) {
  check_positive(nprobe, "nprobe");
  check_positive(rprobe, "rprobe");
}

template <class T, class F>
auto collect(const std::vector<T> &recs, const F &field) {
  using R = std::decay_t<decltype(field(recs[0]))>;
  using Arr =
      std::conditional_t<std::is_same_v<R, Vector3d>, Matrix3Xd, ArrayX<R>>;
  Arr out;
  if constexpr (std::is_same_v<R, Vector3d>) {
    out.resize(3, recs.size());
    for (size_t i = 0; i < recs.size(); ++i)
      out.col(i) = field(recs[i]);
  } else {
    out.resize(recs.size());
    for (size_t i = 0; i < recs.size(); ++i)
      out[i] = field(recs[i]);
  }
  return eigen_as_numpy(out);
}

py::dict csr_dict(const internal::CSR &csr) {
  py::dict d;
  d["adj"] = eigen_as_numpy(csr.adj());
  d["off"] = eigen_as_numpy(csr.off());
  return d;
}

struct SasResult {
  internal::SaPrep sa;
  internal::SasGeometry geo;
};

template <class F>
SasResult compute_sas(py::handle py_pts, py::handle py_radii, double rp,
                      py::handle py_active, const F &then) {
  auto pts = py_array_cast<3>(py_pts);
  auto radii = py_array_cast<E::Dynamic, 1>(py_radii);
  const int n = static_cast<int>(radii.eigen().size());

  if (pts.eigen().cols() != n) {
    throw py::value_error(absl::StrCat("number of points (", pts.eigen().cols(),
                                       ") does not match number of radii (", n,
                                       ")"));
  }
  if (!(radii.eigen().array() > 0).all())
    throw py::value_error("all radii must be positive values");
  check_positive(rp, "rp");

  ArrayXb active = ArrayXb::Constant(n, true);
  if (!py_active.is_none()) {
    auto mask = py_array_cast<E::Dynamic, 1, bool>(py_active);
    if (mask.eigen().size() != n)
      throw py::value_error("active mask size does not match number of atoms");
    active = mask.eigen();
  }

  std::optional<internal::SaPrep> sa;
  internal::SasGeometry geo;
  {
    py::gil_scoped_release rel;
    Matrix3Xd p = pts.eigen();
    ArrayXd sar = radii.eigen().array() + rp;
    sa = internal::prepare(p, sar, active, rp);
    if (sa) {
      internal::SasDelaunay del = internal::triangulate(*sa);
      geo = internal::build_sas(*sa, del);
      then(*sa, del, geo);
    }
  }
  if (!sa)
    throw py::value_error("preparation failed; see log for details");

  return { std::move(*sa), std::move(geo) };
}

py::dict sas_dict(const internal::SaPrep &sa,
                  const internal::SasGeometry &geo) {
  py::dict d;
  d["pts"] = eigen_as_numpy(sa.pts);
  d["sar"] = eigen_as_numpy(sa.sar);
  d["order"] = eigen_as_numpy(sa.order);
  d["g"] = csr_dict(sa.g);
  d["d"] = eigen_as_numpy(sa.d);
  d["n_active"] = sa.n_active;
  d["n_solve"] = sa.n_solve;
  d["n_enum"] = sa.n_enum;

  py::dict circ;
  circ["i"] = collect(geo.circles, [](const auto &c) { return c.i; });
  circ["j"] = collect(geo.circles, [](const auto &c) { return c.j; });
  circ["a"] = collect(geo.circles, [](const auto &c) { return c.a; });
  circ["rl"] = collect(geo.circles, [](const auto &c) { return c.rl; });
  circ["axis"] = collect(geo.circles, [](const auto &c) { return c.axis; });
  circ["cntr"] = collect(geo.circles, [](const auto &c) { return c.cntr; });
  circ["e1"] = collect(geo.circles, [](const auto &c) { return c.e1; });
  circ["e2"] = collect(geo.circles, [](const auto &c) { return c.e2; });
  d["circles"] = circ;

  py::dict caps;
  caps["h"] = csr_dict(geo.caps.h);
  caps["axis"] = eigen_as_numpy(geo.caps.axis);
  caps["cosa"] = eigen_as_numpy(geo.caps.cosa);
  caps["sina"] = eigen_as_numpy(geo.caps.sina);
  d["caps"] = caps;

  py::dict probes;
  probes["atoms"] = csr_dict(geo.probes.atoms);
  probes["pos"] = eigen_as_numpy(geo.probes.pos);
  probes["tan"] = eigen_as_numpy(geo.probes.tan);
  probes["tan_off"] = eigen_as_numpy(geo.probes.tan_off.off());
  probes["n_active"] = geo.probes.n_active;
  d["probes"] = probes;

  py::dict arcs;
  arcs["phi"] = collect(geo.arcs, [](const auto &a) { return a.phi; });
  arcs["dphi"] = collect(geo.arcs, [](const auto &a) { return a.dphi; });
  arcs["circ"] = collect(geo.arcs, [](const auto &a) { return a.circ; });
  arcs["beg"] = collect(geo.arcs, [](const auto &a) { return a.beg; });
  arcs["end"] = collect(geo.arcs, [](const auto &a) { return a.end; });
  arcs["n_active"] = geo.n_active_arcs;
  d["arcs"] = arcs;

  d["area"] = eigen_as_numpy(geo.area);
  return d;
}

py::dict ses_dict(const internal::SesGeometry &ses) {
  py::dict d;
  d["rp"] = ses.rp;
  d["convex_area"] = eigen_as_numpy(ses.convex_area);
  d["saddle_beta"] = eigen_as_numpy(ses.saddle_beta);
  d["saddle_integral"] = eigen_as_numpy(ses.saddle_integral);
  d["saddle_area"] = eigen_as_numpy(ses.saddle_area);
  d["face_area"] = eigen_as_numpy(ses.face_area);
  d["face_off"] = eigen_as_numpy(ses.face_off.off());
  d["face_axis"] = eigen_as_numpy(ses.face_axis);
  d["face_cosa"] = eigen_as_numpy(ses.face_cosa);
  d["face_sina"] = eigen_as_numpy(ses.face_sina);
  return d;
}

py::dict sas_geometry(py::handle py_pts, py::handle py_radii, double rp,
                      py::handle py_active) {
  SasResult res = compute_sas(py_pts, py_radii, rp, py_active,
                              [](const auto &, const auto &, const auto &) { });
  return sas_dict(res.sa, res.geo);
}

py::dict ses_geometry(py::handle py_pts, py::handle py_radii, double rp,
                      py::handle py_active) {
  internal::SesGeometry ses;
  SasResult res = compute_sas(py_pts, py_radii, rp, py_active,
                              [&](const internal::SaPrep &sa,
                                  const internal::SasDelaunay &del,
                                  const internal::SasGeometry &geo) {
                                ses = internal::build_ses(sa, del, geo, rp);
                              });
  py::dict d = sas_dict(res.sa, res.geo);
  d["ses"] = ses_dict(ses);
  return d;
}

py::dict ses_dots(py::handle py_pts, py::handle py_radii, double rp,
                  double density, py::handle py_active) {
  check_positive(density, "density");

  internal::SesDots dots;
  compute_sas(py_pts, py_radii, rp, py_active,
              [&](const internal::SaPrep &sa, const internal::SasDelaunay &del,
                  const internal::SasGeometry &geo) {
                const internal::SesGeometry ses =
                    internal::build_ses(sa, del, geo, rp);
                dots = internal::sample_ses(sa, geo, ses, density);
              });

  py::dict d;
  d["pts"] = eigen_as_numpy(dots.pts);
  d["nrm"] = eigen_as_numpy(dots.nrm);
  d["area"] = eigen_as_numpy(dots.area);
  d["atom"] = eigen_as_numpy(dots.atom);
  d["kind_off"] = eigen_as_numpy(dots.kind.off());
  d["rp"] = dots.rp;
  d["dropped_area"] = dots.dropped_area;
  return d;
}

NURI_PYTHON_MODULE(m) {
  m.def("_sas_geometry", &sas_geometry, py::arg("pts"), py::arg("radii"),
        py::arg("rp"), py::arg("active") = py::none(), R"doc(
Experimental: analytic SAS geometry of a set of spheres. Returns a dict of
flat arrays mirroring the C++ internal structures; atoms are reordered by
``order`` (new to old).
)doc");

  m.def("_ses_geometry", &ses_geometry, py::arg("pts"), py::arg("radii"),
        py::arg("rp"), py::arg("active") = py::none(), R"doc(
Experimental: ``_sas_geometry`` plus the SES patch areas under ``"ses"``.
)doc");

  m.def("_ses_dots", &ses_dots, py::arg("pts"), py::arg("radii"), py::arg("rp"),
        py::arg("density"), py::arg("active") = py::none(),
        R"doc(
Experimental: dots sampled on the SES at roughly ``density`` dots per square
angstrom, as a dict of flat arrays: ``pts`` and ``nrm`` (outward unit
normals), ``area`` (weights summing to each patch's analytic area), ``atom``
(original index of the owning atom), ``kind_off`` (offsets of the convex,
toroidal and concave blocks), ``rp`` and ``dropped_area``.
)doc");

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
