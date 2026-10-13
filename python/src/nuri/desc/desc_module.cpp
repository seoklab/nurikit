//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <optional>
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

struct Spheres {
  Matrix3Xd pts;
  ArrayXd radii;
};

Spheres cast_spheres(py::handle py_pts, py::handle py_radii) {
  auto pts = py_array_cast<3>(py_pts);
  auto radii = py_array_cast<E::Dynamic, 1>(py_radii);

  if (pts.eigen().cols() != radii.eigen().size()) {
    throw py::value_error(absl::StrCat("number of points (", pts.eigen().cols(),
                                       ") does not match number of radii (",
                                       radii.eigen().size(), ")"));
  }
  if (!(radii.eigen().array() > 0).all())
    throw py::value_error("all radii must be positive values");

  return { pts.eigen(), radii.eigen().array() };
}

ArrayXb cast_active(py::handle py_active, const int n) {
  if (py_active.is_none())
    return ArrayXb::Constant(n, true);

  auto mask = py_array_cast<E::Dynamic, 1, bool>(py_active);
  if (mask.eigen().size() != n)
    throw py::value_error("active mask size does not match number of atoms");
  return mask.eigen();
}

/**
 * Runs `then(surface)` without the GIL on the initialized surface of the
 * given spheres.
 */
template <class F>
auto with_surface(py::handle py_pts, py::handle py_radii, double rp,
                  py::handle py_active, const F &then) {
  const Spheres spheres = cast_spheres(py_pts, py_radii);
  const int n = static_cast<int>(spheres.radii.size());
  check_positive(rp, "rp");
  const ArrayXb active = cast_active(py_active, n);

  std::optional<decltype(then(std::declval<const AnalyticSurface &>()))> res;
  {
    py::gil_scoped_release rel;
    AnalyticSurface surf(spheres.pts, spheres.radii);
    if (surf.initialize(rp, active))
      res = then(surf);
  }
  if (!res)
    throw py::value_error("preparation failed; see log for details");
  return std::move(*res);
}

py::dict geometry_dict(const internal::SurfaceGeometry &geo) {
  py::dict d;
  d["kind"] = geo.kind() == SurfaceKind::kSas ? "sas" : "ses";
  d["rp"] = geo.rp();
  d["n_atoms"] = geo.n_atoms();
  d["order"] = eigen_as_numpy(geo.order());
  d["area_convex"] = eigen_as_numpy(geo.area(SurfacePatch::kConvex));
  d["area_toroidal"] = eigen_as_numpy(geo.area(SurfacePatch::kToroidal));
  d["area_concave"] = eigen_as_numpy(geo.area(SurfacePatch::kConcave));
  d["total_area"] = geo.total_area();
  return d;
}

py::dict dots_dict(const SurfaceDots &dots) {
  Array4i kind_off;
  kind_off << dots.offset(SurfacePatch::kConvex),
      dots.offset(SurfacePatch::kToroidal), dots.offset(SurfacePatch::kConcave),
      dots.n();

  py::dict d;
  d["pts"] = eigen_as_numpy(dots.pts());
  d["nrm"] = eigen_as_numpy(dots.nrm());
  d["area"] = eigen_as_numpy(dots.area());
  d["atom"] = eigen_as_numpy(dots.atom());
  d["kind_off"] = eigen_as_numpy(kind_off);
  d["rp"] = dots.rp();
  d["dropped_area"] = dots.dropped_area();
  return d;
}

py::dict sas_geometry(py::handle py_pts, py::handle py_radii, double rp,
                      py::handle py_active) {
  return geometry_dict(with_surface(py_pts, py_radii, rp, py_active,
                                    [](const AnalyticSurface &surf) {
                                      return surf.build_sas();
                                    }));
}

py::dict ses_geometry(py::handle py_pts, py::handle py_radii, double rp,
                      py::handle py_active) {
  return geometry_dict(with_surface(py_pts, py_radii, rp, py_active,
                                    [](const AnalyticSurface &surf) {
                                      return surf.build_ses();
                                    }));
}

py::dict sas_dots(py::handle py_pts, py::handle py_radii, double rp,
                  double density, py::handle py_active) {
  check_positive(density, "density");
  return dots_dict(with_surface(py_pts, py_radii, rp, py_active,
                                [&](const AnalyticSurface &surf) {
                                  SurfaceSampler sampler(density);
                                  return sampler.sample(surf.build_sas());
                                }));
}

py::dict ses_dots(py::handle py_pts, py::handle py_radii, double rp,
                  double density, py::handle py_active) {
  check_positive(density, "density");
  return dots_dict(with_surface(py_pts, py_radii, rp, py_active,
                                [&](const AnalyticSurface &surf) {
                                  SurfaceSampler sampler(density);
                                  return sampler.sample(surf.build_ses());
                                }));
}

py::dict sc_side_dict(const ScSide &s) {
  py::dict d;
  d["n_atoms"] = s.n_atoms;
  d["n_active"] = s.n_active;
  d["n_dots"] = s.n_dots;
  d["n_buried"] = s.n_buried;
  d["n_trimmed"] = s.n_trimmed;
  d["trimmed_area"] = s.trimmed_area;
  d["d_median"] = s.d_median;
  d["s_median"] = s.s_median;
  return d;
}

py::dict shape_complementarity_py(py::handle pts_a, py::handle radii_a,
                                  py::handle pts_b, py::handle radii_b,
                                  double rp, double density, double weight,
                                  double band, double sep, double clamp) {
  Spheres a = cast_spheres(pts_a, radii_a), b = cast_spheres(pts_b, radii_b);
  if (a.radii.size() == 0 || b.radii.size() == 0)
    throw py::value_error("each side must have at least one atom");

  const ScParams params {
    check_positive(rp, "rp"),
    check_positive(density, "density"),
    check_positive(weight, "weight"),
    check_positive(band, "band"),
    check_positive(sep, "sep"),
    check_interval(clamp, "clamp", Bounds::kLeftOpen, 0.0, 1.0),
  };

  std::optional<ScResult> res;
  {
    py::gil_scoped_release rel;
    res = shape_complementarity(a.pts, a.radii, b.pts, b.radii, params);
  }
  if (!res) {
    throw py::value_error(
        "no interface dots survive trimming, or surface "
        "preparation failed; see log for details");
  }

  py::list sides;
  for (const ScSide &s: res->sides)
    sides.append(sc_side_dict(s));

  py::dict d;
  d["sc"] = res->sc;
  d["distance"] = res->distance;
  d["area"] = res->area;
  d["sides"] = std::move(sides);
  return d;
}

NURI_PYTHON_MODULE(m) {
  m.def("_sas_geometry", &sas_geometry, py::arg("pts"), py::arg("radii"),
        py::arg("rp"), py::arg("active") = py::none(), R"doc(
Experimental: analytic SAS of a set of spheres, as a dict of the public
view: ``kind``, ``rp``, ``n_atoms`` (kept spheres), ``order`` (kept sphere
to original atom), ``area_convex`` (per active sphere), ``area_toroidal``,
``area_concave`` and ``total_area``.
)doc");
  m.def("_ses_geometry", &ses_geometry, py::arg("pts"), py::arg("radii"),
        py::arg("rp"), py::arg("active") = py::none(), R"doc(
Experimental: ``_sas_geometry`` for the SES; ``area_toroidal`` is per active
arc and ``area_concave`` per active probe.
)doc");
  m.def("_sas_dots", &sas_dots, py::arg("pts"), py::arg("radii"), py::arg("rp"),
        py::arg("density"), py::arg("active") = py::none(),
        R"doc(
Experimental: ``_ses_dots`` on the SAS; the dots are the probe centres, so only
the convex block is populated.
)doc");
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

  m.def("_shape_complementarity", &shape_complementarity_py, py::arg("pts_a"),
        py::arg("radii_a"), py::arg("pts_b"), py::arg("radii_b"),
        py::arg("rp") = 1.7, py::arg("density") = 15.0, py::arg("weight") = 0.5,
        py::arg("band") = 1.5, py::arg("sep") = 8.0, py::arg("clamp") = 0.999,
        R"doc(
Experimental: shape complementarity statistic of Lawrence and Colman (1993)
between two sets of spheres, as a dict with ``sc``, ``distance``, ``area`` and
``sides`` (two dicts of per-side counts, ``trimmed_area``, ``d_median`` and
``s_median``). Raises ``ValueError`` when no interface dot survives trimming.
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
