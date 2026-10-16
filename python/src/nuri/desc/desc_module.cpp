//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <absl/strings/str_cat.h>
#include <Eigen/Dense>
#include <pybind11/gil.h>
#include <pybind11/native_enum.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/typing.h>

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

std::string_view kind_str(SurfaceKind kind) {
  return kind == SurfaceKind::kSas ? "sas" : "ses";
}

void bind_surface_patch(py::module &m) {
  py::native_enum<SurfacePatch>(m, "SurfacePatch", "enum.IntEnum", R"doc(
Kind of an analytic surface patch: :attr:`Convex` patches lie on the atom
spheres, :attr:`Toroidal` patches on the probe rolling over two atoms and
:attr:`Concave` patches on the probe resting on three or more atoms.
)doc")
      .value("Convex", SurfacePatch::kConvex)
      .value("Toroidal", SurfacePatch::kToroidal)
      .value("Concave", SurfacePatch::kConcave)
      .finalize();
}

void bind_surface_geometry(py::module &m) {
  py::class_<internal::SurfaceGeometry> sg(m, "SurfaceGeometry", R"doc(
Analytic patch areas of a solvent-accessible or solvent-excluded surface, as
returned by :func:`analytic_surface`.
)doc");
  sg.def_property_readonly(
        "kind",
        [](const internal::SurfaceGeometry &geo) {
          return kind_str(geo.kind());
        },
        R"doc(
The kind of the surface, ``"sas"`` or ``"ses"``.
)doc")
      .def(
          "area",
          [](py::handle self, SurfacePatch patch) {
            const auto &geo = self.cast<const internal::SurfaceGeometry &>();
            return eigen_as_numpy_view(geo.area(patch), self);
          },
          kReturnsSubobject, py::arg("patch"), R"doc(
Areas of the patches of one kind, in angstroms squared.

:param patch: The kind of patches.
:returns: One area per patch. Convex areas are parallel to :attr:`order`;
  toroidal and concave areas are empty for a SAS.
)doc")
      .def(
          "total_area",
          [](const internal::SurfaceGeometry &geo,
             std::optional<SurfacePatch> patch) {
            return patch ? geo.total_area(*patch) : geo.total_area();
          },
          py::arg("patch") = py::none(), R"doc(
Total area of the surface, in angstroms squared.

:param patch: If given, sum only the patches of this kind.
:returns: The total area.
)doc");
  def_property_readonly_subobject(
      sg, "order",
      [](py::handle self) {
        const auto &geo = self.cast<const internal::SurfaceGeometry &>();
        auto h = geo.order().head(geo.convex().area.size());
        return eigen_as_numpy_view(h, self);
      },
      R"doc(
The original atom index of each convex patch, parallel to
``area(SurfacePatch.Convex)``. Atoms that own no surface (inactive ones) are
absent.
)doc");
}

void bind_surface_dots(py::module &m) {
  py::class_<SurfaceDots> sd(m, "SurfaceDots", R"doc(
Dots sampled on a solvent-accessible or solvent-excluded surface, grouped by
patch kind in the order of :class:`SurfacePatch`.
)doc");
  sd.def_property_readonly(
        "kind", [](const SurfaceDots &dots) { return kind_str(dots.kind()); },
        R"doc(
The kind of the surface, ``"sas"`` or ``"ses"``.
)doc")
      .def_property_readonly("dropped_area", &SurfaceDots::dropped_area,
                             R"doc(
Area of the patches too small to receive a dot, in angstroms squared.
)doc")
      .def("__len__", &SurfaceDots::n)
      .def("offset", &SurfaceDots::offset, py::arg("patch"), R"doc(
Index of the first dot of the patches of one kind.

:param patch: The kind of patches.
:returns: The offset of the block; the blocks follow the order of
  :class:`SurfacePatch`.
)doc")
      .def("count", &SurfaceDots::count, py::arg("patch"), R"doc(
Number of dots on the patches of one kind.

:param patch: The kind of patches.
:returns: The size of the block.
)doc");
  def_property_readonly_subobject(
      sd, "pts",
      [](py::handle self) {
        const auto &dots = self.cast<const SurfaceDots &>();
        return eigen_as_numpy_view(dots.pts(), self);
      },
      R"doc(
The dot positions, as a 2D array of shape ``(N, 3)``.
)doc");
  def_property_readonly_subobject(
      sd, "normals",
      [](py::handle self) {
        const auto &dots = self.cast<const SurfaceDots &>();
        return eigen_as_numpy_view(dots.nrm(), self);
      },
      R"doc(
The outward unit normals of the dots, as a 2D array of shape ``(N, 3)``.
)doc");
  def_property_readonly_subobject(
      sd, "areas",
      [](py::handle self) {
        const auto &dots = self.cast<const SurfaceDots &>();
        return eigen_as_numpy_view(dots.area(), self);
      },
      R"doc(
The area each dot stands for, in angstroms squared; the dots of a patch sum to
its analytic area.
)doc");
  def_property_readonly_subobject(
      sd, "atoms",
      [](py::handle self) {
        const auto &dots = self.cast<const SurfaceDots &>();
        return eigen_as_numpy_view(dots.atom(), self);
      },
      R"doc(
The index of the input atom owning each dot, as a 1D integer array.
)doc");
}

void bind_sc_side(py::module &m) {
  py::class_<ScSide> side(m, "ScSide", R"doc(
One side of a shape complementarity calculation; see
:func:`shape_complementarity`.
)doc");
  side.def_readonly("n_atoms", &ScSide::n_atoms, R"doc(
The number of atoms of this side.
)doc")
      .def_readonly("n_active", &ScSide::n_active, R"doc(
The number of atoms taking part in the interface.
)doc")
      .def_readonly("n_dots", &ScSide::n_dots, R"doc(
The number of dots sampled on the surface of the interface atoms.
)doc")
      .def_readonly("n_buried", &ScSide::n_buried, R"doc(
The number of dots buried by the other side.
)doc")
      .def_readonly("n_trimmed", &ScSide::n_trimmed, R"doc(
The number of buried dots kept after the peripheral trim.
)doc")
      .def_readonly("trimmed_area", &ScSide::trimmed_area, R"doc(
The area of the kept dots, in angstroms squared.
)doc")
      .def_readonly("d_median", &ScSide::d_median, R"doc(
The median nearest-dot distance of the kept dots, in angstroms.
)doc")
      .def_readonly("s_median", &ScSide::s_median, R"doc(
The median score of the kept dots.
)doc")
      .def_readonly("dots", &ScSide::dots, R"doc(
The kept dots of this side.
)doc");
  def_property_readonly_subobject(
      side, "distance",
      [](py::handle self) {
        const auto &s = self.cast<const ScSide &>();
        return eigen_as_numpy_view(s.distance, self);
      },
      R"doc(
Distance of each kept dot to the nearest kept dot of the other side, parallel
to :attr:`dots`.
)doc");
  def_property_readonly_subobject(
      side, "score",
      [](py::handle self) {
        const auto &s = self.cast<const ScSide &>();
        return eigen_as_numpy_view(s.score, self);
      },
      R"doc(
Score of each kept dot against the nearest kept dot of the other side,
parallel to :attr:`dots`.
)doc");
}

void bind_sc_result(py::module &m) {
  py::class_<ScResult> res(m, "ScResult", R"doc(
Shape complementarity of an interface; see :func:`shape_complementarity`.
)doc");
  res.def_readonly("sc", &ScResult::sc, R"doc(
The shape complementarity statistic: the mean of the two sides' median scores.
)doc")
      .def_readonly("distance", &ScResult::distance, R"doc(
The mean of the two sides' median nearest-dot distances, in angstroms.
)doc")
      .def_readonly("area", &ScResult::area, R"doc(
The total area of the kept dots of both sides, in angstroms squared.
)doc")
      // keep_alive cannot attach to a tuple, so sides is a plain property.
      .def_property_readonly(
          "sides",
          [](py::handle self) -> pyt::Tuple<ScSide, ScSide> {
            const auto &r = self.cast<const ScResult &>();
            return py::make_tuple(
                py::cast(r.sides[0], rvp::reference_internal, self),
                py::cast(r.sides[1], rvp::reference_internal, self));
          },
          R"doc(
The per-side summaries, in the order of the arguments.
)doc");
}

NURI_PYTHON_MODULE(m) {
  bind_surface_patch(m);
  bind_surface_geometry(m);
  bind_surface_dots(m);
  bind_sc_side(m);
  bind_sc_result(m);

  m.def(
      "analytic_surface",
      [](py::handle py_pts, py::handle py_radii, double rp,
         std::string_view kind, py::handle py_active) {
        SurfaceKind sk;
        if (kind == "sas") {
          sk = SurfaceKind::kSas;
        } else if (kind == "ses") {
          sk = SurfaceKind::kSes;
        } else {
          throw py::value_error(absl::StrCat(
              "kind must be \"sas\" or \"ses\", got \"", kind, "\""));
        }

        return with_surface(py_pts, py_radii, rp, py_active,
                            [sk](const AnalyticSurface &surf) {
                              return sk == SurfaceKind::kSas ? surf.build_sas()
                                                             : surf.build_ses();
                            });
      },
      py::arg("pts"), py::arg("radii"), py::arg("rp") = 1.4,
      py::arg("kind") = "ses", py::arg("active") = py::none(), R"doc(
Build the analytic solvent-accessible (SAS) or solvent-excluded (SES) surface
of a set of spheres.

The SES is the molecular surface of
Richards\ :footcite:`desc:surface:richards-1977` in the patch decomposition of
Connolly\ :footcite:`desc:surface:connolly-1983`, built from the
characterization of Quan and Stamm\ :footcite:`desc:surface:quan-stamm-2016`.

.. note::
  The triangulation inserts the spheres in a randomized order, so the
  numbering of patches and the last bits of per-atom areas (about 1e-14)
  change between calls; totals agree to rounding. Call
  :func:`nuri.seed_thread` first for reproducible output.

:param pts: The sphere centers, as a 2D array of shape ``(N, 3)``.
:param radii: The sphere radii, as a 1D array of shape ``(N,)``.
:param rp: The probe radius, in angstroms. Default is 1.4 angstroms.
:param kind: ``"sas"`` or ``"ses"``. A SAS has convex patches only.
:param active: Optional boolean mask of shape ``(N,)``. Active spheres own
  surface patches; the others only occlude. All spheres are active by default.
:returns: The patch areas of the surface.
:raises ValueError: If any of the shapes mismatch, any of the `radii` values or
  `rp` are not positive, `kind` is unknown, or the inputs are rejected.

.. note::
  No coordinate may exceed :math:`10^4` angstroms in absolute value; subtract
  one common shift from all inputs first.
)doc");

  m.def(
      "shape_complementarity",
      [](py::handle pts_a, py::handle radii_a, py::handle pts_b,
         py::handle radii_b, double rp, double density, double weight,
         double band, double sep, double clamp, py::handle py_active_a,
         py::handle py_active_b) {
        Spheres a = cast_spheres(pts_a, radii_a),
                b = cast_spheres(pts_b, radii_b);
        if (a.radii.size() == 0 || b.radii.size() == 0)
          throw py::value_error("each side must have at least one atom");

        const ArrayXb active_a = cast_active(py_active_a,
                                             static_cast<int>(a.radii.size())),
                      active_b = cast_active(py_active_b,
                                             static_cast<int>(b.radii.size()));

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
          res = shape_complementarity(a.pts, a.radii, active_a, b.pts, b.radii,
                                      active_b, params);
        }
        if (!res) {
          throw py::value_error(
              "no interface dot survives trimming, or the surface inputs were "
              "rejected; see log for details");
        }
        return std::move(*res);
      },
      py::arg("pts_a"), py::arg("radii_a"), py::arg("pts_b"),
      py::arg("radii_b"), py::kw_only(), py::arg("rp") = 1.7,
      py::arg("density") = 15.0, py::arg("weight") = 0.5, py::arg("band") = 1.5,
      py::arg("sep") = 8.0, py::arg("clamp") = 0.999,
      py::arg("active_a") = py::none(), py::arg("active_b") = py::none(),
      R"doc(
Shape complementarity statistic of Lawrence and
Colman\ :footcite:`desc:sc:lawrence-colman-1993` between two sets of spheres.

Each side's solvent-excluded surface is built with probe radius `rp` and
sampled at `density` dots per square angstrom. An atom takes part in the
interface when its center lies within `sep` of an atom of the other side; a
dot is buried when its probe center lies inside a solvent-accessible sphere of
the other side, and a buried dot within `band` of an exposed dot of its own
side is trimmed as peripheral. Each remaining dot scores
:math:`-(n \cdot n') \exp(-\text{weight} \, d^2)` against the nearest
remaining dot of the other side, clamped to ``[-clamp, clamp]``; the statistic
is the mean of the two sides' median scores.

:param pts_a: The atom centers of the first side, as a 2D array of shape
  ``(N, 3)``.
:param radii_a: The van der Waals radii of the first side, as a 1D array of
  shape ``(N,)``.
:param pts_b: The atom centers of the second side, as a 2D array of shape
  ``(M, 3)``.
:param radii_b: The van der Waals radii of the second side, as a 1D array of
  shape ``(M,)``.
:param rp: The probe radius, in angstroms.
:param density: The number of surface dots per square angstrom.
:param weight: The weight of the squared distance in the score.
:param band: The peripheral trim band, in angstroms.
:param sep: The interface atom cutoff, in angstroms.
:param clamp: The magnitude the scores are clamped to, in ``(0, 1]``.
:param active_a: Optional boolean mask of shape ``(N,)``. Only these atoms of
  the first side may take part in the interface; the others only occlude. All
  atoms may take part by default.
:param active_b: Optional boolean mask of shape ``(M,)`` for the second side.
:returns: The statistic with its per-side summaries.
:raises ValueError: If any of the shapes mismatch, a side is empty, any radii
  or `rp`, `density`, `weight`, `band` or `sep` are not positive, `clamp` is
  outside ``(0, 1]``, the surface inputs are rejected, or no interface dot of
  either side survives trimming (which includes a mask that leaves no atom).

.. note::
  The default parameters are those of Rosetta's ``sc`` application.

.. note::
  No coordinate may exceed :math:`10^4` angstroms in absolute value; subtract
  one common shift from all inputs first.

.. note::
  The surface triangulation inserts the spheres in a randomized order, so the
  dot order and the last bits of the result (about 1e-14) change between
  calls. Call :func:`nuri.seed_thread` first for reproducible output.
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
            Spheres sph = cast_spheres(py_pts, py_radii);
            sr_sasa_validate_common_args(nprobe, rprobe);

            ArrayXd sasa;
            {
              py::gil_scoped_release rel;
              ArrayXd rcuts = sph.radii + rprobe;
              sasa = internal::sr_sasa_impl(sph.pts, rcuts, nprobe,
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
