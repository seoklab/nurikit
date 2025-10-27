//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <optional>
#include <utility>
#include <vector>

#include <absl/strings/str_cat.h>
#include <Eigen/Dense>
#include <pybind11/cast.h>
#include <pybind11/eigen.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/typing.h>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/python/core/core_module.h"
#include "nuri/python/utils.h"
#include "nuri/tools/galign.h"

namespace nuri {
namespace python_internal {
namespace {
const Matrix3Xd &galign_try_get_conf(const PyMol &mol,
                                     std::optional<int> conf) {
  if (mol->confs().empty()) {
    throw pybind11::value_error(
        "GAlign requires at least one 3D conformation in the molecule");
  }

  int ci = py_check_index(static_cast<int>(mol->confs().size()),
                          conf.value_or(0), "Invalid conformation index");
  return mol->confs()[ci];
}

GARigidMolInfo galign_init(const PyMol &mol, std::optional<int> conf,
                           double vdw_scale, double hetero_scale, int dcut) {
  if (mol->size() < 3) {
    throw pybind11::value_error(absl::StrCat(
        "GAlign requires at least 3 atoms in the template molecule, got ",
        mol->size()));
  }

  const Matrix3Xd &ref = galign_try_get_conf(mol, conf);

  if (vdw_scale <= 0.0) {
    throw pybind11::value_error(
        absl::StrCat("vdw_scale must be positive, got ", vdw_scale));
  }
  if (hetero_scale <= 0.0) {
    throw pybind11::value_error(
        absl::StrCat("hetero_scale must be positive, got ", hetero_scale));
  }
  if (dcut < 1) {
    throw pybind11::value_error(
        absl::StrCat("dcut must be at least 1 angstrom, got ", dcut));
  }

  GARigidMolInfo galign(*mol, ref, vdw_scale, hetero_scale, dcut);
  return galign;
}

pyt::List<GAlignResult>
galign_align(const GARigidMolInfo &self, const PyMol &query, bool flexible,
             int max_conf, std::optional<int> conf, double max_trs,
             double max_rot, double max_tors, double rigid_min_rmsd,
             int rigid_max_conf, int pool_size, int sample_size, int max_gen,
             int patience, int mut_cnt, double mut_prob, double ftol,
             int max_iters) {
  const Matrix3Xd &seed = galign_try_get_conf(query, conf);

  if (max_conf < 1) {
    throw pybind11::value_error(
        absl::StrCat("max_confs must be at least 1, got ", max_conf));
  }

  GAMinimizeArgs margs;
  if (flexible) {
    if (max_trs < 0.0) {
      throw pybind11::value_error(
          absl::StrCat("max_translation must be nonnegative, got ", max_trs));
    }
    if (max_rot < 0.0) {
      throw pybind11::value_error(
          absl::StrCat("max_rotation must be nonnegative, got ", max_rot));
    }
    if (max_tors < 0.0) {
      throw pybind11::value_error(
          absl::StrCat("max_torsion must be nonnegative, got ", max_tors));
    }
    if (pool_size < 1) {
      throw pybind11::value_error(
          absl::StrCat("pool_size must be at least 1, got ", pool_size));
    }
    if (sample_size < 1) {
      throw pybind11::value_error(
          absl::StrCat("sample_size must be at least 1, got ", sample_size));
    }
    if (max_gen < 1) {
      throw pybind11::value_error(
          absl::StrCat("max_generations must be at least 1, got ", max_gen));
    }
    if (mut_cnt < 0) {
      throw pybind11::value_error(
          absl::StrCat("n_mutation must be nonnegative, got ", mut_cnt));
    }
    if (mut_prob < 0.0 || mut_prob > 1.0) {
      throw pybind11::value_error(
          absl::StrCat("p_mutation must be between 0 and 1, got ", mut_prob));
    }
    if (ftol <= 0.0) {
      throw pybind11::value_error(
          absl::StrCat("opt_ftol must be positive, got ", ftol));
    }
    if (max_iters < 1) {
      throw pybind11::value_error(
          absl::StrCat("opt_max_iters must be at least 1, got ", max_iters));
    }

    margs.ftol = ftol;
    margs.max_iters = max_iters;
  }

  std::vector<GAlignResult> results =
      galign(*query, seed, self, flexible, max_conf,
             { max_trs, max_rot, max_tors, rigid_min_rmsd * rigid_min_rmsd,
               rigid_max_conf, pool_size, sample_size, max_gen, patience,
               mut_cnt, mut_prob },
             margs);

  py::gil_scoped_acquire lock;
  pyt::List<GAlignResult> py_results(results.size());
  for (int i = 0; i < results.size(); ++i)
    py_results[i] = std::move(results[i]);
  return py_results;
}

NURI_PYTHON_MODULE(m) {
  // For types
  py::module_::import("nuri.core");

  py::class_<GAlignResult>(m, "GAlignResult")
      .def_property_readonly(
          "pos", [](const GAlignResult &r) { return eigen_as_numpy(r.conf); },
          R"doc(
A copy of the aligned conformation as a 2D numpy array of shape
``(N, 3)``, where ``N`` is the number of atoms in the query molecule.
)doc")
      .def_readonly("score", &GAlignResult::align_score,
                    R"doc(
The alignment score (shape overlap) of this result.
)doc");

  py::class_<GARigidMolInfo>(m, "GAlign")
      .def(py::init(&galign_init), py::arg("templ"),
           py::kw_only(),  //
           py::arg("conf") = py::none(), py::arg("vdw_scale") = 0.8,
           py::arg("hetero_scale") = 0.7, py::arg("dcut") = 6,
           R"doc(
Prepare GAlign algorithm with the given template structure.

:param templ: The template structure. Must have at least 3 atoms and 3D
  coordinates.
:param conf: The conformation index to use as the template. If not provided,
  the first conformation is used.
:param vdw_scale: The scale factor for van der Waals radii when calculating
  shape overlap score.
:param hetero_scale: The scale factor for atom type mismatch when calculating
  shape overlap score.
:param dcut: The distance cutoff for neighbor search, in angstroms.

:raises ValueError: If the template structure has less than 3 atoms or no 3D
  conformation, or if invalid parameters are provided (e.g., negative dcut).
:raises IndexError: If the provided conformation index is out of range.
)doc",
           kThreadSafe)
      .def("align", galign_align, py::arg("query"), py::arg("flexible") = true,
           py::arg("max_confs") = 1,
           py::kw_only(),  //
           py::arg("conf") = py::none(), py::arg("max_translation") = 2.5,
           py::arg("max_rotation") = deg2rad(120),
           py::arg("max_torsion") = deg2rad(120),
           py::arg("rigid_min_msd") = 9.0, py::arg("rigid_max_confs") = 4,
           py::arg("pool_size") = 10, py::arg("sample_size") = 30,
           py::arg("max_generations") = 50, py::arg("patience") = 5,
           py::arg("n_mutation") = 5, py::arg("p_mutation") = 0.5,
           py::arg("opt_ftol") = 1e-2, py::arg("opt_max_iters") = 300,
           R"doc(
Align the given query molecule to the template structure.

:param query: The query molecule to be aligned. Must have at least one 3D
  conformation.
:param flexible: Whether to perform flexible alignment. When ``False``, only
  rigid alignment is performed and the flexible alignment parameters are ignored.
:param max_confs: The maximum number of alignment results to return.
:param conf: The conformation index to use as the query structure. If not
  provided, the first conformation is used.
:param vdw_scale: The scale factor for van der Waals radii when calculating
  shape overlap score.
:param hetero_scale: The scale factor for atom type mismatch when calculating
  shape overlap score.
:param dcut: The distance cutoff for neighbor search, in angstroms.
:param max_translation: The maximum translation allowed during flexible
  alignment, in angstroms.
:param max_rotation: The maximum rotation allowed during flexible alignment,
  in radians.
:param max_torsion: The maximum torsion angle change allowed during flexible
  alignment, in radians.
:param rigid_min_rmsd: The minimum root-mean-squared deviation between different
  conformations to consider them as distinct during rigid alignment.
:param rigid_max_confs: The maximum number of conformations to consider for
  initial rigid alignment. Ignored if in rigid mode; set ``max_confs`` instead.
:param pool_size: The size of the population pool during flexible alignment.
:param sample_size: The number of new trial conformations to sample in each
  generation.
:param max_generations: The maximum number of generations to run.
:param patience: The number of generations to wait for improvement before
  early stopping.
:param n_mutation: The number of mutation operations to perform when generating
  new trial conformations.
:param p_mutation: The probability of mutation when generating new trial
  conformations.
:param opt_ftol: The function tolerance for the Nelder-Mead optimization.
:param opt_max_iters: The maximum number of iterations for the Nelder-Mead
  optimization.

:returns: At most ``max_confs`` alignment results as a list of
  :class:`GAlignResult` objects, sorted by their alignment scores in
  descending order.

:raises ValueError: If the query molecule has no 3D conformation, or if
  invalid parameters are provided (e.g., negative max_translation).
:raises IndexError: If the provided conformation index is out of range.
)doc",
           kThreadSafe);

  m.def(
      "galign",
      [](const PyMol &query, const PyMol &templ, bool flexible, int max_conf,
         std::optional<int> qconf, std::optional<int> tconf, double vdw_scale,
         double hetero_scale, int dcut, double max_trs, double max_rot,
         double max_tors, double rigid_min_rmsd, int rigid_max_conf,
         int pool_size, int sample_size, int max_gen, int patience, int mut_cnt,
         double mut_prob, double ftol, int max_iters) {
        const GARigidMolInfo tinfo =
            galign_init(templ, tconf, vdw_scale, hetero_scale, dcut);

        return galign_align(tinfo, query, flexible, max_conf, qconf, max_trs,
                            max_rot, max_tors, rigid_min_rmsd, rigid_max_conf,
                            pool_size, sample_size, max_gen, patience, mut_cnt,
                            mut_prob, ftol, max_iters);
      },
      py::arg("query"), py::arg("templ"), py::arg("flexible") = true,
      py::arg("max_confs") = 1,
      py::kw_only(),  //
      py::arg("qconf") = py::none(), py::arg("tconf") = py::none(),
      py::arg("vdw_scale") = 0.8, py::arg("hetero_scale") = 0.7,
      py::arg("dcut") = 6, py::arg("max_translation") = 2.5,
      py::arg("max_rotation") = deg2rad(120),
      py::arg("max_torsion") = deg2rad(120), py::arg("rigid_min_msd") = 9.0,
      py::arg("rigid_max_confs") = 4, py::arg("pool_size") = 10,
      py::arg("sample_size") = 30, py::arg("max_generations") = 50,
      py::arg("patience") = 5, py::arg("n_mutation") = 5,
      py::arg("p_mutation") = 0.5, py::arg("opt_ftol") = 1e-2,
      py::arg("opt_max_iters") = 300,
      R"doc(
Align the given query molecule to the template structure.

:param query: The query molecule to be aligned. Must have at least one 3D
  conformation.
:param templ: The template structure. Must have at least 3 atoms and 3D
  coordinates.
:param flexible: Whether to perform flexible alignment. When ``False``, only
  rigid alignment is performed and the flexible alignment parameters are ignored.
:param max_confs: The maximum number of alignment results to return.
:param qconf: The conformation index to use as the query structure. If not
  provided, the first conformation is used.
:param tconf: The conformation index to use as the template structure. If not
  provided, the first conformation is used.
:param vdw_scale: The scale factor for van der Waals radii when calculating
  shape overlap score.
:param hetero_scale: The scale factor for atom type mismatch when calculating
  shape overlap score.
:param dcut: The distance cutoff for neighbor search, in angstroms.
:param max_translation: The maximum translation allowed during flexible
  alignment, in angstroms.
:param max_rotation: The maximum rotation allowed during flexible alignment,
  in radians.
:param max_torsion: The maximum torsion angle change allowed during flexible
  alignment, in radians.
:param rigid_min_rmsd: The minimum root-mean-squared deviation between different
  conformations to consider them as distinct during rigid alignment.
:param rigid_max_confs: The maximum number of conformations to consider for
  initial rigid alignment. Ignored if in rigid mode; set ``max_confs`` instead.
:param pool_size: The size of the population pool during flexible alignment.
:param sample_size: The number of new trial conformations to sample in each
  generation.
:param max_generations: The maximum number of generations to run.
:param patience: The number of generations to wait for improvement before
  early stopping.
:param n_mutation: The number of mutation operations to perform when generating
  new trial conformations.
:param p_mutation: The probability of mutation when generating new trial
  conformations.
:param opt_ftol: The function tolerance for the Nelder-Mead optimization.
:param opt_max_iters: The maximum number of iterations for the Nelder-Mead
  optimization.

:returns: At most ``max_confs`` alignment results as a list of
  :class:`GAlignResult` objects, sorted by their alignment scores in
  descending order.

:raises ValueError: If the query or template molecule is invalid, or if
  any of the parameters are invalid (e.g., negative max_translation).
:raises IndexError: If the provided conformation index is out of range.
)doc",
      kThreadSafe);
}
}  // namespace
}  // namespace python_internal
}  // namespace nuri
