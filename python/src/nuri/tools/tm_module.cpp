//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <absl/algorithm/container.h>
#include <absl/strings/str_cat.h>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/typing.h>

#include "nuri/eigen_config.h"
#include "nuri/python/utils.h"
#include "nuri/tools/tm.h"
#include "nuri/utils.h"

namespace nuri {
namespace python_internal {
namespace {
template <class... Args>
TMAlign tmalign_try_construct_init(ConstRef<Matrix3Xd> x, ConstRef<Matrix3Xd> y,
                                   Args &&...args) {
  if (x.cols() < 5 || y.cols() < 5) {
    throw py::value_error(
        absl::StrCat("TM-align requires at least 5 residues (query: ", x.cols(),
                     ", templ: ", y.cols(), ")"));
  }

  TMAlign tm(x, y);
  if (!tm.initialize(std::forward<Args>(args)...))
    throw py::value_error("TM-align initialization failed");
  if (!tm.initialized()) {
    throw std::runtime_error(
        "TM-align initialization succeeded but no alignment was found");
  }

  return tm;
}

Eigen::Map<const ArrayXc> str_eigen_view(std::string_view str) {
  return Eigen::Map<const ArrayXc>(
      reinterpret_cast<const std::int8_t *>(str.data()),
      static_cast<Eigen::Index>(str.size()));
}

ArrayXc resolve_sec_str(ConstRef<Matrix3Xd> pts,
                        std::optional<std::string_view> py_ss, Matrix3Xd &buf,
                        std::string_view type) {
  if (!py_ss)
    return internal::assign_secstr_approx_full(pts, buf);

  if (py_ss->size() != pts.cols()) {
    throw py::value_error(absl::StrCat(
        "Secondary structure of the ", type,
        " structure must have the same length as the ", type,
        " structure (got ", py_ss->size(), ", expected ", pts.cols(), ")"));
  }

  return str_eigen_view(*py_ss);
}

TMAlign tmalign_init(py::handle query, py::handle templ,
                     std::optional<std::string_view> py_secx,
                     std::optional<std::string_view> py_secy, bool gt, bool ss,
                     bool local, bool lpss, bool fgt) {
  TMAlign::InitFlags flags = TMAlign::InitFlags::kNone;
  if (gt)
    flags |= TMAlign::InitFlags::kGaplessThreading;
  if (ss)
    flags |= TMAlign::InitFlags::kSecStr;
  if (local)
    flags |= TMAlign::InitFlags::kLocal;
  if (lpss)
    flags |= TMAlign::InitFlags::kLocalPlusSecStr;
  if (fgt)
    flags |= TMAlign::InitFlags::kFragmentGaplessThreading;

  if (flags == TMAlign::InitFlags::kNone)
    throw py::value_error("At least one initialization flag must be set");

  auto xa = py_array_cast<3>(query), ya = py_array_cast<3>(templ);
  auto x = xa.eigen(), y = ya.eigen();

  if ((!ss && !lpss) || (!py_secx && !py_secy))
    return tmalign_try_construct_init(x, y, flags);

  Matrix3Xd buf(3, nuri::max(x.cols(), y.cols()));
  ArrayXc secx = resolve_sec_str(x, py_secx, buf, "query"),
          secy = resolve_sec_str(y, py_secy, buf, "template");
  return tmalign_try_construct_init(x, y, flags, secx, secy);
}

TMAlign tmalign_init_aln(py::handle query, py::handle templ, py::handle aln) {
  auto xa = py_array_cast<3>(query), ya = py_array_cast<3>(templ);
  auto x = xa.eigen(), y = ya.eigen();
  ArrayXi y2x(y.cols());

  if (aln.is_none()) {
    if (x.cols() != y.cols()) {
      throw py::value_error(
          absl::StrCat("Query and template structures must have the same "
                       "length when no alignment is provided (got ",
                       x.cols(), " != ", y.cols(), ")"));
    }

    absl::c_iota(y2x, 0);
    return tmalign_try_construct_init(x, y, y2x);
  }

  auto xya = py_array_cast<2, Eigen::Dynamic, int>(aln);
  auto xy = xya.eigen();

  y2x.setConstant(-1);
  for (Eigen::Index k = 0; k < xy.cols(); ++k) {
    const int i = xy(0, k), j = xy(1, k);
    if (i < 0 || i >= x.cols() || j < 0 || j >= y.cols()) {
      throw py::value_error(absl::StrCat(
          "Alignment contains out-of-range indice(s) at row ", k, " (got ", i,
          " -> ", j, ", expected [0, ", x.cols(), ") -> [0, ", y.cols(), "))"));
    }
    y2x[j] = i;
  }

  return tmalign_try_construct_init(x, y, y2x);
}

pyt::Tuple<py::array_t<double>, double>
tmalign_convert_result(const std::pair<Affine3d, double> &result) {
  if (result.second < 0)
    throw py::value_error("TM-align failed to calculate TM-score");

  auto xform = empty_like(result.first.matrix());
  xform.eigen().transpose() = result.first.matrix();
  return py::make_tuple(std::move(xform).numpy(), result.second);
}

void bind_tmalign(py::module &m) {
  py::class_<TMAlign>(m, "TMAlign")
      .def(py::init(&tmalign_init), py::arg("query"), py::arg("templ"),
           py::arg("query_ss") = py::none(), py::arg("templ_ss") = py::none(),
           py::kw_only(),  //
           py::arg("gapless") = true, py::arg("sec_str") = true,
           py::arg("local_sup") = true, py::arg("local_with_ss") = true,
           py::arg("fragment_gapless") = true, R"doc(
Prepare TM-align algorithm with the given structures.

:param query: The query structure, in which each residue is represented by a
  single atom (usually ``CA``). Must be representable as a 2D numpy array of
  shape ``(N, 3)``, where ``N`` is the number of residues.
:param templ: The template structure, in which each residue is represented by a
  single atom (usually ``CA``). Must be representable as a 2D numpy array of
  shape ``(M, 3)``, where ``M`` is the number of residues.
:param query_ss: The secondary structure of the query structure. When provided,
  must be an ASCII string of length ``N``.
:param templ_ss: The secondary structure of the template structure. When
  provided, must be an ASCII string of length ``M``.
:param gapless: Enable gapless threading.
:param sec_str: Enable secondary structure assignment.
:param local_sup: Enable local superposition. Note that this is the most
  expensive initialization method due to the exhaustive pairwise distance
  calculation. Consider disabling this flag if alignment takes too long.
:param local_with_ss: Enable local superposition with secondary structure-based
  alignment.
:param fragment_gapless: Enable fragment gapless threading.

:raises ValueError: If:

  - The query or template structure has less than 5 residues.
  - The secondary structure of the query or template structure has a different
    length than the structure.
  - No initialization flag is set.
  - The initialization fails (for any other reason).

.. note::
  If the secondary structure is not provided, it will be assigned using the
  approximate secondary structure assignment algorithm defined in the TM-align
  code. When both ``sec_str`` and ``local_with_ss`` flags are not set, the
  secondary structures are ignored.
)doc")
      .def_static("from_alignment", tmalign_init_aln, py::arg("query"),
                  py::arg("templ"), py::arg("alignment") = py::none(),
                  R"doc(
Prepare TM-align algorithm with the given structures and user-provided alignment.

:param query: The query structure, in which each residue is represented by a
  single atom (usually ``CA``). Must be representable as a 2D numpy array of
  shape ``(N, 3)``, where ``N`` is the number of residues.
:param templ: The template structure, in which each residue is represented by a
  single atom (usually ``CA``). Must be representable as a 2D numpy array of
  shape ``(M, 3)``, where ``M`` is the number of residues.
:param alignment: Pairwise alignment of the query and template structures. Must
  be in a form representable as a 2D numpy array of shape ``(L, 2)``, in which
  rows must contain (query index, template index) pairs. If not provided, query
  and template must have same length and assumed to be aligned in order.
:returns: A :class:`TMAlign` object initialized with the given alignment.

:raises ValueError: If:

  - The query or template structure has less than 5 residues.
  - The alignment contains out-of-range indices.
  - Alignment is not provided and the query and template structures have
    different lengths.
  - The initialization fails (for any other reason).

.. tip::
  When initialized by this method, the result is equivalent to the "TM-score"
  program in the TM-tools suite.

.. note::
  Duplicate values in ``alignment`` are not checked and may result in invalid
  alignment.
)doc")
      .def(
          "score",
          [](TMAlign &self, std::optional<int> l_norm,
             std::optional<double> d0) {
            return tmalign_convert_result(
                self.tm_score(l_norm.value_or(-1), d0.value_or(-1)));
          },
          py::arg("l_norm") = py::none(),  //
          py::kw_only(),                   //
          py::arg("d0") = py::none(), R"doc(
Calculate TM-score using the current alignment.

:param l_norm: Length normalization factor. If not specified, the length of the
  template structure is used.
:param d0: Distance scale factor. If not specified, calculated based on the
  length normalization factor.

:returns: A pair of the transformation tensor and the TM-score of the alignment.
)doc")
      .def(
          "rmsd",
          [](TMAlign &self) {
            return std::sqrt(nonnegative(self.aligned_msd()));
          },
          R"doc(
The RMSD of the aligned pairs.
)doc")
      .def(
          "aligned_pairs",
          [](TMAlign &self) {
            auto aln = empty_numpy<2, Eigen::Dynamic, int>({ 2, self.l_ali() });
            auto aln_mat = aln.eigen();

            int k = 0;
            for (int j = 0; j < self.templ_to_query().size(); ++j) {
              const int i = self.templ_to_query()[j];
              if (i >= 0) {
                if (k >= aln_mat.cols())
                  throw std::runtime_error("Alignment buffer overflow");

                aln_mat(0, k) = i;
                aln_mat(1, k) = j;
                k++;
              }
            }
            if (k != aln_mat.cols())
              throw std::runtime_error("Alignment buffer size mismatch");

            return std::move(aln).numpy();
          },
          R"doc(
Get pairwise alignment of the query and template structures.

:returns: A 2D numpy array of shape ``(L, 2)``, where ``L`` is the number of
  aligned pairs. Each row is a (query index, template index) pair.

.. tip::
  This will always return the same alignment once the :class:`TMAlign` object is
  created.

.. note::
  Even if the :class:`TMAlign` object is created with :meth:`from_alignment`,
  the returned pairs from this method may not be the same as the input
  alignment. This is because the TM-align algorithm filters out far-apart pairs
  when calculating the final alignment.
)doc");

  m.def(
       "tm_align",
       [](py::handle query, py::handle templ, std::optional<int> l_norm,
          std::optional<std::string_view> py_secx,
          std::optional<std::string_view> py_secy, std::optional<double> d0,
          bool gt, bool ss, bool local, bool lpss, bool fgt) {
         TMAlign tm = tmalign_init(query, templ, py_secx, py_secy, gt, ss,
                                   local, lpss, fgt);
         return tmalign_convert_result(
             tm.tm_score(l_norm.value_or(-1), d0.value_or(-1)));
       },
       py::arg("query"), py::arg("templ"), py::arg("l_norm") = py::none(),
       py::arg("query_ss") = py::none(), py::arg("templ_ss") = py::none(),
       py::kw_only(),  //
       py::arg("d0") = py::none(), py::arg("gapless") = true,
       py::arg("sec_str") = true, py::arg("local_sup") = true,
       py::arg("local_with_ss") = true, py::arg("fragment_gapless") = true,
       R"doc(
Run TM-align algorithm with the given structures and parameters.

:param query: The query structure, in which each residue is represented by a
  single atom (usually ``CA``). Must be representable as a 2D numpy array of
  shape ``(N, 3)``, where ``N`` is the number of residues.
:param templ: The template structure, in which each residue is represented by a
  single atom (usually ``CA``). Must be representable as a 2D numpy array of
  shape ``(M, 3)``, where ``M`` is the number of residues.
:param l_norm: Length normalization factor. If not specified, the length of the
  template structure is used.
:param query_ss: The secondary structure of the query structure. When provided,
  must be an ASCII string of length ``N``.
:param templ_ss: The secondary structure of the template structure. When
  provided, must be an ASCII string of length ``M``.
:param d0: Distance scale factor. If not specified, calculated based on the
  length normalization factor.
:param gapless: Enable gapless threading.
:param sec_str: Enable secondary structure assignment.
:param local_sup: Enable local superposition. Note that this is the most
  expensive initialization method due to the exhaustive pairwise distance
  calculation. Consider disabling this flag if alignment takes too long.
:param local_with_ss: Enable local superposition with secondary structure-based
  alignment.
:param fragment_gapless: Enable fragment gapless threading.
:returns: A pair of the transformation tensor and the TM-score of the alignment.

:raises ValueError: If:

  - The query or template structure has less than 5 residues.
  - The secondary structure of the query or template structure has a different
    length than the structure.
  - No initialization flag is set.
  - The initialization fails (for any other reason).

.. tip::
  If want to calculate TM-score for multiple ``l_norm`` or ``d0`` values, or
  want more details such as RMSD or aligned pairs, consider using the
  :class:`TMAlign` object directly.

.. note::
  If the secondary structure is not provided, it will be assigned using the
  approximate secondary structure assignment algorithm defined in the TM-align
  code. When both ``sec_str`` and ``local_with_ss`` flags are not set, the
  secondary structures are ignored.

.. seealso::
  :class:`TMAlign`, :meth:`TMAlign.__init__`, :meth:`TMAlign.score`
)doc")
      .def(
          "tm_score",
          [](py::handle query, py::handle templ, py::handle aln,
             std::optional<int> l_norm, std::optional<double> d0) {
            TMAlign tm = tmalign_init_aln(query, templ, aln);
            return tmalign_convert_result(
                tm.tm_score(l_norm.value_or(-1), d0.value_or(-1)));
          },
          py::arg("query"), py::arg("templ"), py::arg("alignment") = py::none(),
          py::arg("l_norm") = py::none(),
          py::kw_only(),  //
          py::arg("d0") = py::none(),
          R"doc(
Run TM-align algorithm with the given structures and alignment. This is also
known as the "TM-score" program in the TM-tools suite, from which the function
got its name.

:param query: The query structure, in which residues are represented by a single
  atom (usually ``CA``). Must be representable as a 2D numpy array of shape
  ``(N, 3)`` where ``N`` is the number of residues.
:param templ: The template structure, in which residues are represented by a
  single atom (usually ``CA``). Must be representable as a 2D numpy array of
  shape ``(M, 3)`` where ``M`` is the number of residues.
:param alignment: Pairwise alignment of the query and template structures. Must
  be in a form representable as a 2D numpy array of shape ``(L, 2)``, in which
  rows must contain (query index, template index) pairs. If not provided, query
  and template must have same length and assumed to be aligned in order.
:param l_norm: Length normalization factor. If not specified, the length of the
  template structure is used.
:param d0: Distance scale factor. If not specified, calculated based on the
  length normalization factor.
:returns: A pair of the transformation tensor and the TM-score of the alignment.

:raises ValueError: If:

  - The query or template structure has less than 5 residues.
  - The alignment contains out-of-range indices.
  - Alignment is not provided and the query and template structures have
    different lengths.
  - The initialization fails (for any other reason).


.. tip::
  If want to calculate TM-score for multiple ``l_norm`` or ``d0`` values, or
  want more details such as RMSD or aligned pairs, consider using the
  :class:`TMAlign` object directly.

.. note::
  Duplicate values in ``alignment`` are not checked and may result in invalid
  alignment.

.. seealso::
  :class:`TMAlign`, :meth:`TMAlign.from_alignment`, :meth:`TMAlign.score`
)doc");
}

NURI_PYTHON_MODULE(m) {
  bind_tmalign(m);
}
}  // namespace
}  // namespace python_internal
}  // namespace nuri
