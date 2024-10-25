//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <stdexcept>
#include <string_view>
#include <utility>

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <absl/strings/str_cat.h>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/python/utils.h"

namespace nuri {
namespace python_internal {
namespace {
template <class Impl>
double align_rmsd(Impl impl, std::string_view name, const PyMatrixMap<3> &query,
                  const PyMatrixMap<3> &templ, bool reflection) {
  auto [_, msd] = impl(query, templ, AlignMode::kMsdOnly, reflection);
  if (msd < 0)
    throw std::runtime_error(
        absl::StrCat("Alignment method '", name, "' failed to calculate RMSD"));

  return std::sqrt(msd);
}

template <class Impl>
std::pair<Affine3d, double>
align_both(Impl impl, std::string_view name, const PyMatrixMap<3> &query,
           const PyMatrixMap<3> &templ, bool reflection) {
  std::pair<Affine3d, double> result =
      impl(query, templ, AlignMode::kBoth, reflection);
  if (result.second < 0) {
    throw std::runtime_error(
        absl::StrCat("Alignment method '", name,
                     "' failed to calculate transformation tensor"));
  }

  result.second = std::sqrt(result.second);
  return result;
}

std::pair<PyMatrixMap<3>, PyMatrixMap<3>>
check_convert_points(const py::handle &py_q, const py::handle &py_t) {
  PyMatrixMap<3> query = map_py_matrix<3>(py_q), templ = map_py_matrix<3>(py_t);

  if (query.cols() != templ.cols()) {
    throw py::value_error(
        absl::StrCat("two sets of points must have the same size; got ",
                     query.cols(), " and ", templ.cols()));
  }

  if (query.cols() < 3)
    throw py::value_error("too few points to align, need at least 3 points");

  return { query, templ };
}

auto default_qcp(ConstRef<Matrix3Xd> query, ConstRef<Matrix3Xd> templ,
                 AlignMode mode, bool reflection) {
  return qcp(query, templ, mode, reflection);
}

NURI_PYTHON_MODULE(m) {
  m.def(
      "align_points",
      [](const py::handle &py_q, const py::handle &py_t,
         std::string_view method, bool reflection) {
        auto [query, templ] = check_convert_points(py_q, py_t);

        std::pair<Affine3d, double> result;
        if (method == "qcp") {
          result = align_both(default_qcp, method, query, templ, reflection);
        } else if (method == "kabsch") {
          result = align_both(kabsch, method, query, templ, reflection);
        } else {
          throw py::value_error(
              absl::StrCat("Unknown alignment method: ", method));
        }

        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> xform =
            result.first.matrix();
        return std::make_pair(xform, result.second);
      },
      py::arg("query"), py::arg("template"), py::arg("method") = "qcp",
      py::arg("reflection") = false, R"doc(
Find a 4x4 best-fit rigid-body transformation tensor, to align ``query`` to
``template``.

:param query: The query points. Must be representable by a 2D numpy array of
  shape ``(N, 3)``.
:param template: The template points. Must be representable by a 2D numpy array
  of shape ``(N, 3)``.
:param method: The alignment method to use. Defaults to ``"qcp"``. Currently
  supported methods are:

  - ``"qcp"``: The Quaternion Characteristic Polynomial (QCP) method, based on
    the implementation of Liu and Theobald
    :footcite:`core:geom:qcp-2005,core:geom:qcp-2010,core:geom:qcp-2011`. Unlike
    the original implementation, this version can also handle reflection
    based on the observations of :footcite:ts:`core:geom:qcp-2004`.

  - ``"kabsch"``: The Kabsch algorithm.
    :footcite:`core:geom:kabsch-1976,core:geom:kabsch-1978` This implementation
    is based on the implementation in TM-align. :footcite:`tm-align`

:param reflection: Whether to allow reflection in the alignment. Defaults to
  ``False``.

:returns: A tuple of the transformation tensor and the RMSD of the alignment.
)doc");

  m.def(
      "align_rmsd",
      [](const py::handle &py_q, const py::handle &py_t,
         std::string_view method, bool reflection) {
        auto [query, templ] = check_convert_points(py_q, py_t);

        if (method == "qcp")
          return align_rmsd(default_qcp, method, query, templ, reflection);

        if (method == "kabsch")
          return align_rmsd(kabsch, method, query, templ, reflection);

        throw py::value_error(
            absl::StrCat("Unknown alignment method: ", method));
      },
      py::arg("query"), py::arg("template"), py::arg("method") = "qcp",
      py::arg("reflection") = false, R"doc(
Calculate the RMSD of the best-fit rigid-body alignment of ``query`` to
``template``.

:param query: The query points. Must be representable by a 2D numpy array of
  shape ``(N, 3)``.
:param template: The template points. Must be representable by a 2D numpy array
  of shape ``(N, 3)``.
:param method: The alignment method to use. Defaults to ``"qcp"``. Currently
  supported methods are:

  - ``"qcp"``: The Quaternion Characteristic Polynomial (QCP) method, based on
    the implementation of Liu and Theobald
    :footcite:`core:geom:qcp-2005,core:geom:qcp-2010,core:geom:qcp-2011`. Unlike
    the original implementation, this version can also handle reflection
    based on the observations of :footcite:ts:`core:geom:qcp-2004`.

  - ``"kabsch"``: The Kabsch algorithm.
    :footcite:`core:geom:kabsch-1976,core:geom:kabsch-1978` This implementation
    is based on the implementation in TM-align. :footcite:`tm-align`

:param reflection: Whether to allow reflection in the alignment. Defaults to
  ``False``.

:returns: The RMSD of the alignment.
)doc");

  m.def(
      "transform",
      [](const py::handle &obj, const py::handle &py_pts) {
        auto mat_t = map_py_matrix<4, 4>(obj);
        auto pts = map_py_matrix<3>(py_pts);

        // unlike most operations that simply transposing the matrix gives the
        // correct column-major matrix, the transformation matrix should not
        // be transposed.
        Affine3d xform;
        xform.matrix() = mat_t.transpose();

        Transposed<Matrix3Xd> result(pts.cols(), 3);
        result.transpose().noalias() = xform * pts;
        return result;
      },
      py::arg("tensor"), py::arg("pts"), R"doc(
Transform a set of points using a 4x4 transformation tensor.

Effectively, this function is roughly equivalent to the following Python code:

.. code-block:: python

  def transform(tensor, pts):
    rotated = tensor[:3, :3] @ pts.T
    translated = rotated + tensor[:3, 3, None]
    return translated.T

:param tensor: The transformation tensor. Must be representable by a 2D numpy
  array of shape ``(4, 4)``.
:param pts: The points to transform. Must be representable by a 2D numpy array
  of shape ``(N, 3)``.
:returns: The transformed points.

:warning: This function does not check if the transformation tensor is a valid
  affine transformation matrix.
)doc");
}
}  // namespace
}  // namespace python_internal
}  // namespace nuri
