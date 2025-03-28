//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <stdexcept>
#include <string_view>
#include <utility>

#include <absl/strings/str_cat.h>
#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/python/utils.h"

namespace nuri {
namespace python_internal {
namespace {
template <class Impl>
double align_rmsd(Impl impl, std::string_view method,
                  const PyMatrixMap<3> &query, const PyMatrixMap<3> &templ,
                  bool reflection) {
  auto [_, msd] = impl(query, templ, AlignMode::kMsdOnly, reflection);
  if (msd < 0)
    throw std::runtime_error(absl::StrCat("Alignment method '", method,
                                          "' failed to calculate RMSD"));

  return std::sqrt(msd);
}

template <class Impl>
std::pair<Affine3d, double>
align_both(Impl impl, std::string_view method, const PyMatrixMap<3> &query,
           const PyMatrixMap<3> &templ, bool reflection) {
  // cols < 2 already handled in implementations
  if (method == "qcp" && query.cols() == 2)
    throw std::runtime_error(absl::StrCat("Alignment method '", method,
                                          "' requires at least 3 points"));

  std::pair<Affine3d, double> result =
      impl(query, templ, AlignMode::kBoth, reflection);
  if (result.second < 0) {
    throw std::runtime_error(
        absl::StrCat("Alignment method '", method,
                     "' failed to calculate transformation tensor"));
  }

  result.second = std::sqrt(result.second);
  return result;
}

std::pair<PyMatrixMap<3>, PyMatrixMap<3>>
check_convert_points(const NpArrayWrapper<3> &q_arr,
                     const NpArrayWrapper<3> &t_arr) {
  auto query = q_arr.eigen(), templ = t_arr.eigen();
  if (query.cols() != templ.cols()) {
    throw py::value_error(
        absl::StrCat("two sets of points must have the same size; got ",
                     query.cols(), " and ", templ.cols()));
  }

  if (!query.array().isFinite().all() || !templ.array().isFinite().all())
    throw py::value_error("NaN or infinite values in the points");

  return { query, templ };
}

auto default_qcp(ConstRef<Matrix3Xd> query, ConstRef<Matrix3Xd> templ,
                 AlignMode mode, bool reflection) {
  return qcp(query, templ, mode, reflection);
}

NURI_PYTHON_MODULE(m) {
  m.def(
      "align_points",
      [](const py::handle &q_py, const py::handle &t_py,
         std::string_view method, bool reflection) {
        auto q_arr = py_array_cast<3>(q_py), t_arr = py_array_cast<3>(t_py);
        auto [query, templ] = check_convert_points(q_arr, t_arr);

        std::pair<Affine3d, double> result;
        if (method == "qcp") {
          result = align_both(default_qcp, method, query, templ, reflection);
        } else if (method == "kabsch") {
          result = align_both(kabsch, method, query, templ, reflection);
        } else {
          throw py::value_error(
              absl::StrCat("Unknown alignment method: ", method));
        }

        auto xform = empty_like(result.first.matrix());
        // Unlike most operations that simply transposing the matrix gives the
        // correct column-major matrix, the transformation matrix should not
        // be transposed.
        xform.eigen().transpose() = result.first.matrix();
        return std::make_pair(std::move(xform).numpy(), result.second);
      },
      py::arg("query"), py::arg("template"), py::arg("method") = "qcp",
      py::arg("reflection") = false, R"doc(
Find a 4x4 best-fit rigid-body transformation tensor, to align ``query`` to
``template``.

:param query: The query points. Must be representable as a 2D numpy array of
  shape ``(N, 3)``.
:param template: The template points. Must be representable as a 2D numpy array
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
      [](const py::handle &q_py, const py::handle &t_py,
         std::string_view method, bool reflection) {
        auto q_arr = py_array_cast<3>(q_py), t_arr = py_array_cast<3>(t_py);
        auto [query, templ] = check_convert_points(q_arr, t_arr);

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

:param query: The query points. Must be representable as a 2D numpy array of
  shape ``(N, 3)``.
:param template: The template points. Must be representable as a 2D numpy array
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
        auto mat_arr = py_array_cast<4, 4>(obj);
        auto mat_t = mat_arr.eigen();

        // unlike most operations that simply transposing the matrix gives the
        // correct column-major matrix, the transformation matrix should not
        // be transposed.
        Affine3d xform;
        xform.matrix() = mat_t.transpose();

        auto pts_arr = py_array_cast<3>(py_pts);
        auto pts = pts_arr.eigen();
        auto result = empty_like(pts);
        result.eigen().noalias() = xform * pts;
        return std::move(result).numpy();
      },
      py::arg("tensor"), py::arg("pts"), R"doc(
Transform a set of points using a 4x4 transformation tensor.

Effectively, this function is roughly equivalent to the following Python code:

.. code-block:: python

  def transform(tensor, pts):
      rotated = tensor[:3, :3] @ pts.T
      translated = rotated + tensor[:3, 3, None]
      return translated.T

:param tensor: The transformation tensor. Must be representable as a 2D numpy
  array of shape ``(4, 4)``.
:param pts: The points to transform. Must be representable as a 2D numpy array
  of shape ``(N, 3)``.
:returns: The transformed points.

:warning: This function does not check if the transformation tensor is a valid
  affine transformation matrix.
)doc");
}
}  // namespace
}  // namespace python_internal
}  // namespace nuri
