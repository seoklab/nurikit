//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <absl/strings/str_cat.h>
#include <Eigen/Dense>
#include <pybind11/cast.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

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
std::pair<Isometry3d, double>
align_both(Impl impl, std::string_view method, const PyMatrixMap<3> &query,
           const PyMatrixMap<3> &templ, bool reflection) {
  // cols < 2 already handled in implementations
  if (method == "qcp" && query.cols() == 2)
    throw std::runtime_error(absl::StrCat("Alignment method '", method,
                                          "' requires at least 3 points"));

  std::pair<Isometry3d, double> result =
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

struct OCTreeWrapper {
  OCTree tree;
  Matrix3Xd pts;
};

template <class T>
auto vector_as_eigen(const std::vector<T> &v) {
  return E::Map<const E::Array<T, E::Dynamic, 1>>(
      v.data(), static_cast<E::Index>(v.size()));
}

void check_cutoff(double cutoff) {
  if (!(cutoff > 0)) {
    throw py::value_error(
        absl::StrCat("cutoff distance must be positive; got ", cutoff));
  }
}

void find_neighbors_d(const OCTree &octree, const Vector3d &query, double d,
                      int /* k */, std::vector<int> &idxs,
                      std::vector<double> &distsq) {
  octree.find_neighbors_d(query, d, idxs, distsq);
}

void find_neighbors_kd(const OCTree &octree, const Vector3d &query, double d,
                       int k, std::vector<int> &idxs,
                       std::vector<double> &distsq) {
  octree.find_neighbors_kd(query, k, idxs, distsq, d);
}

NURI_PYTHON_MODULE(m) {
  m.def(
      "align_points",
      [](const py::handle &q_py, const py::handle &t_py,
         std::string_view method, bool reflection) {
        auto q_arr = py_array_cast<3>(q_py), t_arr = py_array_cast<3>(t_py);
        auto [query, templ] = check_convert_points(q_arr, t_arr);

        std::pair<Isometry3d, double> result;
        if (method == "qcp") {
          result = align_both(default_qcp, method, query, templ, reflection);
        } else if (method == "kabsch") {
          result = align_both(kabsch, method, query, templ, reflection);
        } else {
          throw py::value_error(
              absl::StrCat("Unknown alignment method: ", method));
        }

        // Unlike most operations that simply transposing the matrix gives the
        // correct column-major matrix, the transformation matrix should not
        // be transposed.
        return std::make_pair(eigen_as_numpy(result.first.matrix().transpose()),
                              result.second);
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
        Isometry3d xform;
        xform.matrix() = mat_t.transpose();

        auto pts_arr = py_array_cast<3>(py_pts);
        auto pts = pts_arr.eigen();
        auto result = empty_like(pts);
        inplace_transform(result.eigen(), xform, pts);
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

  py::class_<OCTreeWrapper>(m, "Octree", R"doc(
An octree implementation for 3D point clouds.

The octree partitions 3D space into axis-aligned boxes recursively, allowing
efficient spatial queries such as nearest neighbor search.

This implementation is designed for static point clouds; each octree instance
keeps a copy of the original point set that could not be modified in any way.
To update the point set, one must :py:meth:`rebuild()` the octree.
)doc")
      .def(py::init([](const py::handle &obj, int bucket_size) {
             auto py_arr = py_array_cast<3>(obj);
             OCTreeWrapper self { OCTree(), py_arr.eigen() };
             self.tree.rebuild(self.pts, bucket_size);
             return self;
           }),
           py::arg("pts"), py::arg("bucket_size") = 32, R"doc(
Initialize the octree with a set of points.

:param pts: The points to build the octree with. Must be representable as a 2D
  numpy array of shape ``(N, 3)``.
:param bucket_size: The maximum number of points in each leaf node of the
  octree. Defaults to 32.
)doc")
      .def(
          "rebuild",
          [](OCTreeWrapper &self, const py::handle &obj, int bucket_size) {
            auto py_arr = py_array_cast<3>(obj);
            self.pts = py_arr.eigen();
            self.tree.rebuild(self.pts, bucket_size);
          },
          py::arg("pts"), py::arg("bucket_size") = 32, R"doc(
Rebuild the octree with a new set of points.

:param pts: The points to build the octree with. Must be representable as a 2D
  numpy array of shape ``(N, 3)``.
:param bucket_size: The maximum number of points in each leaf node of the
  octree. Defaults to 32.
)doc")
      .def(
          "find_neighbors",
          [](const OCTreeWrapper &self, const py::handle &obj,
             std::optional<double> xd, std::optional<int> xk) {
            auto py_arr = py_array_cast<3>(obj);
            if (!xd && !xk) {
              throw py::value_error(
                  "either cutoff distance or number of neighbors must be "
                  "specified");
            }

            void (*impl)(const OCTree &, const Vector3d &, double, int,
                         std::vector<int> &, std::vector<double> &);
            if (!xk) {
              impl = find_neighbors_d;
            } else {
              impl = find_neighbors_kd;
            }
            const double d = xd.value_or(-1);
            const int k = xk.value_or(0);

            const auto pts = py_arr.eigen();
            std::vector<std::vector<int>> all_idxs(pts.cols());
            std::vector<std::vector<double>> all_distsq(pts.cols());
            py::ssize_t total_neighbors = 0;
            for (E::Index i = 0; i < pts.cols(); ++i) {
              impl(self.tree, pts.col(i), d, k, all_idxs[i], all_distsq[i]);
              total_neighbors += static_cast<py::ssize_t>(all_idxs[i].size());
            }

            auto py_idxs =
                empty_numpy<2, E::Dynamic, int>({ 2, total_neighbors });
            auto py_dist =
                empty_numpy<E::Dynamic, 1, double>({ total_neighbors });

            E::Index left = 0;
            for (int i = 0; i < pts.cols(); ++i) {
              const auto &idxs = all_idxs[i];
              const auto &distsq = all_distsq[i];

              auto iblk = py_idxs.eigen().middleCols(left, idxs.size());
              iblk.row(0).setConstant(i);
              iblk.row(1) = vector_as_eigen(idxs);

              py_dist.eigen().segment(left, idxs.size()) =
                  vector_as_eigen(distsq);

              left += static_cast<py::ssize_t>(idxs.size());
            }
            py_dist.eigen() = py_dist.eigen().cwiseMax(0.0).cwiseSqrt();

            return std::make_pair(std::move(py_idxs).numpy(),
                                  std::move(py_dist).numpy());
          },
          py::arg("query"), py::kw_only(), py::arg("d") = py::none(),
          py::arg("k") = py::none(), R"doc(
Find neighbors of each point in the octree.

:param query: The query points. Must be representable as a 2D numpy array of
  shape ``(M, 3)``.
:param d: The cutoff distance. If specified, only neighbors within this
  distance will be returned.
:param k: The number of nearest neighbors to return. If specified, at most k
  nearest neighbors will be returned, from the closest to the farthest. If both
  ``d`` and ``k`` are specified, only neighbors within distance ``d`` will be
  considered when finding the k nearest neighbors.
:returns: A tuple of two numpy arrays. The first array contains the indices of
  the neighbors in the original point set, and the second array contains the
  distances to the neighbors. If the query contains ``M`` points and a total of
  ``N`` neighbors are found, the first array will have shape ``(N, 2)``, where
  each row is a pair of (query index, neighbor index), and the second array will
  have shape ``(N,)``, where each element is the distance to the corresponding
  neighbor.
)doc")
      .def(
          "query_tree",
          [](const OCTreeWrapper &self, const OCTreeWrapper &other,
             double d) -> pyt::List<py::array_t<int>> {
            check_cutoff(d);

            std::vector<std::vector<int>> idxs =
                self.tree.find_neighbors_tree(other.tree, d);

            auto ret = py::list(self.pts.cols());
            for (int i = 0; i < idxs.size(); ++i) {
              const auto &nbrs = idxs[i];
              auto py_nbrs = empty_numpy<E::Dynamic, 1, int>(
                  { static_cast<py::ssize_t>(nbrs.size()) });
              py_nbrs.eigen() = vector_as_eigen(nbrs);
              ret[i] = std::move(py_nbrs).numpy();
            }
            return ret;
          },
          py::arg("other"), py::kw_only(), py::arg("d"), R"doc(
Find all neighbors in another octree.

:param other: The other octree to query.
:param d: The cutoff distance for neighbors. Must be non-negative.
:returns: A list of numpy arrays. For point ``i`` in the original octree,
  ``results[i]`` is a 1D numpy array containing the indices of its neighbors in
  the ``other`` octree.
)doc")
      .def(
          "query_pairs",
          [](const OCTreeWrapper &self, double d) {
            check_cutoff(d);

            std::vector<int> is, js;
            self.tree.find_neighbors_self(d, is, js);

            auto py_idxs = empty_numpy<2, E::Dynamic, int>(
                { 2, static_cast<py::ssize_t>(is.size()) });
            if (is.empty())
              return std::move(py_idxs).numpy();

            py_idxs.eigen().row(0) = vector_as_eigen(is);
            py_idxs.eigen().row(1) = vector_as_eigen(js);
            return std::move(py_idxs).numpy();
          },
          py::kw_only(), py::arg("d"), R"doc(
Find all non-redundant pairs of neighbors in the octree.

:param d: The cutoff distance for neighbors. Must be non-negative.
:returns: A numpy array of shape ``(N, 2)``, where each row is a pair of
  neighbor indices in the original point set. The pairs are non-redundant,
  meaning that if :math:`(i, j)` is in the array, then :math:`(j, i)` will not
  be in the array, and :math:`i \neq j`.
)doc");

  py::class_<VoxelGrid>(m, "VoxelGrid", R"doc(
A uniform voxel-grid index for 3D point clouds with a fixed cutoff.

The voxel grid partitions 3D space into uniform cells, allowing efficient
distance queries with the cutoff specified at construction time.
)doc")
      .def(py::init([](const py::handle &obj, double cutoff) {
             check_cutoff(cutoff);

             auto py_arr = py_array_cast<3>(obj);
             return VoxelGrid(py_arr.eigen(), cutoff);
           }),
           py::arg("pts"), py::arg("cutoff"), R"doc(
Initialize the voxel grid with a set of points and cutoff distance.

:param pts: The points to build the grid with. Must be representable as a 2D
  numpy array of shape ``(N, 3)``.
:param cutoff: The cutoff distance for neighbor queries. Must be positive.
)doc")
      .def(
          "rebuild",
          [](VoxelGrid &self, const py::handle &obj,
             std::optional<double> xcutoff) {
            double cutoff = -1.0;
            if (xcutoff) {
              cutoff = *xcutoff;
              check_cutoff(cutoff);
            }

            auto py_arr = py_array_cast<3>(obj);
            self.rebuild(py_arr.eigen(), cutoff);
          },
          py::arg("pts"), py::arg("cutoff") = py::none(), R"doc(
Rebuild the voxel grid with a new set of points.

:param pts: The points to build the grid with. Must be representable as a 2D
  numpy array of shape ``(N, 3)``.
:param cutoff: The cutoff distance for neighbor queries. If omitted, the
  current cutoff is reused. Must be positive when specified.
)doc")
      .def_property_readonly(
          "cutoff", [](const VoxelGrid &self) { return self.cutoff(); },
          "The cutoff distance used by this voxel grid.")
      .def(
          "find_neighbors",
          [](const VoxelGrid &self, const py::handle &obj) {
            auto py_arr = py_array_cast<3>(obj);
            const auto pts = py_arr.eigen();

            std::vector<std::vector<int>> all_idxs(pts.cols());
            std::vector<std::vector<double>> all_distsq(pts.cols());
            py::ssize_t total_neighbors = 0;
            for (E::Index i = 0; i < pts.cols(); ++i) {
              self.find_neighbors_d(pts.col(i), all_idxs[i], all_distsq[i]);
              total_neighbors += static_cast<py::ssize_t>(all_idxs[i].size());
            }

            auto py_idxs =
                empty_numpy<2, E::Dynamic, int>({ 2, total_neighbors });
            auto py_dist =
                empty_numpy<E::Dynamic, 1, double>({ total_neighbors });

            E::Index left = 0;
            for (int i = 0; i < pts.cols(); ++i) {
              const auto &idxs = all_idxs[i];
              const auto &distsq = all_distsq[i];

              auto iblk = py_idxs.eigen().middleCols(left, idxs.size());
              iblk.row(0).setConstant(i);
              iblk.row(1) = vector_as_eigen(idxs);

              py_dist.eigen().segment(left, idxs.size()) =
                  vector_as_eigen(distsq);

              left += static_cast<py::ssize_t>(idxs.size());
            }
            py_dist.eigen() = py_dist.eigen().cwiseMax(0.0).cwiseSqrt();

            return std::make_pair(std::move(py_idxs).numpy(),
                                  std::move(py_dist).numpy());
          },
          py::arg("query"), R"doc(
Find neighbors of each point in the voxel grid.

:param query: The query points. Must be representable as a 2D numpy array of
  shape ``(M, 3)``.
:returns: A tuple of two numpy arrays. The first array contains the indices of
  the neighbors in the original point set, and the second array contains the
  distances to the neighbors. If the query contains ``M`` points and a total of
  ``N`` neighbors are found, the first array will have shape ``(N, 2)``, where
  each row is a pair of (query index, neighbor index), and the second array will
  have shape ``(N,)``, where each element is the distance to the corresponding
  neighbor.
)doc")
      .def(
          "query_grid",
          [](const VoxelGrid &self,
             const VoxelGrid &other) -> pyt::List<py::array_t<int>> {
            std::vector<std::vector<int>> idxs =
                self.find_neighbors_grid(other);

            auto ret = py::list(self.pts().cols());
            for (int i = 0; i < idxs.size(); ++i) {
              const auto &nbrs = idxs[i];
              auto py_nbrs = empty_numpy<E::Dynamic, 1, int>(
                  { static_cast<py::ssize_t>(nbrs.size()) });
              py_nbrs.eigen() = vector_as_eigen(nbrs);
              ret[i] = std::move(py_nbrs).numpy();
            }
            return ret;
          },
          py::arg("other"), R"doc(
Find all neighbors in another voxel grid.

The cutoff of this voxel grid is used for the query.

:param other: The other voxel grid to query.
:returns: A list of numpy arrays. For point ``i`` in the original point set,
  ``results[i]`` is a 1D numpy array containing the indices of its neighbors in
  the ``other`` voxel grid.
)doc")
      .def(
          "query_pairs",
          [](const VoxelGrid &self) {
            std::vector<int> is, js;
            self.find_neighbors_self(is, js);

            auto py_idxs = empty_numpy<2, E::Dynamic, int>(
                { 2, static_cast<py::ssize_t>(is.size()) });
            if (is.empty())
              return std::move(py_idxs).numpy();

            py_idxs.eigen().row(0) = vector_as_eigen(is);
            py_idxs.eigen().row(1) = vector_as_eigen(js);
            return std::move(py_idxs).numpy();
          },
          R"doc(
Find all non-redundant pairs of neighbors in the voxel grid.

:returns: A numpy array of shape ``(N, 2)``, where each row is a pair of
  neighbor indices in the original point set. The pairs are non-redundant,
  meaning that if :math:`(i, j)` is in the array, then :math:`(j, i)` will not
  be in the array, and :math:`i \neq j`.
)doc");
}
}  // namespace
}  // namespace python_internal
}  // namespace nuri
