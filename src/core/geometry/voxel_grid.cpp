//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <absl/log/absl_check.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/utils.h"

namespace nuri {
namespace {
  Array3i voxel_of(const Vector3d &p, const Vector3d &origin, double cutoff) {
    return ((p - origin).array() / cutoff).floor().cast<int>();
  }

  // Yield linear cell indices for the 3x3x3 stencil around `c`, clipped at
  // `[0, dims - 1]` on each axis. Out-of-range query voxels (whose stencil
  // does not intersect the grid) yield zero cell indices.
  template <class Func>
  void for_each_neighbor_cell(const Array3i &c, const Array3i &dims,
                              const Func &f) {
    Array3i imin = (c - 1).max(0);
    Array3i imax = (c + 1).min(dims - 1);

    const int nx = dims.x();
    const int ny = dims.y();
    for (int z = imin.z(); z <= imax.z(); ++z) {
      const int basey = ny * z;
      for (int y = imin.y(); y <= imax.y(); ++y) {
        const int basex = nx * (y + basey);
        for (int x = imin.x(); x <= imax.x(); ++x)
          f(basex + x);
      }
    }
  }
}  // namespace

void VoxelGrid::rebuild_impl(const Points &src) {
  ABSL_DCHECK_GT(cutoff_, 0.0);

  const int n = static_cast<int>(src.cols());
  if (n == 0) {
    pts_.resize(3, 0);
    origin_.setZero();
    dims_.setZero();
    cell_offset_.setZero(1);
    cell_pts_.resize(0);
    return;
  }

  origin_ = src.rowwise().minCoeff();
  Vector3d extent = src.rowwise().maxCoeff() - origin_;
  dims_ = (extent.array() / cutoff_).ceil().cast<int>().max(1);
  const int ncells = dims_.prod();

  cell_offset_.setZero(ncells + 1);
  cell_pts_.resize(n);
  ArrayXi cursor(ncells), pt_cell(n);

  {
    internal::AllowEigenMallocScoped<false> ems;

    const int nx = dims_.x();
    const int ny = dims_.y();
    const Array3i upper = dims_ - 1;

    for (int i = 0; i < n; ++i) {
      Array3i c = voxel_of(src.col(i), origin_, cutoff_).min(upper).max(0);
      int v = c.x() + nx * (c.y() + ny * c.z());
      pt_cell[i] = v;
      ++cell_offset_[v + 1];
    }
    max_occ_ = cell_offset_.maxCoeff();

    for (int v = 0; v < ncells; ++v)
      cell_offset_[v + 1] += cell_offset_[v];

    cursor = cell_offset_.head(ncells);
    for (int i = 0; i < n; ++i) {
      const int slot = cursor[pt_cell[i]]++;
      cell_pts_[slot] = i;
    }
  }

  pts_ = src(Eigen::all, cell_pts_);
}

void VoxelGrid::find_neighbors_d(const Vector3d &pt, std::vector<int> &idxs,
                                 std::vector<double> &distsq) const {
  idxs.clear();
  distsq.clear();

  if (pts_.cols() == 0)
    return;

  const double cutsq = cutoff_ * cutoff_;
  Array3i c = voxel_of(pt, origin_, cutoff_);

  for_each_neighbor_cell(c, dims_, [&](int v) {
    const int beg = cell_offset_[v];
    const int end = cell_offset_[v + 1];
    for (int p = beg; p < end; ++p) {
      double d2 = (pts_.col(p) - pt).squaredNorm();
      if (d2 <= cutsq) {
        idxs.push_back(cell_pts_[p]);
        distsq.push_back(d2);
      }
    }
  });
}

void VoxelGrid::find_neighbors_grid(const VoxelGrid &grid,
                                    std::vector<int> &self,
                                    std::vector<int> &other) const {
  self.clear();
  other.clear();

  const int n = static_cast<int>(pts_.cols());
  const int m = static_cast<int>(grid.pts_.cols());
  if (n == 0 || m == 0)
    return;

  const double cutsq = cutoff_ * cutoff_;

  for (int j = 0; j < m; ++j) {
    const Vector3d pj = grid.pts_.col(j);
    Array3i cj = voxel_of(pj, origin_, cutoff_);

    for_each_neighbor_cell(cj, dims_, [&](int v) {
      const int beg = cell_offset_[v];
      const int end = cell_offset_[v + 1];
      for (int p = beg; p < end; ++p) {
        double d2 = (pts_.col(p) - pj).squaredNorm();
        if (d2 <= cutsq) {
          self.push_back(cell_pts_[p]);
          other.push_back(grid.cell_pts_[j]);
        }
      }
    });
  }
}

std::vector<std::vector<int>>
VoxelGrid::find_neighbors_grid(const VoxelGrid &grid) const {
  std::vector<int> is, js;
  find_neighbors_grid(grid, is, js);

  std::vector<std::vector<int>> idxs(pts_.cols());
  for (int p = 0; p < static_cast<int>(is.size()); ++p)
    idxs[is[p]].push_back(js[p]);
  return idxs;
}

namespace {
  // Forward half-shell of 13 neighbor cells, ordered so each unordered cell
  // pair is visited exactly once: (dx > 0, dy = 0, dz = 0), then any
  // (dy > 0, dz = 0), then any (dz > 0).
  constexpr int kHalfShell[13][3] = {
    {  1,  0, 0 },
    { -1,  1, 0 },
    {  0,  1, 0 },
    {  1,  1, 0 },
    { -1, -1, 1 },
    {  0, -1, 1 },
    {  1, -1, 1 },
    { -1,  0, 1 },
    {  0,  0, 1 },
    {  1,  0, 1 },
    { -1,  1, 1 },
    {  0,  1, 1 },
    {  1,  1, 1 },
  };

  // Branchless emit of pairs (i_global, jsrc[k]) for which `dsqbuf[k] <=
  // cutsq`. Compacts surviving j-indices into `jbuf[0..n)` using
  // value_if-predication, then appends them to (left, right) with one
  // vector::insert per call.
  ABSL_ATTRIBUTE_ALWAYS_INLINE inline void
  emit_filtered(std::vector<int> &left, std::vector<int> &right,
                const VectorXd &dsqbuf, ArrayXi &jbuf, const int i,
                const int *jsrc, int count, double cutsq) {
    int n = 0;
    for (int k = 0; k < count; ++k) {
      jbuf[n] = jsrc[k];
      n += value_if(dsqbuf[k] <= cutsq);
    }
    left.insert(left.end(), n, i);
    right.insert(right.end(), jbuf.data(), jbuf.data() + n);
  }
}  // namespace

void VoxelGrid::find_neighbors_self(std::vector<int> &left,
                                    std::vector<int> &right) const {
  left.clear();
  right.clear();

  if (pts_.cols() == 0)
    return;

  VectorXd dsqbuf(max_occ_);
  ArrayXi jbuf(max_occ_);

  const double cutsq = cutoff_ * cutoff_;
  const int nx = dims_.x();
  const int ny = dims_.y();
  const int nz = dims_.z();

  for (int cz = 0; cz < nz; ++cz) {
    for (int cy = 0; cy < ny; ++cy) {
      for (int cx = 0; cx < nx; ++cx) {
        const int va = cx + nx * (cy + ny * cz);
        const int ab = cell_offset_[va];
        const int ae = cell_offset_[va + 1];
        const int na = ae - ab;
        if (na == 0)
          continue;

        auto pts = pts_.middleCols(ab, na);
        for (int p = 0; p < na - 1; ++p) {
          const int cnt = na - p - 1;
          const Vector3d pi = pts.col(p);
          dsqbuf.head(cnt) = (pts.middleCols(p + 1, cnt).colwise() - pi)
                                 .colwise()
                                 .squaredNorm();

          emit_filtered(left, right, dsqbuf, jbuf, cell_pts_[ab + p],
                        cell_pts_.data() + ab + p + 1, cnt, cutsq);
        }

        for (const auto &d: kHalfShell) {
          const int bx = cx + d[0];
          const int by = cy + d[1];
          const int bz = cz + d[2];
          if (bx < 0 || bx >= nx || by < 0 || by >= ny || bz < 0 || bz >= nz)
            continue;

          const int vb = bx + nx * (by + ny * bz);
          const int bb = cell_offset_[vb];
          const int be = cell_offset_[vb + 1];
          const int nb = be - bb;
          if (nb == 0)
            continue;

          auto qts = pts_.middleCols(bb, nb);
          for (int p = 0; p < na; ++p) {
            const Vector3d pi = pts.col(p);
            dsqbuf.head(nb) = (qts.colwise() - pi).colwise().squaredNorm();
            emit_filtered(left, right, dsqbuf, jbuf, cell_pts_[ab + p],
                          cell_pts_.data() + bb, nb, cutsq);
          }
        }
      }
    }
  }
}

}  // namespace nuri
