//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <absl/base/attributes.h>
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
}  // namespace

void VoxelGrid::rebuild_impl(Points src) {
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
  Array3i imin = (c - 1).max(0);
  Array3i imax = (c + 1).min(dims_ - 1);

  VectorXd dsqbuf(3 * max_occ_);
  ArrayXi jbuf(dsqbuf.size());

  const int nx = dims_.x();
  const int ny = dims_.y();
  for (int z = imin.z(); z <= imax.z(); ++z) {
    const int basey = ny * z;
    for (int y = imin.y(); y <= imax.y(); ++y) {
      const int basex = nx * (y + basey);
      const int beg = cell_offset_[imin.x() + basex];
      const int end = cell_offset_[imax.x() + 1 + basex];
      const int cnt = end - beg;
      if (cnt == 0)
        continue;

      dsqbuf.head(cnt) =
          (pts_.middleCols(beg, cnt).colwise() - pt).colwise().squaredNorm();

      int n = 0;
      for (int k = 0; k < cnt; ++k) {
        jbuf[n] = cell_pts_[beg + k];
        const double dsq = dsqbuf[n] = dsqbuf[k];
        n += value_if(dsq <= cutsq);
      }
      idxs.insert(idxs.end(), jbuf.data(), jbuf.data() + n);
      distsq.insert(distsq.end(), dsqbuf.data(), dsqbuf.data() + n);
    }
  }
}

namespace {
  Array2Xi axis_cell_range(int ldim, double lorg, double lcut,  //
                           int rdim, double rorg, double rcut) {
    Array2Xi lohi(2, ldim);

    const double rel = lorg - rorg;
    const double inv = 1.0 / rcut;
    const int upper = rdim - 1;

    lohi.row(0) = ArrayXd::LinSpaced(ldim, inv * (rel - lcut),
                                     inv * (rel + lcut * (ldim - 2)))
                      .floor()
                      .cast<int>()
                      .max(0);
    lohi.row(1) = ArrayXd::LinSpaced(ldim, inv * (rel + lcut * 2),
                                     inv * (rel + lcut * (ldim + 1)))
                      .floor()
                      .cast<int>()
                      .min(upper);

    return lohi;
  }

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

void VoxelGrid::find_neighbors_grid(const VoxelGrid &grid,
                                    std::vector<int> &self,
                                    std::vector<int> &other) const {
  self.clear();
  other.clear();

  const int n = static_cast<int>(pts_.cols());
  const int m = static_cast<int>(grid.pts_.cols());
  if (n == 0 || m == 0)
    return;

  const int nx = dims_.x();
  const int ny = dims_.y();
  const int nz = dims_.z();
  const int gnx = grid.dims_.x();
  const int gny = grid.dims_.y();
  const int gnz = grid.dims_.z();

  Array2Xi xlim = axis_cell_range(nx, origin_.x(), cutoff_,  //
                                  gnx, grid.origin_.x(), grid.cutoff_),
           ylim = axis_cell_range(ny, origin_.y(), cutoff_,  //
                                  gny, grid.origin_.y(), grid.cutoff_),
           zlim = axis_cell_range(nz, origin_.z(), cutoff_,  //
                                  gnz, grid.origin_.z(), grid.cutoff_);

  const int max_xrange = (xlim.row(1) - xlim.row(0) + 1).cwiseMax(0).maxCoeff();
  VectorXd dsqbuf(max_xrange * grid.max_occ_);
  ArrayXi jbuf(dsqbuf.size());
  const double cutsq = cutoff_ * cutoff_;

  for (int cz = 0; cz < nz; ++cz) {
    E::Array2i czlim = zlim.col(cz);
    if (czlim[0] > czlim[1])
      continue;

    for (int cy = 0; cy < ny; ++cy) {
      E::Array2i cylim = ylim.col(cy);
      if (cylim[0] > cylim[1])
        continue;

      for (int cx = 0; cx < nx; ++cx) {
        E::Array2i cxlim = xlim.col(cx);
        if (cxlim[0] > cxlim[1])
          continue;

        const int va = cx + nx * (cy + ny * cz);
        const int ab = cell_offset_[va];
        const int ae = cell_offset_[va + 1];
        const int na = ae - ab;
        if (na == 0)
          continue;

        auto pts = pts_.middleCols(ab, na);

        for (int kz = czlim[0]; kz <= czlim[1]; ++kz) {
          const int basey = gny * kz;
          for (int ky = cylim[0]; ky <= cylim[1]; ++ky) {
            const int basex = gnx * (ky + basey);
            const int bb = grid.cell_offset_[cxlim[0] + basex];
            const int be = grid.cell_offset_[cxlim[1] + 1 + basex];
            const int nb = be - bb;
            if (nb == 0)
              continue;

            auto qts = grid.pts_.middleCols(bb, nb);
            for (int p = 0; p < na; ++p) {
              const Vector3d pi = pts.col(p);
              dsqbuf.head(nb) = (qts.colwise() - pi).colwise().squaredNorm();
              emit_filtered(self, other, dsqbuf, jbuf, cell_pts_[ab + p],
                            grid.cell_pts_.data() + bb, nb, cutsq);
            }
          }
        }
      }
    }
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
  // Forward half-shell of 13 neighbor cells, grouped into 5 rows of contiguous
  // dx ranges per (dy, dz). Each row enumerates every unordered cell pair
  // exactly once: the (dy=0, dz=0) row spans dx > 0 only, the four (dy != 0 or
  // dz > 0) rows span dx ∈ {-1, 0, 1}. Fusing the dx range into one wider qts
  // slab amortises per-call overhead vs. the unfused 13-entry walk.
  constexpr int kHalfShellRows[5][4] = {
    // { dy, dz, dx_lo, dx_hi }
    {  0, 0,  1, 1 },
    {  1, 0, -1, 1 },
    { -1, 1, -1, 1 },
    {  0, 1, -1, 1 },
    {  1, 1, -1, 1 },
  };
}  // namespace

void VoxelGrid::find_neighbors_self(std::vector<int> &left,
                                    std::vector<int> &right) const {
  left.clear();
  right.clear();

  if (pts_.cols() == 0)
    return;

  const int nx = dims_.x();
  const int ny = dims_.y();
  const int nz = dims_.z();

  VectorXd dsqbuf(3 * max_occ_);
  ArrayXi jbuf(dsqbuf.size());

  const double cutsq = cutoff_ * cutoff_;

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

        for (auto [dy, dz, dxlo, dxhi]: kHalfShellRows) {
          const int by = cy + dy;
          const int bz = cz + dz;
          const int bxlo = nuri::max(cx + dxlo, 0);
          const int bxhi = nuri::min(cx + dxhi, nx - 1);
          if (by < 0 || by >= ny || bz < 0 || bz >= nz || bxlo > bxhi)
            continue;

          const int baserow = nx * (by + ny * bz);
          const int bb = cell_offset_[bxlo + baserow];
          const int be = cell_offset_[bxhi + baserow + 1];
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
