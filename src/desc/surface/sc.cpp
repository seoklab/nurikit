//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include <absl/log/absl_check.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/desc/surface.h"
#include "nuri/utils.h"

namespace nuri {
namespace {
  using internal::SesDots;

  constexpr double kNearestCellSpacings = 3;

  class Neighbors {
  public:
    template <class Pred>
    bool any(const VoxelGrid &grid, const Vector3d &pt, const Pred &pred) {
      grid.find_neighbors_d(pt, idxs_, distsq_, grid_scratch_);
      for (size_t m = 0; m < idxs_.size(); ++m) {
        if (pred(idxs_[m], distsq_[m]))
          return true;
      }
      return false;
    }

    std::pair<int, double> nearest(const VoxelGrid &grid, const Vector3d &pt) {
      return grid.nearest(pt, grid_scratch_);
    }

  private:
    std::vector<int> idxs_;
    std::vector<double> distsq_;
    VoxelGrid::Scratch grid_scratch_;
  };

  ArrayXb active_atoms(const Matrix3Xd &pts, const Matrix3Xd &other,
                       const double sep) {
    const VoxelGrid grid(other, sep);
    const double sepsq = sep * sep;
    Neighbors nbrs;

    ArrayXb active(pts.cols());
    for (int i = 0; i < pts.cols(); ++i) {
      active[i] = nbrs.any(grid, pts.col(i),
                           [&](int, double dsq) { return dsq < sepsq; });
    }
    return active;
  }

  ArrayXb buried_mask(const SesDots &dots, const Matrix3Xd &other,
                      const ArrayXd &sar_other) {
    const VoxelGrid grid(other, sar_other.maxCoeff());
    const ArrayXd sarsq = sar_other.square();
    Neighbors nbrs;

    ArrayXb buried(dots.n());
    for (int k = 0; k < dots.n(); ++k) {
      const Vector3d probe = dots.pts.col(k) + dots.rp * dots.nrm.col(k);
      buried[k] = nbrs.any(grid, probe,
                           [&](int j, double dsq) { return dsq < sarsq[j]; });
    }
    return buried;
  }

  ArrayXb trim_peripheral(const SesDots &dots, const ArrayXb &buried,
                          const double band) {
    const int n_buried = static_cast<int>(buried.count()),
              n_exposed = dots.n() - n_buried;
    if (n_buried == 0 || n_exposed == 0)
      return buried;

    Matrix3Xd exposed(3, n_exposed);
    for (int k = 0, w = 0; k < dots.n(); ++k) {
      if (!buried[k])
        exposed.col(w++) = dots.pts.col(k);
    }

    const VoxelGrid grid(exposed, band);
    const double bandsq = band * band;
    Neighbors nbrs;

    ArrayXb keep = buried;
    for (int k = 0; k < dots.n(); ++k) {
      if (!buried[k])
        continue;
      keep[k] = !nbrs.any(grid, dots.pts.col(k),
                          [&](int, double dsq) { return dsq <= bandsq; });
    }
    return keep;
  }

  struct Side {
    int n_atoms, n_active, n_dots, n_buried;
    SesDots trimmed;
  };

  std::optional<Side> build_side(const Matrix3Xd &pts, const ArrayXd &radii,
                                 const Matrix3Xd &other,
                                 const ArrayXd &other_radii,
                                 const ScParams &params) {
    const ArrayXb active = active_atoms(pts, other, params.sep);
    const std::optional<internal::SurfaceSpheres> ss =
        internal::prepare(pts, radii, active, params.rp);
    if (!ss)
      return std::nullopt;

    const internal::SasDelaunay del = internal::triangulate(*ss);
    const internal::SasGeometry geo = internal::build_sas(*ss, del);
    const internal::SesGeometry ses = internal::build_ses(*ss, del, geo);
    SesDots dots = internal::sample_ses(*ss, geo, ses, params.density);

    const ArrayXb buried = buried_mask(dots, other, other_radii + params.rp);
    const ArrayXb kept = trim_peripheral(dots, buried, params.band);
    const int n_dots = dots.n();
    dots.keep(kept);
    return Side {
      static_cast<int>(pts.cols()),
      static_cast<int>(active.count()),
      n_dots,
      static_cast<int>(buried.count()),
      std::move(dots),
    };
  }

  double median(std::vector<double> &v) {
    ABSL_DCHECK(!v.empty());

    const auto mid = v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2);
    std::nth_element(v.begin(), mid, v.end());
    if (v.size() % 2 == 1)
      return *mid;
    return 0.5 * (*std::max_element(v.begin(), mid) + *mid);
  }

  ScSide side_stats(const Side &mine, const Side &theirs,
                    const ScParams &params) {
    const SesDots &a = mine.trimmed, &b = theirs.trimmed;
    const VoxelGrid grid(b.pts,
                         kNearestCellSpacings / std::sqrt(params.density));
    Neighbors nbrs;

    std::vector<double> d(a.n()), s(a.n());
    for (int k = 0; k < a.n(); ++k) {
      const auto [j, dsq] = nbrs.nearest(grid, a.pts.col(k));
      ABSL_DCHECK_GE(j, 0);
      const double dot = a.nrm.col(k).dot(b.nrm.col(j));
      d[k] = std::sqrt(dsq);
      s[k] = nuri::clamp(-dot * std::exp(-params.weight * dsq), -params.clamp,
                         params.clamp);
    }

    return {
      mine.n_atoms, mine.n_active, mine.n_dots, mine.n_buried,
      a.n(),        a.area.sum(),  median(d),   median(s),
    };
  }

  void check_side(const Matrix3Xd &pts, const ArrayXd &radii) {
    ABSL_DCHECK_GT(pts.cols(), 0);
    ABSL_DCHECK_EQ(pts.cols(), radii.size());
    ABSL_DCHECK((radii > 0).all());
  }
}  // namespace

std::optional<ScResult> shape_complementarity(const Matrix3Xd &pts_a,
                                              const ArrayXd &radii_a,
                                              const Matrix3Xd &pts_b,
                                              const ArrayXd &radii_b,
                                              const ScParams &params) {
  check_side(pts_a, radii_a);
  check_side(pts_b, radii_b);
  ABSL_DCHECK_GT(params.rp, 0);
  ABSL_DCHECK_GT(params.density, 0);
  ABSL_DCHECK_GT(params.weight, 0);
  ABSL_DCHECK_GT(params.band, 0);
  ABSL_DCHECK_GT(params.sep, 0);
  ABSL_DCHECK(params.clamp > 0 && params.clamp <= 1);

  const std::optional<Side> a =
      build_side(pts_a, radii_a, pts_b, radii_b, params);
  if (!a || a->trimmed.n() == 0)
    return std::nullopt;

  const std::optional<Side> b =
      build_side(pts_b, radii_b, pts_a, radii_a, params);
  if (!b || b->trimmed.n() == 0)
    return std::nullopt;

  ScResult res;
  res.sides = { side_stats(*a, *b, params), side_stats(*b, *a, params) };
  res.sc = 0.5 * (res.sides[0].s_median + res.sides[1].s_median);
  res.distance = 0.5 * (res.sides[0].d_median + res.sides[1].d_median);
  res.area = res.sides[0].trimmed_area + res.sides[1].trimmed_area;
  return res;
}
}  // namespace nuri
