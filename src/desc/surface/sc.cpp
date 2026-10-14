//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

#include <absl/log/absl_check.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/desc/surface.h"
#include "nuri/utils.h"

namespace nuri {
namespace {
  using internal::ScScratch;

  // Three mean dot spacings: a 0.5..2.0 angstrom sweep at density 15 bottomed
  // at 0.75 angstrom on a shallow curve.
  constexpr double kNearestCellSpacings = 3;

  template <class Pred>
  bool any_neighbor(const VoxelGrid &grid, const Vector3d &pt,
                    ScScratch &scratch, const Pred &pred) {
    grid.find_neighbors_d(pt, scratch.idxs, scratch.distsq, scratch.grid);
    for (size_t m = 0; m < scratch.idxs.size(); ++m) {
      if (pred(scratch.idxs[m], scratch.distsq[m]))
        return true;
    }
    return false;
  }

  ArrayXb active_atoms(ConstRef<Matrix3Xd> pts, ConstRef<ArrayXb> mask,
                       ConstRef<Matrix3Xd> other, const double sep,
                       ScScratch &scratch) {
    const VoxelGrid grid(other, sep);
    const double sepsq = sep * sep;

    ArrayXb active(pts.cols());
    for (int i = 0; i < pts.cols(); ++i) {
      active[i] = mask[i]
                  && any_neighbor(grid, pts.col(i), scratch,
                                  [&](int, double dsq) { return dsq < sepsq; });
    }
    return active;
  }

  ArrayXb buried_mask(const SurfaceDots &dots, ConstRef<Matrix3Xd> other,
                      const ArrayXd &sar_other, ScScratch &scratch) {
    const VoxelGrid grid(other, sar_other.maxCoeff());
    const ArrayXd sarsq = sar_other.square();

    ArrayXb buried(dots.n());
    for (int k = 0; k < dots.n(); ++k) {
      const Vector3d probe = dots.pts().col(k) + dots.rp() * dots.nrm().col(k);
      buried[k] = any_neighbor(grid, probe, scratch, [&](int j, double dsq) {
        return dsq < sarsq[j];
      });
    }
    return buried;
  }

  ArrayXb trim_peripheral(const SurfaceDots &dots, const ArrayXb &buried,
                          const double band, ScScratch &scratch) {
    const int n_buried = static_cast<int>(buried.count()),
              n_exposed = dots.n() - n_buried;
    if (n_buried == 0 || n_exposed == 0)
      return buried;

    Matrix3Xd exposed(3, n_exposed);
    for (int k = 0, w = 0; k < dots.n(); ++k) {
      if (!buried[k])
        exposed.col(w++) = dots.pts().col(k);
    }

    const VoxelGrid grid(exposed, band);
    const double bandsq = band * band;

    ArrayXb keep = buried;
    for (int k = 0; k < dots.n(); ++k) {
      if (!buried[k])
        continue;
      keep[k] = !any_neighbor(grid, dots.pts().col(k), scratch,
                              [&](int, double dsq) { return dsq <= bandsq; });
    }
    return keep;
  }

  /**
   * The trimmed SES dots of one side, with the side's counts; false when the
   * surface cannot be prepared or no dot survives.
   */
  bool build_side(const ScParams &params, SurfaceSampler &sampler,
                  ScScratch &scratch, ConstRef<Matrix3Xd> pts,
                  ConstRef<ArrayXd> radii, ConstRef<ArrayXb> mask,
                  ConstRef<Matrix3Xd> other, ConstRef<ArrayXd> other_radii,
                  ScSide &side) {
    const ArrayXb active = active_atoms(pts, mask, other, params.sep, scratch);
    AnalyticSurface surf(pts, radii);
    if (!surf.initialize(params.rp, active))
      return false;

    SurfaceDots &dots = side.dots;
    dots = sampler.sample(surf.build_ses());

    const ArrayXb buried =
        buried_mask(dots, other, other_radii.array() + params.rp, scratch);
    const ArrayXb kept = trim_peripheral(dots, buried, params.band, scratch);
    side.n_atoms = static_cast<int>(pts.cols());
    side.n_active = static_cast<int>(active.count());
    side.n_dots = dots.n();
    side.n_buried = static_cast<int>(buried.count());
    dots.keep(kept);
    side.n_trimmed = dots.n();
    side.trimmed_area = dots.area().sum();
    return dots.n() > 0;
  }

  double median(std::vector<double> &v) {
    ABSL_DCHECK(!v.empty());

    const auto mid = v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2);
    std::nth_element(v.begin(), mid, v.end());
    if (v.size() % 2 == 1)
      return *mid;
    return 0.5 * (*std::max_element(v.begin(), mid) + *mid);
  }

  /**
   * Nearest-dot distance and score of every dot of `a` against `b`, and
   * their medians. An array formulation measured neutral, and an IndexedView
   * inside a colwise redux copies the index per column, O(n^2).
   */
  void side_stats(const ScParams &params, ScScratch &scratch,
                  const SurfaceDots &theirs, ScSide &side) {
    const SurfaceDots &a = side.dots, &b = theirs;
    const VoxelGrid grid(b.pts(),
                         kNearestCellSpacings / std::sqrt(params.density));

    ArrayXd &d = side.distance, &s = side.score;
    d.resize(a.n());
    s.resize(a.n());
    for (int k = 0; k < a.n(); ++k) {
      const auto [j, dsq] = grid.nearest(a.pts().col(k), scratch.grid);
      ABSL_DCHECK_GE(j, 0);
      const double dot = a.nrm().col(k).dot(b.nrm().col(j));
      d[k] = std::sqrt(dsq);
      s[k] = nuri::clamp(-dot * std::exp(-params.weight * dsq), -params.clamp,
                         params.clamp);
    }

    scratch.d.assign(d.begin(), d.end());
    scratch.s.assign(s.begin(), s.end());
    side.d_median = median(scratch.d);
    side.s_median = median(scratch.s);
  }

  void check_side(ConstRef<Matrix3Xd> pts, ConstRef<ArrayXd> radii,
                  ConstRef<ArrayXb> mask) {
    ABSL_DCHECK_GT(pts.cols(), 0);
    ABSL_DCHECK_EQ(pts.cols(), radii.size());
    ABSL_DCHECK_EQ(pts.cols(), mask.size());
    ABSL_DCHECK(pts.allFinite());
    ABSL_DCHECK((radii.array() > 0).all());
  }
}  // namespace

ShapeComplementarity::ShapeComplementarity(ScParams params)
    : params_(params), sampler_(params.density) {
  ABSL_DCHECK_GT(params.rp, 0);
  ABSL_DCHECK_GT(params.density, 0);
  ABSL_DCHECK_GT(params.weight, 0);
  ABSL_DCHECK_GT(params.band, 0);
  ABSL_DCHECK_GT(params.sep, 0);
  ABSL_DCHECK(params.clamp > 0 && params.clamp <= 1);
}

std::optional<ScResult> ShapeComplementarity::compute(
    ConstRef<Matrix3Xd> pts_a, ConstRef<ArrayXd> radii_a,
    ConstRef<Matrix3Xd> pts_b, ConstRef<ArrayXd> radii_b) {
  return compute(pts_a, radii_a, ArrayXb::Constant(pts_a.cols(), true), pts_b,
                 radii_b, ArrayXb::Constant(pts_b.cols(), true));
}

std::optional<ScResult> ShapeComplementarity::compute(
    ConstRef<Matrix3Xd> pts_a, ConstRef<ArrayXd> radii_a,
    ConstRef<ArrayXb> active_a, ConstRef<Matrix3Xd> pts_b,
    ConstRef<ArrayXd> radii_b, ConstRef<ArrayXb> active_b) {
  check_side(pts_a, radii_a, active_a);
  check_side(pts_b, radii_b, active_b);

  ScResult res;
  if (!build_side(params_, sampler_, scratch_, pts_a, radii_a, active_a, pts_b,
                  radii_b, res.sides[0]))
    return std::nullopt;
  if (!build_side(params_, sampler_, scratch_, pts_b, radii_b, active_b, pts_a,
                  radii_a, res.sides[1]))
    return std::nullopt;

  side_stats(params_, scratch_, res.sides[1].dots, res.sides[0]);
  side_stats(params_, scratch_, res.sides[0].dots, res.sides[1]);
  res.sc = 0.5 * (res.sides[0].s_median + res.sides[1].s_median);
  res.distance = 0.5 * (res.sides[0].d_median + res.sides[1].d_median);
  res.area = res.sides[0].trimmed_area + res.sides[1].trimmed_area;
  return res;
}

std::optional<ScResult> shape_complementarity(ConstRef<Matrix3Xd> pts_a,
                                              ConstRef<ArrayXd> radii_a,
                                              ConstRef<Matrix3Xd> pts_b,
                                              ConstRef<ArrayXd> radii_b,
                                              const ScParams &params) {
  return ShapeComplementarity(params).compute(pts_a, radii_a, pts_b, radii_b);
}

std::optional<ScResult>
shape_complementarity(ConstRef<Matrix3Xd> pts_a, ConstRef<ArrayXd> radii_a,
                      ConstRef<ArrayXb> active_a, ConstRef<Matrix3Xd> pts_b,
                      ConstRef<ArrayXd> radii_b, ConstRef<ArrayXb> active_b,
                      const ScParams &params) {
  return ShapeComplementarity(params).compute(pts_a, radii_a, active_a, pts_b,
                                              radii_b, active_b);
}
}  // namespace nuri
