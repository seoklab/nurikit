//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <absl/functional/function_ref.h>
#include <absl/log/absl_check.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/desc/surface.h"

namespace nuri {
namespace internal {
  namespace {
    using constants::kPi;
    using constants::kTwoPi;
  }  // namespace

  ArrangementSolver::ArrangementSolver(const int mcap, const int kcap,
                                       const int acap) {
    cosa_.resize(mcap);

    dirs_.resize(3, kcap);
    ea_.resize(3, kcap);
    eb_.resize(3, kcap);

    arcs_.reserve(acap);
    succ_.resize(acap);
    seen_.resize(acap);
    darts_.resize(kcap);
  }

  void ArrangementSolver::begin(const double radius) {
    radius_ = radius;
    m_ = k_ = 0;
    arcs_.clear();
  }

  int ArrangementSolver::add_cap(const double cosa) {
    ABSL_DCHECK_LT(m_, cosa_.size());

    cosa_[m_] = cosa;
    return m_++;
  }

  int ArrangementSolver::add_vertex(const Vector3d &dir) {
    ABSL_DCHECK_LT(k_, dirs_.cols());

    dirs_.col(k_) = dir;
    return k_++;
  }

  /**
   * Two darts per arc end: the departing tangent at `beg` and the reversed
   * arriving tangent at `end`. A vertex ends exactly one arc on each of its
   * two circles, so it carries one in-dart and one out-dart; the corner from
   * the out-dart to the in-dart is the signed angle in `(−π/2, 3π/2]`, and
   * the two arcs are linked into one loop. Every corner also joins its two
   * caps' components.
   */
  std::pair<int, double> ArrangementSolver::walk(UnionFind &uf) {
    const int k = k_, na = static_cast<int>(arcs_.size());

    auto angle_at = [&](int v, const Vector3d &t) {
      return std::atan2(t.dot(eb_.col(v)), t.dot(ea_.col(v)));
    };

    std::fill(darts_.begin(), darts_.begin() + k, Darts {});
    for (int a = 0; a < na; ++a) {
      const Arc &arc = arcs_[a];
      if (arc.beg < 0)
        continue;

      Darts &out = darts_[arc.end], &in = darts_[arc.beg];
      ABSL_DCHECK_LT(out.out, 0) << "two out-darts at vertex " << arc.end;
      ABSL_DCHECK_LT(in.in, 0) << "two in-darts at vertex " << arc.beg;
      out.out = a;
      out.aout = angle_at(arc.end, -arc.tend);
      in.in = a;
      in.ain = angle_at(arc.beg, arc.tbeg);
    }

    auto succ = succ_.head(na);
    succ = ArrayXi::LinSpaced(na, 0, na - 1);
    double turn_sum = 0;
    for (int v = 0; v < k; ++v) {
      const Darts &d = darts_[v];
      ABSL_DCHECK_GE(d.in, 0) << "no in-dart at vertex " << v;
      ABSL_DCHECK_GE(d.out, 0) << "no out-dart at vertex " << v;

      double iota = d.ain - d.aout;
      iota += kTwoPi * static_cast<double>(iota <= -kPi / 2);
      iota -= kTwoPi * static_cast<double>(iota > 3 * kPi / 2);
      ABSL_DCHECK_LE(iota, kPi + kSurfaceAngleEps)
          << "reflex corner at vertex " << v;
      succ[d.in] = d.out;
      turn_sum += kPi - iota;
      uf.merge(arcs_[d.in].cap, arcs_[d.out].cap);
    }

    auto seen = seen_.head(na);
    seen.setConstant(false);
    int n_loops = 0;
    for (int a = 0; a < na; ++a) {
      if (seen[a])
        continue;

      ++n_loops;
      for (int i = a; !seen[i]; i = succ[i])
        seen[i] = true;
    }

    return { n_loops, turn_sum };
  }

  namespace {
    void build_frames(Matrix3Xd &xs, Matrix3Xd &ys, const Matrix3Xd &zs,
                      const int cols) {
      for (int j = 0; j < cols; ++j) {
        const Vector3d z = zs.col(j);
        Vector3d x = xs.col(j) = any_perpendicular(z);
        ys.col(j) = z.cross(x);
      }
    }
  }  // namespace

  /**
   * `n_patches = 1 + n_loops − n_components` (§2 step 7): the components of
   * the union of all caps come from the corners plus the exact disc
   * intersection test on the remaining pairs. Caps without any arc leave the
   * accessible region without boundary, so it is empty.
   */
  double
  ArrangementSolver::solve(absl::FunctionRef<bool(int, int)> intersects) {
    const double r2 = radius_ * radius_;
    if (m_ == 0)
      return 2 * kTwoPi * r2;
    if (arcs_.empty())
      return 0;

    build_frames(ea_, eb_, dirs_, k_);

    UnionFind uf(m_);
    auto [n_loops, turn_sum] = walk(uf);
    int n_sets = uf.n_sets();
    for (int j = 0; j < m_ && n_sets > 1; ++j) {
      for (int l = j + 1; l < m_ && n_sets > 1; ++l) {
        if (uf.find(j) != uf.find(l) && intersects(j, l)) {
          uf.merge(j, l);
          --n_sets;
        }
      }
    }

    double geo_sum = 0;
    for (const Arc &arc: arcs_)
      geo_sum += arc.dphi * cosa_[arc.cap];

    const int n_patches = 1 + n_loops - n_sets;
    const int chi = 2 * n_patches - n_loops;
    const double area = r2 * (kTwoPi * chi - turn_sum + geo_sum);
    ABSL_DCHECK_GE(area, -kSurfaceLengthEps * r2);
    return area;
  }
}  // namespace internal
}  // namespace nuri
