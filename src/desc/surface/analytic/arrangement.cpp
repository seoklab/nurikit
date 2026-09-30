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
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  namespace {
    using constants::kPi;
    using constants::kTwoPi;
  }  // namespace

  ArrangementSolver::ArrangementSolver(const int mcap, const int kcap,
                                       const int acap)
      : off_(nuri::max(mcap, kcap)) {
    cosa_.resize(mcap);

    dirs_.resize(3, kcap);
    ea_.resize(3, kcap);
    eb_.resize(3, kcap);

    arcs_.reserve(acap);
    order_.resize(2L * acap);
    succ_.resize(acap);
    seen_.resize(acap);
    keys_.reserve(2L * acap);
    darts_.reserve(2L * acap);
    dring_.reserve(2L * acap + 1);
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
   * arriving tangent at `end`, as supplied with the arc. At every vertex the
   * darts alternate in/out around the vertex; the corner between an in-dart and
   * the out-dart before it is the signed angle in `(−π/2, 3π/2]`, and the two
   * arcs are linked into one loop. Every corner also joins its two caps'
   * components.
   */
  std::pair<int, double> ArrangementSolver::walk(UnionFind &uf) {
    const int k = k_, na = static_cast<int>(arcs_.size());

    auto angle_at = [&](int v, const Vector3d &t) {
      return std::atan2(t.dot(eb_.col(v)), t.dot(ea_.col(v)));
    };

    darts_.clear();
    keys_.clear();
    for (int a = 0; a < na; ++a) {
      const Arc &arc = arcs_[a];
      if (arc.beg < 0)
        continue;

      const double out = angle_at(arc.end, -arc.tend),
                   in = angle_at(arc.beg, arc.tbeg);
      darts_.push_back({ out, a, false });
      keys_.push_back(arc.end);
      darts_.push_back({ in, a, true });
      keys_.push_back(arc.beg);
    }

    argsort_bucket(order_, off_.off().head(k + 1), eigen_map(keys_));

    auto succ = succ_.head(na);
    succ = ArrayXi::LinSpaced(na, 0, na - 1);
    double turn_sum = 0;
    for (int v = 0; v < k; ++v) {
      const int nv = off_.degree(v);
      if (nv == 0)
        continue;

      dring_.resize(nv + 1);
      for (int i = 0; i < nv; ++i)
        dring_[i + 1] = darts_[order_[off_[v] + i]];
      std::sort(dring_.begin() + 1, dring_.end(),
                [](const Dart &a, const Dart &b) {
                  return std::make_pair(a.angle, a.is_in)
                         < std::make_pair(b.angle, b.is_in);
                });
      dring_[0] = dring_[nv];

      for (int i = 0; i < nv; ++i) {
        const Dart &d = dring_[i + 1], &prev = dring_[i];
        ABSL_DCHECK_NE(d.is_in, prev.is_in) << "darts do not alternate";
        if (!d.is_in)
          continue;

        double iota = d.angle - prev.angle;
        iota += kTwoPi * static_cast<double>(iota <= -kPi / 2);
        iota -= kTwoPi * static_cast<double>(iota > 3 * kPi / 2);
        ABSL_CHECK_LE(iota, kPi + kSurfaceAngleEps)
            << "reflex corner at vertex " << v;
        succ[d.arc] = prev.arc;
        turn_sum += kPi - iota;
        uf.merge(arcs_[d.arc].cap, arcs_[prev.arc].cap);
      }
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
