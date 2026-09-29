//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>
#include <vector>

#include <absl/log/absl_check.h>
#include <absl/types/span.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/desc/surface.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  namespace {
    using constants::kTwoPi;
  }  // namespace

  ArrangementSolver::ArrangementSolver(const int mcap, const int kcap,
                                       const int ecap)
      : off_(nuri::max(mcap, kcap)) {
    const int icap = 4 * ecap, acap = icap + mcap, dcap = 2 * acap;

    axis_.resize(3, mcap);
    e1_.resize(3, mcap);
    e2_.resize(3, mcap);
    cosa_.resize(mcap);
    sina_.resize(mcap);

    reps_.resize(3, kcap);
    ea_.resize(3, kcap);
    eb_.resize(3, kcap);
    accessible_.resize(kcap);

    edges_.reserve(ecap);
    incs_.reserve(icap);

    crossing_.resize(mcap, mcap);
    order_.resize(nuri::max(icap, dcap));
    succ_.resize(acap);
    seen_.resize(acap);
    keys_.reserve(nuri::max(icap, dcap));
    ring_.reserve(icap + 1);
    darts_.reserve(dcap);
    dring_.reserve(dcap + 1);
  }

  void ArrangementSolver::begin(const double radius) {
    radius_ = radius;
    m_ = k_ = 0;
    edges_.clear();
    incs_.clear();
  }

  int ArrangementSolver::add_cap(const Vector3d &axis, const double cosa,
                                 const double sina) {
    ABSL_DCHECK_LT(m_, axis_.cols());

    axis_.col(m_) = axis;
    cosa_[m_] = cosa;
    sina_[m_] = sina;
    return m_++;
  }

  int ArrangementSolver::add_vertex(const Vector3d &rep,
                                    const bool accessible) {
    ABSL_DCHECK_LT(k_, reps_.cols());

    reps_.col(k_) = rep;
    accessible_[k_] = accessible;
    return k_++;
  }

  double ArrangementSolver::cap_arcs(std::vector<SasArc> &arcs,
                                     const E::Map<ArrayXX<bool>> &crossing) {
    const int m = m_, k = k_, ni = static_cast<int>(incs_.size());
    auto axis = axis_.leftCols(m);
    auto cosa = cosa_.head(m);

    keys_.resize(ni);
    for (int i = 0; i < ni; ++i)
      keys_[i] = incs_[i].first;
    argsort_bucket(order_, off_.off().head(m + 1), eigen_map(keys_));

    double geo_sum = 0;
    for (int j = 0; j < m; ++j) {
      const Vector3d n = axis.col(j), e1 = e1_.col(j), e2 = e2_.col(j);

      ring_.clear();
      for (int i = off_[j]; i < off_[j + 1]; ++i) {
        const int v = incs_[order_[i]].second;
        if (accessible_[v])
          ring_.push_back({ 0.0, v });
      }
      std::sort(ring_.begin(), ring_.end(),
                [](const RingVertex &a, const RingVertex &b) {
                  return a.v < b.v;
                });
      ring_.erase(std::unique(ring_.begin(), ring_.end(),
                              [](const RingVertex &a, const RingVertex &b) {
                                return a.v == b.v;
                              }),
                  ring_.end());
      for (RingVertex &rv: ring_) {
        const Vector3d u = reps_.col(rv.v);
        rv.phi = std::atan2(u.dot(e2), u.dot(e1));
      }
      std::sort(ring_.begin(), ring_.end(),
                [](const RingVertex &a, const RingVertex &b) {
                  return a.phi < b.phi;
                });
      if (ring_.empty())
        ring_.push_back({ 0.0, k });

      const int nv = static_cast<int>(ring_.size());
      ring_.push_back(ring_.front());
      for (int i = 0; i < nv; ++i) {
        const RingVertex &beg = ring_[i], &end = ring_[i + 1];
        double dphi = end.phi - beg.phi;
        dphi += kTwoPi * static_cast<double>(dphi <= 0);

        const double mid = beg.phi + 0.5 * dphi;
        const Vector3d radial = std::cos(mid) * e1 + std::sin(mid) * e2;
        const Vector3d pt = cosa[j] * n + sina_[j] * radial;
        const bool inside =
            ((axis.transpose() * pt).array() > cosa && crossing.col(j)).any();
        if (inside)
          continue;

        arcs.push_back({ beg.phi, dphi, j, beg.v, end.v });
        geo_sum += dphi * cosa[j];
      }
    }

    return geo_sum;
  }

  void ArrangementSolver::snap_ring(absl::Span<Dart> ring) {
    std::sort(ring.begin(), ring.end(),
              [](const Dart &a, const Dart &b) { return a.angle < b.angle; });

    const int n = static_cast<int>(ring.size());
    const double raw0 = ring[0].angle, raw_last = ring[n - 1].angle;
    double prev_raw = raw0, snapped = raw0;
    int last_beg = 0;
    for (int i = 1; i < n; ++i) {
      const double raw = ring[i].angle;
      if (raw - prev_raw >= kSurfaceAngleEps) {
        snapped = raw;
        last_beg = i;
      }
      ring[i].angle = snapped;
      prev_raw = raw;
    }
    if (raw0 + kTwoPi - raw_last < kSurfaceAngleEps) {
      for (int i = last_beg; i < n; ++i)
        ring[i].angle = raw0;
    }

    std::sort(ring.begin(), ring.end(), [](const Dart &a, const Dart &b) {
      return std::make_tuple(a.angle, a.kappa, 2 * a.arc + a.is_in)
             < std::make_tuple(b.angle, b.kappa, 2 * b.arc + b.is_in);
    });
  }

  std::pair<int, double>
  ArrangementSolver::walk(const std::vector<SasArc> &arcs, const int a0) {
    const int k = k_, na = static_cast<int>(arcs.size()) - a0;

    auto angle_at = [&](int v, const Vector3d &t) {
      return std::atan2(t.dot(eb_.col(v)), t.dot(ea_.col(v)));
    };

    darts_.clear();
    keys_.clear();
    for (int a = 0; a < na; ++a) {
      const SasArc &arc = arcs[a0 + a];
      if (arc.beg == k)
        continue;

      const Vector3d n = axis_.col(arc.circ);
      const double cot = cosa_[arc.circ] / sina_[arc.circ];
      const Vector3d ub = reps_.col(arc.beg), ue = reps_.col(arc.end);
      darts_.push_back({ angle_at(arc.end, -n.cross(ue)), -cot, a, false });
      keys_.push_back(arc.end);
      darts_.push_back({ angle_at(arc.beg, n.cross(ub)), cot, a, true });
      keys_.push_back(arc.beg);
    }

    argsort_bucket(order_, off_.off().head(k + 1), eigen_map(keys_));

    ABSL_DCHECK_LE(na, succ_.size());
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
      snap_ring(absl::MakeSpan(dring_).subspan(1, nv));
      dring_[0] = dring_[nv];

      for (int i = 0; i < nv; ++i) {
        const Dart &d = dring_[i + 1], &prev = dring_[i];
        ABSL_DCHECK_NE(d.is_in, prev.is_in) << "darts do not alternate";
        if (!d.is_in)
          continue;

        double iota = d.angle - prev.angle;
        iota += kTwoPi * static_cast<double>(iota < 0);
        succ[d.arc] = prev.arc;
        turn_sum += constants::kPi - iota;
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

    auto build_crossing(ArrayXX<bool> &buffer,
                        const std::vector<std::pair<int, int>> &edges,
                        const int m) {
      auto crossing = take_buffer(buffer, m, m);
      crossing.setConstant(false);
      for (auto [a, b]: edges)
        crossing(a, b) = crossing(b, a) = true;
      return crossing;
    }
  }  // namespace

  double ArrangementSolver::solve(std::vector<SasArc> &arcs) {
    const int a0 = static_cast<int>(arcs.size());

    build_frames(e1_, e2_, axis_, m_);
    build_frames(ea_, eb_, reps_, k_);
    auto crossing = build_crossing(crossing_, edges_, m_);

    UnionFind uf(m_);
    for (auto [a, b]: edges_)
      uf.merge(a, b);
    const int n_components = uf.n_sets();

    const double geo_sum = cap_arcs(arcs, crossing);
    auto [n_loops, turn_sum] = walk(arcs, a0);

    const int n_patches = 1 + n_loops - n_components;
    const int chi = 2 * n_patches - n_loops;
    const double area = radius_ * radius_ * (kTwoPi * chi - turn_sum + geo_sum);
    ABSL_DCHECK_GE(area, -kSurfaceLengthEps * radius_ * radius_);
    return area;
  }
}  // namespace internal
}  // namespace nuri
