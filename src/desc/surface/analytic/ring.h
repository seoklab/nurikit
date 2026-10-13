//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_DESC_SURFACE_ANALYTIC_RING_H_
#define NURI_DESC_SURFACE_ANALYTIC_RING_H_

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <absl/log/absl_check.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/desc/surface.h"
#include "stages.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  /**
   * One cut point of a circle: `vertex` is the caller's id of the point
   * (probe or root), `face` the caller's handle of the face whose root it
   * is, resolved by the `Ops` object; `leave` whether the circle leaves the
   * cutting ball here. `decide_wrap` sets `dec` when the arc to the next
   * vertex contains the reference ray and `coinc` when the two coincide.
   */
  struct RingVertex {
    int vertex, face;
    bool plus, leave;
    bool dec = false, coinc = false;
    int cls = 0;
    double phi = 0;
  };

  /**
   * The cut points of one circle in cyclic order with the frame `e1`, `e2`
   * of `SasArc`. The `Ops` object of the templates below answers the exact
   * queries on a vertex: `Vector3d offset(const RingVertex &)`, `int
   * half_plane(const RingVertex &)`, `Sgn ccw(const RingVertex &, const
   * RingVertex &)` and `bool antipode_accessible(const RingVertex &)`.
   * `covered` is whether any other ball cuts or contains the circle.
   */
  struct Ring {
    std::vector<RingVertex> verts;
    bool covered;
    Vector3d e1, e2;
  };

  using RingTangents = std::vector<std::pair<Vector3d, Vector3d>>;

  inline Vector3d ring_tangent(const Ring &ring, const double psi) {
    return -std::sin(psi) * ring.e1 + std::cos(psi) * ring.e2;
  }

  /**
   * Frame of a circle about unit `axis` with reference ray `d × e_k`.
   */
  inline std::pair<Vector3d, Vector3d> circle_frame(const Vector3d &d,
                                                    const Vector3d &axis) {
    const Vector3d e1 = reference_ray(d).normalized();
    return { e1, axis.cross(e1) };
  }

  /**
   * Angle of every vertex in `[0, 2π]` from the frame's reference ray, with
   * the exact half-plane class overriding the rounding of `atan2` at the
   * ray and at `π`, so the numeric angles are monotone in the exact order.
   */
  template <class Ops>
  void ring_angles(Ring &ring, const Ops &ops) {
    for (RingVertex &rv: ring.verts) {
      const Vector3d u = ops.offset(rv);
      const double phi = std::atan2(u.dot(ring.e2), u.dot(ring.e1));
      rv.cls = ops.half_plane(rv);
      switch (rv.cls) {
      case 0:
        rv.phi = 0;
        break;
      case 1:
        rv.phi = std::abs(phi);
        break;
      case 2:
        rv.phi = constants::kPi;
        break;
      default:
        rv.phi = constants::kTwoPi - std::abs(phi);
        break;
      }
    }
  }

  /**
   * Exact order within every run of consecutive vertices of one class whose
   * numeric angles are chained within `eps` (`kRingSortEps`; all of a class
   * `0` or `2` run); `verts` is sorted by `(cls, phi)` on entry.
   */
  template <class Ops>
  void sort_near_ties(std::vector<RingVertex> &verts, const Ops &ops,
                      const double eps) {
    const int n = static_cast<int>(verts.size());
    for (int beg = 0, end; beg < n; beg = end) {
      for (end = beg + 1; end < n && verts[end].cls == verts[beg].cls
                          && verts[end].phi - verts[end - 1].phi <= eps;
           ++end) { }

      for (int i = beg + 1; i < end; ++i) {
        for (int j = i; j > beg && ops.ccw(verts[j], verts[j - 1]) == Sgn::kPos;
             --j)
          std::swap(verts[j], verts[j - 1]);
      }
    }
  }

  /**
   * Which consecutive pair's arc contains the reference ray (its `dphi`
   * gets `+2π`) and which pairs coincide. A same-class pair of class `1` or
   * `3` is decided by its numeric angles when they differ by more than
   * `eps`, by the exact `ccw` otherwise. A ring without any decrease is a
   * single-point window: the `2π` goes to an arc of the accessibility of
   * the antipode.
   */
  template <class Ops>
  void decide_wrap(Ring &ring, const Ops &ops, const double eps) {
    const int n = static_cast<int>(ring.verts.size());

    int n_dec = 0;
    for (int k = 0; k < n; ++k) {
      RingVertex &p = ring.verts[k];
      const RingVertex &r = ring.verts[(k + 1) % n];
      ABSL_DCHECK_NE(p.leave, r.leave) << "ring does not alternate";
      p.dec = p.coinc = false;
      if (p.cls != r.cls) {
        p.dec = r.cls < p.cls;
      } else if (p.cls == 0 || p.cls == 2) {
        p.coinc = true;
      } else if (const double gap = r.phi - p.phi; std::abs(gap) > eps) {
        p.dec = gap < 0;
      } else {
        const Sgn s = ops.ccw(p, r);
        p.coinc = s == Sgn::kZero;
        p.dec = s == Sgn::kNeg;
      }
      n_dec += static_cast<int>(p.dec);
    }
    if (n_dec == 1)
      return;
    ABSL_DCHECK_EQ(n_dec, 0) << "ring wraps twice";

    const bool accessible = ops.antipode_accessible(ring.verts[0]);
    for (RingVertex &p: ring.verts) {
      if (p.leave == accessible) {
        p.dec = true;
        p.coinc = false;
        return;
      }
    }
  }

  /**
   * `decide_wrap` with every same-class pair decided by the exact `ccw`.
   */
  template <class Ops>
  void decide_wrap(Ring &ring, const Ops &ops) {
    decide_wrap(ring, ops, std::numeric_limits<double>::infinity());
  }

  /**
   * Accessible arcs of circle `circ`: from each leaving vertex to the next
   * vertex, with the departing tangent at the start and the arriving
   * tangent at the end from the frame and each end's own exact-class angle.
   * An empty ring is an accessible full circle iff nothing cuts or contains
   * the circle.
   */
  inline void emit_arcs(const Ring &ring, const int circ,
                        std::vector<SasArc> &arcs, RingTangents &tangents) {
    const int n = static_cast<int>(ring.verts.size());
    if (n == 0) {
      if (!ring.covered) {
        arcs.push_back({ 0.0, constants::kTwoPi, circ, -1, -1 });
        tangents.emplace_back(Vector3d::Zero(), Vector3d::Zero());
      }
      return;
    }

    for (int k = 0; k < n; ++k) {
      const RingVertex &p = ring.verts[k], &r = ring.verts[(k + 1) % n];
      if (!p.leave)
        continue;

      double dphi = 0;
      if (!p.coinc) {
        dphi =
            nuri::max(r.phi - p.phi + (p.dec ? constants::kTwoPi : 0.0), 0.0);
      }
      const double phi = p.phi > constants::kPi ? p.phi - constants::kTwoPi
                                                : p.phi;
      arcs.push_back({ phi, dphi, circ, p.vertex, r.vertex });
      tangents.emplace_back(ring_tangent(ring, p.phi),
                            ring_tangent(ring, r.phi));
    }
  }
}  // namespace internal
}  // namespace nuri

#endif /* NURI_DESC_SURFACE_ANALYTIC_RING_H_ */
