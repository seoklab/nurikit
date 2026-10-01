//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_DESC_SURFACE_ANALYTIC_RING_H_
#define NURI_DESC_SURFACE_ANALYTIC_RING_H_

#include <cmath>
#include <utility>
#include <vector>

#include <absl/log/absl_check.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/desc/surface.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  /**
   * One cut point of a circle: `vertex` is the caller's id of the point
   * (probe or root), `face` the caller's handle of the face whose root it
   * is, resolved by the `Ops` object; `leave` whether the circle leaves the
   * cutting ball here.
   */
  struct RingVertex {
    int vertex, face, cls;
    bool plus, leave;
    double phi;
  };

  /**
   * The cut points of one circle in cyclic order with the frame `e1`, `e2`
   * of `SasArc`. The `Ops` object of the templates below answers the exact
   * queries on a vertex: `Vector3d offset(const RingVertex &)`, `int
   * half_plane(const RingVertex &)`, `Sgn ccw(const RingVertex &, const
   * RingVertex &)` and `bool antipode_accessible(const RingVertex &)`.
   */
  struct Ring {
    std::vector<RingVertex> verts;
    std::vector<bool> dec, coinc;
    bool cut_any, inside_any;
    Vector3d e1, e2;
  };

  using RingTangents = std::vector<std::pair<Vector3d, Vector3d>>;

  inline Vector3d ring_tangent(const Ring &ring, const double psi) {
    return -std::sin(psi) * ring.e1 + std::cos(psi) * ring.e2;
  }

  /**
   * Frame of a circle about unit `axis` with reference ray `d × e_k`.
   */
  inline void ring_frame(Ring &ring, const Vector3d &d, const Vector3d &axis) {
    ring.e1 =
        d.cross(Vector3d::Unit(BallExact::reference_axis(d))).normalized();
    ring.e2 = axis.cross(ring.e1);
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
   * Which consecutive pair's arc contains the reference ray (its `dphi`
   * gets `+2π`) and which pairs coincide. A ring without any decrease is
   * a single-point window: the `2π` goes to an arc of the accessibility of
   * the antipode.
   */
  template <class Ops>
  void decide_wrap(Ring &ring, const Ops &ops) {
    const int n = static_cast<int>(ring.verts.size());
    ring.dec.assign(n, false);
    ring.coinc.assign(n, false);

    int n_dec = 0;
    for (int k = 0; k < n; ++k) {
      const RingVertex &p = ring.verts[k], &r = ring.verts[(k + 1) % n];
      ABSL_DCHECK_NE(p.leave, r.leave) << "ring does not alternate";
      if (p.cls != r.cls) {
        ring.dec[k] = r.cls < p.cls;
      } else if (p.cls == 0 || p.cls == 2) {
        ring.coinc[k] = true;
      } else {
        const Sgn s = ops.ccw(p, r);
        ring.coinc[k] = s == Sgn::kZero;
        ring.dec[k] = s == Sgn::kNeg;
      }
      n_dec += static_cast<int>(ring.dec[k]);
    }
    if (n_dec == 1)
      return;
    ABSL_DCHECK_EQ(n_dec, 0) << "ring wraps twice";

    const bool accessible = ops.antipode_accessible(ring.verts[0]);
    for (int k = 0; k < n; ++k) {
      if (ring.verts[k].leave == accessible) {
        ring.dec[k] = true;
        ring.coinc[k] = false;
        return;
      }
    }
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
      if (!ring.cut_any && !ring.inside_any) {
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
      if (!ring.coinc[k]) {
        dphi = nuri::max(
            r.phi - p.phi + (ring.dec[k] ? constants::kTwoPi : 0.0), 0.0);
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
