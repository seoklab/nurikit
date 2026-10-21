//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

#include <absl/log/absl_check.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "ring.h"
#include "stages.h"
#include "nuri/core/container/csr.h"
#include "nuri/core/geometry.h"
#include "nuri/desc/surface.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  namespace {
    using constants::kTwoPi;

    ArrayXd convex_areas(const SurfaceSpheres &ss, const SasGeometry &geo) {
      const int n = ss.n_active;
      return geo.area.head(n) * (ss.r.head(n) / ss.sar.head(n)).square();
    }

    struct Beta {
      double angle, sin;
    };

    struct SaddleRanges {
      Array4d beta;
      E::Array2d integral;
    };

    /**
     * Valid ranges `(lo0, hi0, lo1, hi1)` of the generating arc angle `beta`
     * on circle `(i, j)` and the integral `rl (hi - lo) - rp (sin hi - sin
     * lo)` of each. `beta` runs from `-theta_i` to `theta_j` with
     * `sin theta = a / R`; on a spindle (`rl < rp`) the part `|beta| < beta0`
     * lies beyond the axis and is cut out, `beta0 = 0` otherwise. Absent
     * parts are zero-width. Endpoints are selected together with their
     * sines, so no sine of a selected angle is ever evaluated.
     */
    SaddleRanges saddle_ranges(const double rl, const double rp,
                               const double a_i, const double a_j,
                               const double sas_i, const double sas_j) {
      const Beta lo { -std::atan2(a_i, rl), -a_i / sas_i },
          hi { std::atan2(a_j, rl), a_j / sas_j };
      const double root = std::sqrt(nuri::max(rp * rp - rl * rl, 0.0));
      const Beta b0 { std::atan2(root, rl), root / rp },
          nb0 { -b0.angle, -b0.sin };

      const Beta below_inner = hi.angle < nb0.angle ? hi : nb0,
                 above_inner = lo.angle > b0.angle ? lo : b0;
      const Beta below_hi = lo.angle > below_inner.angle ? lo : below_inner,
                 above_lo = hi.angle < above_inner.angle ? hi : above_inner;

      SaddleRanges out;
      out.beta << lo.angle, below_hi.angle, above_lo.angle, hi.angle;
      out.integral
          << rl * (below_hi.angle - lo.angle) - rp * (below_hi.sin - lo.sin),
          rl * (hi.angle - above_lo.angle) - rp * (hi.sin - above_lo.sin);
      return out;
    }

    void saddles(SesGeometry &ses, const SurfaceSpheres &ss,
                 const SasGeometry &geo) {
      const double rp = ss.rp;
      const int n_circ = ss.g.offset(ss.n_active);
      ses.saddle_beta.resize(4, n_circ);
      ses.saddle_integral.resize(2, n_circ);
      for (int q = 0; q < n_circ; ++q) {
        const SasCircle &c = geo.circles[q];
        const SaddleRanges sr = saddle_ranges(c.rl, rp, c.aij[0], c.aij[1],
                                              ss.sar[c.i], ss.sar[c.j]);
        ses.saddle_beta.col(q) = sr.beta;
        ses.saddle_integral.col(q) = sr.integral;
      }

      ses.saddle_area.resize(geo.n_active_arcs);
      for (int r = 0; r < geo.n_active_arcs; ++r) {
        const RingArc &arc = geo.arcs[r];
        ses.saddle_area[r] =
            arc.dphi * rp * ses.saddle_integral.col(arc.circ).sum();
      }
    }

    double segment_distance(const Vector3d &x, const Vector3d &p,
                            const Vector3d &q) {
      const Vector3d e = q - p;
      const double t = nuri::clamp((x - p).dot(e) / e.squaredNorm(), 0.0, 1.0);
      return (x - p - t * e).norm();
    }

    /**
     * Per probe: whether its ball reaches its contact triangle (`low`) and,
     * for low three-atom probes (`planar`), the unit normal of the contact
     * plane pointing from the probe toward the triangle with `cos b = h /
     * rp` for the plane distance `h < rp` (Corollary 1(a), (c) of the
     * analytic_surface page). The side is the root's: the probe is root
     * `sigma` of its face, `x = c_a + y_perp + sigma sqrt(D) u / |u|^2`
     * with `u = cross(c_b - c_a, c_c - c_a)` in the kernel's vertex order,
     * so the normal toward the plane is `-sigma u / |u|`, whatever the
     * rounded height. Probes of more than three atoms, or merged from
     * several roots (`members > 1`), are low and never planar.
     */
    struct ProbeHeights {
      ArrayXb low, planar;
      Matrix3Xd normal;
      ArrayXd cos_b, sin_b;
    };

    ProbeHeights probe_heights(const SurfaceSpheres &ss, const BallExact &ex,
                               const SasProbes &probes) {
      const double rp = ss.rp;
      const int np = static_cast<int>(probes.pos.cols());
      ProbeHeights hts { ArrayXb::Constant(np, true),
                         ArrayXb::Constant(np, false), Matrix3Xd::Zero(3, np),
                         ArrayXd::Zero(np), ArrayXd::Zero(np) };

      const auto centers = ex.centers();
      for (int p = 0; p < np; ++p) {
        if (probes.atoms.degree(p) != 3 || probes.members[p] != 1)
          continue;

        const Matrix3d tri = ss.pts(E::all, probes.atoms.nbrs(p));
        const Array3i f = probes.face.col(p);
        const Vector3d x = probes.pos.col(p), ca = centers.col(f[0]),
                       u = (centers.col(f[1]) - ca)
                               .cross(centers.col(f[2]) - ca)
                               .normalized(),
                       nrm = (tri.col(1) - tri.col(0))
                                 .cross(tri.col(2) - tri.col(0))
                                 .normalized();
        const double h = std::abs((x - ca).dot(u));

        bool inside = true;
        for (int m = 0; m < 3; ++m) {
          const Vector3d pm = tri.col(m), qm = tri.col((m + 1) % 3);
          inside &= (qm - pm).cross(x - pm).dot(nrm) >= 0;
        }

        double edge = segment_distance(x, tri.col(0), tri.col(1));
        for (int m = 1; m < 3; ++m) {
          edge = nuri::min(edge, segment_distance(x, tri.col(m),
                                                  tri.col((m + 1) % 3)));
        }

        const double dist = inside ? h : edge;
        hts.low[p] = dist < rp;
        if (!hts.low[p])
          continue;

        hts.planar[p] = true;
        hts.normal.col(p) = probes.plus[p] ? -u : u;
        hts.cos_b[p] = h / rp;
        hts.sin_b[p] = std::sqrt(1 - hts.cos_b[p] * hts.cos_b[p]);
      }
      return hts;
    }

    /**
     * Per unmerged probe (`members == 1`) with exactly three departure
     * tangents whose determinant clears `kTriangleDetMin` (`triangular`):
     * the corners
     * `p_m = +-normalize(cross(t_{m+1}, t_{m+2}))`, `dot(t_m, p_m) <= 0`, of
     * the spherical triangle `{d : dot(d, t_m) <= 0}`, in columns
     * `3p .. 3p + 2`, and the angle `H` within which every direction of the
     * exact triangle lies from the computed one.
     */
    struct FaceTriangles {
      ArrayXb triangular;
      Matrix3Xd corners;
      ArrayXd sin_h, cos_h;
    };

    auto probe_tangents(const SasProbes &probes, const int p) {
      return probes.tan.middleCols<3>(probes.tan_off[p]);
    }

    FaceTriangles face_triangles(const SasProbes &probes) {
      const int np = static_cast<int>(probes.pos.cols());
      FaceTriangles tri { ArrayXb::Constant(np, false),
                          Matrix3Xd::Zero(3, 3L * np), ArrayXd::Zero(np),
                          ArrayXd::Ones(np) };

      for (int p = 0; p < np; ++p) {
        if (probes.tan_off.degree(p) != 3 || probes.members[p] != 1)
          continue;

        const auto t = probe_tangents(probes, p);
        const double det = std::abs(t.col(0).dot(t.col(1).cross(t.col(2))));
        if (det < kTriangleDetMin)
          continue;

        tri.triangular[p] = true;
        tri.sin_h[p] = 3 * kTangentEps / (det - kTriangleDetRoundoff);
        tri.cos_h[p] = std::sqrt(1 - tri.sin_h[p] * tri.sin_h[p]);
        for (int m = 0; m < 3; ++m) {
          Vector3d v =
              t.col((m + 1) % 3).cross(t.col((m + 2) % 3)).normalized();
          if (t.col(m).dot(v) > 0)
            v = -v;
          tri.corners.col(3L * p + m) = v;
        }
      }
      return tri;
    }

    /**
     * Whether the cap `(axis, cos a)` on the sphere of `p` overlaps the cap
     * beyond the contact plane, `dot(d, normal) > cos b`: the angle between the
     * axes is below `a + b` (Lemma 9 of the analytic_surface page). Probes
     * without a plane pass.
     */
    bool meets_plane_cap(const ProbeHeights &hts, const int p,
                         const Vector3d &axis, const double cos_a,
                         const double sin_a) {
      return !hts.planar[p]
             || axis.dot(hts.normal.col(p))
                    > cos_a * hts.cos_b[p] - sin_a * hts.sin_b[p];
    }

    /**
     * Whether the cap `(axis, cos a)` on the sphere of `p`, widened by the
     * triangle's slack `H`, meets its computed spherical triangle: the axis
     * is inside, or within `a + H` of a corner, or within `a + H` of an edge
     * arc (Lemma 10 of the analytic_surface page). The foot of the axis on
     * the great circle of `t_m` lies on the arc iff it is on the triangle's
     * side of the other two tangents, `along_k - along_m dot(t_k, t_m) <= 0`.
     * Comparisons are on squared cosines, valid while `a + H < pi/2`; a
     * wider cap and probes without a triangle pass.
     */
    bool meets_triangle(const SasProbes &probes, const FaceTriangles &tri,
                        const int p, const Vector3d &axis, const double cos_a,
                        const double sin_a) {
      if (!tri.triangular[p])
        return true;

      const double cos_ah = cos_a * tri.cos_h[p] - sin_a * tri.sin_h[p];
      if (cos_ah <= 0)
        return true;

      const auto t = probe_tangents(probes, p);
      const auto c = tri.corners.middleCols<3>(3L * p);
      const Vector3d along = t.transpose() * axis;
      if ((along.array() <= 0).all())
        return true;

      const double cos2_ah = cos_ah * cos_ah;
      for (int m = 0; m < 3; ++m) {
        if (c.col(m).dot(axis) > cos_ah)
          return true;

        const int k1 = (m + 1) % 3, k2 = (m + 2) % 3;
        const bool on_arc = along[k1] - along[m] * t.col(k1).dot(t.col(m)) <= 0
                            && along[k2] - along[m] * t.col(k2).dot(t.col(m))
                                   <= 0;
        const double cos2_foot = 1 - along[m] * along[m];
        if (on_arc && cos2_foot > cos2_ah)
          return true;
      }
      return false;
    }

    struct PairCaps {
      ArrayXi left, right;
      Matrix3Xd diff, axis;
      ArrayXd cosa;
      int n;
    };

    /**
     * The first `n` entries are the probe pairs strictly within `2 rp`
     * whose caps may cut each other's face: both low and each cap meets the
     * other probe's beyond-plane cap and spherical triangle. `diff` runs
     * from `left` to `right`, taken from the exact kernel on the probes'
     * roots so its direction is right however close the probes are; `axis`
     * is its direction. The distance test on the positions is a prefilter;
     * the face solver decides cap existence exactly.
     */
    PairCaps pair_caps(const BallExact &ex, const SasProbes &probes,
                       const double rp, const ProbeHeights &hts,
                       const FaceTriangles &tri) {
      VoxelGrid grid(probes.pos, 2 * rp);
      std::vector<int> lbuf, rbuf;
      grid.find_neighbors_self(lbuf, rbuf);
      canonical_pairs(lbuf, rbuf);
      const int m = static_cast<int>(lbuf.size());
      auto left = eigen_map(lbuf), right = eigen_map(rbuf);

      const int np = static_cast<int>(probes.pos.cols());
      ArrayXi order(m), by_right(m);
      OffsetTable off(np);
      argsort_bucket(by_right, off.off().head(np), right);
      argsort_bucket(order, off, left, by_right);

      PairCaps pc { ArrayXi(m),      ArrayXi(m), Matrix3Xd(3, m),
                    Matrix3Xd(3, m), ArrayXd(m), 0 };
      int w = 0;
      for (const int k: order) {
        const int l = left[k], r = right[k];
        if (!hts.low[l] || !hts.low[r])
          continue;

        const auto [axis, dist] =
            ex.difference(ball_triple(probes.face.col(r)), probes.plus[r],
                          ball_triple(probes.face.col(l)), probes.plus[l]);
        const double cos_a = dist / (2 * rp);
        // The grid distance on rounded positions and the exact difference may
        // disagree by an ulp.
        if (cos_a >= 1)
          continue;

        const Vector3d diff = dist * axis;
        const double sin_a = std::sqrt(1 - cos_a * cos_a);
        if (!meets_plane_cap(hts, l, axis, cos_a, sin_a)
            || !meets_plane_cap(hts, r, -axis, cos_a, sin_a)
            || !meets_triangle(probes, tri, l, axis, cos_a, sin_a)
            || !meets_triangle(probes, tri, r, -axis, cos_a, sin_a))
          continue;

        pc.left[w] = l;
        pc.right[w] = r;
        pc.diff.col(w) = diff;
        pc.axis.col(w) = axis;
        pc.cosa[w] = cos_a;
        ++w;
      }
      pc.n = w;
      return pc;
    }

    /**
     * Every cap of the active faces lifted to a ball relative to its probe
     * (`solve_ses_face`): `(t, 0)` for a hemisphere, `(w, |w|^2)` for a
     * neighbor at `w`, the hemisphere of its direction once `|w|^2` is 0.
     */
    struct LiftedCaps {
      Matrix3Xd n;
      ArrayXd h;
    };

    /**
     * Caps of every active face, hemispheres of the departure tangents
     * first, then the surviving neighbor caps.
     */
    LiftedCaps faces(SesGeometry &ses, const SasProbes &probes,
                     const PairCaps &pc) {
      const int na = probes.n_active, m = pc.n;

      ArrayXi &off = ses.face_off.off();
      off = ArrayXi::Zero(na + 1);
      for (int p = 0; p < na; ++p)
        off[p + 1] = probes.tan_off.degree(p);
      for (int k = 0; k < m; ++k) {
        for (const int p: { pc.left[k], pc.right[k] }) {
          if (p < na)
            ++off[p + 1];
        }
      }
      std::inclusive_scan(off.begin(), off.end(), off.begin());

      const int total = off[na];
      ses.face_axis.resize(3, total);
      ses.face_cosa.resize(total);
      LiftedCaps lifted { Matrix3Xd(3, total), ArrayXd(total) };

      ArrayXi cur = off.head(na);
      for (int p = 0; p < na; ++p) {
        const int nt = probes.tan_off.degree(p);
        ses.face_axis.middleCols(cur[p], nt) =
            probes.tan.middleCols(probes.tan_off[p], nt);
        ses.face_cosa.segment(cur[p], nt).setZero();
        lifted.n.middleCols(cur[p], nt) = ses.face_axis.middleCols(cur[p], nt);
        lifted.h.segment(cur[p], nt).setZero();
        cur[p] += nt;
      }

      for (int k = 0; k < m; ++k) {
        const std::array<int, 2> ends { pc.left[k], pc.right[k] };
        for (int side = 0; side < 2; ++side) {
          const int p = ends[side];
          if (p >= na)
            continue;

          const Vector3d axis = pc.axis.col(k), diff = pc.diff.col(k);
          const double h = diff.squaredNorm();
          const Vector3d lift = h > 0 ? diff : axis;
          ses.face_axis.col(cur[p]) = side == 0 ? axis : Vector3d(-axis);
          ses.face_cosa[cur[p]] = pc.cosa[k];
          lifted.n.col(cur[p]) = side == 0 ? lift : Vector3d(-lift);
          lifted.h[cur[p]] = h;
          ++cur[p];
        }
      }
      return lifted;
    }

    struct FacePair {
      int m, l;
    };

    struct FaceRoot {
      int pair;
      bool plus;
    };

    /**
     * Exact queries of `ring.h` on circle `(0, m)` of a face's lifted balls;
     * a vertex's `face` is its index into `pairs`. The antipode is
     * accessible iff it is outside every other live cap.
     */
    // NOLINTBEGIN(*-unused-member-function)
    class FaceRingOps {
    public:
      FaceRingOps(const BallExact &ex, const std::vector<FacePair> &pairs,
                  const ArrayXb &live, int m)
          : ex_(&ex), pairs_(&pairs), live_(&live), m_(m) { }

      BallTriple face(const RingVertex &v) const {
        const FacePair &pr = (*pairs_)[v.face];
        return { 0, pr.m, pr.l };
      }

      Vector3d offset(const RingVertex &v) const {
        return ex_->offset(0, m_, face(v), v.plus);
      }

      int half_plane(const RingVertex &v) const {
        return ex_->half_plane(0, m_, face(v), v.plus);
      }

      Sgn ccw(const RingVertex &p, const RingVertex &r) const {
        return ex_->ccw_perturbed(0, m_, face(p), p.plus, face(r), r.plus);
      }

      bool antipode_accessible(const RingVertex &v) const {
        const BallTriple f = face(v);
        bool accessible = true;
        for (int k = 1; k < ex_->n(); ++k) {
          if (k != m_ && (*live_)[k - 1])
            accessible &= ex_->antipode(0, m_, f, v.plus, k) == Sgn::kPos;
        }
        return accessible;
      }

    private:
      const BallExact *ex_;
      const std::vector<FacePair> *pairs_;
      const ArrayXb *live_;
      int m_;
    };
    // NOLINTEND(*-unused-member-function)

    /**
     * Face solver with its buffers; cap `k` of the input is ball `k + 1`,
     * the probe ball 0. Faces are `(0, m, l)`, `m < l`, circles `(0, m)`
     * oriented about the cap axis `n_m`, so the root `+` leaves ball `l`
     * along circle `(0, m)` and enters it along `(0, l)` (`face_parity` of
     * sas.cpp).
     */
    class FaceSolver {
    public:
      explicit FaceSolver(const int mcap)
          : solver_(mcap, mcap * (mcap - 1), mcap * mcap), c_(3, mcap + 1),
            h_(mcap + 1), slot_(mcap + 1), cap_of_slot_(mcap),
            covered_(mcap + 1) {
        ex_.reserve(mcap + 1);
      }

      double solve(ConstBlock<Matrix3Xd, 3, E::Dynamic, true> n,
                   ConstVecBlock<ArrayXd> h, ConstVecBlock<ArrayXd> cosa,
                   const double rp, ArrayXb &live) {
        const int m = static_cast<int>(h.size());
        ABSL_DCHECK_LE(m + 1, c_.cols());
        ABSL_DCHECK_GE(live.size(), m);

        c_.col(0).setZero();
        h_[0] = 0;
        c_.middleCols(1, m) = n;
        h_.segment(1, m) = h;
        ex_.load(std::as_const(c_).leftCols(m + 1),
                 std::as_const(h_).head(m + 1), rp * rp);

        // the pair stage only prefilters; the exact overlap decides
        for (int k = 1; k <= m; ++k)
          live[k - 1] = ex_.overlap(0, k) == Sgn::kPos;
        // caps on one circle, opposite sides: the sphere is covered; same
        // side or equal balls (closer than the certificate resolves): one cap
        for (int b = 1; b <= m; ++b) {
          for (int c = b + 1; c <= m; ++c) {
            if (!live[b - 1] || !live[c - 1])
              continue;
            const int s = ex_.shared_circle({ 0, b, c });
            if (s == 0)
              return 0;
            if (s > 0)
              live[c - 1] = false;
          }
        }

        solver_.begin(rp);
        for (int b = 1; b <= m; ++b) {
          if (!live[b - 1])
            continue;
          slot_[b] = solver_.add_cap(cosa[b - 1]);
          cap_of_slot_[slot_[b]] = b;
        }

        collect_roots(m, live);
        for (int b = 1; b <= m; ++b) {
          if (live[b - 1])
            ring_arcs(b, live);
        }

        return solver_.solve([&](int j, int l) {
          return ex_.discs_intersect(0, cap_of_slot_[j], cap_of_slot_[l]);
        });
      }

    private:
      /**
       * Roots of every cutting pair of live caps that lie outside every
       * other live cap, in order, so that a root's index is its solver
       * vertex; per circle whether any pair cuts it or buries it.
       */
      void collect_roots(const int m, const ArrayXb &live) {
        covered_.head(m + 1).setConstant(false);
        pairs_.clear();
        roots_.clear();

        for (int b = 1; b <= m; ++b) {
          for (int c = b + 1; c <= m; ++c) {
            if (!live[b - 1] || !live[c - 1])
              continue;

            ex_.load(*k_, { 0, b, c });
            if (ex_.cuts(*k_) != Sgn::kPos) {
              covered_[b] |= ex_.side(0, b, c) == Sgn::kNeg;
              covered_[c] |= ex_.side(0, c, b) == Sgn::kNeg;
              continue;
            }
            covered_[b] = covered_[c] = true;

            const int pair = static_cast<int>(pairs_.size());
            pairs_.push_back({ b, c });
            for (const bool plus: { true, false }) {
              bool accepted = true;
              for (int k = 1; k <= m && accepted; ++k) {
                accepted = k == b || k == c || !live[k - 1]
                           || ex_.accept(*k_, plus, k) == Sgn::kPos;
              }
              if (!accepted)
                continue;

              roots_.push_back({ pair, plus });
              solver_.add_vertex(ex_.root(*k_, plus).normalized());
            }
          }
        }
      }

      /**
       * Ring of circle `(0, b)` sorted by exact class, then by the numeric
       * angle with the perturbed orientation deciding the near-ties within
       * `kRingSortEps` (coincident roots are generic here), and its arcs.
       */
      void ring_arcs(const int b, const ArrayXb &live) {
        ring_.verts.clear();
        ring_.covered = covered_[b];
        for (int r = 0; r < static_cast<int>(roots_.size()); ++r) {
          const FaceRoot &rt = roots_[r];
          const FacePair &pr = pairs_[rt.pair];
          if (pr.m == b)
            ring_.verts.push_back({ r, rt.pair, rt.plus, rt.plus });
          else if (pr.l == b)
            ring_.verts.push_back({ r, rt.pair, rt.plus, !rt.plus });
        }

        const int nv = static_cast<int>(ring_.verts.size());
        if (nv > 0) {
          ABSL_DCHECK_EQ(nv % 2, 0) << "odd ring on cap " << b;
          const FaceRingOps ops(ex_, pairs_, live, b);
          const Vector3d axis = c_.col(b);
          const auto [e1, e2] = circle_frame(axis, axis.normalized());
          ring_.e1 = e1;
          ring_.e2 = e2;
          ring_angles(ring_, ops);
          std::sort(ring_.verts.begin(), ring_.verts.end(),
                    [](const RingVertex &p, const RingVertex &r) {
                      return p.cls < r.cls || (p.cls == r.cls && p.phi < r.phi);
                    });
          sort_near_ties(ring_.verts, ops, kRingSortEps);
          decide_wrap(ring_, ops, kRingSortEps);
        }

        arcs_.clear();
        tangents_.clear();
        emit_arcs(ring_, b, arcs_, tangents_);
        for (int a = 0; a < static_cast<int>(arcs_.size()); ++a) {
          const RingArc &arc = arcs_[a];
          const auto &[tb, te] = tangents_[a];
          solver_.add_arc(slot_[b], arc.beg, arc.end, arc.dphi, tb, te);
        }
      }

      ArrangementSolver solver_;
      BallExact ex_;
      FaceKernelPtr k_ = face_kernel();
      Matrix3Xd c_;
      ArrayXd h_;
      ArrayXi slot_, cap_of_slot_;
      ArrayXb covered_;
      std::vector<FacePair> pairs_;
      std::vector<FaceRoot> roots_;
      Ring ring_;
      std::vector<RingArc> arcs_;
      RingTangents tangents_;
    };

    /**
     * Area of face `p` with caps `beg .. beg + m - 1`: high faces with a
     * triangle get their closed form (Corollary 1(c)), every other face is
     * solved exactly on its lifted caps.
     */
    double face_area(FaceSolver &solver, const SasProbes &probes,
                     const LiftedCaps &lifted, const ProbeHeights &hts,
                     const FaceTriangles &tri, const ArrayXd &face_cosa,
                     const double rp, const int p, const int beg, const int m,
                     ArrayXb &live) {
      if (hts.low[p] || !tri.triangular[p]) {
        return solver.solve(lifted.n.middleCols(beg, m),
                            lifted.h.segment(beg, m), face_cosa.segment(beg, m),
                            rp, live);
      }

      const auto t = probe_tangents(probes, p);
      double total = 0;
      for (int k = 0; k < 3; ++k) {
        const Vector3d t1 = t.col(k), t2 = t.col((k + 1) % 3);
        total += std::atan2(t1.cross(t2).norm(), t1.dot(t2));
      }
      return rp * rp * (kTwoPi - total);
    }

    /**
     * Area of every active face; the cap lists are compacted to the caps
     * that survived hygiene.
     */
    void solve_faces(SesGeometry &ses, const SasProbes &probes,
                     const LiftedCaps &lifted, const ProbeHeights &hts,
                     const FaceTriangles &tri, const double rp) {
      const int na = probes.n_active;
      ses.face_area.resize(na);
      if (na == 0)
        return;

      const int mcap = ses.face_off.max_deg();
      FaceSolver solver(mcap);
      ArrayXb live(mcap);

      ArrayXi &off = ses.face_off.off();
      int w = 0;
      for (int p = 0; p < na; ++p) {
        const int beg = off[p], m = off[p + 1] - beg;
        off[p] = w;
        live.head(m).setConstant(true);

        ses.face_area[p] = face_area(solver, probes, lifted, hts, tri,
                                     ses.face_cosa, rp, p, beg, m, live);

        for (int k = 0; k < m; ++k) {
          if (!live[k])
            continue;
          ses.face_axis.col(w) = ses.face_axis.col(beg + k);
          ses.face_cosa[w] = ses.face_cosa[beg + k];
          ++w;
        }
      }
      off[na] = w;
      ses.face_axis.conservativeResize(3, w);
      ses.face_cosa.conservativeResize(w);
    }
  }  // namespace

  double solve_ses_face(const Matrix3Xd &n, const ArrayXd &h,
                        const ArrayXd &cosa, const double rp, ArrayXb &live) {
    const int m = static_cast<int>(h.size());
    FaceSolver solver(m);
    live.resize(m);
    return solver.solve(n.leftCols(m), h.head(m), cosa.head(m), rp, live);
  }

  SesGeometry build_ses(const SurfaceSpheres &ss, const SasDelaunay &del,
                        const SasGeometry &geo) {
    const double rp = ss.rp;
    SesGeometry ses;
    ses.convex_area = convex_areas(ss, geo);
    saddles(ses, ss, geo);

    const SasProbes &probes = geo.probes;
    const ProbeHeights hts = probe_heights(ss, del.ex, probes);
    const FaceTriangles tri = face_triangles(probes);
    const PairCaps pc = pair_caps(del.ex, probes, rp, hts, tri);
    const LiftedCaps lifted = faces(ses, probes, pc);
    solve_faces(ses, probes, lifted, hts, tri, rp);
    return ses;
  }
}  // namespace internal
}  // namespace nuri
