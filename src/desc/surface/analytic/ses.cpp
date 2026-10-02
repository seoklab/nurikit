//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include <absl/log/absl_check.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "ring.h"
#include "nuri/core/geometry.h"
#include "nuri/desc/surface.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  namespace {
    using constants::kTwoPi;

    ArrayXd convex_areas(const SaPrep &sa, const SasGeometry &geo,
                         const double rp) {
      const auto sar = sa.sar.head(sa.n_active);
      return geo.area.head(sa.n_active) * ((sar - rp) / sar).square();
    }

    struct Beta {
      double angle, sin;
    };

    Beta pick(const bool cond, const Beta &x, const Beta &y) {
      return cond ? x : y;
    }

    struct SaddleRanges {
      Array4d beta;
      E::Array2d integral;
    };

    /**
     * Valid ranges `(lo0, hi0, lo1, hi1)` of the generating arc angle `β`
     * on circle `(i, j)` and the integral `rl (hi − lo) − rp (sin hi − sin
     * lo)` of each. `β` runs from `−θ_i` to `θ_j` with `sin θ = a / R`; on
     * a spindle (`rl < rp`) the part `|β| < β0` lies beyond the axis and is
     * cut out, `β0 = 0` otherwise. Absent parts are zero-width. Endpoints
     * are selected together with their sines, so no sine of a selected
     * angle is ever evaluated.
     */
    SaddleRanges saddle_ranges(const double rl, const double rp,
                               const double a_i, const double a_j,
                               const double sas_i, const double sas_j) {
      const Beta lo { -std::atan2(a_i, rl), -a_i / sas_i },
          hi { std::atan2(a_j, rl), a_j / sas_j };
      const double root = std::sqrt(nuri::max(rp * rp - rl * rl, 0.0));
      const Beta b0 { std::atan2(root, rl), root / rp },
          nb0 { -b0.angle, -b0.sin };

      Beta m = pick(hi.angle < nb0.angle, hi, nb0);
      const Beta below_hi = pick(lo.angle > m.angle, lo, m);
      m = pick(lo.angle > b0.angle, lo, b0);
      const Beta above_lo = pick(hi.angle < m.angle, hi, m);

      SaddleRanges out;
      out.beta << lo.angle, below_hi.angle, above_lo.angle, hi.angle;
      out.integral
          << rl * (below_hi.angle - lo.angle) - rp * (below_hi.sin - lo.sin),
          rl * (hi.angle - above_lo.angle) - rp * (hi.sin - above_lo.sin);
      return out;
    }

    void saddles(SesGeometry &ses, const SaPrep &sa, const SasGeometry &geo,
                 const double rp) {
      const int n_circ = sa.g.offset(sa.n_active);
      ses.saddle_beta.resize(4, n_circ);
      ses.saddle_integral.resize(2, n_circ);
      for (int q = 0; q < n_circ; ++q) {
        const SasCircle &c = geo.circles[q];
        const SaddleRanges sr = saddle_ranges(c.rl, rp, c.a, sa.d[q] - c.a,
                                              sa.sar[c.i], sa.sar[c.j]);
        ses.saddle_beta.col(q) = sr.beta;
        ses.saddle_integral.col(q) = sr.integral;
      }

      ses.saddle_area.resize(geo.n_active_arcs);
      for (int r = 0; r < geo.n_active_arcs; ++r) {
        const SasArc &arc = geo.arcs[r];
        ses.saddle_area[r] =
            arc.dphi * rp * ses.saddle_integral.col(arc.circ).sum();
      }
    }

    double sign(const double x) {
      return static_cast<double>(x > 0) - static_cast<double>(x < 0);
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
     * rp` for the plane distance `h < rp` (Lemma 1(a), (c)). Probes of more
     * than three atoms are low and never planar.
     */
    struct ProbeHeights {
      ArrayXb low, planar;
      Matrix3Xd normal;
      ArrayXd cos_b, sin_b;
    };

    ProbeHeights probe_heights(const SaPrep &sa, const SasProbes &probes,
                               const double rp) {
      const int np = static_cast<int>(probes.pos.cols());
      ProbeHeights hts { ArrayXb::Constant(np, true),
                         ArrayXb::Constant(np, false), Matrix3Xd::Zero(3, np),
                         ArrayXd::Zero(np), ArrayXd::Zero(np) };

      for (int p = 0; p < np; ++p) {
        if (probes.atoms.degree(p) != 3)
          continue;

        const Matrix3d tri = sa.pts(E::all, probes.atoms.nbrs(p));
        const Vector3d x = probes.pos.col(p),
                       nrm = (tri.col(1) - tri.col(0))
                                 .cross(tri.col(2) - tri.col(0))
                                 .normalized();
        const double signed_h = (x - tri.col(0)).dot(nrm),
                     h = std::abs(signed_h);

        bool inside = true;
        double edge = std::numeric_limits<double>::infinity();
        for (int m = 0; m < 3; ++m) {
          const Vector3d pm = tri.col(m), qm = tri.col((m + 1) % 3);
          inside &= (qm - pm).cross(x - pm).dot(nrm) >= 0;
          edge = nuri::min(edge, segment_distance(x, pm, qm));
        }

        const double dist = inside ? h : edge;
        hts.low[p] = dist < rp;
        if (!hts.low[p])
          continue;

        hts.planar[p] = true;
        hts.normal.col(p) = -sign(signed_h) * nrm;
        hts.cos_b[p] = h / rp;
        hts.sin_b[p] = std::sqrt(1 - hts.cos_b[p] * hts.cos_b[p]);
      }
      return hts;
    }

    /**
     * Per probe with exactly three linearly independent departure tangents
     * (`triangular`): the corners `p_m = ±normalize(t_{m+1} × t_{m+2})`,
     * `t_m · p_m ≤ 0`, of the spherical triangle `{d : d · t_m ≤ 0}`, in
     * columns `3p .. 3p + 2`.
     */
    struct FaceTriangles {
      ArrayXb triangular;
      Matrix3Xd corners;
    };

    auto probe_tangents(const SasProbes &probes, const int p) {
      return probes.tan.middleCols<3>(probes.tan_off[p]);
    }

    FaceTriangles face_triangles(const SasProbes &probes) {
      const int np = static_cast<int>(probes.pos.cols());
      FaceTriangles tri { ArrayXb::Constant(np, false),
                          Matrix3Xd::Zero(3, 3L * np) };

      for (int p = 0; p < np; ++p) {
        if (probes.tan_off.degree(p) != 3)
          continue;

        const auto t = probe_tangents(probes, p);
        if (t.col(0).dot(t.col(1).cross(t.col(2))) == 0.0)
          continue;

        tri.triangular[p] = true;
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

    double triangle_area(const SasProbes &probes, const int p,
                         const double rp) {
      const auto t = probe_tangents(probes, p);
      double total = 0;
      for (int m = 0; m < 3; ++m) {
        const Vector3d t1 = t.col(m), t2 = t.col((m + 1) % 3);
        total += std::atan2(t1.cross(t2).norm(), t1.dot(t2));
      }
      return rp * rp * (kTwoPi - total);
    }

    /**
     * Whether the cap `(axis, cos a)` on the sphere of `p` overlaps the cap
     * beyond the contact plane, `d · normal > cos b`: the angle between the
     * axes is below `a + b` (Lemma 5). Probes without a plane pass.
     */
    bool meets_plane_cap(const ProbeHeights &hts, const int p,
                         const Vector3d &axis, const double cos_a,
                         const double sin_a) {
      return !hts.planar[p]
             || axis.dot(hts.normal.col(p))
                    > cos_a * hts.cos_b[p] - sin_a * hts.sin_b[p];
    }

    /**
     * Whether the cap `(axis, cos a)` on the sphere of `p` meets its
     * spherical triangle: the axis is inside, or within `a` of a corner, or
     * within `a` of an edge arc (Lemma 6). The foot of the axis on the
     * great circle of `t_m` lies on the arc iff it is on the arc's side of
     * the corners' bisector `p + q`; the corners are perpendicular to `t_m`,
     * so the foot's component along the bisector is the axis's own. All
     * comparisons are on squared cosines. Probes without a triangle pass.
     */
    bool meets_triangle(const SasProbes &probes, const FaceTriangles &tri,
                        const int p, const Vector3d &axis, const double cos_a) {
      if (!tri.triangular[p])
        return true;

      const auto t = probe_tangents(probes, p);
      const auto c = tri.corners.middleCols<3>(3L * p);
      const Vector3d along = t.transpose() * axis;
      if ((along.array() <= 0).all())
        return true;

      const double cos2_a = cos_a * cos_a;
      for (int m = 0; m < 3; ++m) {
        if (c.col(m).dot(axis) > cos_a)
          return true;

        const Vector3d pc = c.col((m + 1) % 3), mid = pc + c.col((m + 2) % 3);
        const double axis_mid = axis.dot(mid), p_mid = pc.dot(mid),
                     cos2_foot = 1 - along[m] * along[m];
        const bool on_arc = axis_mid >= 0
                            && axis_mid * axis_mid >= cos2_foot * p_mid * p_mid;
        if (on_arc && cos2_foot > cos2_a)
          return true;
      }
      return false;
    }

    struct PairCaps {
      ArrayXi left, right;
      Matrix3Xd diff, axis;
      ArrayXd cosa, sina;
    };

    BallTriple probe_face(const SasProbes &probes, const int p) {
      return { probes.face(0, p), probes.face(1, p), probes.face(2, p) };
    }

    /**
     * Probe pairs strictly within `2 rp` whose caps may cut each other's
     * face: both low and each cap meets the other probe's beyond-plane cap
     * and spherical triangle. `diff` runs from `left` to `right`, taken
     * from the exact kernel on the probes' roots so its direction is right
     * however close the probes are; `axis` is its direction. The distance
     * test on the positions is a prefilter; the face solver decides cap
     * existence exactly.
     */
    PairCaps pair_caps(const BallExact &ex, const SasProbes &probes,
                       const double rp, const ProbeHeights &hts,
                       const FaceTriangles &tri) {
      VoxelGrid grid(probes.pos, 2 * rp);
      std::vector<int> lbuf, rbuf;
      grid.find_neighbors_self(lbuf, rbuf);
      const int m = static_cast<int>(lbuf.size());

      PairCaps pc { ArrayXi(m),  ArrayXi(m), Matrix3Xd(3, m), Matrix3Xd(3, m),
                    ArrayXd(m), ArrayXd(m) };
      int w = 0;
      for (int k = 0; k < m; ++k) {
        const int l = lbuf[k], r = rbuf[k];
        if (!hts.low[l] || !hts.low[r])
          continue;

        const Vector3d diff =
            ex.difference(probe_face(probes, r), probes.plus[r],
                          probe_face(probes, l), probes.plus[l]);
        const double dist = diff.norm(), cos_a = dist / (2 * rp);
        ABSL_DCHECK_GT(dist, 0);
        if (cos_a >= 1)
          continue;

        const Vector3d axis = diff / dist;
        const double sin_a = std::sqrt(1 - cos_a * cos_a);
        if (!meets_plane_cap(hts, l, axis, cos_a, sin_a)
            || !meets_plane_cap(hts, r, -axis, cos_a, sin_a)
            || !meets_triangle(probes, tri, l, axis, cos_a)
            || !meets_triangle(probes, tri, r, -axis, cos_a))
          continue;

        pc.left[w] = l;
        pc.right[w] = r;
        pc.diff.col(w) = diff;
        pc.axis.col(w) = axis;
        pc.cosa[w] = cos_a;
        pc.sina[w] = sin_a;
        ++w;
      }

      pc.left.conservativeResize(w);
      pc.right.conservativeResize(w);
      pc.diff.conservativeResize(3, w);
      pc.axis.conservativeResize(3, w);
      pc.cosa.conservativeResize(w);
      pc.sina.conservativeResize(w);
      return pc;
    }

    /**
     * Every cap of the active faces lifted to a ball relative to its probe
     * (`solve_ses_face`): `(t, 0)` for a hemisphere, `(w, |w|²)` for a
     * neighbour at `w`.
     */
    struct LiftedCaps {
      Matrix3Xd n;
      ArrayXd h;
    };

    /**
     * Caps of every active face, hemispheres of the departure tangents
     * first, then the surviving neighbour caps; high faces with a triangle
     * get their closed-form area (Lemma 1(c)), every other face NaN for the
     * solver.
     */
    LiftedCaps faces(SesGeometry &ses, const SaPrep &sa, const SasDelaunay &del,
                     const SasGeometry &geo, const double rp) {
      const SasProbes &probes = geo.probes;
      const int na = probes.n_active;
      const ProbeHeights hts = probe_heights(sa, probes, rp);
      const FaceTriangles tri = face_triangles(probes);
      const PairCaps pc = pair_caps(del.ex, probes, rp, hts, tri);
      const int m = static_cast<int>(pc.left.size());

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
      ses.face_sina.resize(total);
      ses.face_area.resize(na);
      LiftedCaps lifted { Matrix3Xd(3, total), ArrayXd(total) };

      ArrayXi cur = off.head(na);
      for (int p = 0; p < na; ++p) {
        const int nt = probes.tan_off.degree(p);
        ses.face_axis.middleCols(cur[p], nt) =
            probes.tan.middleCols(probes.tan_off[p], nt);
        ses.face_cosa.segment(cur[p], nt).setZero();
        ses.face_sina.segment(cur[p], nt).setOnes();
        lifted.n.middleCols(cur[p], nt) = ses.face_axis.middleCols(cur[p], nt);
        lifted.h.segment(cur[p], nt).setZero();
        cur[p] += nt;

        ses.face_area[p] = !hts.low[p] && tri.triangular[p]
                               ? triangle_area(probes, p, rp)
                               : std::numeric_limits<double>::quiet_NaN();
      }

      for (int k = 0; k < m; ++k) {
        const std::array<int, 2> ends { pc.left[k], pc.right[k] };
        for (int side = 0; side < 2; ++side) {
          const int p = ends[side];
          if (p >= na)
            continue;

          ses.face_axis.col(cur[p]) = (1 - 2 * side) * pc.axis.col(k);
          ses.face_cosa[cur[p]] = pc.cosa[k];
          ses.face_sina[cur[p]] = pc.sina[k];
          lifted.n.col(cur[p]) = (1 - 2 * side) * pc.diff.col(k);
          lifted.h[cur[p]] = pc.diff.col(k).squaredNorm();
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

    /**
     * Face solver with its buffers; cap `k` of the input is ball `k + 1`,
     * the probe ball 0. Faces are `(0, m, l)`, `m <
     * l`, circles `(0, m)` oriented about the cap axis `n_m`, so the root
     * `+` leaves ball `l` along circle `(0, m)` and enters it along `(0, l)`
     * (`face_parity` of sas.cpp).
     */
    class FaceSolver {
    public:
      explicit FaceSolver(const int mcap)
          : solver_(mcap, mcap * (mcap - 1), mcap * mcap), c_(3, mcap + 1),
            h_(mcap + 1), slot_(mcap + 1), cap_of_slot_(mcap),
            cut_any_(mcap + 1), inside_any_(mcap + 1) { }

      double solve(const Matrix3Xd &n, const ArrayXd &h, const ArrayXd &cosa,
                   const double rp, ArrayXb &live) {
        const int m = static_cast<int>(h.size());
        ABSL_DCHECK_LE(m + 1, c_.cols());
        ABSL_DCHECK_GE(live.size(), m);

        c_.col(0).setZero();
        h_[0] = 0;
        c_.middleCols(1, m) = n;
        h_.segment(1, m) = h;
        const BallExact ex =
            BallExact::make(c_.leftCols(m + 1), h_.head(m + 1), rp * rp);

        for (int k = 1; k <= m; ++k)
          live[k - 1] = ex.overlap(0, k) == Sgn::kPos;
        for (int b = 1; b <= m; ++b) {
          for (int c = b + 1; c <= m; ++c) {
            if (!live[b - 1] || !live[c - 1])
              continue;
            const int s = ex.shared_circle({ 0, b, c });
            if (s == 0)
              return 0;
            if (s > 0)
              live[c - 1] = false;
          }
        }

        solver_.begin(rp);
        for (int b = 1; b <= m; ++b) {
          slot_[b] = live[b - 1] ? solver_.add_cap(cosa[b - 1]) : -1;
          if (slot_[b] >= 0)
            cap_of_slot_[slot_[b]] = b;
        }

        collect_roots(ex, m, live);
        for (int b = 1; b <= m; ++b) {
          if (live[b - 1])
            ring_arcs(ex, b, live);
        }

        return solver_.solve([&](int j, int l) {
          return ex.discs_intersect(0, cap_of_slot_[j], cap_of_slot_[l]);
        });
      }

    private:
      /**
       * Roots of every cutting pair of live caps that lie outside every
       * other live cap, in order, so that a root's index is its solver
       * vertex; per circle whether any pair cuts it or buries it.
       */
      void collect_roots(const BallExact &ex, const int m,
                         const ArrayXb &live) {
        cut_any_.head(m + 1).setConstant(false);
        inside_any_.head(m + 1).setConstant(false);
        pairs_.clear();
        roots_.clear();

        for (int b = 1; b <= m; ++b) {
          for (int c = b + 1; c <= m; ++c) {
            if (!live[b - 1] || !live[c - 1])
              continue;

            const BallTriple f { 0, b, c };
            if (ex.cuts(f) != Sgn::kPos) {
              inside_any_[b] |= ex.side(0, b, c) == Sgn::kNeg;
              inside_any_[c] |= ex.side(0, c, b) == Sgn::kNeg;
              continue;
            }
            cut_any_[b] = cut_any_[c] = true;

            const int pair = static_cast<int>(pairs_.size());
            pairs_.push_back({ b, c });
            for (const bool plus: { true, false }) {
              bool accepted = true;
              for (int k = 1; k <= m && accepted; ++k) {
                accepted = k == b || k == c || !live[k - 1]
                           || ex.accept(f, plus, k) == Sgn::kPos;
              }
              if (!accepted)
                continue;

              roots_.push_back({ pair, plus });
              solver_.add_vertex(ex.root(f, plus).normalized());
            }
          }
        }
      }

      /**
       * Ring of circle `(0, b)` sorted by exact class, then by the perturbed
       * orientation (coincident roots are generic here), and its arcs.
       */
      void ring_arcs(const BallExact &ex, const int b, const ArrayXb &live) {
        ring_.verts.clear();
        ring_.cut_any = cut_any_[b];
        ring_.inside_any = inside_any_[b];
        for (int r = 0; r < static_cast<int>(roots_.size()); ++r) {
          const FaceRoot &rt = roots_[r];
          const FacePair &pr = pairs_[rt.pair];
          if (pr.m == b)
            ring_.verts.push_back({ r, rt.pair, 0, rt.plus, rt.plus, 0.0 });
          else if (pr.l == b)
            ring_.verts.push_back({ r, rt.pair, 0, rt.plus, !rt.plus, 0.0 });
        }

        const int nv = static_cast<int>(ring_.verts.size());
        if (nv > 0) {
          ABSL_DCHECK_EQ(nv % 2, 0) << "odd ring on cap " << b;
          const FaceRingOps ops(ex, pairs_, live, b);
          const Vector3d axis = c_.col(b);
          const auto [e1, e2] = circle_frame(axis, axis.normalized());
          ring_frame(ring_, e1, e2);
          ring_angles(ring_, ops);
          std::sort(ring_.verts.begin(), ring_.verts.end(),
                    [&](const RingVertex &p, const RingVertex &r) {
                      return p.cls < r.cls
                             || (p.cls == r.cls && p.vertex != r.vertex
                                 && ops.ccw(p, r) == Sgn::kPos);
                    });
          decide_wrap(ring_, ops);
        }

        arcs_.clear();
        tangents_.clear();
        emit_arcs(ring_, b, arcs_, tangents_);
        for (int a = 0; a < static_cast<int>(arcs_.size()); ++a) {
          const SasArc &arc = arcs_[a];
          const auto &[tb, te] = tangents_[a];
          solver_.add_arc(slot_[b], arc.beg, arc.end, arc.dphi, tb, te);
        }
      }

      ArrangementSolver solver_;
      Matrix3Xd c_;
      ArrayXd h_;
      ArrayXi slot_, cap_of_slot_;
      ArrayXb cut_any_, inside_any_;
      std::vector<FacePair> pairs_;
      std::vector<FaceRoot> roots_;
      Ring ring_;
      std::vector<SasArc> arcs_;
      RingTangents tangents_;
    };

    /**
     * Every face left NaN by `faces` solved exactly on its lifted caps; the
     * cap lists are compacted to the caps that survived hygiene.
     */
    void solve_faces(SesGeometry &ses, const SasProbes &probes,
                     const LiftedCaps &lifted, const double rp) {
      const int na = probes.n_active;
      if (na == 0)
        return;

      const int mcap = ses.face_off.max_deg();
      FaceSolver solver(mcap);
      ArrayXb live(mcap);

      ArrayXi &off = ses.face_off.off();
      const ArrayXi old = off;
      int w = 0;
      for (int p = 0; p < na; ++p) {
        const int beg = old[p], m = old[p + 1] - beg;
        off[p] = w;
        live.head(m).setConstant(true);

        if (std::isnan(ses.face_area[p])) {
          ses.face_area[p] =
              solver.solve(lifted.n.middleCols(beg, m), lifted.h.segment(beg, m),
                           ses.face_cosa.segment(beg, m), rp, live);
        }

        for (int k = 0; k < m; ++k) {
          if (!live[k])
            continue;
          ses.face_axis.col(w) = ses.face_axis.col(beg + k);
          ses.face_cosa[w] = ses.face_cosa[beg + k];
          ses.face_sina[w] = ses.face_sina[beg + k];
          ++w;
        }
      }
      off[na] = w;
      ses.face_axis.conservativeResize(3, w);
      ses.face_cosa.conservativeResize(w);
      ses.face_sina.conservativeResize(w);
    }
  }  // namespace

  double solve_ses_face(const Matrix3Xd &n, const ArrayXd &h,
                        const ArrayXd &cosa, const double rp, ArrayXb &live) {
    const int m = static_cast<int>(h.size());
    FaceSolver solver(m);
    live.resize(m);
    return solver.solve(n, h, cosa, rp, live);
  }

  SesGeometry build_ses(const SaPrep &sa, const SasDelaunay &del,
                        const SasGeometry &geo, const double rp) {
    SesGeometry ses;
    ses.convex_area = convex_areas(sa, geo, rp);
    saddles(ses, sa, geo, rp);
    const LiftedCaps lifted = faces(ses, sa, del, geo, rp);
    solve_faces(ses, geo.probes, lifted, rp);
    return ses;
  }
}  // namespace internal
}  // namespace nuri
