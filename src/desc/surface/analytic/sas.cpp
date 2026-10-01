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

#include <absl/base/optimization.h>
#include <absl/log/absl_check.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/desc/surface.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  namespace {
    using constants::kTwoPi;
    using Array3Xi = E::Array<int, 3, E::Dynamic>;

    std::vector<SasCircle> circles(const SaPrep &sa) {
      const int n_circ = sa.g.offset(sa.n_enum);
      std::vector<SasCircle> result(n_circ);

      for (int i = 0; i < sa.n_enum; ++i) {
        double ri = sa.sar[i];
        Vector3d pi = sa.pts.col(i);

        for (auto it = sa.g.begin(i), ei = sa.g.end(i); it < ei; ++it) {
          const int j = *it, q = sa.g.eid(it);
          const double rj = sa.sar[j];
          const double d = sa.d[q];

          Vector3d axis = (sa.pts.col(j) - pi) / d;
          double a = (d * d + ri * ri - rj * rj) / (2 * d);
          double rl = std::sqrt(nuri::max(ri * ri - a * a, 0.0));
          Vector3d cntr = pi + a * axis;

          result[q] = { axis, cntr, a, rl, i, j };
        }
      }

      return result;
    }

    int pair_id(const CSR &g, const int i, const int j) {
      const auto beg = g.begin(i), end = g.end(i);
      const auto it = std::lower_bound(beg, end, j);
      return it != end && *it == j ? g.eid(it) : -1;
    }

    bool has_edge(const SasDelaunay &del, const int va, const int vb) {
      const auto end = del.nbrs.end(va),
                 it = std::lower_bound(del.nbrs.begin(va), end, vb);
      return it != end && *it == vb;
    }

    int pair_id_unordered(const CSR &g, const int i, const int j) {
      auto [a, b] = nuri::minmax(i, j);
      return pair_id(g, a, b);
    }

    int icirc(int tag) {
      return tag >> 1;
    }

    int iside(int tag) {
      return tag & 1;
    }

    struct VertexMap {
      ArrayXi v_of_new, new_of_v;
    };

    VertexMap map_vertices(const SaPrep &sa, const SasDelaunay &del) {
      const int n = sa.g.n();
      ABSL_DCHECK_EQ(n, del.nbrs.n());

      VertexMap vm { ArrayXi(n), ArrayXi::Constant(del.ex.n(), -1) };
      for (int p = 0; p < n; ++p) {
        ABSL_DCHECK_LT(sa.order[p], del.vertex.size());
        const int v = del.vertex[sa.order[p]];
        ABSL_DCHECK_GE(v, 0);
        vm.v_of_new[p] = v;
        vm.new_of_v[v] = p;
      }
      return vm;
    }

    /**
     * Caps of sphere `s` are its overlapping Delaunay neighbours: every other
     * overlapping sphere cuts a cap inside their union (Theorem 2), so the
     * accessible region and its component structure are unchanged.
     */
    SasCaps neighbor_caps(const SaPrep &sa, const SasDelaunay &del,
                          const VertexMap &vm,
                          const std::vector<SasCircle> &circ) {
      const int n_enum = sa.n_enum, bound = del.nbrs.m();

      SasCaps caps { CSR(ArrayXi(bound), OffsetTable(n_enum)),
                     Matrix3Xd(3, bound), ArrayXd(bound), ArrayXd(bound) };
      ArrayXi &tags = caps.h.adj(), &off = caps.h.off();

      int w = 0;
      for (int s = 0; s < n_enum; ++s) {
        off[s] = w;
        const double rs = sa.sar[s];

        for (int v: del.nbrs.nbrs(vm.v_of_new[s])) {
          const int t = vm.new_of_v[v];
          if (t < 0)
            continue;

          auto [a, b] = nuri::minmax(s, t);
          const int q = pair_id(sa.g, a, b);
          if (q < 0)
            continue;

          const SasCircle &c = circ[q];
          const int side = value_if(s != a);
          const double d = sa.d[q];
          tags[w] = 2 * q + side;
          caps.axis.col(w) = (1 - 2 * side) * c.axis;
          caps.cosa[w] = (c.a + (d - 2 * c.a) * side) / rs;
          caps.sina[w] = c.rl / rs;
          ++w;
        }
      }
      off[n_enum] = w;

      tags.conservativeResize(w);
      caps.axis.conservativeResize(3, w);
      caps.cosa.conservativeResize(w);
      caps.sina.conservativeResize(w);
      return caps;
    }

    /**
     * Per triangulation face: its sorted vertices (-1 if it holds a corner or
     * a non-overlapping pair), whether its third sphere cuts the circle, and
     * the probe of each root (0 = plus, 1 = minus; -1 if rejected).
     */
    struct Faces {
      Array3Xi verts;
      ArrayXb cut;
      Array2Xi root;
    };

    struct RawProbe {
      Array3i atoms;
      Vector3d pos;
    };

    SasFace sorted_face(Array3i fv) {
      std::sort(fv.begin(), fv.end());
      return { fv[0], fv[1], fv[2] };
    }

    /**
     * The face opposite local vertex `lf` of cell `c`, sorted, if all three
     * spheres are kept, pairwise overlapping and one of them gets caps.
     */
    bool sphere_face(const SaPrep &sa, const SasDelaunay &del,
                     const VertexMap &vm, const int c, const int lf,
                     Array3i &fv, Array3i &abc) {
      const Array4i tv = del.tets.col(c);
      for (int lv = 0, m = 0; lv < 4; ++lv)
        if (lv != lf)
          fv[m++] = tv[lv];
      abc = vm.new_of_v(fv);
      if ((abc < 0).any())
        return false;

      std::sort(abc.begin(), abc.end());
      return abc[0] < sa.n_enum && pair_id(sa.g, abc[0], abc[1]) >= 0
             && pair_id(sa.g, abc[0], abc[2]) >= 0
             && pair_id(sa.g, abc[1], abc[2]) >= 0;
    }

    int opposite_vertex(const SasDelaunay &del, const int c2,
                        const Array3i &fv) {
      ABSL_DCHECK_GE(c2, 0) << "sphere face on the hull";
      for (int lv = 0; lv < 4; ++lv) {
        const int v = del.tets(lv, c2);
        if ((fv != v).all())
          return v;
      }
      ABSL_UNREACHABLE();
    }

    /**
     * An apex that overlaps none of the face spheres, or is a bounding point
     * (Lemma 0), has non-negative power on their surfaces, zero only at a
     * tangency point, and never rejects a root.
     */
    bool apex_may_reject(const SaPrep &sa, const VertexMap &vm,
                         const Array3i &abc, const int lv) {
      const int l = vm.new_of_v[lv];
      return l >= 0 && std::all_of(abc.begin(), abc.end(), [&](int s) {
               return pair_id_unordered(sa.g, s, l) >= 0;
             });
    }

    /**
     * Every face of a finite cell once, from the lower cell. A cut point is a
     * probe iff its power against both apexes is non-negative, decided
     * exactly for the perturbed weights (Theorem 1).
     */
    Faces accept_faces(const SaPrep &sa, const SasDelaunay &del,
                       const VertexMap &vm, std::vector<RawProbe> &raw) {
      const int nf = static_cast<int>(del.tets.cols());
      const SasExact &ex = del.ex;

      Faces fs { Array3Xi::Constant(3, del.n_faces, -1),
                 ArrayXb::Constant(del.n_faces, false),
                 Array2Xi::Constant(2, del.n_faces, -1) };
      raw.reserve(3L * sa.n_enum);

      for (int c = 0; c < nf; ++c) {
        for (int lf = 0; lf < 4; ++lf) {
          const int c2 = del.adj(lf, c);
          if (c2 >= 0 && c2 < c)
            continue;

          Array3i fv, abc;
          if (!sphere_face(sa, del, vm, c, lf, fv, abc))
            continue;

          const SasFace face = sorted_face(fv);
          const int f = del.face(lf, c);
          fs.verts.col(f) << face.a, face.b, face.c;
          if (ex.cuts(face) != Sgn::kPos)
            continue;
          fs.cut[f] = true;

          const int l1 = del.tets(lf, c), l2 = opposite_vertex(del, c2, fv);
          const bool test1 = apex_may_reject(sa, vm, abc, l1),
                     test2 = apex_may_reject(sa, vm, abc, l2);
          const auto [xp, xm] = ex.roots(face);
          for (const bool plus: { true, false }) {
            if ((test1 && ex.accept(face, plus, l1) != Sgn::kPos)
                || (test2 && ex.accept(face, plus, l2) != Sgn::kPos))
              continue;

            fs.root(plus ? 0 : 1, f) = static_cast<int>(raw.size());
            raw.push_back({ abc, plus ? xp : xm });
          }
        }
      }

      return fs;
    }

    /**
     * Probes owner-sorted (owner = smallest atom); `pid` maps raw ids.
     */
    SasProbes make_probes(const SaPrep &sa, const std::vector<RawProbe> &raw,
                          ArrayXi &pid) {
      const int nr = static_cast<int>(raw.size());

      ArrayXi owner(nr), order(nr);
      for (int r = 0; r < nr; ++r)
        owner[r] = raw[r].atoms[0];
      OffsetTable own_off(sa.n_enum);
      argsort_bucket(order, own_off, owner);

      pid.resize(nr);
      SasProbes probes { CSR(ArrayXi(3L * nr), OffsetTable(nr)),
                         Matrix3Xd(3, nr), Matrix3Xd(), OffsetTable(nr),
                         own_off[sa.n_active] };
      for (int p = 0; p < nr; ++p) {
        const int r = order[p];
        pid[r] = p;
        probes.atoms.off()[p] = 3 * p;
        probes.atoms.adj().segment(3L * p, 3) = raw[r].atoms;
        probes.pos.col(p) = raw[r].pos;
      }
      probes.atoms.off()[nr] = 3 * nr;
      return probes;
    }

    /**
     * Accessible arcs by circle with, per arc, the departing tangent at
     * `beg` and the arriving tangent at `end` from the circle frame and the
     * arc's own angles, so darts stay consistent with the arcs they bound
     * even on a circle the probe positions cannot resolve.
     */
    struct Arcs {
      std::vector<SasArc> arcs;
      std::vector<std::pair<Vector3d, Vector3d>> tangents;
      OffsetTable off;
    };

    struct RingVertex {
      int probe, face, cls;
      bool plus, leave;
      double phi;
    };

    int local_index(const Array4i &tv, const int v) {
      for (int lv = 0; lv < 4; ++lv)
        if (tv[lv] == v)
          return lv;
      ABSL_UNREACHABLE();
    }

    bool even_permutation(std::array<int, 4> p) {
      int swaps = 0;
      for (int i = 0; i < 4; ++i) {
        while (p[i] != i) {
          std::swap(p[i], p[p[i]]);
          ++swaps;
        }
      }
      return swaps % 2 == 0;
    }

    /**
     * Sign of the permutation `(a, b, c)` of the sorted face; the root
     * `−ε` enters ball `c` along circle `(a, b)` oriented about `a → b`,
     * the root `+ε` leaves it.
     */
    int face_parity(const Array3i &sorted, const int a, const int b) {
      int ia = 0, ib = 0;
      for (int k = 0; k < 3; ++k) {
        ia = sorted[k] == a ? k : ia;
        ib = sorted[k] == b ? k : ib;
      }
      return (ib - ia + 3) % 3 == 1 ? 1 : -1;
    }

    /**
     * Cells around edge `(a, b)` counter-clockwise about `a → b`; positively
     * oriented cells make `(a, b, d, c)` even iff `d → c` turns
     * counter-clockwise. `face()` is the face `(a, b, third())` about to be
     * crossed.
     */
    class FanWalker {
    public:
      FanWalker(const SasDelaunay &del, int va, int vb)
          : del_(&del), va_(va), vb_(vb) {
        const auto row_beg = del.nbrs.begin(va), row_end = del.nbrs.end(va);
        const auto it = std::lower_bound(row_beg, row_end, vb);
        ABSL_DCHECK(it != row_end && *it == vb);
        t0_ = t_ = del.edge_cell[del.nbrs.eid(it)];

        const Array4i tv = del.tets.col(t_);
        const int la = local_index(tv, va), lb = local_index(tv, vb);
        std::array<int, 2> others;
        for (int lv = 0, m = 0; lv < 4; ++lv)
          if (lv != la && lv != lb)
            others[m++] = lv;
        ld_ = others[0];
        lc_ = others[1];
        if (!even_permutation({ la, lb, ld_, lc_ }))
          std::swap(ld_, lc_);
        guard_ = del.nbrs.degree(va) + 1;
      }

      int face() const { return del_->face(ld_, t_); }
      int third() const { return del_->tets(lc_, t_); }

      bool advance() {
        const int vc = third(), t2 = del_->adj(ld_, t_);
        ABSL_DCHECK_GE(t2, 0) << "sphere edge on the hull";
        ABSL_DCHECK_GT(guard_--, 0) << "fan does not close";

        const Array4i tv = del_->tets.col(t2);
        const int la = local_index(tv, va_), lb = local_index(tv, vb_),
                  ld = local_index(tv, vc);
        t_ = t2;
        ld_ = ld;
        lc_ = 6 - la - lb - ld;
        return t_ != t0_;
      }

    private:
      const SasDelaunay *del_;
      int va_, vb_, t0_, t_, ld_, lc_, guard_;
    };

    struct Ring {
      std::vector<RingVertex> verts;
      std::vector<int> fan;
      std::vector<bool> dec, coinc;
      bool cut_any, inside_any;
      Vector3d e1, e2;
    };

    Vector3d ring_tangent(const Ring &ring, const double psi) {
      return -std::sin(psi) * ring.e1 + std::cos(psi) * ring.e2;
    }

    SasFace face_of(const Faces &fs, const int f) {
      const Array3i fv = fs.verts.col(f);
      return { fv[0], fv[1], fv[2] };
    }

    /**
     * Walk the cell fan around edge `(a, b)` counter-clockwise (Lemma F);
     * each face `(a, b, c)` with a sphere `c` overlapping both contributes
     * its accepted roots as (enter ball `c`, leave ball `c`).
     */
    void collect_ring(Ring &ring, const SaPrep &sa, const SasDelaunay &del,
                      const VertexMap &vm, const Faces &fs, const ArrayXi &pid,
                      const int i, const int j) {
      const int va = vm.v_of_new[i], vb = vm.v_of_new[j],
                n_sphere_v = del.nbrs.n();
      ring.verts.clear();
      ring.fan.clear();
      ring.cut_any = ring.inside_any = false;

      FanWalker walk(del, va, vb);
      do {
        const int vc = walk.third();
        if (vc >= n_sphere_v)
          continue;
        const int cn = vm.new_of_v[vc];
        if (cn < 0 || pair_id_unordered(sa.g, i, cn) < 0
            || pair_id_unordered(sa.g, j, cn) < 0)
          continue;

        const int f = walk.face();
        ABSL_DCHECK_GE(fs.verts(0, f), 0);
        ring.fan.push_back(vc);
        if (!fs.cut[f]) {
          ring.inside_any |= del.ex.side(va, vb, vc) == Sgn::kNeg;
          continue;
        }

        ring.cut_any = true;
        const int eps = face_parity(fs.verts.col(f), va, vb);
        for (const int sigma: { -eps, eps }) {
          const bool plus = sigma > 0;
          const int r = fs.root(plus ? 0 : 1, f);
          if (r >= 0)
            ring.verts.push_back({ pid[r], f, 0, plus, sigma == eps, 0.0 });
        }
      } while (walk.advance());
    }

    /**
     * Angle of every vertex in `[0, 2π]` from the frame's reference ray, with
     * the exact half-plane class overriding the rounding of `atan2` at the
     * ray and at `π`, so the numeric angles are monotone in the exact order.
     */
    void ring_angles(Ring &ring, const SaPrep &sa, const SasExact &ex,
                     const VertexMap &vm, const SasCircle &c, const Faces &fs,
                     const SasProbes &probes) {
      const int va = vm.v_of_new[c.i], vb = vm.v_of_new[c.j];
      const Vector3d d = sa.pts.col(c.j) - sa.pts.col(c.i);
      ring.e1 =
          d.cross(Vector3d::Unit(SasExact::reference_axis(d))).normalized();
      ring.e2 = c.axis.cross(ring.e1);
      for (RingVertex &rv: ring.verts) {
        const Vector3d u = probes.pos.col(rv.probe) - c.cntr;
        const double phi = std::atan2(u.dot(ring.e2), u.dot(ring.e1));
        rv.cls = ex.half_plane(va, vb, face_of(fs, rv.face), rv.plus);
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
          rv.phi = kTwoPi - std::abs(phi);
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
    void decide_wrap(Ring &ring, const SasExact &ex, const VertexMap &vm,
                     const SasCircle &c, const Faces &fs) {
      const int va = vm.v_of_new[c.i], vb = vm.v_of_new[c.j],
                n = static_cast<int>(ring.verts.size());
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
          const Sgn s = ex.ccw(va, vb, face_of(fs, p.face), p.plus,
                               face_of(fs, r.face), r.plus);
          ring.coinc[k] = s == Sgn::kZero;
          ring.dec[k] = s == Sgn::kNeg;
        }
        n_dec += static_cast<int>(ring.dec[k]);
      }
      if (n_dec == 1)
        return;
      ABSL_DCHECK_EQ(n_dec, 0) << "ring wraps twice";

      const RingVertex &r0 = ring.verts[0];
      bool accessible = true;
      for (const int vc: ring.fan) {
        accessible &= ex.antipode(va, vb, face_of(fs, r0.face), r0.plus, vc)
                      == Sgn::kPos;
      }
      for (int k = 0; k < n; ++k) {
        if (ring.verts[k].leave == accessible) {
          ring.dec[k] = true;
          ring.coinc[k] = false;
          return;
        }
      }
    }

    /**
     * Accessible arcs of every solved circle, grouped by circle: from each
     * leaving vertex to the next vertex; an empty ring is an accessible full
     * circle iff no fan face cuts or contains the circle.
     */
    Arcs fan_rings(const SaPrep &sa, const SasDelaunay &del,
                   const VertexMap &vm, const std::vector<SasCircle> &circ,
                   const Faces &fs, const ArrayXi &pid,
                   const SasProbes &probes) {
      Arcs out;
      out.off = OffsetTable(sa.g.m());
      Ring ring;

      for (int i = 0; i < sa.n_solve; ++i) {
        for (auto it = sa.g.begin(i), ei = sa.g.end(i); it < ei; ++it) {
          const int j = *it, q = sa.g.eid(it);
          out.off.off()[q] = static_cast<int>(out.arcs.size());

          if (!has_edge(del, vm.v_of_new[i], vm.v_of_new[j]))
            continue;

          collect_ring(ring, sa, del, vm, fs, pid, i, j);
          const int n = static_cast<int>(ring.verts.size());
          if (n == 0) {
            if (!ring.cut_any && !ring.inside_any) {
              out.arcs.push_back({ 0.0, kTwoPi, q, -1, -1 });
              out.tangents.emplace_back(Vector3d::Zero(), Vector3d::Zero());
            }
            continue;
          }
          ABSL_DCHECK_EQ(n % 2, 0) << "odd ring on circle " << q;

          ring_angles(ring, sa, del.ex, vm, circ[q], fs, probes);
          decide_wrap(ring, del.ex, vm, circ[q], fs);

          for (int k = 0; k < n; ++k) {
            const RingVertex &p = ring.verts[k], &r = ring.verts[(k + 1) % n];
            if (!p.leave)
              continue;

            double dphi = 0;
            if (!ring.coinc[k]) {
              dphi =
                  nuri::max(r.phi - p.phi + (ring.dec[k] ? kTwoPi : 0.0), 0.0);
            }
            const double phi = p.phi > constants::kPi ? p.phi - kTwoPi : p.phi;
            out.arcs.push_back({ phi, dphi, q, p.probe, r.probe });
            out.tangents.emplace_back(ring_tangent(ring, p.phi),
                                      ring_tangent(ring, p.phi + dphi));
          }
        }
      }

      for (int q = sa.g.offset(sa.n_solve); q <= sa.g.m(); ++q)
        out.off.off()[q] = static_cast<int>(out.arcs.size());
      return out;
    }

    ArrayXd sweep_spheres(const SaPrep &sa, const SasDelaunay &del,
                          const VertexMap &vm,
                          const std::vector<SasCircle> &circ,
                          const SasCaps &caps, const Arcs &arcs,
                          const SasProbes &probes) {
      const int n_solve = sa.n_solve, mcap = caps.h.max_deg(),
                np = static_cast<int>(probes.pos.cols());
      const SasExact &ex = del.ex;

      int acap = 0;
      for (int s = 0; s < n_solve; ++s) {
        int n_arcs = 0;
        for (auto it = caps.h.begin(s), e = caps.h.end(s); it < e; ++it)
          n_arcs += arcs.off.degree(icirc(*it));
        acap = nuri::max(acap, n_arcs);
      }

      ArrayXd area(n_solve);
      ArrangementSolver solver(mcap, 2 * acap, acap);
      ArrayXi vloc = ArrayXi::Constant(np, -1), vlist(2 * acap + 1),
              partner(mcap);

      for (int s = 0; s < n_solve; ++s) {
        if (del.nbrs.degree(vm.v_of_new[s]) == 0) {
          area[s] = 0;
          continue;
        }

        solver.begin(sa.sar[s]);
        int k = 0;
        auto local = [&](int p) {
          if (vloc[p] < 0) {
            vloc[p] = solver.add_vertex(
                (probes.pos.col(p) - sa.pts.col(s)).normalized());
            vlist[k++] = p;
          }
          return vloc[p];
        };

        for (auto it = caps.h.begin(s), e = caps.h.end(s); it < e; ++it) {
          const int p = caps.h.eid(it), tag = *it, q = icirc(tag),
                    side = iside(tag);
          const int slot = solver.add_cap(caps.cosa[p]);
          const SasCircle &c = circ[q];
          partner[slot] = c.i == s ? c.j : c.i;

          for (int r = arcs.off[q]; r < arcs.off[q + 1]; ++r) {
            const SasArc &arc = arcs.arcs[r];
            const auto &[tb, te] = arcs.tangents[r];
            if (arc.beg < 0) {
              solver.add_arc(slot, -1, -1, arc.dphi, tb, te);
              continue;
            }
            const int vb = local(arc.beg), ve = local(arc.end);
            if (side == 0)
              solver.add_arc(slot, vb, ve, arc.dphi, tb, te);
            else
              solver.add_arc(slot, ve, vb, arc.dphi, -te, -tb);
          }
        }

        const int vs = vm.v_of_new[s];
        area[s] = solver.solve([&](int j, int l) {
          return pair_id_unordered(sa.g, partner[j], partner[l]) >= 0
                 && ex.discs_intersect(vs, vm.v_of_new[partner[j]],
                                       vm.v_of_new[partner[l]]);
        });

        for (int v = 0; v < k; ++v)
          vloc[vlist[v]] = -1;
      }

      return area;
    }

    void tangents(SasProbes &pr, const Arcs &arcs) {
      const int np = static_cast<int>(pr.pos.cols());

      ArrayXi &toff = pr.tan_off.off();
      toff.setZero();
      for (const SasArc &arc: arcs.arcs) {
        if (arc.beg < 0)
          continue;

        ++toff[arc.beg + 1];
        ++toff[arc.end + 1];
      }
      std::inclusive_scan(toff.begin(), toff.end(), toff.begin());

      pr.tan.resize(3, toff[np]);
      ArrayXi cur = toff.head(np);
      for (int r = 0; r < static_cast<int>(arcs.arcs.size()); ++r) {
        const SasArc &arc = arcs.arcs[r];
        if (arc.beg < 0)
          continue;

        const auto &[tb, te] = arcs.tangents[r];
        pr.tan.col(cur[arc.beg]++) = tb;
        pr.tan.col(cur[arc.end]++) = -te;
      }
    }
  }  // namespace

  SasGeometry build_sas(const SaPrep &sa, const SasDelaunay &del) {
    if (sa.n_enum == 0)
      return {};

    std::vector circ = circles(sa);
    VertexMap vm = map_vertices(sa, del);
    SasCaps caps = neighbor_caps(sa, del, vm, circ);

    std::vector<RawProbe> raw;
    Faces fs = accept_faces(sa, del, vm, raw);
    ArrayXi pid;
    SasProbes probes = make_probes(sa, raw, pid);

    Arcs arcs = fan_rings(sa, del, vm, circ, fs, pid, probes);
    ArrayXd area = sweep_spheres(sa, del, vm, circ, caps, arcs, probes);
    tangents(probes, arcs);

    const int n_active_arcs = arcs.off[sa.g.offset(sa.n_active)];
    return { std::move(circ),      std::move(caps), std::move(probes),
             std::move(arcs.arcs), n_active_arcs,   std::move(area) };
  }
}  // namespace internal
}  // namespace nuri
