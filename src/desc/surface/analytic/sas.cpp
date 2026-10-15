//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include <absl/base/optimization.h>
#include <absl/log/absl_check.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "ring.h"
#include "stages.h"
#include "nuri/core/geometry.h"
#include "nuri/desc/surface.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  namespace {
    std::vector<SasCircle> circles(const SurfaceSpheres &ss) {
      const int n_circ = ss.g.offset(ss.n_enum);
      std::vector<SasCircle> result(n_circ);

      for (int i = 0; i < ss.n_enum; ++i) {
        double ri2 = ss.sar2[i];
        Vector3d pi = ss.pts.col(i);

        for (auto it = ss.g.begin(i), ei = ss.g.end(i); it < ei; ++it) {
          const int j = *it, q = ss.g.eid(it);
          const double rj2 = ss.sar2[j];
          const double d = ss.d[q];

          Vector3d axis = (ss.pts.col(j) - pi) / d;
          const auto [e1, e2] = circle_frame(ss.pts.col(j) - pi, axis);
          double a = (d * d + ri2 - rj2) / (2 * d);
          // The one rounding guard (design rules of the analytic_surface
          // page): an exact overlap of sub-ulp depth.
          double rl = std::sqrt(nuri::max(ri2 - a * a, 0.0));
          Vector3d cntr = pi + a * axis;

          result[q] = {
            axis, cntr, e1, e2, { a, d - a },
                rl, i, j
          };
        }
      }

      return result;
    }

    int pair_id_unordered(const CSR &g, const int i, const int j) {
      auto [a, b] = nuri::minmax(i, j);
      return g.find(a, b);
    }

    struct VertexMap {
      ArrayXi v_of_new, new_of_v;
    };

    VertexMap map_vertices(const SurfaceSpheres &ss, const SasDelaunay &del) {
      const int n = ss.g.n();
      ABSL_DCHECK_EQ(n, del.nbrs.n());

      VertexMap vm { ArrayXi(n), ArrayXi::Constant(del.ex.n(), -1) };
      for (int p = 0; p < n; ++p) {
        ABSL_DCHECK_LT(ss.order[p], del.vertex.size());
        const int v = del.vertex[ss.order[p]];
        ABSL_DCHECK_GE(v, 0);
        vm.v_of_new[p] = v;
        vm.new_of_v[v] = p;
      }
      return vm;
    }

    /**
     * Edges bucketed by their lower sphere meet that sphere's row of `ss.g`
     * through a dense mark over sphere indices, then the pair row of each
     * sphere meets its `nbrs` row the same way; index 0 absorbs the corners.
     */
    SasEdgeTables edge_tables(const SurfaceSpheres &ss, const SasDelaunay &del,
                              const VertexMap &vm) {
      const int n = ss.g.n(), m = del.nbrs.m();

      ArrayXi lo(m), hi(m);
      for (int v = 0; v < n; ++v) {
        const int s = vm.new_of_v[v];
        for (auto it = del.nbrs.begin(v), e = del.nbrs.end(v); it < e; ++it) {
          const int k = del.nbrs.eid(it);
          const auto [a, b] = nuri::minmax(s, vm.new_of_v[*it]);
          lo[k] = a + 1;
          hi[k] = b + 1;
        }
      }

      ArrayXi order(m);
      OffsetTable off(n + 1);
      argsort_bucket(order, off, lo);

      SasEdgeTables tbl { ArrayXi::Constant(m, -1),
                          ArrayXi::Constant(ss.g.m(), -1) };
      ArrayXi mark = ArrayXi::Constant(n + 1, -1);
      for (int a = 0; a < n; ++a) {
        for (auto it = ss.g.begin(a), e = ss.g.end(a); it < e; ++it)
          mark[*it + 1] = ss.g.eid(it);
        for (const int k: order.segment(off[a + 1], off.degree(a + 1)))
          tbl.pair_of_edge[k] = mark[hi[k]];
        for (auto it = ss.g.begin(a), e = ss.g.end(a); it < e; ++it)
          mark[*it + 1] = -1;
      }

      for (int s = 0; s < n; ++s) {
        const int v = vm.v_of_new[s];
        for (auto it = del.nbrs.begin(v), e = del.nbrs.end(v); it < e; ++it)
          mark[vm.new_of_v[*it] + 1] = del.nbrs.eid(it);
        for (auto it = ss.g.begin(s), e = ss.g.end(s); it < e; ++it)
          tbl.edge_of_pair[ss.g.eid(it)] = mark[*it + 1];
        for (auto it = del.nbrs.begin(v), e = del.nbrs.end(v); it < e; ++it)
          mark[vm.new_of_v[*it] + 1] = -1;
      }

      return tbl;
    }

    /**
     * Caps of sphere `s` are its overlapping Delaunay neighbors: every other
     * overlapping sphere cuts a cap inside their union (Theorem 2 of the
     * analytic_surface page), so the accessible region and its component
     * structure are unchanged.
     */
    SasCaps neighbor_caps(const SurfaceSpheres &ss, const SasDelaunay &del,
                          const VertexMap &vm, const ArrayXi &pair_of_edge,
                          const std::vector<SasCircle> &circ) {
      const int n_enum = ss.n_enum;

      int n_caps = 0;
      for (int s = 0; s < n_enum; ++s) {
        const int v = vm.v_of_new[s];
        for (auto it = del.nbrs.begin(v), e = del.nbrs.end(v); it < e; ++it)
          n_caps += value_if(pair_of_edge[del.nbrs.eid(it)] >= 0);
      }

      SasCaps caps { CSR(ArrayXi(n_caps), OffsetTable(n_enum)), ArrayXi(n_caps),
                     Matrix3Xd(3, n_caps), ArrayXd(n_caps) };
      ArrayXi &circ_of = caps.h.adj(), &off = caps.h.off();

      int w = 0;
      for (int s = 0; s < n_enum; ++s) {
        off[s] = w;
        const double rs = ss.sar[s];
        const int v = vm.v_of_new[s];

        for (auto it = del.nbrs.begin(v), e = del.nbrs.end(v); it < e; ++it) {
          const int q = pair_of_edge[del.nbrs.eid(it)];
          if (q < 0)
            continue;

          const SasCircle &c = circ[q];
          const bool first = c.i == s;
          circ_of[w] = q;
          caps.partner[w] = first ? c.j : c.i;
          caps.axis.col(w) = first ? c.axis : Vector3d(-c.axis);
          caps.cosa[w] = c.aij[value_if(!first)] / rs;
          ++w;
        }
      }
      off[n_enum] = w;
      ABSL_DCHECK_EQ(w, n_caps);
      return caps;
    }

    /**
     * One triangulation face: its sorted vertices (-1 if it holds a corner
     * or a non-overlapping pair), the probe of each root (0 = plus, 1 =
     * minus; -1 if rejected), whether its three vertices are kept spheres
     * carrying all three overlapping pairs, and whether its third sphere
     * cuts the circle.
     */
    struct FaceRec {
      Array3i verts = Array3i::Constant(-1);
      E::Array2i root = E::Array2i::Constant(-1);
      bool full = false, cut = false;
    };

    /**
     * A probe at `pos`, the root `plus` of the sorted face `face`.
     */
    struct RawProbe {
      Vector3d pos;
      Array3i face;
      bool plus;
    };

    Array3i face_vertices(const SasDelaunay &del, const int c, const int lf) {
      const Array4i tv = del.tets.col(c);
      Array3i fv;
      for (int lv = 0, m = 0; lv < kCellVerts; ++lv)
        if (lv != lf)
          fv[m++] = tv[lv];
      return fv;
    }

    /**
     * The face opposite local vertex `lf` of cell `c` if all three vertices
     * are kept spheres: `fv` its vertices, `abc` their sorted indices.
     */
    bool face_spheres(const SasDelaunay &del, const VertexMap &vm, const int c,
                      const int lf, Array3i &fv, Array3i &abc) {
      fv = face_vertices(del, c, lf);
      abc = vm.new_of_v(fv);
      if ((abc < 0).any())
        return false;

      abc = sort3(abc);
      return true;
    }

    constexpr std::array<int, 3> face_slots(const int lf) {
      std::array<int, 3> lv {};
      for (int l = 0, m = 0; l < kCellVerts; ++l)
        if (l != lf)
          lv[m++] = l;
      return { kCellPairSlot[lv[0]][lv[1]], kCellPairSlot[lv[0]][lv[2]],
               kCellPairSlot[lv[1]][lv[2]] };
    }

    constexpr std::array<std::array<int, 3>, 4> kFaceSlots {
      face_slots(0), face_slots(1), face_slots(2), face_slots(3)
    };

    /**
     * Every face of a finite cell once, from the lower cell, as
     * `f(c, lf, c2)`.
     */
    template <class F>
    void for_each_face(const SasDelaunay &del, const F &f) {
      const int nf = static_cast<int>(del.tets.cols());
      for (int c = 0; c < nf; ++c) {
        for (int lf = 0; lf < kCellVerts; ++lf) {
          const int c2 = del.adj(lf, c);
          if (c2 < 0 || c2 > c)
            f(c, lf, c2);
        }
      }
    }

    void mark_full_faces(const SasDelaunay &del, const VertexMap &vm,
                         const ArrayXi &pair_of_edge,
                         std::vector<FaceRec> &fs) {
      for_each_face(del, [&](int c, int lf, int) {
        if ((vm.new_of_v(face_vertices(del, c, lf)) < 0).any())
          return;

        const auto &[s0, s1, s2] = kFaceSlots[lf];
        fs[del.face(lf, c)].full = pair_of_edge[del.cell_edge(s0, c)] >= 0
                                   && pair_of_edge[del.cell_edge(s1, c)] >= 0
                                   && pair_of_edge[del.cell_edge(s2, c)] >= 0;
      });
    }

    /**
     * The face opposite local vertex `lf` of cell `c`, sorted, if all three
     * spheres are kept, pairwise overlapping and one of them gets caps.
     */
    bool sphere_face(const SurfaceSpheres &ss, const SasDelaunay &del,
                     const VertexMap &vm, const std::vector<FaceRec> &fs,
                     const int c, const int lf, Array3i &fv, Array3i &abc) {
      return face_spheres(del, vm, c, lf, fv, abc) && abc[0] < ss.n_enum
             && fs[del.face(lf, c)].full;
    }

    int opposite_local(const SasDelaunay &del, const int c2,
                       const Array3i &fv) {
      ABSL_DCHECK_GE(c2, 0) << "sphere face on the hull";
      for (int lv = 0; lv < kCellVerts; ++lv) {
        if ((fv != del.tets(lv, c2)).all())
          return lv;
      }
      ABSL_UNREACHABLE();
    }

    /**
     * An apex that misses one of the face spheres, or is a bounding point
     * (Lemma 1(a) of the analytic_surface page), has non-negative power on
     * that surface, zero only at a tangency point, and never rejects a
     * root. Apex `lv` of cell `c` overlaps all three iff the other faces of
     * the cell carry every pair.
     */
    bool apex_may_reject(const std::vector<FaceRec> &fs, const SasDelaunay &del,
                         const int c, const int lv) {
      bool full = true;
      for (int lf = 0; lf < kCellVerts; ++lf)
        full &= lf == lv || fs[del.face(lf, c)].full;
      return full;
    }

    /**
     * A cut point is a probe iff its power against both apexes is
     * non-negative, decided exactly for the perturbed weights (Theorem 1 of
     * the analytic_surface page).
     */
    std::vector<FaceRec> accept_faces(const SurfaceSpheres &ss,
                                      const SasDelaunay &del,
                                      const VertexMap &vm,
                                      const ArrayXi &pair_of_edge,
                                      std::vector<RawProbe> &raw) {
      const BallExact &ex = del.ex;
      FaceKernelPtr k = face_kernel();

      std::vector<FaceRec> fs(del.n_faces);
      mark_full_faces(del, vm, pair_of_edge, fs);
      raw.reserve(3L * ss.n_enum);

      for_each_face(del, [&](int c, int lf, int c2) {
        Array3i fv, abc;
        if (!sphere_face(ss, del, vm, fs, c, lf, fv, abc))
          return;

        FaceRec &rec = fs[del.face(lf, c)];
        rec.verts = sort3(fv);
        ex.load(*k, ball_triple(rec.verts));
        if (ex.cuts(*k) != Sgn::kPos)
          return;
        rec.cut = true;

        const int lv2 = opposite_local(del, c2, fv);
        const int l1 = del.tets(lf, c), l2 = del.tets(lv2, c2);
        const bool test1 = apex_may_reject(fs, del, c, lf),
                   test2 = apex_may_reject(fs, del, c2, lv2);
        for (const bool plus: { true, false }) {
          if ((test1 && ex.accept(*k, plus, l1) != Sgn::kPos)
              || (test2 && ex.accept(*k, plus, l2) != Sgn::kPos))
            continue;

          rec.root[plus ? 0 : 1] = static_cast<int>(raw.size());
          raw.push_back({ ex.root(*k, plus), rec.verts, plus });
        }
      });

      return fs;
    }

    /**
     * Raw probes owner-sorted (owner = smallest atom): `order` maps a probe
     * to its raw id, `pid` a raw id to its probe.
     */
    struct ProbeOrder {
      ArrayXi order, pid;
    };

    ProbeOrder make_probes(const SurfaceSpheres &ss, const VertexMap &vm,
                           const std::vector<RawProbe> &raw) {
      const int nr = static_cast<int>(raw.size());

      ArrayXi owner(nr);
      for (int r = 0; r < nr; ++r)
        owner[r] = vm.new_of_v(raw[r].face).minCoeff();

      ProbeOrder po { ArrayXi(nr), ArrayXi(nr) };
      OffsetTable own_off(ss.n_enum);
      argsort_bucket(po.order, own_off, owner);
      for (int p = 0; p < nr; ++p)
        po.pid[po.order[p]] = p;
      return po;
    }

    /**
     * Accessible arcs by circle with, per arc, the departing tangent at
     * `beg` and the arriving tangent at `end` from the circle frame and each
     * end's own exact-class angle, so a vertex's darts depend only on its own
     * offset even on a circle the probe positions cannot resolve.
     */
    struct Arcs {
      std::vector<RingArc> arcs;
      RingTangents tangents;
      OffsetTable off;
    };

    int local_index(const Array4i &tv, const int v) {
      ABSL_DCHECK_EQ((tv == v).count(), 1);
      return value_if(tv[1] == v) + 2 * value_if(tv[2] == v)
             + 3 * value_if(tv[3] == v);
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
     * `-eps` enters ball `c` along circle `(a, b)` oriented about `a -> b`,
     * the root `+eps` leaves it.
     */
    int face_parity(const Array3i &sorted, const int a, const int b) {
      const int ia = value_if(sorted[1] == a) + 2 * value_if(sorted[2] == a),
                ib = value_if(sorted[1] == b) + 2 * value_if(sorted[2] == b),
                turn = (ib - ia + 3) % 3;
      ABSL_DCHECK_NE(turn, 0);
      return 3 - 2 * turn;
    }

    /**
     * Cells around edge `(a, b)` counter-clockwise about `a -> b`; positively
     * oriented cells make `(a, b, d, c)` even iff `d -> c` turns
     * counter-clockwise. `face()` is the face `(a, b, third())` about to be
     * crossed.
     */
    class FanWalker {
    public:
      FanWalker(const SasDelaunay &del, int va, int vb, int edge)
          : del_(&del), va_(va), vb_(vb) {
        ABSL_DCHECK_GE(edge, del.nbrs.offset(va));
        ABSL_DCHECK_LT(edge, del.nbrs.offset(va + 1));
        ABSL_DCHECK_EQ(del.nbrs.adj()[edge], vb);
        t0_ = t_ = del.edge_cell[edge];

        const Array4i tv = del.tets.col(t_);
        const int la = local_index(tv, va), lb = local_index(tv, vb);
        std::array<int, 2> others;
        for (int lv = 0, m = 0; lv < kCellVerts; ++lv)
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

    /**
     * The accepted roots of cut face `f` on circle `(va, vb)` as (enter
     * ball `c`, leave ball `c`).
     */
    void push_face_roots(std::vector<RingVertex> &verts, const FaceRec &rec,
                         const ArrayXi &pid, const int f, const int va,
                         const int vb) {
      const int eps = face_parity(rec.verts, va, vb);
      for (const int sigma: { -eps, eps }) {
        const bool plus = sigma > 0;
        const int r = rec.root[plus ? 0 : 1];
        if (r >= 0)
          verts.push_back({ pid[r], f, plus, sigma == eps });
      }
    }

    /**
     * Walk the cell fan around edge `(a, b)` counter-clockwise (Lemma 2 of
     * the analytic_surface page);
     * each face `(a, b, c)` with a sphere `c` overlapping both contributes
     * its accepted roots as (enter ball `c`, leave ball `c`). `fan` collects
     * the spheres `c`.
     */
    void collect_ring(Ring &ring, std::vector<int> &fan, const SasDelaunay &del,
                      const std::vector<FaceRec> &fs, const ArrayXi &pid,
                      const int va, const int vb, const int edge) {
      ring.verts.clear();
      ring.covered = false;
      fan.clear();

      FanWalker walk(del, va, vb, edge);
      do {
        const int f = walk.face();
        const FaceRec &rec = fs[f];
        if (!rec.full)
          continue;

        const int vc = walk.third();
        ABSL_DCHECK_GE(rec.verts[0], 0);
        fan.push_back(vc);
        if (!rec.cut) {
          ring.covered |= del.ex.side(va, vb, vc) == Sgn::kNeg;
          continue;
        }

        ring.covered = true;
        push_face_roots(ring.verts, rec, pid, f, va, vb);
      } while (walk.advance());
    }

    /**
     * Exact queries of `ring.h` on circle `(va, vb)` of the triangulation;
     * the antipode is accessible iff it is outside every fan sphere.
     */
    // NOLINTBEGIN(*-unused-member-function)
    class SasRingOps {
    public:
      SasRingOps(const BallExact &ex, const std::vector<FaceRec> &fs,
                 const std::vector<int> &fan, int va, int vb)
          : ex_(&ex), fs_(&fs), fan_(&fan), va_(va), vb_(vb) { }

      BallTriple face(const RingVertex &v) const {
        return ball_triple((*fs_)[v.face].verts);
      }

      Vector3d offset(const RingVertex &v) const {
        return ex_->offset(va_, vb_, face(v), v.plus);
      }

      int half_plane(const RingVertex &v) const {
        return ex_->half_plane(va_, vb_, face(v), v.plus);
      }

      Sgn ccw(const RingVertex &p, const RingVertex &r) const {
        return ex_->ccw(va_, vb_, face(p), p.plus, face(r), r.plus);
      }

      bool antipode_accessible(const RingVertex &v) const {
        const BallTriple f = face(v);
        bool accessible = true;
        for (const int vc: *fan_)
          accessible &= ex_->antipode(va_, vb_, f, v.plus, vc) == Sgn::kPos;
        return accessible;
      }

    private:
      const BallExact *ex_;
      const std::vector<FaceRec> *fs_;
      const std::vector<int> *fan_;
      int va_, vb_;
    };
    // NOLINTEND(*-unused-member-function)

    /**
     * Accessible arcs of every circle with caps, grouped by circle.
     */
    Arcs fan_rings(const SurfaceSpheres &ss, const SasDelaunay &del,
                   const VertexMap &vm, const ArrayXi &edge_of_pair,
                   const std::vector<SasCircle> &circ,
                   const std::vector<FaceRec> &fs, const ArrayXi &pid) {
      Arcs out;  // reserving `out.arcs` measured as noise (+-1 %)
      out.off = OffsetTable(ss.g.m());
      Ring ring;
      std::vector<int> fan;

      for (int i = 0; i < ss.n_enum; ++i) {
        for (auto it = ss.g.begin(i), ei = ss.g.end(i); it < ei; ++it) {
          const int j = *it, q = ss.g.eid(it);
          out.off.off()[q] = static_cast<int>(out.arcs.size());

          const int edge = edge_of_pair[q];
          if (edge < 0)
            continue;

          const int va = vm.v_of_new[i], vb = vm.v_of_new[j];
          collect_ring(ring, fan, del, fs, pid, va, vb, edge);
          const int n = static_cast<int>(ring.verts.size());
          if (n > 0) {
            ABSL_DCHECK_EQ(n % 2, 0) << "odd ring on circle " << q;

            const SasRingOps ops(del.ex, fs, fan, va, vb);
            ring.e1 = circ[q].e1;
            ring.e2 = circ[q].e2;
            ring_angles(ring, ops);
            decide_wrap(ring, ops);
          }
          emit_arcs(ring, q, out.arcs, out.tangents);
        }
      }

      for (int q = ss.g.offset(ss.n_enum); q <= ss.g.m(); ++q)
        out.off.off()[q] = static_cast<int>(out.arcs.size());
      return out;
    }

    using Cell = std::array<std::int64_t, 3>;

    Cell operator+(const Cell &a, const Cell &b) {
      return { a[0] + b[0], a[1] + b[1], a[2] + b[2] };
    }

    /**
     * Every pair of exactly coincident probes joined in `uf`. Candidates
     * are the pairs within `probe_merge_tol`, a certified superset: probes
     * are bucketed on a lattice of pitch `tau` and sorted by cell, so a
     * pair within `tau` shares a cell or sits in cells adjacent along each
     * axis; a constant cell shift preserves the sorted order, so the 13
     * forward neighbor shifts are 13 merge joins of the order with itself.
     * `BallExact::coincide` decides each candidate not already linked
     * (coincidence is transitive, so the skipped pairs are known). The
     * result is the partition into exact points, whatever the probe order.
     * Cell indices fit `int64`: the span is `<= 2 (kSurfaceMaxCoord + rmax)`
     * and `prepare` keeps `rmax >= 2 kSurfaceLengthEps`, so
     * `span / tau < 2^43`.
     */
    void merge_near_probes(const SurfaceSpheres &ss, const BallExact &ex,
                           const std::vector<RawProbe> &raw,
                           const ArrayXi &order, UnionFind &uf) {
      const int np = static_cast<int>(order.size());
      if (np < 2)
        return;

      const double tau = probe_merge_tol(ss.rmax), tau2 = tau * tau;
      Vector3d lo = raw[0].pos;
      for (const RawProbe &r: raw)
        lo = lo.cwiseMin(r.pos);
      auto pos = [&](const int p) -> const Vector3d & {
        return raw[order[p]].pos;
      };
      std::vector<Cell> cells(np);
      for (int p = 0; p < np; ++p) {
        for (int k = 0; k < 3; ++k) {
          cells[p][k] =
              static_cast<std::int64_t>(std::floor((pos(p)[k] - lo[k]) / tau));
        }
      }

      ArrayXi by_cell = ArrayXi::LinSpaced(np, 0, np - 1);
      std::sort(by_cell.begin(), by_cell.end(),
                [&](int p, int r) { return cells[p] < cells[r]; });

      auto run_end = [&](const int i) {
        int j = i + 1;
        while (j < np && cells[by_cell[j]] == cells[by_cell[i]])
          ++j;
        return j;
      };
      auto coincident = [&](const int p, const int r) {
        const RawProbe &rp = raw[order[p]], &rr = raw[order[r]];
        return (pos(p) - pos(r)).squaredNorm() <= tau2
               && uf.find(p) != uf.find(r)
               && ex.coincide(ball_triple(rp.face), rp.plus,
                              ball_triple(rr.face), rr.plus);
      };
      auto join = [&](const int bi, const int ei, const int bj, const int ej) {
        for (int a = bi; a < ei; ++a) {
          for (int b = bj; b < ej; ++b) {
            if (coincident(by_cell[a], by_cell[b]))
              uf.merge(by_cell[a], by_cell[b]);
          }
        }
      };

      for (int i = 0; i < np;) {
        const int e = run_end(i);
        for (int a = i; a < e; ++a)
          join(a, a + 1, a + 1, e);
        i = e;
      }

      for (std::int64_t dx = -1; dx <= 1; ++dx) {
        for (std::int64_t dy = -1; dy <= 1; ++dy) {
          for (std::int64_t dz = -1; dz <= 1; ++dz) {
            const Cell shift { dx, dy, dz };
            if (!(Cell { 0, 0, 0 } < shift))
              continue;

            for (int i = 0, j = 0; i < np && j < np;) {
              const Cell ci = cells[by_cell[i]] + shift, cj = cells[by_cell[j]];
              if (ci < cj) {
                i = run_end(i);
              } else if (cj < ci) {
                j = run_end(j);
              } else {
                const int ei = run_end(i), ej = run_end(j);
                join(i, ei, j, ej);
                i = ei;
                j = ej;
              }
            }
          }
        }
      }
    }

    ArrayXd sweep_spheres(const SurfaceSpheres &ss, const SasDelaunay &del,
                          const VertexMap &vm, const SasCaps &caps,
                          const Arcs &arcs, const std::vector<RawProbe> &raw,
                          const ArrayXi &order) {
      const int n_solve = ss.n_solve, mcap = caps.h.max_deg(),
                np = static_cast<int>(order.size());
      const BallExact &ex = del.ex;

      int acap = 0;
      for (int s = 0; s < n_solve; ++s) {
        int n_arcs = 0;
        for (auto it = caps.h.begin(s), e = caps.h.end(s); it < e; ++it)
          n_arcs += arcs.off.degree(*it);
        acap = nuri::max(acap, n_arcs);
      }

      ArrayXd area(n_solve);
      ArrangementSolver solver(mcap, 2 * acap, acap);
      ArrayXi vloc = ArrayXi::Constant(np, -1), vlist(2 * acap + 1);

      for (int s = 0; s < n_solve; ++s) {
        if (del.nbrs.degree(vm.v_of_new[s]) == 0) {
          area[s] = 0;
          continue;
        }

        solver.begin(ss.sar[s]);
        int k = 0;
        auto local = [&](int p) {
          if (vloc[p] < 0) {
            vloc[p] = solver.add_vertex(
                (raw[order[p]].pos - ss.pts.col(s)).normalized());
            vlist[k++] = p;
          }
          return vloc[p];
        };

        for (auto it = caps.h.begin(s), e = caps.h.end(s); it < e; ++it) {
          const int p = caps.h.eid(it), q = *it;
          const int slot = solver.add_cap(caps.cosa[p]);
          const bool first = s < caps.partner[p];

          for (int r = arcs.off[q]; r < arcs.off[q + 1]; ++r) {
            const RingArc &arc = arcs.arcs[r];
            const auto &[tb, te] = arcs.tangents[r];
            if (arc.beg < 0) {
              solver.add_arc(slot, -1, -1, arc.dphi, tb, te);
              continue;
            }
            const int vb = local(arc.beg), ve = local(arc.end);
            if (first)
              solver.add_arc(slot, vb, ve, arc.dphi, tb, te);
            else
              solver.add_arc(slot, ve, vb, arc.dphi, -te, -tb);
          }
        }

        const int vs = vm.v_of_new[s], beg = caps.h.offset(s);
        area[s] = solver.solve([&](int j, int l) {
          const int pj = caps.partner[beg + j], pl = caps.partner[beg + l];
          return pair_id_unordered(ss.g, pj, pl) >= 0
                 && ex.discs_intersect(vs, vm.v_of_new[pj], vm.v_of_new[pl]);
        });

        for (int v = 0; v < k; ++v)
          vloc[vlist[v]] = -1;
      }

      return area;
    }

    /**
     * Probes linked in `uf` become one probe on the sorted union of their
     * atoms at the position and root of their lowest member, owner-sorted
     * like `make_probes`; arc ends are relabeled to the merged ids.
     */
    SasProbes merge_probes(const SurfaceSpheres &ss,
                           const std::vector<RawProbe> &raw,
                           const ArrayXi &order, const VertexMap &vm,
                           UnionFind &uf, Arcs &arcs) {
      const int np = static_cast<int>(order.size()), nc = uf.relabel();
      const auto label = uf.labels();

      ArrayXi mem(np);
      OffsetTable moff(nc);
      argsort_bucket(mem, moff, label);

      ArrayXi catoms(3L * np), owner(nc);
      OffsetTable coff(nc);
      int n = coff.off()[0] = 0;
      for (int c = 0; c < nc; ++c) {
        const int beg = n;
        for (const int p: mem.segment(moff[c], moff.degree(c))) {
          catoms.segment(n, 3) = sort3(vm.new_of_v(raw[order[p]].face));
          n += 3;
        }
        const auto first = catoms.begin() + beg, last = catoms.begin() + n;
        std::sort(first, last);
        n = static_cast<int>(std::unique(first, last) - catoms.begin());
        coff.off()[c + 1] = n;
        owner[c] = catoms[beg];
      }

      ArrayXi by_owner(nc);
      OffsetTable own_off(ss.n_enum);
      argsort_bucket(by_owner, own_off, owner);

      SasProbes merged { CSR(ArrayXi(n), OffsetTable(nc)),
                         Matrix3Xd(3, nc),
                         Matrix3Xd(),
                         OffsetTable(nc),
                         Array3Xi(3, nc),
                         ArrayXb(nc),
                         ArrayXi(nc),
                         own_off[ss.n_active] };
      ArrayXi &adj = merged.atoms.adj(), &aoff = merged.atoms.off();
      ArrayXi cid(nc);
      int w = aoff[0] = 0;
      for (int q = 0; q < nc; ++q) {
        const int c = by_owner[q], d = coff.degree(c);
        cid[c] = q;
        adj.segment(w, d) = catoms.segment(coff[c], d);
        w += d;
        aoff[q + 1] = w;
        const RawProbe &first = raw[order[mem[moff[c]]]];
        merged.pos.col(q) = first.pos;
        merged.face.col(q) = first.face;
        merged.plus[q] = first.plus;
        merged.members[q] = moff.degree(c);
      }

      for (RingArc &arc: arcs.arcs) {
        if (arc.beg < 0)
          continue;

        arc.beg = cid[label[arc.beg]];
        arc.end = cid[label[arc.end]];
      }
      return merged;
    }

    /**
     * An arc with both ends in one merged probe joins two roots at one exact
     * point: a diagonal of a k-fold point (`dphi` exactly 0) that contributes
     * no tangent, or the full loop grazing the vertex (`dphi >= pi`), which
     * contributes both.
     */
    bool has_tangents(const RingArc &arc) {
      return arc.beg >= 0 && (arc.beg != arc.end || arc.dphi >= constants::kPi);
    }

    void tangents(SasProbes &pr, const Arcs &arcs) {
      const int np = static_cast<int>(pr.pos.cols());

      ArrayXi &toff = pr.tan_off.off();
      toff.setZero();
      for (const RingArc &arc: arcs.arcs) {
        if (!has_tangents(arc))
          continue;

        ++toff[arc.beg + 1];
        ++toff[arc.end + 1];
      }
      std::inclusive_scan(toff.begin(), toff.end(), toff.begin());

      pr.tan.resize(3, toff[np]);
      ArrayXi cur = toff.head(np);
      for (int r = 0; r < static_cast<int>(arcs.arcs.size()); ++r) {
        const RingArc &arc = arcs.arcs[r];
        if (!has_tangents(arc))
          continue;

        const auto &[tb, te] = arcs.tangents[r];
        pr.tan.col(cur[arc.beg]++) = tb;
        pr.tan.col(cur[arc.end]++) = -te;
      }
    }
  }  // namespace

  SasEdgeTables edge_tables(const SurfaceSpheres &ss, const SasDelaunay &del) {
    return edge_tables(ss, del, map_vertices(ss, del));
  }

  SasGeometry build_sas(const SurfaceSpheres &ss, const SasDelaunay &del) {
    if (ss.n_enum == 0)
      return {};

    std::vector circ = circles(ss);
    VertexMap vm = map_vertices(ss, del);
    SasEdgeTables tbl = edge_tables(ss, del, vm);
    SasCaps caps = neighbor_caps(ss, del, vm, tbl.pair_of_edge, circ);

    std::vector<RawProbe> raw;
    const std::vector<FaceRec> fs =
        accept_faces(ss, del, vm, tbl.pair_of_edge, raw);
    const ProbeOrder po = make_probes(ss, vm, raw);

    Arcs arcs = fan_rings(ss, del, vm, tbl.edge_of_pair, circ, fs, po.pid);
    ArrayXd area = sweep_spheres(ss, del, vm, caps, arcs, raw, po.order);

    UnionFind uf(static_cast<int>(raw.size()));
    merge_near_probes(ss, del.ex, raw, po.order, uf);
    SasProbes merged = merge_probes(ss, raw, po.order, vm, uf, arcs);
    tangents(merged, arcs);

    const int n_active_arcs = arcs.off[ss.g.offset(ss.n_active)];
    return { std::move(circ),      std::move(caps), std::move(merged),
             std::move(arcs.arcs), n_active_arcs,   std::move(area) };
  }
}  // namespace internal
}  // namespace nuri
