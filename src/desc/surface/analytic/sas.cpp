//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include <absl/log/absl_check.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/desc/surface.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  namespace {
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
          ABSL_DCHECK_GE(ri * ri - a * a, 0);
          double rl = std::sqrt(ri * ri - a * a);
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

    int icirc(int tag) {
      return tag >> 1;
    }

    int iside(int tag) {
      return tag & 1;
    }

    struct TripleCut {
      Vector3d w;
      double wa, amp2, g, h2;
    };

    TripleCut cut_triple(const SasCircle &cij, const Vector3d &pk,
                         const double rk) {
      const Vector3d w = cij.cntr - pk;
      const double wa = w.dot(cij.axis), w2 = w.squaredNorm();
      const double amp2 = w2 - wa * wa;
      const double g = (rk * rk - w2 - cij.rl * cij.rl) / (2 * cij.rl);
      return { w, wa, amp2, g, amp2 - g * g };
    }

    /**
     * Circle `(a, b)` cut by sphere `c`, `a < b < c`; positive iff the two
     * cut points exist, and then the crossing of caps `b`, `c` on sphere `a`.
     */
    template <class Cut>
    bool cut_sorted(const SaPrep &sa, const std::vector<SasCircle> &circ,
                    Array3i abc, const Cut &on_cut) {
      std::sort(abc.begin(), abc.end());
      const int a = abc[0], b = abc[1], c = abc[2];
      if (a >= sa.n_enum)
        return false;

      const int qab = pair_id(sa.g, a, b);
      if (qab < 0 || pair_id(sa.g, a, c) < 0 || pair_id(sa.g, b, c) < 0)
        return false;

      const SasCircle &cij = circ[qab];
      const TripleCut cut = cut_triple(cij, sa.pts.col(c), sa.sar[c]);
      if (cut.h2 <= 0)
        return false;

      on_cut(abc, cij, cut);
      return true;
    }

    struct VertexMap {
      ArrayXi v_of_new, new_of_v;
    };

    VertexMap map_vertices(const SaPrep &sa, const SasDelaunay &del) {
      const int n = sa.g.n();
      ABSL_DCHECK_EQ(n, del.nbrs.n());

      VertexMap vm { ArrayXi(n), ArrayXi::Constant(del.nbrs.n() + 4, -1) };
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
     * Caps `j`, `k` on a sphere cross iff the angle between their axes lies
     * strictly between the difference and the sum of their angular radii,
     * i.e. `|cos g - cos a_j cos a_k| < sin a_j sin a_k`. Pairs this far from
     * the boundary need no exact test; the rest are decided like the
     * enumeration did, by the cut of the circle with the third sphere.
     */
    constexpr double kCrossingSlack = 1e-4;

    struct CapSet {
      SasCaps caps;
      ArrayXb covered;
      Array2Xi xing;
      OffsetTable xoff;
    };

    /**
     * Candidate caps of one sphere; `crossing` and `hidden` are filled by
     * `classify_caps`.
     */
    struct CapScratch {
      ArrayXi sphere, tag;
      ArrayXb hidden;
      Matrix3Xd axis;
      ArrayXd cosa, sina;
      MatrixXd cosg;
      ArrayXX<bool> crossing;
      int m;
    };

    void gather_caps(CapScratch &cs, const SaPrep &sa, const SasDelaunay &del,
                     const VertexMap &vm, const std::vector<SasCircle> &circ,
                     const int s) {
      const double rs = sa.sar[s];

      int m = 0;
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
        cs.sphere[m] = t;
        cs.tag[m] = 2 * q + side;
        cs.axis.col(m) = (1 - 2 * side) * c.axis;
        cs.cosa[m] = (c.a + (d - 2 * c.a) * side) / rs;
        cs.sina[m] = c.rl / rs;
        ++m;
      }
      cs.m = m;
    }

    /**
     * Nested caps are hidden and dropped; returns whether two remaining caps
     * with disjoint boundaries cover the sphere.
     */
    bool classify_caps(CapScratch &cs, const SaPrep &sa,
                       const std::vector<SasCircle> &circ, const int s) {
      const int m = cs.m;
      auto ax = cs.axis.leftCols(m);
      auto cosg = take_buffer(cs.cosg, m, m);
      auto crossing = take_buffer(cs.crossing, m, m);
      cosg.noalias() = ax.transpose() * ax;

      auto crosses = [&](int j, int k) {
        const double dev = std::abs(cosg(j, k) - cs.cosa[j] * cs.cosa[k]),
                     band = cs.sina[j] * cs.sina[k];
        if (std::abs(dev - band) > kCrossingSlack)
          return dev < band;
        return cut_sorted(sa, circ, { s, cs.sphere[j], cs.sphere[k] },
                          [](auto &&...) { });
      };
      for (int j = 0; j < m; ++j) {
        crossing(j, j) = false;
        for (int k = j + 1; k < m; ++k)
          crossing(j, k) = crossing(k, j) = crosses(j, k);
      }

      bool covered = false;
      for (int j = 0; j < m; ++j) {
        bool hid = false;
        for (int k = 0; k < m; ++k) {
          const bool nested = cosg(j, k) > cs.cosa[j] * cs.cosa[k]
                              && !crossing(j, k);
          hid |= nested && cs.cosa[j] > cs.cosa[k];
          covered |= !nested && !crossing(j, k) && cs.cosa[j] + cs.cosa[k] < 0;
        }
        cs.hidden[j] = hid;
      }
      return covered;
    }

    /**
     * Caps of sphere `s` are its overlapping Delaunay neighbours: every other
     * overlapping sphere cuts a cap inside their union, so the accessible
     * region and its component structure are unchanged.
     */
    CapSet neighbor_caps(const SaPrep &sa, const SasDelaunay &del,
                         const VertexMap &vm,
                         const std::vector<SasCircle> &circ) {
      const int n_enum = sa.n_enum, n_solve = sa.n_solve,
                dmax = del.nbrs.max_deg(), bound = del.nbrs.m();

      CapSet cs {
        SasCaps { CSR(ArrayXi(bound), OffsetTable(n_enum)), Matrix3Xd(3, bound),
                 ArrayXd(bound), ArrayXd(bound) },
        ArrayXb(n_solve), Array2Xi(2, 0), OffsetTable(n_solve)
      };
      std::vector<int> xing;

      CapScratch scratch { ArrayXi(dmax),
                           ArrayXi(dmax),
                           ArrayXb(dmax),
                           Matrix3Xd(3, dmax),
                           ArrayXd(dmax),
                           ArrayXd(dmax),
                           MatrixXd(dmax, dmax),
                           ArrayXX<bool>(dmax, dmax),
                           0 };
      ArrayXi slot(dmax);

      ArrayXi &tags = cs.caps.h.adj(), &off = cs.caps.h.off();
      int w = 0;
      for (int s = 0; s < n_enum; ++s) {
        off[s] = w;
        gather_caps(scratch, sa, del, vm, circ, s);
        const bool covered = classify_caps(scratch, sa, circ, s);

        const int m = scratch.m;
        auto crossing = take_buffer(scratch.crossing, m, m);
        for (int j = 0; j < m; ++j) {
          slot[j] = scratch.hidden[j] ? -1 : w - off[s];
          if (scratch.hidden[j])
            continue;

          tags[w] = scratch.tag[j];
          cs.caps.axis.col(w) = scratch.axis.col(j);
          cs.caps.cosa[w] = scratch.cosa[j];
          cs.caps.sina[w] = scratch.sina[j];
          ++w;
        }

        if (s >= n_solve)
          continue;

        cs.covered[s] = covered;
        cs.xoff.off()[s] = static_cast<int>(xing.size()) / 2;
        for (int j = 0; j < m; ++j) {
          for (int k = j + 1; k < m; ++k) {
            if (slot[j] < 0 || slot[k] < 0 || !crossing(j, k))
              continue;

            xing.push_back(slot[j]);
            xing.push_back(slot[k]);
          }
        }
      }
      off[n_enum] = w;
      cs.xoff.off()[n_solve] = static_cast<int>(xing.size()) / 2;

      tags.conservativeResize(w);
      cs.caps.axis.conservativeResize(3, w);
      cs.caps.cosa.conservativeResize(w);
      cs.caps.sina.conservativeResize(w);
      cs.xing = eigen_map(xing).reshaped(2, xing.size() / 2);
      return cs;
    }

    struct RawVertex {
      Array4i atoms;
      Vector3d sum;
      int count;
    };

    constexpr double kInf = std::numeric_limits<double>::infinity();

    /**
     * The two cells sharing a Delaunay face: `c2` and `l2` are -1 past the
     * hull, `l1`/`l2` are the apex spheres in prepared indices (-1 for a
     * bounding point).
     */
    struct FaceApex {
      int c, c2, l1, l2;
    };

    struct VertexSink {
      std::vector<RawVertex> raw;
      ArrayXi tet_slot;
    };

    double sphere_power(const SaPrep &sa, const Vector3d &x, const int l,
                        double &tol) {
      if (l < 0) {
        tol = 0;
        return kInf;
      }
      tol = 2 * sa.sar[l] * kSurfaceLengthEps;
      return (x - sa.pts.col(l)).squaredNorm() - sa.sar[l] * sa.sar[l];
    }

    void merge_orthocenter(VertexSink &sink, const SasDelaunay &del,
                           const VertexMap &vm, const int tet,
                           const Vector3d &x) {
      int &slot = sink.tet_slot[tet];
      if (slot >= 0) {
        sink.raw[slot].sum += x;
        ++sink.raw[slot].count;
        return;
      }

      slot = static_cast<int>(sink.raw.size());
      Array4i atoms = vm.new_of_v(del.tets.col(tet));
      ABSL_DCHECK((atoms >= 0).all());
      std::sort(atoms.begin(), atoms.end());
      sink.raw.push_back({ atoms, x, 1 });
    }

    /**
     * A cut point of circle `ab` with sphere `c` is accessible iff it lies on
     * the power-diagram edge dual to Delaunay face `abc`, i.e. it has
     * non-negative power against the apexes of the two cells sharing the
     * face. A point within tolerance of an apex sphere is that cell's
     * orthocenter; all four faces of the cell find it, so it is merged there.
     */
    void accept_cuts(VertexSink &sink, const SaPrep &sa, const SasDelaunay &del,
                     const VertexMap &vm, const Array3i &abc,
                     const SasCircle &cij, const TripleCut &cut,
                     const FaceApex &apex) {
      const Vector3d wperp = cut.w - cut.wa * cij.axis;
      const double scale = cij.rl / cut.amp2;
      const Vector3d radial = scale * cut.g * wperp,
                     tangent =
                         scale * std::sqrt(cut.h2) * cij.axis.cross(wperp);

      for (int side = 0; side < 2; ++side) {
        const Vector3d x = cij.cntr + radial + (1 - 2 * side) * tangent;
        double tol1, tol2;
        const double p1 = sphere_power(sa, x, apex.l1, tol1),
                     p2 = sphere_power(sa, x, apex.l2, tol2);
        if (p1 < -tol1 || p2 < -tol2)
          continue;

        if (p1 <= tol1) {
          merge_orthocenter(sink, del, vm, apex.c, x);
        } else if (p2 <= tol2) {
          merge_orthocenter(sink, del, vm, apex.c2, x);
        } else {
          sink.raw.push_back({
              { abc[0], abc[1], abc[2], -1 },
              x,
              1,
          });
        }
      }
    }

    std::vector<RawVertex>
    extract_vertices(const SaPrep &sa, const SasDelaunay &del,
                     const VertexMap &vm, const std::vector<SasCircle> &circ) {
      const int nf = static_cast<int>(del.tets.cols());
      VertexSink sink { {}, ArrayXi::Constant(nf, -1) };
      sink.raw.reserve(3L * sa.n_enum);

      for (int c = 0; c < nf; ++c) {
        const Array4i tv = del.tets.col(c);
        for (int lf = 0; lf < 4; ++lf) {
          const int c2 = del.adj(lf, c);
          if (c2 >= 0 && c2 < c)
            continue;

          Array3i fv;
          for (int lv = 0, m = 0; lv < 4; ++lv)
            if (lv != lf)
              fv[m++] = tv[lv];
          const Array3i face = vm.new_of_v(fv);
          if ((face < 0).any())
            continue;

          FaceApex apex { c, c2, vm.new_of_v[tv[lf]], -1 };
          for (int lv = 0; c2 >= 0 && lv < 4; ++lv) {
            const int v = del.tets(lv, c2);
            if ((fv != v).all())
              apex.l2 = vm.new_of_v[v];
          }

          cut_sorted(sa, circ, face,
                     [&](const Array3i &abc, const SasCircle &cij,
                         const TripleCut &cut) {
                       accept_cuts(sink, sa, del, vm, abc, cij, cut, apex);
                     });
        }
      }

      return std::move(sink.raw);
    }

    struct Vertices {
      SasProbes probes;
      CSR by_sphere;
    };

    Vertices order_vertices(const SaPrep &sa,
                            const std::vector<RawVertex> &raw) {
      const int nv = static_cast<int>(raw.size()), n_enum = sa.n_enum,
                n_solve = sa.n_solve;

      ArrayXi owner(nv);
      for (int r = 0; r < nv; ++r)
        owner[r] = raw[r].atoms[0];

      ArrayXi order(nv);
      OffsetTable own_off(n_enum);
      argsort_bucket(order, own_off, owner);

      Vertices vtx {
        SasProbes { CSR(ArrayXi(4L * nv), OffsetTable(nv)), Matrix3Xd(3, nv),
                   Matrix3Xd(), OffsetTable(nv), own_off[sa.n_active] },
        CSR()
      };

      ArrayXi &adj = vtx.probes.atoms.adj(), &aoff = vtx.probes.atoms.off();
      ArrayXi key(4L * nv), val(4L * nv);
      int w = 0, m = 0;
      for (int p = 0; p < nv; ++p) {
        const RawVertex &rv = raw[order[p]];
        vtx.probes.pos.col(p) = rv.sum / rv.count;

        aoff[p] = w;
        for (int a: rv.atoms) {
          if (a < 0)
            continue;

          adj[w++] = a;
          if (a < n_solve) {
            key[m] = a;
            val[m] = p;
            ++m;
          }
        }
      }
      aoff[nv] = w;
      adj.conservativeResize(w);

      ArrayXi perm(m), sadj(m);
      OffsetTable soff(n_solve);
      argsort_bucket(perm, soff, key.head(m));
      sadj = val(perm);
      vtx.by_sphere = CSR(std::move(sadj), std::move(soff));

      return vtx;
    }

    struct Sweep {
      std::vector<SasArc> arcs;
      int n_active_arcs;
      ArrayXd area;
    };

    Sweep sweep_spheres(const SaPrep &sa, const SasDelaunay &del,
                        const VertexMap &vm, const std::vector<SasCircle> &circ,
                        const CapSet &cs, const Vertices &vtx) {
      const SasCaps &caps = cs.caps;
      const int n_solve = sa.n_solve, mcap = caps.h.max_deg(),
                kcap = vtx.by_sphere.max_deg(), ecap = cs.xoff.max_deg();

      Sweep sw { {}, 0, ArrayXd(n_solve) };
      ArrangementSolver solver(mcap, kcap, ecap);
      ArrayXi slot_of = ArrayXi::Constant(sa.g.n(), -1);

      auto solve = [&](int s) {
        const int off = caps.h.offset(s), m = caps.h.degree(s);

        solver.begin(sa.sar[s]);
        for (int p = off; p < off + m; ++p) {
          solver.add_cap(caps.axis.col(p), caps.cosa[p], caps.sina[p]);
          const SasCircle &c = circ[icirc(caps.h.adj()[p])];
          slot_of[c.i == s ? c.j : c.i] = p - off;
        }

        for (int e = cs.xoff[s]; e < cs.xoff[s + 1]; ++e)
          solver.add_crossing(cs.xing(0, e), cs.xing(1, e));

        auto vs = vtx.by_sphere.nbrs(s);
        for (int k = 0; k < vs.size(); ++k) {
          const int v = vs[k];
          solver.add_vertex(
              (vtx.probes.pos.col(v) - sa.pts.col(s)).normalized(), true);
          for (int a: vtx.probes.atoms.nbrs(v)) {
            if (a == s)
              continue;

            ABSL_DCHECK_GE(slot_of[a], 0);
            solver.add_incidence(slot_of[a], k);
          }
        }

        const int n0 = static_cast<int>(sw.arcs.size());
        const double area = solver.solve(sw.arcs);

        for (int p = off; p < off + m; ++p) {
          const SasCircle &c = circ[icirc(caps.h.adj()[p])];
          slot_of[c.i == s ? c.j : c.i] = -1;
        }

        int w = n0;
        for (int r = n0; r < static_cast<int>(sw.arcs.size()); ++r) {
          SasArc &arc = sw.arcs[r];
          const int tag = caps.h.adj()[off + arc.circ];
          arc.circ = icirc(tag);
          arc.beg = arc.beg < vs.size() ? vs[arc.beg] : -1;
          arc.end = arc.end < vs.size() ? vs[arc.end] : -1;
          sw.arcs[w] = arc;
          w += 1 - iside(tag);
        }
        sw.arcs.resize(w);

        return area;
      };

      auto visit = [&](int s) {
        const bool hidden = del.nbrs.degree(vm.v_of_new[s]) == 0;
        sw.area[s] = hidden || cs.covered[s] ? 0.0 : solve(s);
      };

      for (int s = 0; s < sa.n_active; ++s)
        visit(s);
      sw.n_active_arcs = static_cast<int>(sw.arcs.size());
      for (int s = sa.n_active; s < n_solve; ++s)
        visit(s);

      return sw;
    }

    void tangents(const std::vector<SasCircle> &circ, SasProbes &pr,
                  const std::vector<SasArc> &arcs) {
      const int np = static_cast<int>(pr.pos.cols());

      ArrayXi &toff = pr.tan_off.off();
      toff.setZero();
      for (const SasArc &arc: arcs) {
        if (arc.beg < 0)
          continue;

        ++toff[arc.beg + 1];
        ++toff[arc.end + 1];
      }
      std::inclusive_scan(toff.begin(), toff.end(), toff.begin());

      pr.tan.resize(3, toff[np]);
      ArrayXi cur = toff.head(np);
      for (const SasArc &arc: arcs) {
        if (arc.beg < 0)
          continue;

        const SasCircle &c = circ[arc.circ];
        auto depart = [&](int p, double sign) {
          const Vector3d t = c.axis.cross(pr.pos.col(p) - c.cntr);
          pr.tan.col(cur[p]++) = sign * t.normalized();
        };
        depart(arc.beg, 1.0);
        depart(arc.end, -1.0);
      }
    }
  }  // namespace

  SasGeometry build_sas(const SaPrep &sa, const SasDelaunay &del) {
    if (sa.n_enum == 0)
      return {};

    std::vector circ = circles(sa);
    VertexMap vm = map_vertices(sa, del);
    CapSet cs = neighbor_caps(sa, del, vm, circ);
    Vertices vtx = order_vertices(sa, extract_vertices(sa, del, vm, circ));
    Sweep sw = sweep_spheres(sa, del, vm, circ, cs, vtx);
    tangents(circ, vtx.probes, sw.arcs);

    return { std::move(circ),    std::move(cs.caps), std::move(vtx.probes),
             std::move(sw.arcs), sw.n_active_arcs,   std::move(sw.area) };
  }
}  // namespace internal
}  // namespace nuri
