//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <vector>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
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

    int icirc(int tag) {
      return tag >> 1;
    }

    int iside(int tag) {
      return tag & 1;
    }

    std::pair<SasCaps, ArrayXi> cap_rows(const SaPrep &sa,
                                         const std::vector<SasCircle> &circ) {
      const int n_enum = sa.n_enum, n_circ = static_cast<int>(circ.size());

      ArrayXi key(2L * sa.g.m());
      for (int q = 0; q < n_circ; ++q) {
        key[2L * q] = circ[q].i;
        // n_enum = drop bucket
        key[2L * q + 1] = nuri::min(circ[q].j, n_enum);
      }

      ArrayXi tags(2L * n_circ);
      OffsetTable off(n_enum);
      argsort_bucket(tags, off.off(), key.head(2L * n_circ), [&](int p) {
        // side 1 first, then side 0
        return p < n_circ ? 2 * p + 1 : 2 * (p - n_circ);
      });
      const int m = off[n_enum];
      tags.conservativeResize(m);

      key.setConstant(-1);
      for (int p = 0; p < m; ++p)
        key[tags[p]] = p;

      SasCaps caps { CSR(std::move(tags), std::move(off)), Matrix3Xd(3, m),
                     ArrayXd(m), ArrayXd(m) };
      for (int i = 0; i < n_enum; ++i) {
        const double rs = sa.sar[i];
        for (auto it = caps.h.begin(i), ei = caps.h.end(i); it < ei; ++it) {
          const int p = caps.h.eid(it), q = icirc(*it), side = iside(*it);
          const SasCircle &c = circ[q];
          const double d = sa.d[q];

          caps.axis.col(p) = (1 - 2 * side) * c.axis;
          caps.cosa[p] = (c.a + (d - 2 * c.a) * side) / rs;
          caps.sina[p] = c.rl / rs;
        }
      }
      return { std::move(caps), std::move(key) };
    }

    struct Triple {
      Array3i ijk;
      Array3i q;
      E::Array2i cluster = E::Array2i::Constant(-1);
    };

    struct Incidences {
      std::vector<Triple> tri;
      CSR inc;
      Array2Xi slot;
    };

    constexpr int kOtherPair[3][2] = {
      { 0, 1 },
      { 0, 2 },
      { 1, 2 }
    };
    constexpr int kOtherSide[3][2] = {
      { 0, 0 },
      { 1, 0 },
      { 1, 1 }
    };

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

    Incidences incidences(const SaPrep &sa, const std::vector<SasCircle> &circ,
                          const ArrayXi &slot_of) {
      std::vector<Triple> tri;

      sa.g.for_each_triangle(
          sa.n_enum,
          [&](int i, int j, int k, auto pij, auto pik, auto pjk) {
            const int qij = sa.g.eid(pij), qik = sa.g.eid(pik),
                      qjk = sa.g.eid(pjk);
            const double h2 =
                cut_triple(circ[qij], sa.pts.col(k), sa.sar[k]).h2;
            if (h2 > 0) {
              tri.push_back({
                  {   i,   j,   k },
                  { qij, qik, qjk },
              });
            }
          },
          [](int /* i */) { });

      const int nt = static_cast<int>(tri.size());

      ArrayXi key(3L * nt);
      for (int t = 0; t < nt; ++t)
        key.segment(3L * t, 3) = tri[t].ijk.min(sa.n_enum);

      ArrayXi adj(3L * nt);
      OffsetTable off(sa.n_enum);
      argsort_bucket(adj, off, key);
      const int m = off[sa.n_enum];
      adj.conservativeResize(m);

      Array2Xi slot(2, m);
      for (int e = 0; e < m; ++e) {
        const int t = adj[e] / 3, corner = adj[e] % 3;
        const Array3i &q = tri[t].q;
        for (int c = 0; c < 2; ++c) {
          const int s =
              slot_of[2L * q[kOtherPair[corner][c]] + kOtherSide[corner][c]];
          ABSL_DCHECK_GE(s, 0);
          slot(c, e) = s;
        }
      }

      return { std::move(tri), CSR(std::move(adj), std::move(off)),
               std::move(slot) };
    }

    struct CapVisibility {
      ArrayXb hidden;
      ArrayXb covered;
      ArrayXi active;
    };

    CapVisibility hide_caps(const SaPrep &sa, const SasCaps &caps,
                            const Incidences &inc) {
      const int n_t = static_cast<int>(inc.tri.size()), n_enum = sa.n_enum,
                n_solve = sa.n_solve, dmax = caps.h.max_deg();

      CapVisibility vis { ArrayXb(caps.h.m()), ArrayXb(n_solve),
                          ArrayXi::Ones(n_t) };
      MatrixXd cosg(dmax, dmax);
      ArrayXX<bool> crossing(dmax, dmax);

      auto classify = [&](int s) {
        const int off = caps.h.offset(s), m = caps.h.degree(s);
        auto ax = caps.axis.middleCols(off, m);
        auto c = caps.cosa.segment(off, m);
        auto hidden = vis.hidden.segment(off, m);

        cosg.topLeftCorner(m, m).noalias() = ax.transpose() * ax;
        crossing.topLeftCorner(m, m).setConstant(false);
        for (auto it = inc.inc.begin(s), ei = inc.inc.end(s); it < ei; ++it) {
          const int e = inc.inc.eid(it), a = inc.slot(0, e) - off,
                    b = inc.slot(1, e) - off;
          crossing(a, b) = crossing(b, a) = true;
        }

        bool covered = false;
        for (int j = 0; j < m; ++j) {
          bool hid = false;
          for (int k = 0; k < m; ++k) {
            const bool nested = cosg(j, k) > c[j] * c[k] && !crossing(j, k);
            hid |= nested && c[j] > c[k];
            covered |= !nested && !crossing(j, k) && c[j] + c[k] < 0;
          }
          hidden[j] = hid;
        }

        for (auto it = inc.inc.begin(s), ei = inc.inc.end(s); it < ei; ++it) {
          const int e = inc.inc.eid(it), t = *it / 3;
          vis.active[t] &= static_cast<int>(!vis.hidden[inc.slot(0, e)]
                                            && !vis.hidden[inc.slot(1, e)]);
        }

        return covered;
      };

      for (int s = 0; s < n_solve; ++s)
        vis.covered[s] = classify(s);
      for (int s = n_solve; s < n_enum; ++s)
        classify(s);

      return vis;
    }

    struct Vertices {
      ArrayXi tri;
      Matrix3Xd pts;
    };

    Vertices vertex_points(const SaPrep &sa, const std::vector<SasCircle> &circ,
                           const std::vector<Triple> &tri,
                           const ArrayXi &active) {
      const int nv = active.sum();
      Vertices vtx { ArrayXi(nv + 1), Matrix3Xd(3, 2L * nv) };

      int v = 0;
      for (int t = 0; t < active.size(); ++t) {
        vtx.tri[v] = t;
        v += active[t];
      }

      for (v = 0; v < nv; ++v) {
        const Triple &tr = tri[vtx.tri[v]];
        const SasCircle &cij = circ[tr.q[0]];
        const int k = tr.ijk[2];

        const TripleCut cut = cut_triple(cij, sa.pts.col(k), sa.sar[k]);
        ABSL_DCHECK_GT(cut.h2, 0);

        const Vector3d wperp = cut.w - cut.wa * cij.axis;
        const double scale = cij.rl / cut.amp2;
        const Vector3d radial = scale * cut.g * wperp,
                       tangent =
                           scale * std::sqrt(cut.h2) * cij.axis.cross(wperp);

        vtx.pts.col(2L * v) = cij.cntr + radial + tangent;
        vtx.pts.col(2L * v + 1) = cij.cntr + radial - tangent;
      }

      return vtx;
    }

    struct Clusters {
      Matrix3Xd rep;
      CSR atoms;
      OffsetTable own_off;
    };

    Clusters cluster_vertices(const Vertices &vtx, std::vector<Triple> &tri,
                              const int n_enum) {
      const int nr = static_cast<int>(vtx.pts.cols());
      auto triple_of = [&](int r) -> Triple & { return tri[vtx.tri[r / 2]]; };

      std::vector<int> left, right;
      OCTree(vtx.pts).find_neighbors_self(kSurfaceLengthEps, left, right);

      ArrayXi label = ArrayXi::LinSpaced(nr, 0, nr - 1);
      auto root = [&](int x) {
        while (label[x] != x) {
          label[x] = label[label[x]];
          x = label[x];
        }
        return x;
      };
      for (int p = 0; p < left.size(); ++p) {
        auto [lo, hi] = nuri::minmax(root(left[p]), root(right[p]));
        label[hi] = lo;
      }
      for (int r = 0; r < nr; ++r)
        label[r] = root(r);

      int nc = 0;
      for (int r = 0; r < nr; ++r)
        label[r] = label[r] == r ? nc++ : label[label[r]];

      ArrayXi kmin = ArrayXi::Constant(nc, n_enum);
      for (int r = 0; r < nr; ++r)
        kmin[label[r]] = nuri::min(kmin[label[r]], triple_of(r).ijk[0]);

      ArrayXi order(nc);
      OffsetTable own_off(n_enum);
      argsort_bucket(order, own_off, kmin);

      ArrayXi mem(nr);
      OffsetTable moff(nc);
      argsort_bucket(mem, moff, label);

      Clusters cl { Matrix3Xd(3, nc), CSR(ArrayXi(3L * nr), OffsetTable(nc)),
                    std::move(own_off) };
      ArrayXi &adj = cl.atoms.adj(), &aoff = cl.atoms.off();
      int n = aoff[0] = 0;
      for (int pos = 0; pos < nc; ++pos) {
        const int c = order[pos];
        auto ms = mem.segment(moff[c], moff.degree(c));
        auto ps = vtx.pts(E::all, ms);

        cl.rep.col(pos) = ps.rowwise().mean();
        ABSL_DCHECK_LE(
            (ps.colwise() - cl.rep.col(pos)).colwise().squaredNorm().maxCoeff(),
            kSurfaceLengthEps * kSurfaceLengthEps);

        for (int r: ms) {
          Triple &t = triple_of(r);
          t.cluster[r % 2] = pos;
          adj.segment(n, 3) = t.ijk;
          n += 3;
        }

        auto beg = adj.begin() + aoff[pos], end = adj.begin() + n;
        std::sort(beg, end);
        n = static_cast<int>(std::unique(beg, end) - adj.begin());
        aoff[pos + 1] = n;
      }
      adj.conservativeResize(n);

      return cl;
    }

    struct Sweep {
      ArrayXb accessible;
      std::vector<SasArc> arcs;
      int n_active_arcs;
      ArrayXd area;
    };

    Sweep sweep_spheres(const SaPrep &sa, const SasCaps &caps,
                        const Incidences &inc, const CapVisibility &vis,
                        const Clusters &cl) {
      const int n_enum = sa.n_enum, nc = cl.own_off.offset(n_enum),
                dmax = caps.h.max_deg(), vmax = 2 * inc.inc.max_deg(),
                omax = cl.own_off.max_deg();

      Sweep sw { ArrayXb(nc), {}, 0, ArrayXd(sa.n_solve) };

      ArrayXX<bool> inside(omax, dmax);

      ArrangementProblem prob;
      prob.reserve(dmax, vmax);
      ArrayXi local_of(dmax), plist(dmax), vlist(vmax + 1),
          vloc = ArrayXi::Constant(nc, -1);

      auto decide = [&](int s) {
        const int off = caps.h.offset(s), m = caps.h.degree(s),
                  c0 = cl.own_off.offset(s), k = cl.own_off.degree(s);
        auto ax = caps.axis.middleCols(off, m);
        auto c = caps.cosa.segment(off, m);
        auto hid = vis.hidden.segment(off, m);
        auto ins = inside.topLeftCorner(k, m);

        for (int r = 0; r < k; ++r) {
          const Vector3d dir =
              (cl.rep.col(c0 + r) - sa.pts.col(s)).normalized();
          ins.row(r) = ((ax.transpose() * dir).array() > c && !hid).transpose();
        }

        for (auto it = inc.inc.begin(s), ei = inc.inc.end(s); it < ei; ++it) {
          const int e = inc.inc.eid(it), t = *it / 3, corner = *it % 3;
          if (corner != 0 || vis.active[t] == 0)
            continue;

          for (int side = 0; side < 2; ++side) {
            const int r = inc.tri[t].cluster[side] - c0;
            if (static_cast<unsigned>(r) >= static_cast<unsigned>(k))
              continue;

            ins(r, inc.slot(0, e) - off) = ins(r, inc.slot(1, e) - off) = false;
          }
        }

        for (int r = 0; r < k; ++r)
          sw.accessible[c0 + r] = !ins.row(r).any();
      };

      auto solve = [&](int s) {
        const int off = caps.h.offset(s), m = caps.h.degree(s);

        int ml = 0;
        for (int l = 0; l < m; ++l) {
          const int p = off + l;
          const bool keep = !vis.hidden[p];
          local_of[l] = keep ? ml : -1;
          plist[ml] = p;
          prob.axis.col(ml) = caps.axis.col(p);
          prob.cosa[ml] = caps.cosa[p];
          prob.sina[ml] = caps.sina[p];
          ml += static_cast<int>(keep);
        }
        prob.radius = sa.sar[s];
        prob.m = ml;
        prob.k = 0;
        prob.crossing.topLeftCorner(ml, ml).setConstant(false);

        for (auto it = inc.inc.begin(s), ei = inc.inc.end(s); it < ei; ++it) {
          const int e = inc.inc.eid(it), t = *it / 3;
          const int la = local_of[inc.slot(0, e) - off],
                    lb = local_of[inc.slot(1, e) - off];
          if (la < 0 || lb < 0)
            continue;

          prob.crossing(la, lb) = prob.crossing(lb, la) = true;
          if (vis.active[t] == 0)
            continue;

          for (int side = 0; side < 2; ++side) {
            const int cc = inc.tri[t].cluster[side];
            int v = vloc[cc];
            if (v < 0) {
              v = vloc[cc] = prob.k;
              vlist[prob.k++] = cc;
              prob.excused.row(v).head(ml).setConstant(false);
            }
            prob.excused(v, la) = prob.excused(v, lb) = true;
          }
        }

        for (int v = 0; v < prob.k; ++v) {
          const int cc = vlist[v];
          prob.reps.col(v) = (cl.rep.col(cc) - sa.pts.col(s)).normalized();
          prob.accessible[v] = sw.accessible[cc];
          vloc[cc] = -1;
        }
        vlist[prob.k] = nc;

        const int n0 = static_cast<int>(sw.arcs.size());
        const double area = solve_arrangement(prob, sw.arcs);

        int w = n0;
        for (int r = n0; r < static_cast<int>(sw.arcs.size()); ++r) {
          SasArc &arc = sw.arcs[r];
          const int tag = caps.h.adj()[plist[arc.circ]];
          arc.circ = icirc(tag);
          arc.beg = vlist[arc.beg];
          arc.end = vlist[arc.end];
          sw.arcs[w] = arc;
          w += 1 - iside(tag);
        }
        sw.arcs.resize(w);

        return area;
      };

      auto visit = [&](int s) {
        decide(s);
        sw.area[s] = vis.covered[s] ? 0.0 : solve(s);
      };

      for (int s = 0; s < sa.n_active; ++s)
        visit(s);
      sw.n_active_arcs = static_cast<int>(sw.arcs.size());
      for (int s = sa.n_active; s < sa.n_solve; ++s)
        visit(s);
      for (int s = sa.n_solve; s < n_enum; ++s)
        decide(s);

      return sw;
    }
  }  // namespace

  SasGeometry build_sas(const SaPrep &sa) {
    if (sa.n_enum == 0)
      return SasGeometry {};

    std::vector circ = circles(sa);
    auto [caps, slot_of] = cap_rows(sa, circ);
    Incidences inc = incidences(sa, circ, slot_of);
    CapVisibility vis = hide_caps(sa, caps, inc);
    Vertices vtx = vertex_points(sa, circ, inc.tri, vis.active);
    Clusters cl = cluster_vertices(vtx, inc.tri, sa.n_enum);
    Sweep sw = sweep_spheres(sa, caps, inc, vis, cl);

    return SasGeometry {};
  }
}  // namespace internal
}  // namespace nuri
