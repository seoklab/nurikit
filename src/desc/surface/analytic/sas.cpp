//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>

#include "nuri/eigen_config.h"
#include "nuri/desc/surface.h"

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

      ArrayXi tags(2L * n_circ), off(n_enum + 1);
      argsort_bucket(tags, off, key.head(2L * n_circ), [&](int p) {
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

    double triple_h2(const Vector3d &wijk, const Vector3d &aij, double rij,
                     double rk) {
      const double wa = wijk.dot(aij), w2 = wijk.squaredNorm();
      const double amp2 = w2 - wa * wa;
      const double gk = (rk * rk - w2 - rij * rij) / (2 * rij);
      return amp2 - gk * gk;
    }

    Incidences incidences(const SaPrep &sa, const std::vector<SasCircle> &circ,
                          const ArrayXi &slot_of) {
      std::vector<Triple> tri;

      sa.g.for_each_triangle(
          sa.n_enum,
          [&](int i, int j, int k, auto pij, auto pik, auto pjk) {
            const int qij = sa.g.eid(pij), qik = sa.g.eid(pik),
                      qjk = sa.g.eid(pjk);
            const SasCircle &cij = circ[qij];
            const double h2 = triple_h2(cij.cntr - sa.pts.col(k), cij.axis,
                                        cij.rl, sa.sar[k]);
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

      ArrayXi adj(3L * nt), off(sa.n_enum + 1);
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
  }  // namespace

  SasGeometry build_sas(const SaPrep &sa) {
    std::vector circ = circles(sa);
    auto [caps, slot_of] = cap_rows(sa, circ);
    Incidences inc = incidences(sa, circ, slot_of);
    CapVisibility vis = hide_caps(sa, caps, inc);

    return SasGeometry {};
  }
}  // namespace internal
}  // namespace nuri
