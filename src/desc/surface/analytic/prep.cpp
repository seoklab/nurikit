//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include <absl/log/absl_log.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/desc/surface.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  namespace {
    struct NearPairs {
      /*  n_over */
      CSR g;
      ArrayXd dover;

      /* n_near */
      ArrayXi inear, jnear;

      /* n */
      ArrayXi keep;
    };

    template <class I, class J>
    std::pair<CSR, ArrayXi> compile_pairs(const I &i, const J &j, const int n) {
      const int m = static_cast<int>(i.size());

      ArrayXi order(m), adj(m);
      OffsetTable off(n);
      argsort_bucket(adj, off.off().head(n), j);
      argsort_bucket(order, off, i, adj);
      adj = j(order);

      return { CSR(std::move(adj), std::move(off)), std::move(order) };
    }

    NearPairs find_near_pairs(const Matrix3Xd &pts, const ArrayXd &sar,
                              const double rmax) {
      const int n = static_cast<int>(sar.size());

      VoxelGrid grid(pts, 2 * (rmax + kSurfaceLengthEps));
      std::vector<int> lbuf, rbuf;
      grid.find_neighbors_self(lbuf, rbuf);
      auto left = eigen_map(lbuf), right = eigen_map(rbuf);
      int m = static_cast<int>(left.size());
      for (int k = 0; k < m; ++k)
        std::tie(left[k], right[k]) = nuri::minmax(left[k], right[k]);

      ArrayXd d = (pts(E::all, right) - pts(E::all, left)).colwise().norm();

      ArrayXi keep = ArrayXi::Ones(n);
      for (int k = 0; k < m; ++k) {
        int i = left[k], j = right[k];
        if (d[k] <= std::abs(sar[i] - sar[j]) + kSurfaceLengthEps)
          keep[sar[i] < sar[j] ? i : j] = 0;
      }

      ArrayXd touch = sar(left) + sar(right);
      // 0 -> overlap, 1 -> near, 2 -> dropped or far
      ArrayXi key =
          (keep(left) + keep(right) < 2)
              .select(2, (d > touch + 2 * kSurfaceLengthEps).cast<int>()
                             + (d >= touch - kSurfaceLengthEps).cast<int>());
      ArrayXi order(m);
      Array3i off;
      argsort_bucket(order, off, key);

      auto near = order.head(off[2]);
      ArrayXi inear = left(near), jnear = right(near);

      m = off[1];
      auto [g, perm] = compile_pairs(inear.head(m), jnear.head(m), n);

      return {
        std::move(g),     d(order).head(m)(perm),

        std::move(inear), std::move(jnear),

        std::move(keep),
      };
    }

    void drop_shared_circle_middles(ArrayXi &keep, const CSR &g,
                                    const ArrayXd &d, const Matrix3Xd &pts,
                                    const ArrayXd &sar2) {
      constexpr double cutoff = kSurfaceLengthEps * kSurfaceLengthEps;

      Matrix3Xd axis(3, g.max_deg()), cntr(3, g.max_deg());

      g.for_each_triangle(
          g.n(),
          [&](int i, int j, int k, auto pij, auto pik, auto) {
            E::Index ij = pij - g.begin(i), ik = pik - g.begin(i);
            Vector3d uij = axis.col(ij), uik = axis.col(ik);
            if ((cntr.col(ij) - cntr.col(ik)).squaredNorm() >= cutoff
                || sar2[i] * uij.cross(uik).squaredNorm() >= cutoff)
              return;

            const double dot = uij.dot(uik);
            const int mid = dot < 0 ? i
                                    : (d[g.eid(pik)] > d[g.eid(pij)] ? j : k);
            keep[mid] = 0;
          },
          [&](int i) {
            const Vector3d ci = pts.col(i);
            const double r2i = sar2[i];

            auto iax = axis.leftCols(g.degree(i)),
                 icn = cntr.leftCols(g.degree(i));
            auto dij = d.segment(g.offset(i), g.degree(i));

            iax = (pts(E::all, g.nbrs(i)).colwise() - ci).array().rowwise()
                  / dij.transpose();
            icn = (iax.array().rowwise()
                   * ((dij.square() + r2i - sar2(g.nbrs(i))) / (2 * dij))
                         .transpose())
                      .matrix()
                      .colwise()
                  + ci;
          });
    }

    using Array5i = E::Array<int, 5, 1>;

    /**
     * Remap to ranking scheme:
     * 0 -> active, 1 -> need, 2 -> shell, 3 -> occluders, 4 -> ignored
     */
    std::pair<ArrayXi, Array5i> rank_atoms(ArrayXi &keep, const ArrayXi &inear,
                                           const ArrayXi &jnear,
                                           const ArrayXb &active) {
      // alias for clarity: &rank == &keep
      ArrayXi &rank = keep;

      // 0/3/4 split here
      rank = (keep.cast<bool>() && active).select(0, 4 - keep);

      for (int k = 0; k < inear.size(); ++k) {
        int i = inear[k], j = jnear[k];
        if (rank[i] == 4 || rank[j] == 4)
          continue;

        rank[j] = rank[i] == 0 ? nuri::min(rank[j], 1) : rank[j];
        rank[i] = rank[j] == 0 ? nuri::min(rank[i], 1) : rank[i];
      }
      for (int k = 0; k < inear.size(); ++k) {
        int i = inear[k], j = jnear[k];
        if (rank[i] == 4 || rank[j] == 4)
          continue;

        rank[j] = rank[i] <= 1 ? nuri::min(rank[j], 2) : rank[j];
        rank[i] = rank[j] <= 1 ? nuri::min(rank[i], 2) : rank[i];
      }

      ArrayXi order(rank.size());
      Array5i off;
      argsort_bucket(order, off, rank);
      return { std::move(order), off };
    }

    /**
     * Candidate pairs among kept spheres that overlap exactly for the
     * heights the lift will hand to the triangulation; a tangency counts.
     */
    std::pair<CSR, ArrayXd> exact_overlaps(const Matrix3Xd &pts,
                                           const ArrayXd &t, const double wmax,
                                           const ArrayXi &keep,
                                           const ArrayXi &inear,
                                           const ArrayXi &jnear) {
      const int m = static_cast<int>(inear.size());
      ArrayXi li(m), lj(m);
      int q = 0;
      for (int k = 0; k < m; ++k) {
        const int i = inear[k], j = jnear[k];
        if (keep[i] == 0 || keep[j] == 0
            || SasExact::overlap(pts.col(i), t[i], pts.col(j), t[j], wmax)
                   != Sgn::kPos)
          continue;

        li[q] = i;
        lj[q] = j;
        ++q;
      }

      auto [g, perm] =
          compile_pairs(li.head(q), lj.head(q), static_cast<int>(keep.size()));
      ArrayXd d(q);
      for (int k = 0; k < q; ++k)
        d[k] = (pts.col(lj[perm[k]]) - pts.col(li[perm[k]])).norm();
      return { std::move(g), std::move(d) };
    }

    SaPrep compact(const Matrix3Xd &pts, const ArrayXd &sar, const ArrayXd &t,
                   const double wmax, const CSR &g, const ArrayXd &d,
                   ArrayXi &&order, const Array5i &off, ArrayXi &inv) {
      const int n = off[4];

      inv.setConstant(-1);
      for (int p = 0; p < n; ++p)
        inv[order[p]] = p;

      ArrayXi ni(g.m()), nj(g.m());
      ArrayXd dn(g.m());
      int q = 0;
      for (int i = 0; i < g.n(); ++i) {
        const int a = inv[i];
        if (a < 0)
          continue;

        for (auto it = g.begin(i), ei = g.end(i); it < ei; ++it) {
          const int b = inv[*it];
          if (b < 0)
            continue;

          std::tie(ni[q], nj[q]) = nuri::minmax(a, b);
          dn[q] = d[g.eid(it)];
          ++q;
        }
      }

      auto [gn, perm] = compile_pairs(ni.head(q), nj.head(q), n);
      order.conservativeResize(n);

      return {
        pts(E::all, order),
        sar(order),
        t(order),
        wmax,
        std::move(order),
        std::move(gn),
        dn.head(q)(perm),
        off[1],
        off[2],
        off[3],
      };
    }
  }  // namespace

  std::optional<SaPrep> prepare(const Matrix3Xd &pts, const ArrayXd &sar,
                                const ArrayXb &active, double rp) {
    if (pts.cols() == 0)
      return SaPrep {};

    if (rp <= 0) {
      ABSL_LOG(ERROR) << "Probe radius must be positive";
      return std::nullopt;
    }

    const double rmin = sar.minCoeff(), rmax = sar.maxCoeff();
    if (rmin * rmin < 2.0 * rp * rp + 2 * rmax * kSurfaceLengthEps) {
      ABSL_LOG(ERROR) << "Atom radii must be at least (sqrt 2 - 1) rp";
      return std::nullopt;
    }

    ArrayXd sar2 = sar.square();
    auto [g0, dover, inear, jnear, keep] = find_near_pairs(pts, sar, rmax);
    drop_shared_circle_middles(keep, g0, dover, pts, sar2);

    const double wmax = keep.cast<bool>().select(sar2, 0.0).maxCoeff();
    ArrayXd t(sar.size());
    for (int i = 0; i < t.size(); ++i)
      t[i] = std::sqrt(nuri::max(wmax - sar2[i], 0.0));
    auto [g, d] = exact_overlaps(pts, t, wmax, keep, inear, jnear);

    auto [order, off] = rank_atoms(keep, inear, jnear, active);
    return compact(pts, sar, t, wmax, g, d, std::move(order), off, keep);
  }
}  // namespace internal
}  // namespace nuri
