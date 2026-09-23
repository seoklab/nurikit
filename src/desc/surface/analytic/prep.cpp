//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <absl/types/span.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/desc/surface.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  namespace {
    template <class Offset, class Key, class Map>
    // NOLINTNEXTLINE(*-missing-std-forward)
    void argsort_bucket(ArrayXi &idxs, Offset &&off, const Key &key,
                        const Map &map) {
      ABSL_DCHECK_GE(idxs.size(), key.size());

      const int m = static_cast<int>(key.size());

      off.setZero();
      for (int k = 0; k < m; ++k)
        ++off[key[k]];
      std::inclusive_scan(off.begin(), off.end(), off.begin());

      for (int p = m - 1; p >= 0; --p) {
        int k = map(p);
        idxs[--off[key[k]]] = k;
      }
    }

    template <class Offset, class Key>
    void argsort_bucket(ArrayXi &idxs, Offset &&off, const Key &key) {
      argsort_bucket(idxs, std::forward<Offset>(off), key,
                     [](int p) { return p; });
    }

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

      ArrayXi order(m), adj(m), off(n + 1);
      argsort_bucket(adj, off.head(n), j);
      argsort_bucket(order, off, i, adj);
      adj = j(order);

      return { CSR(std::move(adj), std::move(off)), std::move(order) };
    }

    NearPairs find_near_pairs(const Matrix3Xd &pts, const ArrayXd &sar,
                              const double rmax) {
      const int n = static_cast<int>(sar.size());

      VoxelGrid grid(pts, 2 * (rmax + kSurfaceLengthEps));
      std::vector<int> left, right;
      grid.find_neighbors_self(left, right);
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
                             + (d > touch - kSurfaceLengthEps).cast<int>());
      ArrayXi order(m);
      Array3i off;
      argsort_bucket(order, off, key);

      auto near = order.head(off[2]);
      ArrayXi inear = eigen_map(left)(near), jnear = eigen_map(right)(near);

      m = off[1];
      auto [g, perm] = compile_pairs(inear.head(m), jnear.head(m), n);

      return {
        std::move(g),     d(order).head(m)(perm),

        std::move(inear), std::move(jnear),

        std::move(keep),
      };
    }

    template <class F, class B>
    void for_each_triangle(const CSR &g, const F &f, const B &b) {
      for (int i = 0; i < g.n(); ++i) {
        if (g.degree(i) < 2)
          continue;

        b(i);
        const auto ei = g.end(i);
        for (auto pij = g.begin(i); pij < ei; ++pij) {
          const int j = *pij;
          const auto ej = g.end(j);
          for (auto pik = pij + 1, pjk = g.begin(j); pik < ei && pjk < ej;) {
            const int ki = *pik, kj = *pjk;
            if (ki == kj)
              f(i, j, ki, pij, pik, pjk);
            pik += value_if(ki <= kj);
            pjk += value_if(kj <= ki);
          }
        }
      }
    }

    void drop_shared_circle_middles(ArrayXi &keep, const CSR &g,
                                    const ArrayXd &d, const Matrix3Xd &pts,
                                    const ArrayXd &sar2) {
      constexpr double cutoff = kSurfaceLengthEps * kSurfaceLengthEps;

      Matrix3Xd axis(3, g.max_deg()), cntr(3, g.max_deg());

      for_each_triangle(
          g,
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

    SaPrep compact(const Matrix3Xd &pts, const ArrayXd &sar, const CSR &g,
                   const ArrayXd &d, ArrayXi &&order, const Array5i &off,
                   ArrayXi &inv) {
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
        pts(E::all, order), sar(order), std::move(order), std::move(gn),
        dn.head(q)(perm),   off[1],     off[2],           off[3],
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
    auto [g, dover, inear, jnear, keep] = find_near_pairs(pts, sar, rmax);
    drop_shared_circle_middles(keep, g, dover, pts, sar2);
    auto [order, off] = rank_atoms(keep, inear, jnear, active);
    return compact(pts, sar, g, dover, std::move(order), off, keep);
  }
}  // namespace internal
}  // namespace nuri
