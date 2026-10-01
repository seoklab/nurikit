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
    /**
     * Candidate pairs within the near band over every atom, with the exactly
     * contained ones flagged (identical twins included), and the atoms that
     * survive the contained-ball filter.
     */
    struct NearPairs {
      /* n_near */
      ArrayXi inear, jnear;
      ArrayXb contained;

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

    NearPairs find_near_pairs(const SasExact &ex, const ArrayXd &sar,
                              const ArrayXb &active, const double rmax) {
      const Matrix3Xd &pts = ex.centers();
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
      ArrayXb contained = ArrayXb::Constant(m, false);
      for (int k = 0; k < m; ++k) {
        const int i = left[k], j = right[k];
        if (d[k] > std::abs(sar[i] - sar[j]) + kSurfaceLengthEps)
          continue;

        const int in = ex.contained(i, j);
        if (in < 0)
          continue;
        const bool drop_i = in == 0 || (in == 2 && !active[i] && active[j]);
        keep[drop_i ? i : j] = 0;
        contained[k] = true;
      }

      ArrayXd touch = sar(left) + sar(right);
      ArrayXi key = (d > touch + 2 * kSurfaceLengthEps).cast<int>();
      ArrayXi order(m);
      E::Array2i off;
      argsort_bucket(order, off, key);

      auto near = order.head(off[1]);
      return { left(near), right(near), contained(near), std::move(keep) };
    }

    /**
     * Drop the middle sphere of every triple sharing one circle: on each
     * side of the circle plane the outer sphere bulges more, so it has no
     * surface, whether or not an outer sphere is itself contained in a
     * larger one. The band on floating circle centres and axes only selects
     * candidates; `shared_circle` decides.
     */
    void drop_shared_circle_middles(ArrayXi &keep, const CSR &g,
                                    const ArrayXd &d, const ArrayXd &sar2,
                                    const SasExact &ex) {
      constexpr double cutoff = kSurfaceLengthEps * kSurfaceLengthEps;

      const Matrix3Xd &pts = ex.centers();
      Matrix3Xd axis(3, g.max_deg()), cntr(3, g.max_deg());

      g.for_each_triangle(
          g.n(),
          [&](int i, int j, int k, auto pij, auto pik, auto) {
            E::Index ij = pij - g.begin(i), ik = pik - g.begin(i);
            Vector3d uij = axis.col(ij), uik = axis.col(ik);
            if ((cntr.col(ij) - cntr.col(ik)).squaredNorm() >= cutoff
                || sar2[i] * uij.cross(uik).squaredNorm() >= cutoff)
              return;

            const int mid = ex.shared_circle({ i, j, k });
            if (mid >= 0)
              keep[mid == 0 ? i : mid == 1 ? j : k] = 0;
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
     * Candidate pairs over every atom, minus the exactly contained ones, that
     * overlap exactly for the heights the lift will hand to the
     * triangulation; an exact tangency does not. Edges of dropped atoms are
     * removed by `compact`.
     */
    std::pair<CSR, ArrayXd> exact_overlaps(const SasExact &ex,
                                           const NearPairs &np) {
      const Matrix3Xd &pts = ex.centers();
      const int m = static_cast<int>(np.inear.size());
      ArrayXi li(m), lj(m);
      int q = 0;
      for (int k = 0; k < m; ++k) {
        const int i = np.inear[k], j = np.jnear[k];
        if (np.contained[k] || ex.overlap(i, j) != Sgn::kPos)
          continue;

        li[q] = i;
        lj[q] = j;
        ++q;
      }

      auto [g, perm] = compile_pairs(li.head(q), lj.head(q), ex.n());
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
    if (!pts.allFinite() || !sar.allFinite()) {
      ABSL_LOG(ERROR) << "Coordinates and radii must be finite";
      return std::nullopt;
    }
    if (pts.cwiseAbs().maxCoeff() > kSurfaceMaxCoord) {
      ABSL_LOG(ERROR) << "Coordinates must lie within " << kSurfaceMaxCoord
                      << " of the origin";
      return std::nullopt;
    }

    const double rmin = sar.minCoeff(), rmax = sar.maxCoeff();
    if (rmin * rmin < 2.0 * rp * rp + 2 * rmax * kSurfaceLengthEps) {
      ABSL_LOG(ERROR) << "Atom radii must be at least (sqrt 2 - 1) rp";
      return std::nullopt;
    }

    const ArrayXd sar2 = sar.square();
    const double wmax = rmax * rmax;
    const ArrayXd t = (wmax - sar2).sqrt();
    Matrix4Xd lifted(4, sar.size());
    lifted.topRows(3) = pts;
    lifted.row(3) = t.transpose();
    const SasExact ex = SasExact::make(lifted, wmax);

    NearPairs np = find_near_pairs(ex, sar, active, rmax);
    auto [g, d] = exact_overlaps(ex, np);
    drop_shared_circle_middles(np.keep, g, d, sar2, ex);

    auto [order, off] = rank_atoms(np.keep, np.inear, np.jnear, active);
    return compact(pts, sar, t, wmax, g, d, std::move(order), off, np.keep);
  }
}  // namespace internal
}  // namespace nuri
