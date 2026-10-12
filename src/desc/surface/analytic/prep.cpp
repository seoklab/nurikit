//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/desc/surface.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  namespace {
    constexpr int kBandHops = 4;

    /**
     * Candidate pairs within the near band over every atom, the pairs left
     * for `overlap` to decide (both ends within `kBandHops` near hops of an
     * active atom, minus the exactly contained ones, identical twins
     * included), of which the ones below `touch − 2ε` overlap without the
     * predicate (the lifted radius drifts from `sar` by far less, the
     * superset bound of `kSurfaceMaxCoord`), and the atoms that survive the
     * contained-ball filter. Every kept atom lies within three hops, its
     * containing partners and shared-circle triangles within four, so the
     * band changes nothing the ranking keeps.
     */
    struct NearPairs {
      /* n_near */
      ArrayXi inear, jnear;
      ArrayXb band, sure;

      /* n */
      ArrayXi keep;
    };

    bool sweep_hops(ArrayXi &hop, const ArrayXi &inear, const ArrayXi &jnear) {
      const int64_t before = hop.cast<int64_t>().sum();
      for (int k = 0; k < inear.size(); ++k) {
        const int i = inear[k], j = jnear[k];
        const int h = nuri::min(hop[i], hop[j]) + 1;
        hop[i] = nuri::min(hop[i], h);
        hop[j] = nuri::min(hop[j], h);
      }
      return hop.cast<int64_t>().sum() < before;
    }

    ArrayXb active_band(const ArrayXi &inear, const ArrayXi &jnear,
                        const ArrayXb &active) {
      const int n = static_cast<int>(active.size());
      ArrayXi hop = active.select(0, ArrayXi::Constant(n, kBandHops + 1));
      for (int sweep = 0; sweep < kBandHops && !(hop <= kBandHops).all();
           ++sweep) {
        if (!sweep_hops(hop, inear, jnear))
          break;
      }
      return hop <= kBandHops;
    }

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

    /**
     * Atoms whose block of coarse cells holds an active atom, a superset of
     * the atoms within `radius` of one by the grid's binning.
     */
    ArrayXb within_reach(const Matrix3Xd &pts, const ArrayXb &active,
                         const double radius) {
      const VoxelGrid grid(pts, radius);
      const Array3i &dims = grid.dims();
      const ArrayXi &off = grid.cell_offset(), &members = grid.cell_pts();
      const int nc = grid.num_cells();

      ArrayXb marked(nc);
      for (int v = 0; v < nc; ++v)
        marked[v] = active(members.segment(off[v], off[v + 1] - off[v])).any();

      ArrayXb reach = ArrayXb::Constant(pts.cols(), false);
      for (int z = 0, v = 0; z < dims.z(); ++z) {
        for (int y = 0; y < dims.y(); ++y) {
          for (int x = 0; x < dims.x(); ++x, ++v) {
            if (off[v] == off[v + 1])
              continue;

            bool near = false;
            for (int bz = nuri::max(z - 1, 0);
                 bz <= nuri::min(z + 1, dims.z() - 1); ++bz) {
              for (int by = nuri::max(y - 1, 0);
                   by <= nuri::min(y + 1, dims.y() - 1); ++by) {
                for (int bx = nuri::max(x - 1, 0);
                     bx <= nuri::min(x + 1, dims.x() - 1); ++bx)
                  near |= marked[bx + dims.x() * (by + dims.y() * bz)];
              }
            }
            if (near)
              reach(members.segment(off[v], off[v + 1] - off[v])) = true;
          }
        }
      }
      return reach;
    }

    NearPairs find_near_pairs(const BallExact &ex, const ArrayXd &sar,
                              const ArrayXb &active, const double rmax) {
      const Matrix3Xd &pts = ex.centers();
      const int n = static_cast<int>(sar.size());

      VoxelGrid grid(pts, near_cutoff(rmax));
      std::vector<int> lbuf, rbuf;
      grid.find_neighbors_self(lbuf, rbuf);
      auto left = eigen_map(lbuf), right = eigen_map(rbuf);
      const int m = static_cast<int>(left.size());
      for (int k = 0; k < m; ++k)
        std::tie(left[k], right[k]) = nuri::minmax(left[k], right[k]);

      const ArrayXd dist =
          (pts(E::all, right) - pts(E::all, left)).colwise().norm();
      const ArrayXd touch = sar(left) + sar(right);
      const ArrayXi key = (dist > touch + kPairLengthEps).cast<int>();
      ArrayXi order(m);
      E::Array2i off;
      argsort_bucket(order, off, key);
      const auto near = order.head(off[1]);

      NearPairs np;
      np.inear = left(near);
      np.jnear = right(near);
      np.keep = ArrayXi::Ones(n);

      const ArrayXd d = dist(near);
      np.sure = d < touch(near) - kPairLengthEps;
      const ArrayXb banded = active_band(np.inear, np.jnear, active);
      np.band = banded(np.inear) && banded(np.jnear);
      for (int k = 0; k < np.inear.size(); ++k) {
        if (!np.band[k])
          continue;

        const int i = np.inear[k], j = np.jnear[k];
        if (d[k] > std::abs(sar[i] - sar[j]) + kSurfaceLengthEps)
          continue;

        const int in = ex.contained(i, j);
        if (in < 0)
          continue;
        const bool drop_i = in == 0 || (in == 2 && !active[i] && active[j]);
        np.keep[drop_i ? i : j] = 0;
        np.band[k] = false;
      }
      return np;
    }

    bool has_edge(const CSR &g, const int j, const int k) {
      const auto end = g.end(j), it = std::lower_bound(g.begin(j), end, k);
      return it != end && *it == k;
    }

    /**
     * Drop the middle sphere of every triple sharing one circle: on each
     * side of the circle plane the outer sphere bulges more, so it has no
     * surface, whether or not an outer sphere is itself contained in a
     * larger one. Two circles of sphere `i` coincide only if their unsigned
     * plane offsets `|a|` agree within the row band and their centres are
     * collinear with `c_i`, so each row sorted by `|a|` is scanned by a
     * sliding window; `shared_circle` decides.
     */
    void drop_shared_circle_middles(ArrayXi &keep, const CSR &g,
                                    const ArrayXd &d, const ArrayXd &sar2,
                                    const BallExact &ex, const double rmax,
                                    const double wmax) {
      constexpr double cutoff = 2 * kSurfaceLengthEps * kSurfaceLengthEps,
                       c2 = 3 * kSurfaceMaxCoord * kSurfaceMaxCoord;
      const double knum = 1.1 * kUnitRoundoff * (4.2 * c2 + 8.3 * wmax)
                          + 16 * kUnitRoundoff * rmax * rmax,
                   tail = 50 * kUnitRoundoff * rmax;

      const Matrix3Xd &pts = ex.centers();
      const ArrayXi &adj = g.adj();
      ArrayXd key(g.max_deg());
      std::vector<int> idx;
      for (int i = 0; i < g.n(); ++i) {
        const int deg = g.degree(i);
        if (deg < 2)
          continue;

        const int off = g.offset(i);
        const auto de = d.segment(off, deg);
        key.head(deg) =
            ((de.square() + sar2[i] - sar2(g.nbrs(i))) / (2 * de)).abs();
        const double band = 2 * (knum / de.minCoeff() + tail);

        idx.resize(deg);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int p, int q) { return key[p] < key[q]; });

        const Vector3d ci = pts.col(i);
        for (int p = 0; p < deg; ++p) {
          for (int q = p + 1; q < deg && key[idx[q]] - key[idx[p]] <= band;
               ++q) {
            auto [ej, ek] =
                nuri::minmax(off + idx[p], off + idx[q],
                             [&](int a, int b) { return adj[a] < adj[b]; });
            const int j = adj[ej], k = adj[ek];
            const Vector3d dij = pts.col(j) - ci, dik = pts.col(k) - ci;
            const double gij = d[ej], gik = d[ek];
            if (sar2[i] * dij.cross(dik).squaredNorm()
                >= cutoff * (gij * gij) * (gik * gik))
              continue;
            if (!has_edge(g, j, k))
              continue;

            const int mid = ex.shared_circle({ i, j, k });
            if (mid >= 0)
              keep[mid == 0 ? i : mid == 1 ? j : k] = 0;
          }
        }
      }
    }

    using Array5i = E::Array<int, 5, 1>;

    /**
     * Remap to ranking scheme: 0 -> active, 1 -> need, 2 -> shell, 3 ->
     * occluders near a sphere with caps, 4 -> ignored (contained, or an
     * occluder near no sphere with caps; Lemma (occluder crop) in
     * ALGORITHMS.md).
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

      ArrayXb near_caps = ArrayXb::Constant(rank.size(), false);
      for (int k = 0; k < inear.size(); ++k) {
        int i = inear[k], j = jnear[k];
        near_caps[j] = near_caps[j] || rank[i] <= 2;
        near_caps[i] = near_caps[i] || rank[j] <= 2;
      }
      rank = (rank == 3 && !near_caps).select(4, rank);

      ArrayXi order(rank.size());
      Array5i off;
      argsort_bucket(order, off, rank);
      return { std::move(order), off };
    }

    /**
     * Candidate pairs within the band, minus the exactly contained ones, that
     * overlap exactly for the heights the lift will hand to the
     * triangulation; an exact tangency does not. Rows outside the band stay
     * empty; edges of dropped atoms are removed by `compact`.
     */
    std::pair<CSR, ArrayXd> exact_overlaps(const BallExact &ex,
                                           const NearPairs &np) {
      const Matrix3Xd &pts = ex.centers();
      const int m = static_cast<int>(np.inear.size());
      ArrayXi li(m), lj(m);
      int q = 0;
      for (int k = 0; k < m; ++k) {
        const int i = np.inear[k], j = np.jnear[k];
        if (!np.band[k] || (!np.sure[k] && ex.overlap(i, j) != Sgn::kPos))
          continue;
        ABSL_DCHECK(ex.overlap(i, j) == Sgn::kPos);

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

    const ArrayXb reach =
        within_reach(pts, active, (kBandHops + 1) * near_cutoff(rmax));
    ArrayXi near(reach.count());
    for (int i = 0, w = 0; i < reach.size(); ++i) {
      if (reach[i])
        near[w++] = i;
    }
    const Matrix3Xd npts = pts(E::all, near);
    const ArrayXd nsar = sar(near);
    const ArrayXb nactive = active(near);

    const ArrayXd sar2 = nsar.square();
    const double wmax = rmax * rmax;
    const ArrayXd t = (wmax - sar2).sqrt();
    Matrix4Xd lifted(4, nsar.size());
    lifted.topRows(3) = npts;
    lifted.row(3) = t.transpose();
    const BallExact ex = BallExact::make(lifted, wmax);

    NearPairs np = find_near_pairs(ex, nsar, nactive, rmax);
    auto [g, d] = exact_overlaps(ex, np);
    drop_shared_circle_middles(np.keep, g, d, sar2, ex, rmax, wmax);

    auto [order, off] = rank_atoms(np.keep, np.inear, np.jnear, nactive);
    SaPrep sa =
        compact(npts, nsar, t, wmax, g, d, std::move(order), off, np.keep);
    sa.order = near(sa.order);
    return sa;
  }
}  // namespace internal
}  // namespace nuri
