//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <utility>

#include <absl/log/absl_check.h>
#include <Eigen/Dense>
#include <geogram/basic/numeric.h>
#include <geogram/delaunay/delaunay_3d.h>

#include "nuri/eigen_config.h"
#include "nuri/desc/surface.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  namespace {
    constexpr int kBounds = 4;

    /**
     * Lift the spheres as `(x, y, z, sqrt(W - w))`, `w = sar^2`, `W = max w`,
     * then append a weightless tetrahedron around them so the input is never
     * coplanar; it has zero power at every sphere point and hides nothing.
     */
    Matrix4Xd lift(const SaPrep &sa, const ArrayXi &perm) {
      const int n = static_cast<int>(perm.size());

      Matrix4Xd lifted(4, n + kBounds);
      lifted.topRows(3).leftCols(n) = sa.pts(E::all, perm);
      ArrayXd w = sa.sar(perm).square();
      const double wmax = w.maxCoeff();
      lifted.row(3).head(n) = (wmax - w).sqrt().transpose();

      const Vector3d lo = sa.pts.rowwise().minCoeff(),
                     hi = sa.pts.rowwise().maxCoeff();
      const Vector3d center = 0.5 * (lo + hi);
      const double reach = 4 * (0.5 * (hi - lo).norm() + sa.sar.maxCoeff()) + 1;
      const Matrix<double, 3, kBounds> corners {
        { 1,  1, -1, -1 },
        { 1, -1,  1, -1 },
        { 1, -1, -1,  1 },
      };
      lifted.topRows(3).rightCols(kBounds) =
          (reach / std::sqrt(3.0) * corners).colwise() + center;
      lifted.row(3).tail(kBounds).setConstant(std::sqrt(wmax));
      return lifted;
    }

    CSR edges_of(const Array4Xi &tets, const int n) {
      const int nf = static_cast<int>(tets.cols());

      ArrayXi key(12L * nf), val(12L * nf);
      int m = 0;
      for (int c = 0; c < nf; ++c) {
        for (int a = 0; a < 4; ++a) {
          if (tets(a, c) >= n)
            continue;

          for (int b = 0; b < 4; ++b) {
            if (a == b)
              continue;

            key[m] = tets(a, c);
            val[m] = tets(b, c);
            ++m;
          }
        }
      }

      ArrayXi order(m);
      OffsetTable off(n);
      argsort_bucket(order, off, key.head(m));

      ArrayXi adj(m);
      int w = 0;
      for (int v = 0; v < n; ++v) {
        const int beg = off[v], end = off[v + 1];
        off.off()[v] = w;
        auto first = adj.begin() + w;
        for (int e = beg; e < end; ++e)
          adj[w++] = val[order[e]];
        std::sort(first, adj.begin() + w);
        w = static_cast<int>(std::unique(first, adj.begin() + w) - adj.begin());
      }
      off.off()[n] = w;
      adj.conservativeResize(w);
      return { std::move(adj), std::move(off) };
    }
  }  // namespace

  SasDelaunay triangulate(const SaPrep &sa) {
    const int n = sa.g.n();
    ABSL_DCHECK_GT(n, 0);

    ArrayXi perm(n);
    OffsetTable orig(sa.order.maxCoeff() + 1);
    argsort_bucket(perm, orig, sa.order);

    SasDelaunay del;
    del.vertex = ArrayXi::Constant(orig.size(), -1);
    for (int t = 0; t < n; ++t)
      del.vertex[sa.order[perm[t]]] = t;

    const Matrix4Xd lifted = lift(sa, perm);
    GEO::Delaunay3d tri(4);
    tri.set_keeps_infinite(true);
    tri.set_vertices(n + kBounds, lifted.data());

    const int nf = static_cast<int>(tri.nb_finite_cells());
    del.tets.resize(4, nf);
    del.adj.resize(4, nf);
    for (int c = 0; c < nf; ++c) {
      for (int lv = 0; lv < 4; ++lv) {
        const GEO::index_t v = tri.cell_vertex(c, lv),
                           c2 = tri.cell_adjacent(c, lv);
        ABSL_DCHECK_NE(v, GEO::NO_INDEX);
        del.tets(lv, c) = static_cast<int>(v);
        del.adj(lv, c) = c2 < tri.nb_finite_cells() ? static_cast<int>(c2) : -1;
      }
    }

    del.nbrs = edges_of(del.tets, n);
    return del;
  }
}  // namespace internal
}  // namespace nuri
