//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <iterator>
#include <utility>
#include <vector>

#include <absl/log/absl_check.h>
#include <Eigen/Dense>
#include <geogram/basic/numeric.h>
#include <geogram/delaunay/delaunay_3d.h>
#include <geogram/numerics/predicates.h>

#include "nuri/eigen_config.h"
#include "nuri/desc/surface.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  namespace {
    constexpr int kBounds = 4;

    /**
     * Lift the spheres as `(x, y, z, t)`, `t = sqrt(W - sar^2)` from `prepare`,
     * then append a weightless tetrahedron around them so the input is never
     * coplanar; it has positive power at every sphere point and hides nothing.
     */
    Matrix4Xd lift(const SaPrep &sa, const ArrayXi &perm) {
      const int n = static_cast<int>(perm.size());
      const double wmax = sa.wmax;

      Matrix4Xd lifted(4, n + kBounds);
      lifted.topRows(3).leftCols(n) = sa.pts(E::all, perm);
      lifted.row(3).head(n) = sa.t(perm).transpose();

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

    struct Edges {
      CSR nbrs;
      ArrayXi cell;
    };

    Edges edges_of(const Array4Xi &tets, const int n) {
      const int nf = static_cast<int>(tets.cols());

      ArrayXi key(12L * nf), val(12L * nf), cell(12L * nf);
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
            cell[m] = c;
            ++m;
          }
        }
      }

      ArrayXi order(m);
      OffsetTable off(n);
      argsort_bucket(order, off, key.head(m));

      ArrayXi adj(m), ecell(m);
      std::vector<std::pair<int, int>> row;
      int w = 0;
      for (int v = 0; v < n; ++v) {
        const int beg = off[v], end = off[v + 1];
        off.off()[v] = w;

        row.clear();
        for (int e = beg; e < end; ++e)
          row.emplace_back(val[order[e]], cell[order[e]]);
        std::sort(row.begin(), row.end());
        for (auto it = row.begin(); it != row.end(); ++it) {
          if (it != row.begin() && it->first == std::prev(it)->first)
            continue;
          adj[w] = it->first;
          ecell[w] = it->second;
          ++w;
        }
      }
      off.off()[n] = w;
      adj.conservativeResize(w);
      ecell.conservativeResize(w);
      return { CSR(std::move(adj), std::move(off)), std::move(ecell) };
    }

    Array4Xi number_faces(const Array4Xi &adj, int &n_faces) {
      const int nf = static_cast<int>(adj.cols());
      Array4Xi face(4, nf);
      n_faces = 0;
      for (int c = 0; c < nf; ++c) {
        for (int lf = 0; lf < 4; ++lf) {
          const int c2 = adj(lf, c);
          if (c2 < 0 || c < c2) {
            face(lf, c) = n_faces++;
            continue;
          }

          int lf2 = 0;
          while (adj(lf2, c2) != c)
            ++lf2;
          face(lf, c) = face(lf2, c2);
        }
      }
      return face;
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

    for (int c = 0; c < nf; ++c) {
      ABSL_DCHECK_EQ(GEO::PCK::orient_3d(lifted.col(del.tets(0, c)).data(),
                                         lifted.col(del.tets(1, c)).data(),
                                         lifted.col(del.tets(2, c)).data(),
                                         lifted.col(del.tets(3, c)).data()),
                     GEO::POSITIVE)
          << "cell " << c << " is not positively oriented";
    }

    Edges edges = edges_of(del.tets, n);
    del.nbrs = std::move(edges.nbrs);
    del.edge_cell = std::move(edges.cell);
    del.face = number_faces(del.adj, del.n_faces);
    del.ex = BallExact::make(lifted, sa.wmax);
    return del;
  }
}  // namespace internal
}  // namespace nuri
