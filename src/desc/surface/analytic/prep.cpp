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

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/desc/surface.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  namespace {
    class CSR {
    public:
      CSR(ArrayXi &&adj, ArrayXi &&off) noexcept
          : adj_(std::move(adj)), off_(std::move(off)) { }

    private:
      ArrayXi adj_;
      ArrayXi off_;
    };

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

    struct OverlapGraph {
      CSR g;
      ArrayXd d;
      ArrayXi drop;
    };

    OverlapGraph build_overlap_graph(const Matrix3Xd &pts, const ArrayXd &sar,
                                     const double rmax) {
      const int n = static_cast<int>(sar.size());

      VoxelGrid grid(pts, 2 * rmax);
      std::vector<int> left, right;
      grid.find_neighbors_self(left, right);
      int m = static_cast<int>(left.size());
      for (int k = 0; k < m; ++k)
        std::tie(left[k], right[k]) = nuri::minmax(left[k], right[k]);

      ArrayXd d = (pts(E::all, right) - pts(E::all, left)).colwise().norm();
      ArrayXi order(m), off(n + 1);

      // Will be converted in-place to an index mask, hence integer
      ArrayXi drop = ArrayXi::Zero(sar.size());
      for (int k = 0; k < m; ++k) {
        int i = left[k], j = right[k];
        if (d[k] <= std::abs(sar[i] - sar[j]) + kSurfaceLengthEps)
          drop[sar[i] < sar[j] ? i : j] = 1;
      }

      ArrayXb drop_pair = drop(left) + drop(right) > 0;
      ArrayXi key = drop_pair.select(
          1, (d > sar(left) + sar(right) - kSurfaceLengthEps).cast<int>());
      argsort_bucket(order, off.head<2>(), key);

      auto keep = order.head(off[1]);
      ArrayXi iover = eigen_map(left)(keep), jover = eigen_map(right)(keep);
      ArrayXd dover = d(keep);
      m = static_cast<int>(keep.size());

      ArrayXi adj(m);
      argsort_bucket(adj, off.head(n), jover);
      argsort_bucket(order, off, iover, adj);
      adj = jover(order.head(m));

      return { CSR(std::move(adj), std::move(off)), dover(order.head(m)),
               std::move(drop) };
    }

    struct SaPrep { };

    std::optional<SaPrep> prepare(const Matrix3Xd &pts, const ArrayXd &sar,
                                  double rp) {
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

      OverlapGraph og = build_overlap_graph(pts, sar, rmax);

      return SaPrep {};
    }
  }  // namespace
}  // namespace internal
}  // namespace nuri
