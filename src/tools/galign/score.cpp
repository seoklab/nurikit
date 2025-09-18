//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/tools/galign.h"

namespace nuri {
namespace internal {
  double shape_overlap_impl(const GARigidMolInfo &query,
                            const GARigidMolInfo &templ, const ArrayXXd &dists,
                            double scale) {
    const Eigen::Index rows = dists.rows(), cols = dists.cols();

    double total_overlap = 0;
    const double *d_it = dists.data();
    for (int i = 0; i < cols; ++i) {
      const double tr = templ.vdw_radii()[i], tv = templ.vdw_vols()[i];
      const int tt = templ.atom_types()[i];

      for (int j = 0; j < rows; ++j, ++d_it) {
        const double qr = query.vdw_radii()[j], d = *d_it;
        const double rsum = qr + tr, rsum_m_d = rsum - d,
                     qv = query.vdw_vols()[j];

        ABSL_ASSUME(qr > 0 && tr > 0 && d >= 0);
        if (rsum_m_d <= 0)
          continue;

        const double rdiff = qr - tr, vols[] = { tv, qv };
        const int r_min = static_cast<int>(qr < tr);

        double overlap;
        if (d <= std::abs(rdiff)) {
          overlap = vols[r_min];
        } else {
          overlap = constants::kPi / 12 * rsum_m_d * rsum_m_d
                    * (d + 2 * rsum - 3 / d * rdiff * rdiff);
        }

        if (query.atom_types()[j] != tt)
          overlap *= scale;

        total_overlap += overlap;
      }
    }

    return total_overlap;
  }
}  // namespace internal
}  // namespace nuri
