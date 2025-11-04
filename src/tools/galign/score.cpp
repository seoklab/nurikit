//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>

#include <absl/base/optimization.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"  // IWYU pragma: keep, required for E namespace
#include "nuri/core/geometry.h"
#include "nuri/tools/galign.h"

namespace nuri {
namespace internal {
  float shape_overlap(const GARigidMolInfo &query, const GARigidMolInfo &templ,
                      const E::ArrayXXf &dists, float scale) {
    const Eigen::Index rows = dists.rows(), cols = dists.cols();

    float total_overlap = 0;
    const float *d_it = dists.data();
    for (int i = 0; i < cols; ++i) {
      const float tr = templ.vdw_radii()[i], tv = templ.vdw_vols()[i];
      const int tt = templ.atom_types()[i];

      for (int j = 0; j < rows; ++j, ++d_it) {
        const float qr = query.vdw_radii()[j], d = *d_it;
        const float rsum = qr + tr, rsum_m_d = rsum - d,
                    qv = query.vdw_vols()[j];

        ABSL_ASSUME(qr > 0 && tr > 0 && d >= 0);
        if (rsum_m_d <= 0)
          continue;

        const float rdiff = qr - tr, vols[] = { tv, qv };
        const int r_min = static_cast<int>(qr < tr);

        float overlap;
        if (d <= std::abs(rdiff)) {
          overlap = vols[r_min];
        } else {
          overlap = static_cast<float>(constants::kPi) / 12.0F * rsum_m_d
                    * rsum_m_d * (d + 2 * rsum - 3 / d * rdiff * rdiff);
        }

        if (query.atom_types()[j] != tt)
          overlap *= scale;

        total_overlap += overlap;
      }
    }

    return total_overlap;
  }

  float shape_overlap(const GARigidMolInfo &query, const E::Matrix3Xf &qconf,
                      const GARigidMolInfo &templ, const E::Matrix3Xf &tconf,
                      float scale) {
    float total_overlap = 0;

    for (int i = 0; i < templ.n(); ++i) {
      const float tr = templ.vdw_radii()[i], tv = templ.vdw_vols()[i];
      const int tt = templ.atom_types()[i];
      E::Vector3f tpos = tconf.col(i);

      for (int j = 0; j < query.n(); ++j) {
        const float qr = query.vdw_radii()[j], d = (qconf.col(j) - tpos).norm();
        const float rsum = qr + tr, rsum_m_d = rsum - d;

        ABSL_ASSUME(qr > 0 && tr > 0 && d >= 0);
        if (ABSL_PREDICT_FALSE(rsum_m_d <= 0))
          continue;

        const float rdiff = qr - tr;

        float overlap;
        if (ABSL_PREDICT_TRUE(d > std::abs(rdiff))) {
          overlap = static_cast<float>(constants::kPi) / 12.0F * rsum_m_d
                    * rsum_m_d * (d + 2 * rsum - 3 / d * rdiff * rdiff);
        } else {
          const float vols[] = { tv, query.vdw_vols()[j] };
          const int r_min = static_cast<int>(qr < tr);
          overlap = vols[r_min];
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
