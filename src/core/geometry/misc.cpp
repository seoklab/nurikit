//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"

namespace nuri {
Matrix3Xd canonical_fibonacci_lattice(int npts) {
  // 2pi / phi^2
  constexpr double dtheta =
      constants::kTwoPi
      * 0.38196601125010515179541316563436188227969082019423713786455137729473953718;

  const double dz = 2.0 / npts, z0 = 1.0 - 0.5 * dz;

  Matrix3Xd pts(3, npts);
  for (int i = 0; i < npts; ++i) {
    const double theta = i * dtheta;
    const double z = z0 - i * dz;
    const double r = std::sqrt(1 - z * z);

    pts(0, i) = r * std::cos(theta);
    pts(1, i) = r * std::sin(theta);
    pts(2, i) = z;
  }
  return pts;
}
}  // namespace nuri
