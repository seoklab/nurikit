//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"

namespace nuri {
Matrix3Xd canonical_fibonacci_lattice(int npts) {
  constexpr double theta_step =
      constants::kTwoPi
      / 1.61803398874989484820458683436563811772030917980576286213544862270526;

  Matrix3Xd pts(3, npts);
  const double cosphi_step = 2.0 / npts;

  for (int i = 0; i < npts; ++i) {
    const double theta = i * theta_step;

    const double z = pts(2, i) = 1.0 - (i + 0.5) * cosphi_step;
    const double r = std::sqrt(1 - z * z);

    pts(0, i) = r * std::cos(theta);
    pts(1, i) = r * std::sin(theta);
  }

  return pts;
}
}  // namespace nuri
