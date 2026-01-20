//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/random.h"

#include <cmath>
#include <random>

#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"

namespace nuri {
namespace internal {
  // NOLINTNEXTLINE(*-global-variables)
  thread_local std::mt19937 rng {};

  namespace {
    std::seed_seq make_seed_seq(int seed) {
      if (seed >= 0)
        return std::seed_seq({ seed });

      std::random_device rd;
      return std::seed_seq({ rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd() });
    }
  }  // namespace

  void seed_thread(int seed) {
    std::seed_seq seq = make_seed_seq(seed);
    rng.seed(seq);
  }

  Vector3d random_unit() {
    double u = draw_urd(-1.0, 1.0), v = std::sqrt(1.0 - u * u),
           t = draw_urd(0.0, constants::kTwoPi);
    return { v * std::cos(t), v * std::sin(t), u };
  }

  constexpr double kChebyshevCoeffs[] = {
    2.8483461441100055,     2.2478945259472033e-01, 4.7788709002222464e-02,
    1.3504477076024014e-02, 5.2875253824057758e-03, 1.8026786252751091e-03,
  };

  AngleAxisd random_rotation(double max_angle) {
    /**
     * Inverse CDF for uniform angle-axis sampling.
     *
     * Given that y = (x - sin x) / pi (true CDF for angle x in [0, pi]), we
     * want to solve for x in terms of y. We define intermediate variables:
     *
     *   t(y) = cbrt(y)
     *   z(t) = 2 t^2 - 1
     *
     * Now we define an implicit function r(z) such that x = t * r(z). r(z) can
     * be approximated by few terms of Chebyshev polynomials with modest error.
     *
     * The coefficients are determined by fitting r(z) with Chebyshev
     * polynomials up to degree 5 over x in [0, pi], then folded for Horner's
     * method evaluation.
     *
     * Note that y can be interpreted as a "signed" CDF value in [-max_angle/pi,
     * max_angle/pi] for angle in [-max_angle, max_angle], so we draw from
     * [-1, 1) and scale accordingly.
     *
     * CDF was taken from:
     *   H Rummler, The Mathematical Intelligencer 24, 2002, 6-11.
     *   (DOI: https://doi.org/10.1007/BF03025318)
     */
    auto icdf = [max_angle](double p) {
      double t =
          std::cbrt(p * ((max_angle - std::sin(max_angle)) / constants::kPi));
      double z = 2 * t * t - 1;

      double r = z * kChebyshevCoeffs[5];
      r = z * (kChebyshevCoeffs[4] + r);
      r = z * (kChebyshevCoeffs[3] + r);
      r = z * (kChebyshevCoeffs[2] + r);
      r = z * (kChebyshevCoeffs[1] + r);
      r = kChebyshevCoeffs[0] + r;

      return t * r;
    };

    Vector3d axis = random_unit();
    double angle = icdf(draw_urd(-1.0, 1.0));
    return AngleAxisd(angle, axis);
  }
}  // namespace internal
}  // namespace nuri
