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

  AngleAxisd random_rotation(double max_angle) {
    auto icdf = [max_angle](double p) {
      double q = 2 * p - 1;
      return std::copysign(std::pow(std::abs(q), 1.0 / 3.0), q) * max_angle;
    };

    Vector3d axis = random_unit();
    double angle = icdf(draw_urd(1.0));
    return AngleAxisd(angle, axis);
  }
}  // namespace internal
}  // namespace nuri
