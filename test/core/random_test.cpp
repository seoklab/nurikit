//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/random.h"

#include <Eigen/Dense>

#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"

namespace nuri {
namespace internal {
namespace {
TEST(RandomTest, AngleStats) {
  seed_thread(42);

  ArrayXd angles(10000);
  for (int i = 0; i < angles.size(); ++i) {
    AngleAxisd aa = random_rotation(constants::kPi);
    angles[i] = aa.angle();
  }

  double avg = angles.mean();
  EXPECT_NEAR(avg, 0, 5e-3);

  double l1norm = angles.abs().mean();
  EXPECT_NEAR(l1norm, constants::kPi / 2 + 2 / constants::kPi, 1e-2);
}
}  // namespace
}  // namespace internal
}  // namespace nuri
