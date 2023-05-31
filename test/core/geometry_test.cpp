//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/geometry.h"

#include <Eigen/Dense>

#include <absl/log/absl_log.h>
#include <gtest/gtest.h>

#include "nuri/eigen_config.h"

namespace {
TEST(AngleAxisTest, CorrectnessTest) {
  Eigen::AngleAxisd aref(0.1, Eigen::Vector3d(3, 2, 1).normalized());
  Eigen::Vector3d pref(1, 2, 3);

  Eigen::Vector3d prot_ref = aref * pref;

  nuri::AngleAxisd a(aref.angle(), aref.axis());
  nuri::Vector3d p = pref;

  nuri::Vector3d prot = a.to_matrix() * p.transpose();

  for (int i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(prot_ref[i], prot[i]);
  }
}

TEST(AngleAxisTest, RightHandedTest) {
  nuri::Vector3d v(1, 0, 0);
  nuri::AngleAxisd aa(nuri::constants::kPi / 2, nuri::Vector3d(0, 0, 1));

  nuri::Vector3d vrot = aa.to_matrix() * v.transpose();
  EXPECT_TRUE(vrot.isApprox(nuri::Vector3d(0, 1, 0)));
}
}  // namespace
