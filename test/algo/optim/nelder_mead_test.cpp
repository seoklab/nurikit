//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <Eigen/Dense>

#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "test_utils.h"
#include "nuri/algo/optim.h"

namespace nuri {
namespace {
double rosenbrock(ConstRef<ArrayXd> x) {
  const auto nm1 = x.size() - 1;

  double fx = 100 * (x.tail(nm1) - x.head(nm1).square()).square().sum()
              + (1 - x.head(nm1)).square().sum();
  return fx;
}

TEST(NelderMead, Rosenbrock) {
  ArrayXXd data(8, 8);
  data.topRows(7).setConstant(-1);
  data.col(0).head(7).setConstant(0.1);

  auto result = nelder_mead(rosenbrock, data);

  ASSERT_EQ(result.code, OptimResultCode::kSuccess);

  ArrayXd root(8);
  root.head(7) = 1;
  root.tail(1) = 0;
  NURI_EXPECT_EIGEN_EQ_TOL(data.col(result.argmin), root, 1e-3);
}

TEST(NelderMead, InvalidInput) {
  ArrayXXd data;

  auto result = nelder_mead(rosenbrock, data);
  EXPECT_EQ(result.code, OptimResultCode::kInvalidInput);

  data.resize(2, 1);
  result = nelder_mead(rosenbrock, data);
  EXPECT_EQ(result.code, OptimResultCode::kInvalidInput);

  data.resize(2, 2);
  result = nelder_mead(rosenbrock, data, -1, 1e-6, 0);
  EXPECT_EQ(result.code, OptimResultCode::kInvalidInput);

  result = nelder_mead(rosenbrock, data, -1, 1e-6, 1, 0);
  EXPECT_EQ(result.code, OptimResultCode::kInvalidInput);

  result = nelder_mead(rosenbrock, data, -1, 1e-6, 1, 2, 0);
  EXPECT_EQ(result.code, OptimResultCode::kInvalidInput);

  result = nelder_mead(rosenbrock, data, -1, 1e-6, 1, 2, 0.5, 0);
  EXPECT_EQ(result.code, OptimResultCode::kInvalidInput);
}

TEST(NelderMead, PrepareSimplex) {
  ArrayXXd simplex =
      nm_prepare_simplex(Vector3d::UnitX() * -1e-3 + Vector3d::Constant(1e-10));

  ArrayXXd ref(3, 4);
  ref << -1e-3, -1e-3 + 0.05, -1e-3, -1e-3,  //
      1e-10, 1e-10, 0.00025, 1e-10,          //
      1e-10, 1e-10, 1e-10, 0.00025;

  ASSERT_EQ(simplex.rows(), 4);
  ASSERT_EQ(simplex.cols(), 4);

  NURI_EXPECT_EIGEN_EQ(simplex.topRows(3), ref);
}
}  // namespace
}  // namespace nuri
