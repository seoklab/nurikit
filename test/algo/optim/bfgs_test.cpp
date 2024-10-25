//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <Eigen/Dense>

#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "test_utils.h"
#include "nuri/algo/optim.h"

namespace nuri {
namespace {
TEST(BFGSTest, DistanceMaximization) {
  // Distance maximization problem (e.g. forcefield)
  auto fg = [](ArrayXd &gx, ConstRef<ArrayXd> xa) -> double {
    ConstRef<Matrix3Xd> xm = xa.reshaped(3, 20).matrix();
    MutRef<Matrix3Xd> gm = gx.reshaped(3, 20).matrix();

    double fx = 0;
    gm.setZero();

    // Pairwise repulsion
    Vector3d d;
    for (int i = 1; i < 20; ++i) {
      for (int j = 0; j < i; ++j) {
        d = xm.col(i) - xm.col(j);
        double pd_inv = 1 / std::max(d.norm(), 1e-12);
        fx += pd_inv;

        d *= pd_inv * pd_inv * pd_inv;
        gm.col(i) -= d;
        gm.col(j) += d;
      }
    }

    // L2 regularization
    fx += xa.square().sum();
    gx += 2 * xa;

    return fx;
  };

  MatrixXd x {
    {  0.680375,  -0.211234,  0.566198 },
    {   0.59688,   0.823295, -0.604897 },
    { -0.329554,   0.536459, -0.444451 },
    {   0.10794, -0.0452059,  0.257742 },
    { -0.270431,  0.0268018,  0.904459 },
    {   0.83239,   0.271423,  0.434594 },
    { -0.716795,   0.213938, -0.967399 },
    { -0.514226,  -0.725537,  0.608354 },
    { -0.686642,  -0.198111, -0.740419 },
    { -0.782382,   0.997849, -0.563486 },
    { 0.0258648,   0.678224,   0.22528 },
    { -0.407937,   0.275105, 0.0485744 },
    { -0.012834,    0.94555, -0.414966 },
    {  0.542715,    0.05349,  0.539828 },
    { -0.199543,   0.783059, -0.433371 },
    { -0.295083,   0.615449,  0.838053 },
    { -0.860489,   0.898654, 0.0519907 },
    { -0.827888,  -0.615572,  0.326454 },
    {  0.780465,  -0.302214, -0.871657 },
    { -0.959954, -0.0845965, -0.873808 },
  };
  x.transposeInPlace();
  MutRef<ArrayXd> xa = x.reshaped().array();

  auto res = bfgs(fg, xa);
  ASSERT_EQ(res.code, BfgsResultCode::kSuccess);

  // results from scipy.optimize

  MatrixX3d xans {
    {  0.38629347, -0.18648296,  0.28456376 },
    {  0.65357409,  0.11364984, -1.44834129 },
    { -0.42669145,  0.20885128, -0.30625167 },
    {  1.35656259, -0.77176589,  0.67136267 },
    { -0.25086413, -0.89222778,  1.32905227 },
    {   1.5794446,  0.14788839, -0.36549864 },
    { -0.62668712, -0.08360516,  -1.5425326 },
    {  0.28953597, -1.58198176,  0.25232092 },
    { -0.33700667, -1.21478632, -0.97390024 },
    { -0.51182782,  1.21694618, -1.03062277 },
    {   1.1256642,  0.95374507,  0.66976211 },
    { -1.28650381, -0.13874978,  0.94152021 },
    { -0.01593536,  1.56514155,  0.33300894 },
    {  0.62124971,  0.08644108,  1.53914738 },
    {  0.79713537,  1.20073182, -0.68549516 },
    { -0.43040195,  0.78929956,    1.311418 },
    { -1.29390252,  1.02712981,  0.17443237 },
    {  -1.0099036, -1.22261215,  0.17996632 },
    {  0.95804114, -1.00504579, -0.81383169 },
    { -1.57777466, -0.21257005, -0.52008019 },
  };
  double fxans = 145.02890212025355;

  EXPECT_EQ(res.niter, 92);
  EXPECT_NEAR(res.fx, fxans, 1e-6);
  NURI_EXPECT_EIGEN_EQ_TOL(x, xans.transpose(), 1e-6);
}
}  // namespace
}  // namespace nuri
