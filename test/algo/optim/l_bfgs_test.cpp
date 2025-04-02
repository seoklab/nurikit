//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <absl/algorithm/container.h>
#include <absl/base/attributes.h>
#include <absl/base/optimization.h>
#include <absl/log/absl_check.h>
#include <Eigen/Dense>

#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "test_utils.h"
#include "nuri/algo/optim.h"

namespace nuri {
namespace {
TEST(LBFGSTest, LbfgsUnbounded) {
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

  auto [code, iter, fx, gx] = l_bfgs(fg, xa);
  ASSERT_EQ(code, LbfgsResultCode::kSuccess);

  // // Results from lbfgsb original implementation
  EXPECT_EQ(iter, 54);

  MatrixX3d xans {
    {  1.329873, -0.702233,  0.525096 },
    {  0.959901,  0.504039, -1.180017 },
    {  0.000066,  0.000213,  0.000389 },
    {  0.473379, -1.519620, -0.191415 },
    { -0.699817, -0.462164,  1.365490 },
    {  1.532975,  0.410986,  0.077387 },
    { -0.076689, -0.100664, -1.592082 },
    {  0.170076, -1.250362,  0.984563 },
    { -0.365095, -1.160734, -1.038901 },
    { -1.324395,  0.880736, -0.169923 },
    {  0.736751,  1.045224,  0.949381 },
    { -1.523757, -0.162909,  0.449013 },
    {  0.693983,  1.394387, -0.367577 },
    {  0.621756, -0.052286,  1.461228 },
    { -0.434747,  1.039471, -1.115292 },
    { -0.630837,  0.770384,  1.234164 },
    { -0.364141,  1.536944,  0.272608 },
    { -0.942729, -1.267791,  0.131732 },
    {  1.122640, -0.690652, -0.882549 },
    { -1.279187, -0.212978, -0.913293 },
  };
  double fxans = 144.943662;
  MatrixX3d gans {
    {  0.000093,  0.000046, -0.000365 },
    {  0.000654,  0.000675,  0.000634 },
    {  0.000110,  0.001210,  0.001484 },
    {  0.000256, -0.000578,  0.000258 },
    {  0.000145,  0.000925, -0.000211 },
    { -0.000525, -0.000307,  0.000488 },
    { -0.000276,  0.000175, -0.000814 },
    {  0.000221,  0.000032,  0.000199 },
    { -0.000376, -0.000439, -0.000964 },
    {  0.000739,  0.000136,  0.000476 },
    { -0.000958, -0.001127, -0.000077 },
    {  0.000324,  0.000788,  0.000459 },
    {  0.000234, -0.000668, -0.000163 },
    {  0.000023,  0.001554, -0.001037 },
    { -0.000147,  0.000407,  0.000102 },
    {  0.000273, -0.000632, -0.001037 },
    { -0.000611, -0.001219,  0.000600 },
    { -0.000351, -0.000584,  0.000527 },
    {  0.000795, -0.000274, -0.000175 },
    { -0.000610, -0.000138, -0.000381 },
  };

  NURI_EXPECT_EIGEN_EQ(x, xans.transpose());
  EXPECT_NEAR(fx, fxans, 1e-6);
  NURI_EXPECT_EIGEN_EQ(gx.reshaped(3, 20).matrix(), gans.transpose());
}
}  // namespace
}  // namespace nuri
