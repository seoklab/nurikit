//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <Eigen/Dense>

#include <absl/algorithm/container.h>
#include <absl/base/attributes.h>
#include <absl/base/optimization.h>
#include <absl/log/absl_check.h>
#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "test_utils.h"
#include "nuri/algo/optim.h"

namespace nuri {
namespace {
using internal::LbfgsbBounds;

using internal::lbfgs_bmv;
using internal::lbfgsb_cauchy;
using internal::lbfgsb_projgr;
using internal::lbfgsb_subsm;

TEST(LBFGSBTest, Prjgr) {
  ArrayXi nbd {
    { 0, 1, 2, 3, 3 }
  };
  Array2Xd bds {
    { 0.0, 0.0, 0.0, 0.0, 0.0 },
    { 1.0, 1.0, 1.0, 1.0, 1.0 }
  };
  LbfgsbBounds bounds { nbd, bds };

  ArrayXXd x {
    {   0.680375, -0.211234,  0.566198,   0.59688,  0.823295 },
    { -0.0452059,  0.257742, -0.270431, 0.0268018,  0.904459 },
    {  -0.967399, -0.514226, -0.725537,  0.608354, -0.686642 },
    {  0.0258648,  0.678224,   0.22528, -0.407937,  0.275105 },
    {    0.05349,  0.539828, -0.199543,  0.783059, -0.433371 },
  };

  ArrayXXd g {
    { -0.604897, -0.329554,  0.536459, -0.444451,   0.10794 },
    {   0.83239,  0.271423,  0.434594, -0.716795,  0.213938 },
    { -0.198111, -0.740419, -0.782382,  0.997849, -0.563486 },
    { 0.0485744, -0.012834,   0.94555, -0.414966,  0.542715 },
    { -0.295083,  0.615449,  0.838053, -0.860489,  0.898654 },
  };

  double ans[] = { 0.604897, 0.832390, 0.782382, 0.945550, 0.838053 };

  for (int i = 0; i < 5; ++i) {
    double sbgnrm = lbfgsb_projgr(x.row(i), g.row(i), bounds);
    EXPECT_NEAR(sbgnrm, ans[i], 1e-6);
  }
}

TEST(LBFGSBTest, Bmv) {
  MatrixXd vs {
    {  0.680375, -0.211234,  0.566198,   0.59688,   0.823295, -0.604897 },
    { -0.686642, -0.198111, -0.740419, -0.782382,   0.997849, -0.563486 },
    { -0.860489,  0.898654, 0.0519907, -0.827888,  -0.615572,  0.326454 },
    { -0.124725,   0.86367,   0.86162,  0.441905,  -0.431413,  0.477069 },
    { -0.203127,  0.629534,  0.368437,  0.821944, -0.0350187,  -0.56835 },
  };
  std::vector<MatrixXd> sys {
    MatrixXd {
              { 0.329554, 0.536459, 0.444451 },
              { 0.10794, 0.0452059, 0.257742 },
              { 0.270431, 0.0268018, 0.904459 },
              },
    MatrixXd {
              { 0.0258648, 0.678224, 0.22528 },
              { 0.407937, 0.275105, 0.0485744 },
              { 0.012834, 0.94555, 0.414966 },
              },
    MatrixXd {
              { 0.780465, 0.302214, 0.871657 },
              { 0.959954, 0.0845965, 0.873808 },
              { 0.52344, 0.941268, 0.804416 },
              },
    MatrixXd {
              { 0.279958, 0.291903, 0.375723 },
              { 0.668052, 0.119791, 0.76015 },
              { 0.658402, 0.339326, 0.542064 },
              },
    MatrixXd {
              { 0.900505, 0.840257, 0.70468 },
              { 0.762124, 0.282161, 0.136093 },
              { 0.239193, 0.437881, 0.572004 },
              },
  };
  std::vector<MatrixXd> wtts {
    MatrixXd {
              { 0.83239, 0.271423, 0.434594 },
              { -0.716795, 0.213938, -0.967399 },
              { -0.514226, -0.725537, 0.608354 },
              },
    MatrixXd {
              { 0.542715, 0.05349, 0.539828 },
              { -0.199543, 0.783059, -0.433371 },
              { -0.295083, 0.615449, 0.838053 },
              },
    MatrixXd {
              { 0.70184, -0.466669, 0.0795207 },
              { -0.249586, 0.520497, 0.0250707 },
              { 0.335448, 0.0632129, -0.921439 },
              },
    MatrixXd {
              { 0.786745, -0.29928, 0.37334 },
              { 0.912937, 0.17728, 0.314608 },
              { 0.717353, -0.12088, 0.84794 },
              },
    MatrixXd {
              { -0.385084, -0.105933, -0.547787 },
              { -0.624934, -0.447531, 0.112888 },
              { -0.166997, -0.660786, 0.813608 },
              },
  };

  MatrixXd ans {
    {  229.072270, 124.347662, -0.626007, 121.125489, 124.600417, 20.990036 },
    { -824.671415,   2.376489,  1.784288, -10.636928, -35.578127,  9.380931 },
    {   10.910744, 101.231874, -0.064632,  -8.956068,  -5.903956, 10.829057 },
    {  -21.902615,  33.871672, -1.589517,  29.352062, -29.766594,  6.473974 },
    {   -8.402572,  -1.122897, -0.644116,  22.679712, -11.173730,  2.297666 },
  };

  VectorXd p(8);
  ArrayXd xp(5);
  for (int i = 0; i < 5; ++i) {
    bool success = lbfgs_bmv(p.head(6), xp.head(2), vs.row(i).transpose(),
                             sys[i].transpose(), wtts[i]);
    ASSERT_TRUE(success);
    NURI_EXPECT_EIGEN_EQ(p.head(6).transpose(), ans.row(i));
  }
}

TEST(LBFGSBTest, Cauchy) {
  ArrayXi nbd {
    { 0, 1, 2, 3, 3 }
  };
  Array2Xd bds {
    { 0.0, 0.0, 0.0, 0.0, 0.0 },
    { 1.0, 1.0, 1.0, 1.0, 1.0 }
  };
  LbfgsbBounds bounds { nbd, bds };

  ArrayXXd xs {
    {   0.680375, -0.211234,  0.566198,   0.59688,  0.823295 },
    { -0.0452059,  0.257742, -0.270431, 0.0268018,  0.904459 },
    {  -0.967399, -0.514226, -0.725537,  0.608354, -0.686642 },
    {  0.0258648,  0.678224,   0.22528, -0.407937,  0.275105 },
    {    0.05349,  0.539828, -0.199543,  0.783059, -0.433371 },
  };
  xs.transposeInPlace();

  ArrayXXd g {
    { -0.604897, -0.329554,  0.536459, -0.444451,   0.10794 },
    {   0.83239,  0.271423,  0.434594, -0.716795,  0.213938 },
    { -0.198111, -0.740419, -0.782382,  0.997849, -0.563486 },
    { 0.0485744, -0.012834,   0.94555, -0.414966,  0.542715 },
    { -0.295083,  0.615449,  0.838053, -0.860489,  0.898654 },
  };
  g.transposeInPlace();

  MatrixXd vs {
    {  0.680375, -0.211234,  0.566198,   0.59688,   0.823295, -0.604897 },
    { -0.686642, -0.198111, -0.740419, -0.782382,   0.997849, -0.563486 },
    { -0.860489,  0.898654, 0.0519907, -0.827888,  -0.615572,  0.326454 },
    { -0.124725,   0.86367,   0.86162,  0.441905,  -0.431413,  0.477069 },
    { -0.203127,  0.629534,  0.368437,  0.821944, -0.0350187,  -0.56835 },
  };
  vs.transposeInPlace();

  std::vector<MatrixXd> wss {
    MatrixXd {
              { 0.205973, 0.558315, 0.852124 },
              { 0.246163, -0.0436421, 0.808401 },
              { -0.555187, -0.0292421, 0.121329 },
              { -0.662105, 0.109516, -0.809418 },
              { 0.1267, -0.516387, 0.765034 },
              },
    MatrixXd {
              { 0.351302, -0.29183, -0.856863 },
              { 0.696472, 0.63381, 0.478604 },
              { -0.7288, -0.397138, 0.802195 },
              { 0.240472, 0.576201, -0.619467 },
              { 0.0297216, 0.804436, 0.946142 },
              },
    MatrixXd {
              { -0.73719, -0.680994, 0.364371 },
              { 0.586506, -0.734062, 0.141887 },
              { 0.192305, -0.119241, 0.183855 },
              { -0.292377, 0.289764, -0.514299 },
              { 0.924401, 0.375454, 0.332469 },
              },
    MatrixXd {
              { -0.0511112, -0.941338, -0.484279 },
              { 0.837003, 0.928258, -0.195339 },
              { 0.0289407, -0.288715, -0.0169369 },
              { 0.220088, -0.338476, -0.682084 },
              { 0.0774748, 0.504459, 0.185194 },
              },
    MatrixXd {
              { -0.0707946, -0.303899, -0.0702593 },
              { -0.419274, -0.559484, -0.0587686 },
              { -0.2283, 0.554287, -0.127833 },
              { -0.878489, 0.57686, -0.886404 },
              { 0.288349, 0.73028, 0.426932 },
              },
  };
  std::vector<MatrixXd> wys {
    MatrixXd {
              { -0.93306, -0.499743, -0.270015 },
              { -0.340072, 0.273117, 0.530762 },
              { 0.381271, 0.727245, -0.364279 },
              { -0.155027, -0.396689, -0.728543 },
              { -0.58747, -0.950152, -0.786509 },
              },
    MatrixXd {
              { 0.521343, 0.96352, 0.110462 },
              { -0.836655, -0.5623, 0.457093 },
              { 0.101912, -0.0912082, 0.257748 },
              { 0.129304, 0.037308, 0.592256 },
              { 0.486541, 0.147683, 0.167432 },
              },
    MatrixXd {
              { -0.450467, -0.76011, 0.827134 },
              { 0.659196, -0.568939, -0.302879 },
              { 0.82736, 0.777287, -0.434866 },
              { 0.930804, 0.967135, -0.537145 },
              { -0.49583, 0.0343724, -0.0314227 },
              },
    MatrixXd {
              { -0.221375, -0.381537, -0.301115 },
              { 0.984198, 0.844058, -0.362283 },
              { 0.131922, 0.551201, -0.661542 },
              { 0.880537, 0.527255, 0.956633 },
              { 0.113502, -0.118634, -0.770027 },
              },
    MatrixXd {
              { 0.50589, -0.606661, 0.902404 },
              { -0.493834, 0.57314, 0.99776 },
              { 0.889169, -0.131969, -0.762604 },
              { 0.33325, -0.829375, -0.532462 },
              { -0.56303, 0.540275, -0.539385 },
              },
  };

  std::vector<MatrixXd> sys {
    MatrixXd {
              { 0.329554, 0.536459, 0.444451 },
              { 0.10794, 0.0452059, 0.257742 },
              { 0.270431, 0.0268018, 0.904459 },
              },
    MatrixXd {
              { 0.0258648, 0.678224, 0.22528 },
              { 0.407937, 0.275105, 0.0485744 },
              { 0.012834, 0.94555, 0.414966 },
              },
    MatrixXd {
              { 0.780465, 0.302214, 0.871657 },
              { 0.959954, 0.0845965, 0.873808 },
              { 0.52344, 0.941268, 0.804416 },
              },
    MatrixXd {
              { 0.279958, 0.291903, 0.375723 },
              { 0.668052, 0.119791, 0.76015 },
              { 0.658402, 0.339326, 0.542064 },
              },
    MatrixXd {
              { 0.900505, 0.840257, 0.70468 },
              { 0.762124, 0.282161, 0.136093 },
              { 0.239193, 0.437881, 0.572004 },
              },
  };
  std::vector<MatrixXd> wtts {
    MatrixXd {
              { 0.83239, 0.271423, 0.434594 },
              { -0.716795, 0.213938, -0.967399 },
              { -0.514226, -0.725537, 0.608354 },
              },
    MatrixXd {
              { 0.542715, 0.05349, 0.539828 },
              { -0.199543, 0.783059, -0.433371 },
              { -0.295083, 0.615449, 0.838053 },
              },
    MatrixXd {
              { 0.70184, -0.466669, 0.0795207 },
              { -0.249586, 0.520497, 0.0250707 },
              { 0.335448, 0.0632129, -0.921439 },
              },
    MatrixXd {
              { 0.786745, -0.29928, 0.37334 },
              { 0.912937, 0.17728, 0.314608 },
              { 0.717353, -0.12088, 0.84794 },
              },
    MatrixXd {
              { -0.385084, -0.105933, -0.547787 },
              { -0.624934, -0.447531, 0.112888 },
              { -0.166997, -0.660786, 0.813608 },
              },
  };

  ArrayXXd xcpans {
    {  0.680375, 0.000000,  0.566198, 0.596880, 0.823295 },
    { -0.045206, 0.257742, -0.270431, 0.026802, 0.904459 },
    { -0.967399, 0.000000, -0.725537, 0.608354, 0.000000 },
    {  0.001242, 0.684730, -0.254024, 0.210348, 0.000000 },
    {  1.722777, 0.000000, -4.940416, 1.000000, 0.000000 },
  };
  xcpans.transposeInPlace();

  MatrixXd pans {
    { -0.886504, -0.676173, -0.031902, -0.000196, -0.000443, -0.000274 },
    { -0.262569, -0.614618,  0.060677, -0.000001, -0.000484,  0.000412 },
    { -0.162045, -0.909390,  0.117659, -0.001251,  0.000849, -0.001022 },
    {  0.264038, -0.273030,  1.032468, -0.000077, -0.000190,  0.000246 },
    { -0.595891, -0.068418,  0.905387, -0.000170,  0.000554, -0.000086 },
  };
  pans.transposeInPlace();

  MatrixXd vans {
    { -233.283106, -138.285076,  0.035272, -122.259504, -122.693681,
     -26.877552                                                                },
    { -335.409168,    2.992509, -0.146222,   -3.034546,  -14.605031,   4.295200 },
    {  -11.936100, -105.638206, -0.146266,    5.787840,    1.138340, -11.267936 },
    {    2.021730,   -3.600595,  1.420546,   -3.122262,    3.259059,  -0.723479 },
    {   -2.111828,   -1.615625, -1.744323,    5.428624,   -3.573644,   0.861716 },
  };
  vans.transposeInPlace();

  MatrixXd cans {
    {         0,         0,        0,         0,        0,         0 },
    {         0,         0,        0,         0,        0,         0 },
    {         0,         0,        0,         0,        0,         0 },
    {  0.102617, -0.105763, 0.735202, -0.000018, 0.000042,  0.000176 },
    { -3.032081, -0.876366, 4.467648, -0.001000, 0.002708, -0.000328 },
  };
  cans.transposeInPlace();

  ArrayXXi iwhans {
    { -1, 0, 0, 0, 0 },
    { -1, 0, 0, 0, 0 },
    { -1, 0, 0, 0, 0 },
    { -1, 0, 0, 0, 1 },
    { -1, 1, 0, 2, 1 },
  };
  iwhans.transposeInPlace();

  for (int i = 0; i < 5; ++i) {
    LBfgsB lbfgsb(xs.col(i), bounds, 4);

    lbfgsb.update_col(3);
    lbfgsb.update_theta(-1e-3);
    lbfgsb.ws() = wss[i];
    lbfgsb.wy() = wys[i];
    lbfgsb.sy() = sys[i].transpose();
    lbfgsb.wtt() = wtts[i];
    lbfgsb.v() = vs.col(i);

    bool success = lbfgsb_cauchy(lbfgsb, g.col(i), 0.5);
    ASSERT_TRUE(success);

    NURI_EXPECT_EIGEN_EQ(lbfgsb.xcp(), xcpans.col(i));
    NURI_EXPECT_EIGEN_EQ(lbfgsb.p(), pans.col(i));
    NURI_EXPECT_EIGEN_EQ(lbfgsb.v(), vans.col(i));
    NURI_EXPECT_EIGEN_EQ(lbfgsb.c(), cans.col(i));
    EXPECT_PRED2([](const auto &lhs,
                    const auto &rhs) { return (lhs == rhs).all(); },
                 lbfgsb.iwhere(), iwhans.col(i));
  }
}

TEST(LBFGSBTest, Subsm) {
  ArrayXi nbd {
    { 0, 1, 2, 3, 3 }
  };
  Array2Xd bds {
    { 0.0, 0.0, 0.0, 0.0, 0.0 },
    { 1.0, 1.0, 1.0, 1.0, 1.0 }
  };
  LbfgsbBounds bounds { nbd, bds };

  ArrayXXd xs {
    { -0.967399, -0.514226, -0.725537,  0.608354, -0.686642 },
    { 0.0258648,  0.678224,   0.22528, -0.407937,  0.275105 },
    {   0.05349,  0.539828, -0.199543,  0.783059, -0.433371 },
  };
  xs.transposeInPlace();

  ArrayXXd gs {
    { -0.198111, -0.740419, -0.782382,  0.997849, -0.563486 },
    { 0.0485744, -0.012834,   0.94555, -0.414966,  0.542715 },
    { -0.295083,  0.615449,  0.838053, -0.860489,  0.898654 },
  };
  gs.transposeInPlace();

  std::vector<MatrixXd> wss {
    MatrixXd {
              { -0.73719, -0.680994, 0.364371 },
              { 0.586506, -0.734062, 0.141887 },
              { 0.192305, -0.119241, 0.183855 },
              { -0.292377, 0.289764, -0.514299 },
              { 0.924401, 0.375454, 0.332469 },
              },
    MatrixXd {
              { -0.0511112, -0.941338, -0.484279 },
              { 0.837003, 0.928258, -0.195339 },
              { 0.0289407, -0.288715, -0.0169369 },
              { 0.220088, -0.338476, -0.682084 },
              { 0.0774748, 0.504459, 0.185194 },
              },
    MatrixXd {
              { -0.0707946, -0.303899, -0.0702593 },
              { -0.419274, -0.559484, -0.0587686 },
              { -0.2283, 0.554287, -0.127833 },
              { -0.878489, 0.57686, -0.886404 },
              { 0.288349, 0.73028, 0.426932 },
              },
  };
  std::vector<MatrixXd> wys {
    MatrixXd {
              { -0.450467, -0.76011, 0.827134 },
              { 0.659196, -0.568939, -0.302879 },
              { 0.82736, 0.777287, -0.434866 },
              { 0.930804, 0.967135, -0.537145 },
              { -0.49583, 0.0343724, -0.0314227 },
              },
    MatrixXd {
              { -0.221375, -0.381537, -0.301115 },
              { 0.984198, 0.844058, -0.362283 },
              { 0.131922, 0.551201, -0.661542 },
              { 0.880537, 0.527255, 0.956633 },
              { 0.113502, -0.118634, -0.770027 },
              },
    MatrixXd {
              { 0.50589, -0.606661, 0.902404 },
              { -0.493834, 0.57314, 0.99776 },
              { 0.889169, -0.131969, -0.762604 },
              { 0.33325, -0.829375, -0.532462 },
              { -0.56303, 0.540275, -0.539385 },
              },
  };

  ArrayXXd zs {
    { -0.967399, 0.000000, -0.725537, 0.608354, 0.000000 },
    {  0.001242, 0.684730, -0.254024, 0.210348, 0.000000 },
    {  1.722777, 0.000000, -4.940416, 1.000000, 0.000000 },
  };
  zs.transposeInPlace();

  MatrixXd cs {
    {         0,         0,        0,         0,        0,         0 },
    {  0.102617, -0.105763, 0.735202, -0.000018, 0.000042,  0.000176 },
    { -3.032081, -0.876366, 4.467648, -0.001000, 0.002708, -0.000328 },
  };
  cs.transposeInPlace();

  ArrayXi free {
    { 0, 4 }
  };

  ArrayXXd rs {
    {   0.198111, 0.220165, 0, 0, 0 },
    {   0.001037, 1.807676, 0, 0, 0 },
    { -14.078347, 9.587907, 0, 0, 0 },
  };
  rs.transposeInPlace();

  MatrixXd wnt {
    { 1.596070, 0.334823, 1.059997,  0.455904, -0.525436, -0.404465 },
    {        0, 1.358974, 1.322411,  0.769674,  1.016884, -0.061445 },
    {        0,        0, 0.152887, -2.085847,  2.649316,  9.972609 },
    {        0,        0,        0,  2.286466, -2.095094, -9.122658 },
    {        0,        0,        0,         0,  1.992419,  3.931478 },
    {        0,        0,        0,         0,         0,  1.149771 },
  };
  wnt.transposeInPlace();

  ArrayXXd zans {
    { -1.702536,        0, -0.725537, 0.608354, 1 },
    { -1.756607, 0.684730, -0.254024, 0.210348, 1 },
    {  0.547059,        0, -4.940416,        1, 1 },
  };
  zans.transposeInPlace();

  ArrayXXd rans {
    {  -0.735137, 1.153705, 0, 0, 0 },
    {  -1.757849, 3.757679, 0, 0, 0 },
    { -57.737404, 0.000000, 0, 0, 0 },
  };
  rans.transposeInPlace();

  MatrixXd pans {
    { -0.078845,  0.142419, -0.147806,  0.416238,  -0.245800, -0.099450 },
    {  0.270294, -0.365865,  0.305946, -0.427600,   0.636039,  0.574769 },
    { -8.559166,  7.031202, -2.823381, 32.639278, -40.189454, 34.250817 },
  };
  pans.transposeInPlace();

  for (int i = 0; i < 3; ++i) {
    LBfgsB lbfgsb(xs.col(i), bounds, 4);

    lbfgsb.update_col(3);
    lbfgsb.update_theta(0.5);
    lbfgsb.update_nfree(2);
    lbfgsb.free_bound().head(2) = free;

    lbfgsb.z() = zs.col(i);
    lbfgsb.r() = rs.col(i);
    lbfgsb.wnt() = wnt;
    lbfgsb.ws() = wss[i];
    lbfgsb.wy() = wys[i];

    bool success = lbfgsb_subsm(lbfgsb, gs.col(i));
    ASSERT_TRUE(success);

    double tol = i == 2 ? 1e-1 : 1e-3;
    NURI_EXPECT_EIGEN_EQ_TOL(lbfgsb.z(), zans.col(i), tol);
    NURI_EXPECT_EIGEN_EQ_TOL(lbfgsb.rfree().array(),
                             rans.col(i).head(free.size()), tol);
    NURI_EXPECT_EIGEN_EQ_TOL(lbfgsb.p(), pans.col(i), tol);
  }
}

TEST(LBFGSBTest, LbfgsbSquared) {
  auto fg = [](ArrayXd &gx, ConstRef<ArrayXd> xa) -> double {
    double fx = xa.square().sum();
    gx = 2 * xa;
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

  ArrayXi nbd(60);
  nbd.head(20) = 1;
  nbd.segment(20, 20) = 2;
  nbd.tail(20) = 3;

  ArrayXXd bounds {
    {    -0.52344,   0.941268 },
    {     0.70184,   0.804416 },
    {   -0.466669,  0.0795207 },
    {   -0.249586,   0.520497 },
    {   0.0250707,   0.335448 },
    {   -0.921439,  0.0632129 },
    {   -0.124725,    0.86367 },
    {    0.441905,    0.86162 },
    {   -0.431413,   0.477069 },
    {   -0.291903,   0.279958 },
    {   -0.668052,   0.375723 },
    {   -0.119791,    0.76015 },
    {   -0.339326,   0.658402 },
    {   -0.542064,   0.786745 },
    {    -0.29928,    0.37334 },
    {     0.17728,   0.912937 },
    {    0.314608,   0.717353 },
    {    -0.12088,    0.84794 },
    {   -0.203127,   0.629534 },
    {    0.368437,   0.821944 },
    {    -0.56835, -0.0350187 },
    {    0.840257,   0.900505 },
    {    -0.70468,   0.762124 },
    {   -0.136093,   0.282161 },
    {   -0.437881,   0.239193 },
    {   -0.385084,   0.572004 },
    {   -0.547787,  -0.105933 },
    {   -0.624934,  -0.447531 },
    {   -0.166997,   0.112888 },
    {   -0.660786,   0.813608 },
    {   -0.793658,  -0.747849 },
    { -0.00911187,    0.52095 },
    {    0.870008,   0.969503 },
    {   -0.233623,    0.36889 },
    {   -0.262673,   0.499542 },
    {   -0.535477,  -0.411679 },
    {   -0.511175,   0.168977 },
    {    -0.69522,   0.464297 },
    {    -0.74905,   0.586941 },
    {   -0.671796,   0.490143 },
    {    -0.85094,   0.900208 },
    {   -0.894941,  0.0431268 },
    {   -0.647579,  -0.519875 },
    {    0.465309,   0.595596 },
    {    0.313127,    0.93481 },
    {    0.278917,    0.51947 },
    {   -0.813039,  -0.730195 },
    {   -0.843536,  0.0404201 },
    {   -0.860187,   -0.59069 },
    {  -0.0771591,   0.639355 },
    {    0.146637,   0.511162 },
    {   -0.896122,  -0.684386 },
    {   -0.591343,   0.999987 },
    {   -0.749063,   0.779911 },
    {   -0.891885,   0.995598 },
    {   -0.855342,    0.74108 },
    {   -0.991677,   0.846138 },
    {   -0.639255,   0.187784 },
    {   -0.673737,   -0.21662 },
    {     0.63939,   0.826053 },
  };
  bounds.transposeInPlace();

  auto [code, _, fx, gx] = l_bfgs_b(fg, xa, nbd, bounds);
  ASSERT_EQ(code, LbfgsbResultCode::kSuccess);

  EXPECT_TRUE((xa.head(20) >= bounds.row(0).head(20).transpose()).all());
  EXPECT_TRUE(
      (xa.segment(20, 20) <= bounds.row(1).segment(20, 20).transpose()).all());
  EXPECT_TRUE((xa.tail(20) >= bounds.row(0).tail(20).transpose()).all());
  EXPECT_TRUE((xa.tail(20) <= bounds.row(1).tail(20).transpose()).all());

  // Results from scipy.optimize.fmin_l_bfgs_b
  MatrixX3d xans {
    {  2.97450449e-16,  7.01840000e-01,  2.79359746e-16 },
    {  1.70398777e-16,  2.50707000e-02, -2.29439929e-16 },
    { -8.40784200e-17,  4.41905000e-01, -2.98329073e-16 },
    {  6.07533072e-17, -1.85049446e-17,  5.64862103e-17 },
    { -1.52210280e-16,  1.50852139e-17,  2.74444780e-16 },
    {  1.77280000e-01,  3.14608000e-01,  1.56861537e-16 },
    { -1.42084249e-16,  3.68437000e-01, -3.50187000e-02 },
    { -2.89428665e-16, -5.19385951e-16,  1.31056856e-16 },
    { -2.92723838e-16, -1.11505452e-16, -1.05933000e-01 },
    { -4.47531000e-01,  4.96604681e-17, -3.72665478e-16 },
    { -7.47849000e-01,  2.93213224e-16,  9.90417659e-17 },
    { -2.35956292e-16,  1.82596585e-16, -4.11679000e-01 },
    { -8.95825455e-18,  2.05815310e-16, -2.89071989e-16 },
    {  2.75873710e-16,  3.01064888e-17,  2.42736310e-17 },
    { -5.19875000e-01,  4.65309000e-01,  3.13127000e-01 },
    {  2.78917000e-01, -7.30195000e-01,  2.27501830e-17 },
    { -5.90690000e-01,  1.18240810e-16,  1.46637000e-01 },
    { -6.84386000e-01, -2.77322302e-16,  1.28231300e-16 },
    {  3.28257204e-16, -1.14587978e-16, -4.94757427e-16 },
    { -1.55684329e-16, -2.16620000e-01,  6.39390000e-01 },
  };
  double fxans = 4.38648551081218;
  MatrixX3d gans {
    {  5.94900898e-16,  1.40368000e+00,  5.58719491e-16 },
    {  3.40797554e-16,  5.01414000e-02, -4.58879858e-16 },
    { -1.68156840e-16,  8.83810000e-01, -5.96658147e-16 },
    {  1.21506614e-16, -3.70098892e-17,  1.12972421e-16 },
    { -3.04420560e-16,  3.01704278e-17,  5.48889559e-16 },
    {  3.54560000e-01,  6.29216000e-01,  3.13723075e-16 },
    { -2.84168498e-16,  7.36874000e-01, -7.00374000e-02 },
    { -5.78857331e-16, -1.03877190e-15,  2.62113713e-16 },
    { -5.85447677e-16, -2.23010903e-16, -2.11866000e-01 },
    { -8.95062000e-01,  9.93209361e-17, -7.45330956e-16 },
    { -1.49569800e+00,  5.86426447e-16,  1.98083532e-16 },
    { -4.71912585e-16,  3.65193170e-16, -8.23358000e-01 },
    { -1.79165091e-17,  4.11630621e-16, -5.78143979e-16 },
    {  5.51747420e-16,  6.02129776e-17,  4.85472620e-17 },
    { -1.03975000e+00,  9.30618000e-01,  6.26254000e-01 },
    {  5.57834000e-01, -1.46039000e+00,  4.55003660e-17 },
    { -1.18138000e+00,  2.36481621e-16,  2.93274000e-01 },
    { -1.36877200e+00, -5.54644604e-16,  2.56462599e-16 },
    {  6.56514407e-16, -2.29175955e-16, -9.89514853e-16 },
    { -3.11368657e-16, -4.33240000e-01,  1.27878000e+00 },
  };

  NURI_EXPECT_EIGEN_EQ(x, xans.transpose());
  EXPECT_NEAR(fx, fxans, 1e-6);
  NURI_EXPECT_EIGEN_EQ(gx.reshaped(3, 20).matrix(), gans.transpose());
}

TEST(LBFGSBTest, LbfgsbCorrelated) {
  // Distance maximization problem (e.g. forcefield)
  auto fg = [](ArrayXd &gx, ConstRef<ArrayXd> xa) -> double {
    ConstRef<Matrix3Xd> xm = xa.reshaped(3, 20).matrix();
    MutRef<Matrix3Xd> gm = gx.reshaped(3, 20).matrix();

    double fx = 0;
    gm.setZero();

    Vector3d d;
    for (int i = 1; i < 20; ++i) {
      for (int j = 0; j < i; ++j) {
        d = xm.col(i) - xm.col(j);
        double pdsq_inv = 1 / std::max(d.squaredNorm(), 1e-24);
        double pd_inv = std::sqrt(pdsq_inv);
        fx += pd_inv;

        d *= -2 * pdsq_inv * pd_inv;
        gm.col(i) += d;
        gm.col(j) -= d;
      }
    }

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

  ArrayXi nbd = ArrayXi::Constant(60, 3);
  Array2Xd bounds(2, 60);
  bounds.row(0).setConstant(-1);
  bounds.row(1).setConstant(1);

  auto [code, iter, fx, gx] = l_bfgs_b(fg, xa, nbd, bounds, 5);
  ASSERT_EQ(code, LbfgsbResultCode::kSuccess);

  EXPECT_TRUE((xa >= bounds.row(0).transpose()).all());
  EXPECT_TRUE((xa <= bounds.row(1).transpose()).all());

  // Results from scipy.optimize.fmin_l_bfgs_b
  EXPECT_EQ(iter, 24);

  MatrixX3d xans {
    {  1.000000, -1.000000,  1.000000 },
    {  1.000000,  1.000000, -1.000000 },
    {  0.020200, -1.000000, -1.000000 },
    {  1.000000, -0.089951, -1.000000 },
    { -1.000000, -1.000000,  1.000000 },
    {  1.000000,  1.000000,  0.152648 },
    { -1.000000, -0.090330, -1.000000 },
    {  0.078263, -1.000000,  1.000000 },
    {  1.000000, -1.000000,  0.099281 },
    { -1.000000,  1.000000, -1.000000 },
    {  1.000000,  1.000000,  1.000000 },
    { -1.000000, -0.022563,  1.000000 },
    {  0.375690,  1.000000, -1.000000 },
    {  0.072716,  1.000000,  1.000000 },
    { -0.358441,  1.000000, -1.000000 },
    { -1.000000,  1.000000,  1.000000 },
    { -1.000000,  1.000000,  0.107459 },
    { -1.000000, -1.000000,  0.042433 },
    {  1.000000, -1.000000, -1.000000 },
    { -1.000000, -1.000000, -1.000000 },
  };
  double fxans = 101.15066430373199;
  MatrixX3d gans {
    { -4.828977,  2.579002, -5.010159 },
    { -8.551676, -3.889551,  3.573901 },
    {  0.000029,  4.255305,  3.608052 },
    { -4.437316,  0.000095,  3.368040 },
    {  3.861739,  4.454953, -4.712982 },
    { -4.451857, -3.070158, -0.000114 },
    {  4.244722, -0.000062,  3.698355 },
    {  0.000162,  3.328475, -4.156006 },
    { -4.097918,  3.389620,  0.000166 },
    {  8.114099, -3.986511,  3.867640 },
    { -4.876951, -2.445919, -5.497012 },
    {  3.515268, -0.000089, -4.206636 },
    {  0.000159, -3.792768,  3.588330 },
    {  0.000314, -3.163687, -4.431738 },
    {  0.000051, -3.826949,  3.702865 },
    {  3.964431, -4.133211, -5.210774 },
    {  4.113488, -3.750476,  0.000019 },
    {  3.734559,  4.058761, -0.000064 },
    { -4.621804,  4.946700,  3.728489 },
    {  4.317477,  5.046467,  4.089628 },
  };

  NURI_EXPECT_EIGEN_EQ(x, xans.transpose());
  EXPECT_NEAR(fx, fxans, 1e-6);
  NURI_EXPECT_EIGEN_EQ(gx.reshaped(3, 20).matrix(), gans.transpose());
}

TEST(LBFGSBTest, LbfgsbReference) {
  constexpr int n = 25;
  ArrayXd t(n);

  // Taken from the reference implementation (driver1.f90)
  auto fg = [&](ArrayXd &g, ConstRef<ArrayXd> x) -> double {
    t[0] = x[0] - 1;
    t.tail(n - 1) = 2 * (x.tail(n - 1) - x.head(n - 1).square());
    const double fx = t.square().sum();

    t.tail(n - 1) *= 4;

    g[0] = 2 * (t[0] - x[0] * t[1]);
    g.segment(1, n - 2) =
        t.segment(1, n - 2) - 2 * (x.segment(1, n - 2) * t.tail(n - 2));
    g[n - 1] = t[n - 1];

    return fx;
  };

  ArrayXd x = ArrayXd::Constant(n, 3);

  ArrayXi nbd = ArrayXi::Constant(n, 3);
  Array2Xd bounds(2, n);
  bounds(0, Eigen::seq(0, n - 1, 2)) = 1;
  bounds(0, Eigen::seq(1, n - 1, 2)) = -100;
  bounds.row(1) = 100;

  auto [code, iter, fx, gx] = l_bfgs_b(fg, x, nbd, bounds, 5);
  ASSERT_EQ(code, LbfgsbResultCode::kSuccess);

  ArrayXd xans {
    {
     1.0000023258771837, 1.0000030446388724, 1.0000045143926761,
     1.0000050772587232, 1.0000058193063399, 1.0000070253663254,
     1.0000083438107843, 1.0000120681556417, 1.0000196904590828,
     1.0000374605368185, 1.0000770108551211, 1.0001583814754376,
     1.0003231605131711, 1.0006492825285076, 1.0013006127613451,
     1.002601944378863,  1.0052112931969246, 1.010449820256067,
     1.0210095629751152, 1.0424600004275,    1.086723474398011,
     1.1809686991764257, 1.394691851289783,  1.9451626718243049,
     3.7836626570644625, }
  };
  double fans = 1.0834900834300615e-09;
  ArrayXd gans {
    {
     3.0365748651745088e-05,  1.234141889728772e-05,
     5.0625882852877512e-05,  3.7751766268700644e-05,
     3.9131017956764807e-05,  5.4405939303440887e-05,
     2.8257415671835532e-05,  3.4180539567112549e-05,
     -4.8350726165878632e-06, -4.8781455271871458e-05,
     -5.2959691520253405e-05, -6.7145113998150568e-05,
     5.2519369914780871e-06,  -3.1785220170763878e-06,
     2.8593132814416504e-05,  -1.795750106352732e-05,
     3.8476695147693124e-06,  -1.1090233921327606e-05,
     1.4403142695957529e-05,  -1.4591078955444807e-05,
     -8.7499549958468529e-06, -8.4059488955526312e-05,
     9.8250878119067711e-05,  -0.00017205227288462103,
     3.8697646353114123e-05, }
  };

  EXPECT_EQ(iter, 23);
  NURI_EXPECT_EIGEN_EQ(x, xans);
  EXPECT_NEAR(fx, fans, 1e-12);
  NURI_EXPECT_EIGEN_EQ_TOL(gx, gans, 1e-9);
}

TEST(LBFGSBTest, LbfgsbUnbounded) {
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

  ArrayXi nbd = ArrayXi::Zero(60);
  Array2Xd bounds(2, 60);

  auto [code, iter, fx, gx] = l_bfgs_b(fg, xa, nbd, bounds);
  ASSERT_EQ(code, LbfgsbResultCode::kSuccess);

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