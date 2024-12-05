//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/tools/tm.h"

#include <cmath>
#include <string_view>

#include <absl/base/optimization.h>
#include <absl/log/absl_check.h>
#include <Eigen/Dense>

#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "test_utils.h"

namespace nuri {
namespace internal {
namespace {
constexpr double kD0SqInv = 1.0672085240079228,            //
    kD01SqInv = 0.1641760071869689,                        //
    kD0Search = 4.5, kD0SearchSq = kD0Search * kD0Search,  //
    kDCu0 = 4.25, kDCu0Sq = kDCu0 * kDCu0,                 //
    kDCutoff = kD0Search - 1,                              //
    kD8Score = 7.1284182222313817, kD8ScoreSq = kD8Score * kD8Score;

constexpr std::int8_t kPathVert = 1, kPathHorz = 2;

// X coordinates -> 2q7n_A, Y coordinates -> 1x5g_A

TEST(TMAlignComponentTest, ScoreFun8Withd8) {
  Matrix3Xd x(3, 7), y(3, 7);
  x.transpose() << -16.0495, 1.75814, 7.3656,  //
      -13.3227, 2.54396, 4.8601,               //
      -9.55338, 2.99524, 4.99842,              //
      0, 0, 0,                                 //
      -4.03938, 5.6192, 1.83645,               //
      -2.1989, 8.89162, 1.81161,               //
      1.03304, 10.239, 0.408338;
  y.transpose() << -15.904, 0.493, 6.955,  //
      -13.38, 2.503, 4.941,                //
      -9.748, 3.625, 5.041,                //
      -7.877, 6.604, 3.598,                //
      -4.273, 6.286, 2.426,                //
      -2.033, 9.201, 1.455,                //
      1.45, 8.952, -0.051;

  ArrayXi aligned(20);
  auto [n_ali, tmscore] =
      tmalign_score_fun8(x, y, aligned, kDCutoff, kD0SqInv, kD8ScoreSq);

  EXPECT_EQ(n_ali, 6);
  EXPECT_NEAR(tmscore, 3.6427780105500926, 1e-5);
  for (int i = 0; i < 3; ++i)
    EXPECT_EQ(aligned[i], i);
  for (int i = 3; i < n_ali; ++i)
    EXPECT_EQ(aligned[i], i + 1);
}

TEST(TMAlignComponentTest, ScoreFun8NoD8) {
  Matrix3Xd x(3, 7), y(3, 7);
  x.transpose() << -16.0495, 1.75814, 7.3656,  //
      -13.3227, 2.54396, 4.8601,               //
      -9.55338, 2.99524, 4.99842,              //
      0, 0, 0,                                 //
      -4.03938, 5.6192, 1.83645,               //
      -2.1989, 8.89162, 1.81161,               //
      1.03304, 10.239, 0.408338;
  y.transpose() << -15.904, 0.493, 6.955,  //
      -13.38, 2.503, 4.941,                //
      -9.748, 3.625, 5.041,                //
      -7.877, 6.604, 3.598,                //
      -4.273, 6.286, 2.426,                //
      -2.033, 9.201, 1.455,                //
      1.45, 8.952, -0.051;

  ArrayXi aligned(20);
  auto [n_ali, tmscore] = tmalign_score_fun8(x, y, aligned, kDCutoff, kD0SqInv);

  EXPECT_EQ(n_ali, 6);
  EXPECT_NEAR(tmscore, 3.6506164230627633, 1e-5);
  for (int i = 0; i < 3; ++i)
    EXPECT_EQ(aligned[i], i);
  for (int i = 3; i < n_ali; ++i)
    EXPECT_EQ(aligned[i], i + 1);
}

TEST(TMAlignComponentTest, ScoreFun8Far) {
  Matrix3Xd x(3, 7), y(3, 7);
  x.transpose() << -15.904, 8.493, 6.955,  // 8
      -13.38, 6.003, 4.941,                // 3.5
      -9.748, 9.625, 5.041,                // 6
      -7.877, 6.604, 3.598,                // 0
      -4.273, 11.286, 2.426,               // 5
      -2.033, 16.201, 1.455,               // 7
      1.45, 17.952, -0.051;                // 9
  y.transpose() << -15.904, 0.493, 6.955,  //
      -13.38, 2.503, 4.941,                //
      -9.748, 3.625, 5.041,                //
      -7.877, 6.604, 3.598,                //
      -4.273, 6.286, 2.426,                //
      -2.033, 9.201, 1.455,                //
      1.45, 8.952, -0.051;

  ArrayXi aligned(20);
  auto [n_ali, tmscore] =
      tmalign_score_fun8(x, y, aligned, kDCutoff, kD0SqInv, kD8ScoreSq);
  EXPECT_EQ(n_ali, 3);
  EXPECT_NEAR(tmscore, 1.1513156715899149, 1e-5);

  std::vector<int> expected = { 1, 3, 4 };
  for (int i = 0; i < 3; ++i)
    EXPECT_EQ(aligned[i], expected[i]);
}

TEST(TMAlignComponentTest, TMScore8SearchWithd8) {
  Matrix3Xd x(3, 19), y(3, 20);
  x.transpose() << 6.468, 147.247, 1.898,  //
      5.575, 143.722, 3.203,               //
      5.286, 141.68, 6.406,                //
      7.418, 138.579, 5.998,               //
      6.865, 136.019, 8.742,               //
      7.602, 129.767, 12.504,              //
      10.676, 128.272, 13.922,             //
      10.549, 125.504, 16.56,              //
      12.582, 125.983, 19.831,             //
      12.556, 127.662, 23.311,             //
      14.358, 130.868, 22.369,             //
      13.155, 133.223, 24.992,             //
      16.506, 133.627, 26.676,             //
      18.315, 134.029, 23.375,             //
      17.874, 136.187, 20.28,              //
      18.362, 134.986, 16.739,             //
      19.367, 136.896, 13.605,             //
      17.877, 135.802, 10.337,             //
      18.434, 136.495, 6.671;
  y.transpose() << -15.904, 0.493, 6.955,  //
      -13.38, 2.503, 4.941,                //
      -9.748, 3.625, 5.041,                //
      -7.877, 6.604, 3.598,                //
      -4.273, 6.286, 2.426,                //
      -2.033, 9.201, 1.455,                //
      1.45, 8.952, -0.051,                 //
      3.985, 11.104, -1.899,               //
      6.007, 10.672, -5.091,               //
      9.637, 9.482, -4.874,                //
      12.257, 12.088, -5.774,              //
      14.161, 9.281, -7.494,               //
      12.914, 7.62, -10.666,               //
      13.075, 7.523, -14.458,              //
      7.187, 9.652, -14.678,               //
      5.11, 6.995, -12.92,                 //
      2.402, 5.651, -15.227,               //
      0.074, 4.641, -12.394,               //
      0.224, 4.123, -8.629,                //
      -0.587, 0.619, -7.393;

  ArrayXi aligned(20);
  aligned << -1, 0, 1, 2, 3,  //
      4, -1, 5, 6, 7,         //
      8, 9, 10, 12, 13,       //
      14, 15, 16, 17, 18;

  AlignedXY xy(x, y, 19);
  xy.swap_remap(aligned);

  auto [xform, tmscore] =
      tmalign_tmscore8_search(xy, 4, kD0Search, kD8ScoreSq, kD0SqInv);

  EXPECT_NEAR(tmscore, 5.8577381778888604, 1e-5);

  Affine3d xform_ref;
  xform_ref.matrix() << 0.135936, -0.99035, -0.0269758, 132.15,  //
      -0.595673, -0.103459, 0.796536, 19.8124,                   //
      -0.791641, -0.0922089, -0.603989, 24.267,                  //
      0, 0, 0, 1;
  NURI_EXPECT_EIGEN_EQ_TOL(xform.linear(), xform_ref.linear(), 1e-5);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.translation(), xform_ref.translation(), 1e-2);
}

TEST(TMAlignComponentTest, DpIter) {
  Matrix3Xd x(3, 19), y(3, 20);
  x.transpose() << 6.468, 147.247, 1.898,  //
      5.575, 143.722, 3.203,               //
      5.286, 141.68, 6.406,                //
      7.418, 138.579, 5.998,               //
      6.865, 136.019, 8.742,               //
      7.602, 129.767, 12.504,              //
      10.676, 128.272, 13.922,             //
      10.549, 125.504, 16.56,              //
      12.582, 125.983, 19.831,             //
      12.556, 127.662, 23.311,             //
      14.358, 130.868, 22.369,             //
      13.155, 133.223, 24.992,             //
      16.506, 133.627, 26.676,             //
      18.315, 134.029, 23.375,             //
      17.874, 136.187, 20.28,              //
      18.362, 134.986, 16.739,             //
      19.367, 136.896, 13.605,             //
      17.877, 135.802, 10.337,             //
      18.434, 136.495, 6.671;
  y.transpose() << -15.904, 0.493, 6.955,  //
      -13.38, 2.503, 4.941,                //
      -9.748, 3.625, 5.041,                //
      -7.877, 6.604, 3.598,                //
      -4.273, 6.286, 2.426,                //
      -2.033, 9.201, 1.455,                //
      1.45, 8.952, -0.051,                 //
      3.985, 11.104, -1.899,               //
      6.007, 10.672, -5.091,               //
      9.637, 9.482, -4.874,                //
      12.257, 12.088, -5.774,              //
      14.161, 9.281, -7.494,               //
      12.914, 7.62, -10.666,               //
      13.075, 7.523, -14.458,              //
      7.187, 9.652, -14.678,               //
      5.11, 6.995, -12.92,                 //
      2.402, 5.651, -15.227,               //
      0.074, 4.641, -12.394,               //
      0.224, 4.123, -8.629,                //
      -0.587, 0.619, -7.393;

  ArrayXi aligned(20), y2x_best(20);
  aligned << 9, 10, 11, 12, 13,  //
      14, 15, 16, 17, 18,        //
      -1, -1, -1, -1, -1,        //
      -1, -1, -1, -1, -1;

  AlignedXY xy(x, y, 19);
  xy.swap_remap(aligned);

  Affine3d xform;
  xform.matrix() << 0.193389, -0.980792, -0.0254527, 130.005,  //
      -0.562817, -0.132149, 0.815949, 23.3974,                 //
      -0.80364, -0.143471, -0.577563, 31.704,                  //
      0, 0, 0, 1;
  double tmscore = 5.4039298384527008;

  tmalign_dp_iter(xform, tmscore, xy, y2x_best, 0, 2, 30, 40, kD0Search,
                  kD8ScoreSq, kD0SqInv);
  EXPECT_NEAR(tmscore, 5.8577381778888604, 1e-5);

  Affine3d xform_ref;
  xform_ref.matrix() << 0.135936, -0.99035, -0.0269758, 132.15,  //
      -0.595673, -0.103459, 0.796536, 19.8124,                   //
      -0.791641, -0.0922089, -0.603989, 24.267,                  //
      0, 0, 0, 1;
  NURI_EXPECT_EIGEN_EQ_TOL(xform.linear(), xform_ref.linear(), 1e-5);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.translation(), xform_ref.translation(), 1e-2);

  ArrayXi aligned_ref(20);
  aligned_ref << -1, 0, 1, 2, 3,  //
      4, -1, 5, 6, 7,             //
      8, 9, 10, 12, 13,           //
      14, 15, 16, 17, 18;
  for (int i = 0; i < 20; ++i)
    EXPECT_EQ(y2x_best[i], aligned_ref[i]);
}

TEST(TMAlignComponentTest, GetScoreFast) {
  Matrix3Xd x(3, 9), y(3, 9);
  x.transpose() << 6.468, 147.247, 1.898,  //
      5.575, 143.722, 3.203,               //
      5.286, 141.68, 6.406,                //
      7.418, 138.579, 5.998,               //
      6.865, 136.019, 8.742,               //
      7.602, 129.767, 12.504,              //
      10.676, 128.272, 13.922,             //
      10.549, 125.504, 16.56,              //
      12.582, 125.983, 19.831;
  y.transpose() << 14.161, 9.281, -7.494,  //
      12.914, 7.62, -10.666,               //
      13.075, 7.523, -14.458,              //
      7.187, 9.652, -14.678,               //
      5.11, 6.995, -12.92,                 //
      2.402, 5.651, -15.227,               //
      0.074, 4.641, -12.394,               //
      0.224, 4.123, -8.629,                //
      -0.587, 0.619, -7.393;

  double tmscore = tmalign_get_score_fast(x, y, kD0SqInv, kD0Search);
  EXPECT_NEAR(tmscore, 0.98013497007552663, 1e-5);
}

TEST(TMAlignComponentTest, AssignSecStr) {
  Matrix3Xd x(3, 20), buf(3, 18);
  x.transpose() << 34.682, 106.421, 27.027,  //
      35.169, 109.907, 25.557,               //
      33.947, 113.167, 27.045,               //
      33.037, 116.635, 25.787,               //
      32.303, 119.994, 27.4,                 //
      29.802, 122.397, 25.806,               //
      28.392, 125.929, 26.413,               //
      24.901, 125.011, 25.14,                //
      23.872, 128.617, 24.324,               //
      26.951, 128.515, 22.081,               //
      25.558, 125.578, 20.087,               //
      22.079, 126.751, 19.129,               //
      21.964, 125.568, 15.469,               //
      18.246, 125.572, 14.775,               //
      15.919, 123.639, 12.566,               //
      14.856, 125.044, 9.265,                //
      12.216, 127.83, 9.293,                 //
      8.768, 127.083, 7.913,                 //
      5.361, 128.129, 6.619,                 //
      6.356, 131.538, 5.323;

  ArrayXc ss = assign_secstr_approx_full(x, buf);
  std::string_view ss_str { reinterpret_cast<char *>(ss.data()),
                            static_cast<std::size_t>(ss.size()) };
  EXPECT_EQ(ss_str, "CCEEEECTHHCCEECCECCC");
}

class TMAlignTestBase: public ::testing::Test {
public:
  constexpr static int lx = 19, ly = 20, l_min = 19, l_max = 20;

  static void SetUpTestSuite() {
    Matrix3Xd buf(3, l_max);

    x_.transpose() << 6.468, 147.247, 1.898,  //
        5.575, 143.722, 3.203,                //
        5.286, 141.68, 6.406,                 //
        7.418, 138.579, 5.998,                //
        6.865, 136.019, 8.742,                //
        7.602, 129.767, 12.504,               //
        10.676, 128.272, 13.922,              //
        10.549, 125.504, 16.56,               //
        12.582, 125.983, 19.831,              //
        12.556, 127.662, 23.311,              //
        14.358, 130.868, 22.369,              //
        13.155, 133.223, 24.992,              //
        16.506, 133.627, 26.676,              //
        18.315, 134.029, 23.375,              //
        17.874, 136.187, 20.28,               //
        18.362, 134.986, 16.739,              //
        19.367, 136.896, 13.605,              //
        17.877, 135.802, 10.337,              //
        18.434, 136.495, 6.671;
    secx_ = assign_secstr_approx_full(x_, buf);

    y_.transpose() << -15.904, 0.493, 6.955,  //
        -13.38, 2.503, 4.941,                 //
        -9.748, 3.625, 5.041,                 //
        -7.877, 6.604, 3.598,                 //
        -4.273, 6.286, 2.426,                 //
        -2.033, 9.201, 1.455,                 //
        1.45, 8.952, -0.051,                  //
        3.985, 11.104, -1.899,                //
        6.007, 10.672, -5.091,                //
        9.637, 9.482, -4.874,                 //
        12.257, 12.088, -5.774,               //
        14.161, 9.281, -7.494,                //
        12.914, 7.62, -10.666,                //
        13.075, 7.523, -14.458,               //
        7.187, 9.652, -14.678,                //
        5.11, 6.995, -12.92,                  //
        2.402, 5.651, -15.227,                //
        0.074, 4.641, -12.394,                //
        0.224, 4.123, -8.629,                 //
        -0.587, 0.619, -7.393;
    secy_ = assign_secstr_approx_full(y_, buf);
  }

  static const Matrix3Xd &x() { return x_; }
  static const ArrayXc &secx() { return secx_; }

  static const Matrix3Xd &y() { return y_; }
  static const ArrayXc &secy() { return secy_; }

private:
  inline static Matrix3Xd x_ { 3, lx }, y_ { 3, ly };
  inline static ArrayXc secx_, secy_;
};

class TMAlignInitTest: public TMAlignTestBase {
public:
  void SetUp() override {
    y2x_.setConstant(ly, -1);
    y2x_ref_.setConstant(ly, lx);

    rx_.resize(3, lx);
    ry_.resize(3, ly);
    dsqs_.resize(l_min);

    path_.resize(ly + 1, lx + 1);
    path_.col(0).fill(kPathHorz);
    path_.row(0).fill(kPathVert);

    val_.resize(ly + 1, lx + 1);
    val_.col(0).fill(0);
    val_.row(0).fill(0);
  }

  void TearDown() override {
    for (int i = 0; i < ly; ++i)
      EXPECT_EQ(y2x_[i], y2x_ref_[i]);

    EXPECT_TRUE((path_.col(0).tail(ly) == kPathHorz).all());
    EXPECT_TRUE((path_.row(0) == kPathVert).all());

    EXPECT_TRUE((val_.col(0) == 0).all());
    EXPECT_TRUE((val_.row(0) == 0).all());
  }

  ArrayXi y2x_, y2x_ref_;

  Matrix3Xd rx_, ry_;
  ArrayXd dsqs_;
  ArrayXXc path_;
  ArrayXXd val_;

  AlignedXY xy_ { x(), y(), lx };
};

TEST_F(TMAlignInitTest, GaplessThreading) {
  double tmscore =
      tm_initial_gt(rx_, ry_, dsqs_, x(), y(), y2x_, kD0SqInv, kD0SearchSq);
  EXPECT_NEAR(tmscore, 3.2337282632269364, 1e-5);

  y2x_ref_ << 9, 10, 11, 12, 13,  //
      14, 15, 16, 17, 18,         //
      -1, -1, -1, -1, -1,         //
      -1, -1, -1, -1, -1;
}

TEST_F(TMAlignInitTest, SecStr) {
  tm_initial_ss(y2x_, path_, val_, secx(), secy());

  y2x_ref_ << -1, 0, 1, 2, 3,  //
      4, 5, 6, 7, 8,           //
      9, 10, 11, 12, 13,       //
      14, 15, 16, 17, 18;
}

TEST_F(TMAlignInitTest, Local) {
  ArrayXi buf(ly);
  double tmscore = tm_initial_local(rx_, ry_, dsqs_, path_, val_, xy_, y2x_,
                                    buf, kD0SqInv, kD01SqInv, kD0SearchSq);
  EXPECT_NEAR(tmscore, 4.0621778055231612, 1e-5);

  y2x_ref_ << 12, 13, 14, 15, 16,  //
      17, 18, -1, -1, -1,          //
      -1, -1, -1, -1, -1,          //
      -1, -1, -1, -1, -1;
}

TEST_F(TMAlignInitTest, LocalPlusSs) {
  ArrayXi xy0(ly);
  xy0 << -1, 0, 1, 2, 3,  //
      4, -1, 5, 6, 7,     //
      8, 9, 10, 12, 13,   //
      14, 15, 16, 17, 18;
  xy_.swap_remap(xy0);

  bool success = tm_initial_ssplus(rx_, ry_, path_, val_, xy_, y2x_, secx(),
                                   secy(), kD01SqInv);
  EXPECT_TRUE(success);

  y2x_ref_ << -1, 0, 1, 2, 3,  //
      4, 5, 6, 7, 8,           //
      9, 10, 11, 12, 13,       //
      14, 15, 16, 17, 18;
}

TEST_F(TMAlignInitTest, FragmentGTAsymm) {
  double tmscore = tm_initial_fgt(rx_, ry_, dsqs_, x(), y(), y2x_, kDCu0Sq,
                                  kD0SqInv, kD0SearchSq);
  EXPECT_NEAR(tmscore, 5.8222822101865486, 1e-5);

  y2x_ref_ << 11, 12, 13, 14, 15,  //
      16, 17, 18, -1, -1,          //
      -1, -1, -1, -1, -1,          //
      -1, -1, -1, -1, -1;
}

TEST_F(TMAlignInitTest, FragmentGTSymm) {
  constexpr double d0_sq_inv = 1 / (1.1203701738791039 * 1.1203701738791039);

  Matrix3Xd x_same(3, 20), y_same(3, 20);
  x_same.transpose() << 35.308, 102.724, 26.297,  //
      34.682, 106.421, 27.027,                    //
      35.169, 109.907, 25.557,                    //
      33.947, 113.167, 27.045,                    //
      33.037, 116.635, 25.787,                    //
      32.303, 119.994, 27.4,                      //
      29.802, 122.397, 25.806,                    //
      28.392, 125.929, 26.413,                    //
      24.901, 125.011, 25.14,                     //
      23.872, 128.617, 24.324,                    //
      26.951, 128.515, 22.081,                    //
      25.558, 125.578, 20.087,                    //
      22.079, 126.751, 19.129,                    //
      21.964, 125.568, 15.469,                    //
      18.246, 125.572, 14.775,                    //
      15.919, 123.639, 12.566,                    //
      14.856, 125.044, 9.265,                     //
      12.216, 127.83, 9.293,                      //
      8.768, 127.083, 7.913,                      //
      5.361, 128.129, 6.619;
  y_same.transpose() << -4.808, 4.512, -8.76,  //
      -3.433, 7.685, -10.341,                  //
      -3.571, 10.037, -7.356,                  //
      -1.769, 10.802, -4.094,                  //
      -4.837, 9.827, -2.067,                   //
      -7.244, 6.893, -2.307,                   //
      -10.079, 5.637, -0.105,                  //
      -10.862, 1.935, 0.268,                   //
      -14.464, 1.303, 1.313,                   //
      -16.302, -1.945, 1.986,                  //
      -14.274, -3.079, 4.989,                  //
      -14.998, -4.607, 8.396,                  //
      -15.256, -2.612, 11.625,                 //
      -12.352, -2.674, 14.081,                 //
      -10.408, -4.861, 11.652,                 //
      -6.756, -4.474, 10.665,                  //
      -5.672, -4.02, 7.047,                    //
      -2.221, -3.864, 5.457,                   //
      -1.339, -1.297, 2.792,                   //
      1.635, -0.812, 0.473;

  rx_.resize(3, 20);
  ry_.resize(3, 20);
  dsqs_.resize(20);

  double tmscore = tm_initial_fgt(rx_, ry_, dsqs_, x_same, y_same, y2x_,
                                  kDCu0Sq, d0_sq_inv, kD0SearchSq);
  EXPECT_NEAR(tmscore, 5.7017861402047538, 1e-5);

  y2x_ref_ << -1, -1, -1, -1, -1,  //
      -1, -1, -1, -1, -1,          //
      -1, -1, -1, -1, 2,           //
      3, 4, 5, 6, 7;
  for (int i = 0; i < 20; ++i)
    EXPECT_EQ(y2x_[i], y2x_ref_[i]);

  tmscore = tm_initial_fgt(rx_, ry_, dsqs_, y_same, x_same, y2x_, kDCu0Sq,
                           d0_sq_inv, kD0SearchSq);
  EXPECT_NEAR(tmscore, 5.7017861402047538, 1e-5);

  y2x_ref_ << -1, -1, 14, 15, 16,  //
      17, 18, 19, -1, -1,          //
      -1, -1, -1, -1, -1,          //
      -1, -1, -1, -1, -1;
}

class TMAlignTest: public TMAlignTestBase { };

TEST_F(TMAlignTest, InitAll) {
  TMAlign tm_align(x(), y());

  bool success = tm_align.initialize();
  EXPECT_TRUE(success);
  EXPECT_TRUE(tm_align.initialized());

  Affine3d xform_ref;
  xform_ref.matrix() << 0.234082, -0.966243, -0.107608, 128.475,  //
      -0.563769, -0.225079, 0.794672, 36.6465,                    //
      -0.792067, -0.125353, -0.597425, 29.295,                    //
      0, 0, 0, 1;

  auto [xform, tmscore] = tm_align.tm_score(ly);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.linear(), xform_ref.linear(), 1e-5);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.translation(), xform_ref.translation(), 1e-2);
  EXPECT_NEAR(tmscore, 0.19080501444867484, 1e-6);

  std::tie(xform, tmscore) = tm_align.tm_score(lx);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.linear(), xform_ref.linear(), 1e-5);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.translation(), xform_ref.translation(), 1e-2);
  EXPECT_NEAR(tmscore, 0.20084738363018403, 1e-6);

  std::tie(xform, tmscore) = tm_align.tm_score(7);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.linear(), xform_ref.linear(), 1e-5);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.translation(), xform_ref.translation(), 1e-2);
  EXPECT_NEAR(tmscore, 0.54515718413907099, 1e-6);

  xform_ref.matrix() << 0.10261, -0.986975, 0.123899, 130.978,  //
      -0.599276, 0.0380783, 0.799636, -0.10831,                 //
      -0.793939, -0.156301, -0.587564, 32.9484,                 //
      0, 0, 0, 1;

  std::tie(xform, tmscore) = tm_align.tm_score(420);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.linear(), xform_ref.linear(), 1e-5);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.translation(), xform_ref.translation(), 1e-2);
  EXPECT_NEAR(tmscore, 0.029094346551500484, 1e-7);

  std::tie(std::ignore, tmscore) = tm_align.tm_score(0);
  EXPECT_LT(tmscore, 0);

  EXPECT_EQ(tm_align.l_ali(), 13);

  ArrayXi y2x_ref(ly);
  y2x_ref << -1, 0, 1, 2, 3,  //
      4, -1, 5, 6, 7,         //
      8, -1, -1, -1, -1,      //
      -1, 15, 16, 17, 18;
  for (int i = 0; i < ly; ++i)
    EXPECT_EQ(tm_align.templ_to_query()[i], y2x_ref[i]);

  EXPECT_NEAR(tm_align.aligned_msd(), 3.6508720456913579, 1e-5);
}

TEST_F(TMAlignTest, InitSingle) {
  TMAlign tm_align(x(), y());

  bool success = tm_align.initialize(TMAlign::InitFlags::kGaplessThreading);
  EXPECT_TRUE(success);
  EXPECT_TRUE(tm_align.initialized());

  Affine3d xform_ref;
  xform_ref.matrix() << 0.587457, -0.290603, -0.755278, 42.584,  //
      0.268044, 0.95049, -0.157228, -121.782,                    //
      0.763575, -0.110083, 0.636266, -10.1525,                   //
      0, 0, 0, 1;

  auto [xform, tmscore] = tm_align.tm_score(ly);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.linear(), xform_ref.linear(), 1e-5);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.translation(), xform_ref.translation(), 1e-2);
  EXPECT_NEAR(tmscore, 0.15233208382340996, 1e-6);

  std::tie(xform, tmscore) = tm_align.tm_score(lx);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.linear(), xform_ref.linear(), 1e-5);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.translation(), xform_ref.translation(), 1e-2);
  EXPECT_NEAR(tmscore, 0.16034956191937891, 1e-6);

  xform_ref.matrix() << 0.367803, -0.346261, -0.863032, 55.4027,  //
      0.673591, 0.739044, -0.00944722, -103.255,                  //
      0.64109, -0.577856, 0.505061, 57.9584,                      //
      0, 0, 0, 1;

  std::tie(xform, tmscore) = tm_align.tm_score(420);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.linear(), xform_ref.linear(), 1e-5);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.translation(), xform_ref.translation(), 1e-2);
  EXPECT_NEAR(tmscore, 0.017989322151398472, 1e-7);

  EXPECT_EQ(tm_align.l_ali(), 8);

  ArrayXi y2x_ref(ly);
  y2x_ref << -1, -1, 11, 12, 13,  //
      14, 15, 16, 17, 18,         //
      -1, -1, -1, -1, -1,         //
      -1, -1, -1, -1, -1;

  for (int i = 0; i < ly; ++i)
    EXPECT_EQ(tm_align.templ_to_query()[i], y2x_ref[i]);

  EXPECT_NEAR(tm_align.aligned_msd(), 3.5031183875971266, 1e-5);
}

TEST_F(TMAlignTest, InitNoneOrFail) {
  TMAlign tm_align(x(), y());
  EXPECT_FALSE(tm_align.initialized());

  bool success = tm_align.initialize(TMAlign::InitFlags::kNone);
  EXPECT_FALSE(success);

  success = tm_align.initialize(TMAlign::InitFlags::kDefault, secy(), secx());
  EXPECT_FALSE(success);

  EXPECT_FALSE(tm_align.initialized());

  auto [_, tmscore] = tm_align.tm_score(ly);
  EXPECT_LT(tmscore, 0);
}

TEST_F(TMAlignTest, Wrapper) {
  Affine3d xform_ref;
  xform_ref.matrix() << 0.234082, -0.966243, -0.107608, 128.475,  //
      -0.563769, -0.225079, 0.794672, 36.6465,                    //
      -0.792067, -0.125353, -0.597425, 29.295,                    //
      0, 0, 0, 1;

  ArrayXi y2x_ref(ly);
  y2x_ref << -1, 0, 1, 2, 3,  //
      4, -1, 5, 6, 7,         //
      8, -1, -1, -1, -1,      //
      -1, 15, 16, 17, 18;

  auto [xform, y2x, msd, tmscore] = tm_align(x(), y());

  for (int i = 0; i < ly; ++i)
    EXPECT_EQ(y2x[i], y2x_ref[i]);

  NURI_EXPECT_EIGEN_EQ_TOL(xform.linear(), xform_ref.linear(), 1e-5);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.translation(), xform_ref.translation(), 1e-2);

  EXPECT_NEAR(msd, 3.6508720456913579, 1e-5);

  EXPECT_NEAR(tmscore, 0.19080501444867484, 1e-6);
}

TEST_F(TMAlignTest, InitUserWrapper) {
  ArrayXi y2x(ly);
  y2x << -1, 0, 1, 2, 3,  //
      4, -1, 5, 6, 7,     //
      8, 9, 10, 12, 13,   //
      14, 15, 16, 17, 18;

  Affine3d xform_ref;
  xform_ref.matrix() << 0.234082, -0.966243, -0.107608, 128.475,  //
      -0.563769, -0.225079, 0.794672, 36.6465,                    //
      -0.792067, -0.125353, -0.597425, 29.295,                    //
      0, 0, 0, 1;

  auto [xform, y2x_out, msd, tmscore] = tm_align(x(), y(), y2x);
  EXPECT_NEAR(tmscore, 0.19080501444867484, 1e-6);

  NURI_EXPECT_EIGEN_EQ_TOL(xform.linear(), xform_ref.linear(), 1e-5);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.translation(), xform_ref.translation(), 1e-2);

  y2x << -1, 0, 1, 2, 3,  //
      4, -1, 5, 6, 7,     //
      8, -1, -1, -1, -1,  //
      -1, 15, 16, 17, 18;
  for (int i = 0; i < ly; ++i)
    EXPECT_EQ(y2x_out[i], y2x[i]);

  EXPECT_NEAR(msd, 3.6508720456913579, 1e-5);
}

}  // namespace
}  // namespace internal
}  // namespace nuri
