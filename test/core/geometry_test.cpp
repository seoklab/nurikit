//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/geometry.h"

#include <algorithm>
#include <vector>

#include <absl/strings/str_cat.h>
#include <Eigen/Dense>

#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "test_utils.h"
#include "nuri/utils.h"

namespace nuri {
namespace {
// Implementation detail, but should be exposed for testing

const std::array<Vector3d, 8> kOctantMasks = []() {
  std::array<Vector3d, 8> masks;
  for (int i = 0; i < 8; ++i) {
    auto mask_of = [&](int axis) {
      return static_cast<double>(static_cast<bool>(i & (0b100 >> axis)));
    };

    masks[i][0] = mask_of(0);
    masks[i][1] = mask_of(1);
    masks[i][2] = mask_of(2);
  }
  return masks;
}();

Vector3d max_of(int octant, const Vector3d &max, const Vector3d &size) {
  return max - size.cwiseProduct(kOctantMasks[octant]);
}

void verify_oct_range(int idx, const OCTree &oct, const Vector3d &max,
                      const Vector3d &size) {
  const auto &node = oct[idx];
  if (node.leaf()) {
    for (int i = 0; i < node.nleaf(); ++i) {
      const Vector3d &pt = oct.pts().col(node[i]);
      EXPECT_TRUE((pt.array() <= max.array() + 1e6).all()) << "node = " << idx;
      EXPECT_TRUE((pt.array() >= (max - size).array() - 1e6).all())
          << "node = " << idx;
    }
    return;
  }

  Vector3d half = size / 2;
  for (int i = 0; i < 8; ++i) {
    if (node[i] < 0)
      continue;
    verify_oct_range(node[i], oct, max_of(i, max, half), half);
  }
}

TEST(OCTreeTest, Create) {
  Matrix3Xd m = Matrix3Xd::Random(3, 500);
  OCTree tree(m);

  std::vector<int> childs { tree.root() }, idxs;
  for (int i = 0; i < tree.size(); ++i) {
    const auto &node = tree.nodes()[i];
    if (node.leaf()) {
      for (int j = 0; j < node.nleaf(); ++j)
        idxs.push_back(node.children()[j]);
    } else {
      for (int c: node.children())
        if (c >= 0)
          childs.push_back(c);
    }
  }

  verify_oct_range(tree.root(), tree, tree.max(), tree.len());

  ASSERT_EQ(childs.size(), tree.size());
  std::sort(childs.begin(), childs.end());
  for (int i = 0; i < childs.size(); ++i)
    EXPECT_EQ(childs[i], i) << "i = " << i;

  ASSERT_EQ(idxs.size(), m.cols());
  std::sort(idxs.begin(), idxs.end());
  for (int i = 0; i < idxs.size(); ++i)
    EXPECT_EQ(idxs[i], i) << "i = " << i;
}

void verify_octree_neighbor_k(const OCTree &tree, const Vector3d &qry,
                              const int n) {
  const Matrix3Xd &pts = tree.pts();

  std::vector<int> answer(n, -1);
  std::vector<double> dsq_ref(n, 1e+10);
  for (int i = 0; i < pts.cols(); ++i) {
    double dsq = (pts.col(i) - qry).squaredNorm();
    for (int j = 0; j < n; ++j) {
      if (dsq < dsq_ref[j]) {
        for (int k = n - 1; k > j; --k) {
          answer[k] = answer[k - 1];
          dsq_ref[k] = dsq_ref[k - 1];
        }
        answer[j] = i;
        dsq_ref[j] = dsq;
        break;
      }
    }
  }

  ASSERT_TRUE(std::is_sorted(dsq_ref.begin(), dsq_ref.end()));

  std::vector<int> idxs;
  std::vector<double> distsq;
  tree.find_neighbors_k(qry, n, idxs, distsq);
  ASSERT_EQ(idxs.size(), answer.size());

  for (int i = 0; i < idxs.size(); ++i) {
    EXPECT_EQ(idxs[i], answer[i]) << "i = " << i;
    EXPECT_DOUBLE_EQ(distsq[i], dsq_ref[i]) << "i = " << i;
  }

  EXPECT_TRUE(std::is_sorted(distsq.begin(), distsq.end()));
}

TEST(OCTreeTest, FindNeighborByCount) {
  Matrix3Xd m = Matrix3Xd::Random(3, 500);
  Matrix3Xd test = Matrix3Xd::Random(3, 50);
  int k = 10;

  OCTree tree(m);
  for (int i = 0; i < test.cols(); ++i)
    verify_octree_neighbor_k(tree, test.col(i), k);
}

void verify_octree_neighbor_d(const OCTree &tree, const Vector3d &qry,
                              double cutoff) {
  const Matrix3Xd &pts = tree.pts();

  std::vector<int> answer;
  for (int i = 0; i < pts.cols(); ++i)
    if ((pts.col(i) - qry).squaredNorm() <= cutoff * cutoff)
      answer.push_back(i);

  ASSERT_FALSE(answer.empty());

  std::vector<int> idxs;
  std::vector<double> distsq;
  tree.find_neighbors_d(qry, cutoff, idxs, distsq);
  ASSERT_EQ(idxs.size(), answer.size());

  std::sort(idxs.begin(), idxs.end());
  for (int i = 0; i < idxs.size(); ++i)
    EXPECT_EQ(idxs[i], answer[i]) << "i = " << i;
}

TEST(OCTreeTest, FindNeighborByDistance) {
  Matrix3Xd m = Matrix3Xd::Random(3, 500);
  Matrix3Xd test = Matrix3Xd::Random(3, 50);
  double cutoff = 0.5;

  OCTree tree(m);
  for (int i = 0; i < test.cols(); ++i)
    verify_octree_neighbor_d(tree, test.col(i), cutoff);
}

void verify_octree_neighbor_kd(const OCTree &tree, const Vector3d &qry,
                               const int n, const double cutoff,
                               std::string_view onerr) {
  const Matrix3Xd &pts = tree.pts();
  const double cutoffsq = cutoff * cutoff;

  std::vector<int> answer(n, -1);
  std::vector<double> dsq_ref(n, 1e+10);
  for (int i = 0; i < pts.cols(); ++i) {
    double dsq = (pts.col(i) - qry).squaredNorm();
    for (int j = 0; j < n; ++j) {
      if (dsq < dsq_ref[j] && dsq <= cutoffsq) {
        for (int k = n - 1; k > j; --k) {
          answer[k] = answer[k - 1];
          dsq_ref[k] = dsq_ref[k - 1];
        }
        answer[j] = i;
        dsq_ref[j] = dsq;
        break;
      }
    }
  }

  erase_if(answer, [](int idx) { return idx < 0; });
  dsq_ref.resize(answer.size());

  ASSERT_TRUE(std::is_sorted(dsq_ref.begin(), dsq_ref.end())) << onerr;

  std::vector<int> idxs;
  std::vector<double> distsq;
  tree.find_neighbors_kd(qry, n, cutoff, idxs, distsq);
  ASSERT_EQ(idxs.size(), answer.size()) << onerr;

  for (int i = 0; i < idxs.size(); ++i) {
    EXPECT_EQ(idxs[i], answer[i]) << onerr << ", i = " << i;
    EXPECT_DOUBLE_EQ(distsq[i], dsq_ref[i]) << onerr << ", i = " << i;
  }

  EXPECT_TRUE(std::is_sorted(distsq.begin(), distsq.end())) << onerr;
}

TEST(OCTreeTest, FindNeighborByCountAndDistance) {
  Matrix3Xd m = Matrix3Xd::Random(3, 500);
  Matrix3Xd test = Matrix3Xd::Random(3, 50);
  int k = 10;
  // 5 neighbors in average
  double cutoff = std::pow(2.0 * 5 / 500, 1.0 / 3.0);

  OCTree tree(m);
  for (int i = 0; i < test.cols(); ++i)
    verify_octree_neighbor_kd(tree, test.col(i), k, cutoff,
                              absl::StrCat("On line ", __LINE__));

  // 15 neighbors in average
  cutoff = std::pow(2.0 * 15 / 500, 1.0 / 3.0);
  for (int i = 0; i < test.cols(); ++i)
    verify_octree_neighbor_kd(tree, test.col(i), k, cutoff,
                              absl::StrCat("On line ", __LINE__));
}

TEST(FitPlaneTest, CheckCorrectness) {
  Matrix3Xd m(3, 4);
  m.transpose() << 0.83204366, 0.51745906, 0.2127645,  //
      0.1541316, 0.34186033, 0.37958696,               //
      0.09409007, 0.55596287, 0.84068561,              //
      0.16504374, 0.44050598, 0.63581156;

  // From numpy.linalg.svd
  Vector4d ans(0.33410998, -0.84014984, 0.42722215, 0.06480301458722172);
  Vector4d p = fit_plane(m);
  EXPECT_TRUE(p.isApprox(ans, 1e-6));
}

class AlignTest: public ::testing::Test {
public:
  // Random rotation matrix
  inline static const Affine3d xform_ {
    Matrix4d { { 0.82743922, 0.54630659, 0.12997477, -10 },
              { 0.05113706, 0.15719006, -0.98624352, 20 },
              { -0.55922208, 0.8227031, 0.10212871, 30 },
              { 0, 0, 0, 1 } },
  };

  // reflected
  inline static const Matrix3Xd query_ =
      -1
      * MatrixX3d {
          { -18.3397, 72.5541, 64.7727 },
          { -17.5457, 72.7646, 65.8591 },
          { -17.5263, 71.9973, 66.9036 },
          { -18.3203, 70.8982, 66.9881 },
          { -19.2153, 70.5869, 65.8606 },
          { -19.1851, 71.4993, 64.7091 },
          { -19.8932, 71.3140, 63.7366 },
          { -19.8586, 69.4953, 66.1942 },
          { -19.4843, 69.0910, 67.3570 },
          { -18.5535, 69.9074, 67.8928 },
          { -17.8916, 69.7493, 69.2373 },
          { -16.7519, 70.7754, 69.3878 },
          { -15.5030, 70.1100, 69.5871 },
          { -17.1385, 71.5903, 70.6459 },
          { -15.9909, 71.8532, 71.4558 },
          { -18.1166, 70.6286, 71.3678 },
          { -18.8422, 70.0100, 70.2837 },
          { -19.0679, 71.4098, 72.2764 },
          { -20.1784, 71.8813, 71.5104 },
  }.transpose();

  // transformed
  inline static const Matrix3Xd templ_ =
      xform_
      * MatrixX3d {
          {-18.6290, 72.4960, 64.7810},
          {-17.6590, 72.8430, 65.6920},
          {-17.4340, 72.1880, 66.8280},
          {-18.2650, 71.1350, 66.9710},
          {-19.2640, 70.7080, 66.1180},
          {-19.4960, 71.4120, 64.9200},
          {-20.3350, 71.1770, 64.0460},
          {-19.9000, 69.5840, 66.6230},
          {-19.2890, 69.3560, 67.7600},
          {-18.2820, 70.2630, 68.0340},
          {-17.5510, 69.7970, 69.2390},
          {-16.4120, 70.7300, 69.6340},
          {-15.2040, 69.9950, 69.6560},
          {-16.8460, 71.3300, 70.9950},
          {-16.1160, 70.8000, 72.1220},
          {-18.3190, 70.9540, 71.1130},
          {-18.4590, 69.7460, 70.3290},
          {-19.2870, 72.0220, 70.6090},
          {-20.6560, 71.5800, 70.5560},
  }.transpose();

  // Below all calculated manually
  inline static const double msd_ = 0.4516332;
  constexpr static double msd_reflected_ = 3.9094217;
  inline static const Affine3d xform_reflected_ {
    Matrix4d { { -0.994979, 0.091884, -0.0396638, 44.5055 },
              { -0.0531631, -0.149479, 0.987335, 20.6586 },
              { 0.0847914, 0.984486, 0.153613, 184.346 },
              { 0, 0, 0, 1 } },
  };
};

TEST_F(AlignTest, KabschMSDOnly) {
  auto [_, msd] = kabsch(query_, templ_, AlignMode::kMsdOnly);
  EXPECT_NEAR(msd, msd_reflected_, 1e-6);

  std::tie(_, msd) = kabsch(query_, templ_, AlignMode::kMsdOnly, true);
  EXPECT_NEAR(msd, msd_, 1e-6);
}

TEST_F(AlignTest, KabschXformOnly) {
  auto [xform, flag] = kabsch(query_, templ_, AlignMode::kXformOnly);
  ASSERT_GE(flag, 0);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.matrix(), xform_reflected_.matrix(), 1e-3);

  std::tie(xform, flag) = kabsch(query_, templ_, AlignMode::kXformOnly, true);
  ASSERT_GE(flag, 0);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.linear(), -xform_.linear(), 1e-3);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.translation(), xform_.translation(), 1e-3);
}

TEST_F(AlignTest, KabschBoth) {
  auto [xform, msd] = kabsch(query_, templ_, AlignMode::kBoth);
  ASSERT_GE(msd, 0);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.matrix(), xform_reflected_.matrix(), 1e-3);
  EXPECT_NEAR(msd, msd_reflected_, 1e-6);

  std::tie(xform, msd) = kabsch(query_, templ_, AlignMode::kBoth, true);
  ASSERT_GE(msd, 0);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.linear(), -xform_.linear(), 1e-3);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.translation(), xform_.translation(), 1e-3);
  EXPECT_NEAR(msd, msd_, 1e-6);
}

TEST_F(AlignTest, QcpMSDOnly) {
  auto [_, msd] = qcp(query_, templ_, AlignMode::kMsdOnly);
  EXPECT_NEAR(msd, msd_reflected_, 1e-6);

  std::tie(_, msd) = qcp(query_, templ_, AlignMode::kMsdOnly, true);
  EXPECT_NEAR(msd, msd_, 1e-6);
}

TEST_F(AlignTest, QcpXformOnly) {
  auto [xform, flag] = qcp(query_, templ_, AlignMode::kXformOnly);
  ASSERT_GE(flag, 0);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.matrix(), xform_reflected_.matrix(), 1e-3);

  std::tie(xform, flag) = qcp(query_, templ_, AlignMode::kXformOnly, true);
  ASSERT_GE(flag, 0);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.linear(), -xform_.linear(), 1e-3);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.translation(), xform_.translation(), 1e-3);
}

TEST_F(AlignTest, QcpBoth) {
  auto [xform, msd] = qcp(query_, templ_, AlignMode::kBoth);
  ASSERT_GE(msd, 0);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.matrix(), xform_reflected_.matrix(), 1e-3);
  EXPECT_NEAR(msd, msd_reflected_, 1e-6);

  std::tie(xform, msd) = qcp(query_, templ_, AlignMode::kBoth, true);
  ASSERT_GE(msd, 0);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.linear(), -xform_.linear(), 1e-3);
  NURI_EXPECT_EIGEN_EQ_TOL(xform.translation(), xform_.translation(), 1e-3);
  EXPECT_NEAR(msd, msd_, 1e-6);
}

class AlignSingularTest: public ::testing::Test {
public:
  template <class TFunc>
  static void run_test(TFunc f) {
    Matrix3Xd query(3, 20), templ(3, 20);

    for (int axis = 0; axis < 3; ++axis) {
      query.setZero();
      query.row(axis).setRandom();
      query.row(axis).array() -= query.row(axis).mean();
      double msd_ref = query.row(axis).array().square().mean();

      for (int i = 1; i < 3; ++i) {
        Affine3d xform_ref =
            Affine3d::Identity()
            * Eigen::AngleAxisd(constants::kPi, Vector3d::Unit((axis + i) % 3));

        templ.noalias() = 2 * (xform_ref * query);

        auto [xform, msd] = f(query, templ, AlignMode::kBoth);
        ASSERT_GE(msd, 0) << "axis = " << axis << ", i = " << i;

        EXPECT_NEAR(msd, msd_ref, 1e-4) << "axis = " << axis << ", i = " << i;
        NURI_EXPECT_EIGEN_EQ_TOL(2 * (xform * query), templ, 1e-4)
            << "axis = " << axis << ", i = " << i;
      }

      const int a2 = (axis + 1) % 3, a3 = (axis + 2) % 3;

      query.row(a2).setRandom();
      query.row(a2).array() -= query.row(a2).mean();
      msd_ref += query.row(a2).array().square().mean();

      Affine3d xform_ref =
          Affine3d::Identity()
          * Eigen::AngleAxisd(constants::kPi, Vector3d::Unit(a2))
          * Eigen::AngleAxisd(constants::kPi, Vector3d::Unit(a3));

      templ.noalias() = 2 * (xform_ref * query);

      auto [xform, msd] = f(query, templ, AlignMode::kBoth);
      ASSERT_GE(msd, 0) << "axis = " << axis << ", both";

      EXPECT_NEAR(msd, msd_ref, 1e-4) << "axis = " << axis << ", both";
      NURI_EXPECT_EIGEN_EQ_TOL(2 * (xform * query), templ, 1e-4)
          << "axis = " << axis << ", both";
    }
  }
};

TEST_F(AlignSingularTest, Kabsch) {
  run_test([](const auto &q, const auto &t, auto mode) {
    return kabsch(q, t, mode);
  });
}

TEST_F(AlignSingularTest, Qcp) {
  run_test([](const auto &q, const auto &t, auto mode) {
    // might fail on optimized builds if evecprec is too high
    return qcp(q, t, mode, false, 1e-11, 1e-8);
  });
}

TEST(AlignFewPointsTest, ZeroPoints) {
  Matrix3Xd query(3, 0), templ(3, 0);

  {
    auto [xform, msd] = kabsch(query, templ, AlignMode::kBoth);
    EXPECT_TRUE(xform.matrix().isIdentity());
    EXPECT_EQ(msd, 0);
  }

  {
    auto [xform, msd] = qcp(query, templ, AlignMode::kBoth);
    EXPECT_TRUE(xform.matrix().isIdentity());
    EXPECT_EQ(msd, 0);
  }
}

TEST(AlignFewPointsTest, OnePoint) {
  Matrix3Xd query = Matrix3Xd::Random(3, 1), templ = Matrix3Xd::Random(3, 1);

  {
    auto [xform, msd] = kabsch(query, templ, AlignMode::kBoth);
    EXPECT_TRUE(xform.linear().isIdentity());
    NURI_EXPECT_EIGEN_EQ(xform.translation(), templ.col(0) - query.col(0));
    EXPECT_EQ(msd, 0);
  }

  {
    auto [xform, msd] = qcp(query, templ, AlignMode::kBoth);
    EXPECT_TRUE(xform.linear().isIdentity());
    NURI_EXPECT_EIGEN_EQ(xform.translation(), templ.col(0) - query.col(0));
    EXPECT_EQ(msd, 0);
  }
}

TEST(AlignFewPointsTest, TwoPoint) {
  Matrix3Xd query = Matrix3Xd::Random(3, 2), templ(3, 2);
  templ.noalias() = AlignTest::xform_ * query;

  {
    auto [xform, msd] = kabsch(query, templ, AlignMode::kBoth);
    EXPECT_NEAR(msd, 0, 1e-6);
    NURI_EXPECT_EIGEN_EQ_TOL(xform * query, templ, 1e-6);
  }

  {
    // qcp fails on two points
    auto [xform, msd] = qcp(query, templ, AlignMode::kMsdOnly);
    EXPECT_NEAR(msd, 0, 1e-6);
  }
}

TEST(EmbedTest, FromDistance) {
  Matrix3Xd orig(3, 19);
  orig.transpose() << -18.3397, 72.5541, 64.7727,  //
      -17.5457, 72.7646, 65.8591,                  //
      -17.5263, 71.9973, 66.9036,                  //
      -18.3203, 70.8982, 66.9881,                  //
      -19.2153, 70.5869, 65.8606,                  //
      -19.1851, 71.4993, 64.7091,                  //
      -19.8932, 71.3140, 63.7366,                  //
      -19.8586, 69.4953, 66.1942,                  //
      -19.4843, 69.0910, 67.3570,                  //
      -18.5535, 69.9074, 67.8928,                  //
      -17.8916, 69.7493, 69.2373,                  //
      -16.7519, 70.7754, 69.3878,                  //
      -15.5030, 70.1100, 69.5871,                  //
      -17.1385, 71.5903, 70.6459,                  //
      -15.9909, 71.8532, 71.4558,                  //
      -18.1166, 70.6286, 71.3678,                  //
      -18.8422, 70.0100, 70.2837,                  //
      -19.0679, 71.4098, 72.2764,                  //
      -20.1784, 71.8813, 71.5104;

  MatrixXd dsqs = to_square_form(pdistsq(orig), orig.cols());
  Matrix3Xd pts(orig.rows(), orig.cols());
  ASSERT_TRUE(embed_distances_3d(pts, dsqs));

  auto [xform, msd] = kabsch(pts, orig, AlignMode::kBoth, true);
  ASSERT_GE(msd, 0);
  EXPECT_NEAR(msd, 0, 1e-6);
}
}  // namespace
}  // namespace nuri
