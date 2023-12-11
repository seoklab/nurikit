//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/geometry.h"

#include <algorithm>
#include <vector>

#include <absl/strings/str_cat.h>
#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
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
      const Vector3d &pt = oct.pts().row(node[i]);
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
  MatrixX3d m = MatrixX3d::Random(500, 3);
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

  ASSERT_EQ(idxs.size(), m.rows());
  std::sort(idxs.begin(), idxs.end());
  for (int i = 0; i < idxs.size(); ++i)
    EXPECT_EQ(idxs[i], i) << "i = " << i;
}

void verify_octree_neighbor_k(const OCTree &tree, const Vector3d &qry,
                              const int n) {
  const MatrixX3d &pts = tree.pts();

  std::vector<int> answer(n, -1);
  std::vector<double> dsq_ref(n, 1e+10);
  for (int i = 0; i < pts.rows(); ++i) {
    double dsq = (pts.row(i) - qry).squaredNorm();
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
  MatrixX3d m = MatrixX3d::Random(500, 3);
  MatrixX3d test = MatrixX3d::Random(50, 3);
  int k = 10;

  OCTree tree(m);
  for (int i = 0; i < test.rows(); ++i)
    verify_octree_neighbor_k(tree, test.row(i), k);
}

void verify_octree_neighbor_d(const OCTree &tree, const Vector3d &qry,
                              double cutoff) {
  const MatrixX3d &pts = tree.pts();

  std::vector<int> answer;
  for (int i = 0; i < pts.rows(); ++i)
    if ((pts.row(i) - qry).squaredNorm() <= cutoff * cutoff)
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
  MatrixX3d m = MatrixX3d::Random(500, 3);
  MatrixX3d test = MatrixX3d::Random(50, 3);
  double cutoff = 0.5;

  OCTree tree(m);
  for (int i = 0; i < test.rows(); ++i)
    verify_octree_neighbor_d(tree, test.row(i), cutoff);
}

void verify_octree_neighbor_kd(const OCTree &tree, const Vector3d &qry,
                               const int n, const double cutoff,
                               std::string_view onerr) {
  const MatrixX3d &pts = tree.pts();
  const double cutoffsq = cutoff * cutoff;

  std::vector<int> answer(n, -1);
  std::vector<double> dsq_ref(n, 1e+10);
  for (int i = 0; i < pts.rows(); ++i) {
    double dsq = (pts.row(i) - qry).squaredNorm();
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
  MatrixX3d m = MatrixX3d::Random(500, 3);
  MatrixX3d test = MatrixX3d::Random(50, 3);
  int k = 10;
  // 5 neighbors in average
  double cutoff = std::pow(2.0 * 5 / 500, 1.0 / 3.0);

  OCTree tree(m);
  for (int i = 0; i < test.rows(); ++i)
    verify_octree_neighbor_kd(tree, test.row(i), k, cutoff,
                              absl::StrCat("On line ", __LINE__));

  // 15 neighbors in average
  cutoff = std::pow(2.0 * 15 / 500, 1.0 / 3.0);
  for (int i = 0; i < test.rows(); ++i)
    verify_octree_neighbor_kd(tree, test.row(i), k, cutoff,
                              absl::StrCat("On line ", __LINE__));
}
}  // namespace
}  // namespace nuri
