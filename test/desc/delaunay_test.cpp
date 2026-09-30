//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>

#include <geogram/delaunay/delaunay_3d.h>

#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "nuri/random.h"

namespace nuri {
namespace {
using GEO::index_t;
using GEO::NO_INDEX;

Matrix3Xd cube_with_center() {
  Matrix3Xd pts(3, 9);
  for (int i = 0; i < 8; ++i)
    for (int k = 0; k < 3; ++k)
      pts(k, i) = (i & (1 << k)) != 0 ? 1.0 : -1.0;
  pts.col(8).setZero();
  return pts;
}

Matrix4Xd lift(const Matrix3Xd &pts, const ArrayXd &weights) {
  const double wmax = weights.maxCoeff();

  Matrix4Xd lifted(4, pts.cols());
  lifted.topRows(3) = pts;
  lifted.row(3) = (wmax - weights).sqrt().transpose();
  return lifted;
}

void check_closed(const GEO::Delaunay3d &del, const int n_vertices) {
  std::vector<bool> seen(n_vertices, false);
  for (index_t c = 0; c < del.nb_cells(); ++c) {
    for (index_t lv = 0; lv < 4; ++lv) {
      const index_t v = del.cell_vertex(c, lv);
      if (v != NO_INDEX)
        seen[v] = true;

      const index_t c2 = del.cell_adjacent(c, lv);
      ASSERT_NE(c2, NO_INDEX);
      ASSERT_EQ(del.cell_adjacent(c2, del.adjacent_index(c2, c)), c);
    }
  }

  for (int v = 0; v < n_vertices; ++v)
    EXPECT_TRUE(seen[v]) << "vertex " << v << " is in no cell";
}

TEST(DelaunayTest, CubeWithCenter) {
  const Matrix3Xd pts = cube_with_center();

  GEO::Delaunay3d del(3);
  del.set_keeps_infinite(true);
  del.set_vertices(9, pts.data());

  EXPECT_EQ(del.nb_finite_cells(), 12);
  EXPECT_EQ(del.nb_cells(), 24);
  check_closed(del, 9);
}

TEST(DelaunayTest, EqualWeightsMatchUnweighted) {
  const Matrix3Xd pts = cube_with_center();
  const Matrix4Xd lifted = lift(pts, ArrayXd::Constant(9, 0.25));

  GEO::Delaunay3d del(4);
  del.set_keeps_infinite(true);
  del.set_vertices(9, lifted.data());

  EXPECT_EQ(del.nb_finite_cells(), 12);
  EXPECT_EQ(del.nb_cells(), 24);
  check_closed(del, 9);
}

TEST(DelaunayTest, HeavyPointHidesLightNeighbor) {
  Matrix3Xd pts(3, 10);
  pts.leftCols(9) = cube_with_center();
  pts.col(9) << 0.1, 0.0, 0.0;

  ArrayXd weights = ArrayXd::Zero(10);
  weights[8] = 1.0;
  const Matrix4Xd lifted = lift(pts, weights);

  GEO::Delaunay3d del(4);
  del.set_keeps_infinite(true);
  del.set_vertices(10, lifted.data());

  EXPECT_EQ(del.nb_finite_cells(), 12);
  bool light_seen = false, heavy_seen = false;
  for (index_t c = 0; c < del.nb_cells(); ++c) {
    for (index_t lv = 0; lv < 4; ++lv) {
      light_seen |= del.cell_vertex(c, lv) == 9;
      heavy_seen |= del.cell_vertex(c, lv) == 8;
    }
  }
  EXPECT_FALSE(light_seen);
  EXPECT_TRUE(heavy_seen);
}

TEST(DelaunayTest, RandomWeightedIsClosed) {
  internal::seed_thread(42);

  const int n = 200;
  Matrix3Xd pts(3, n);
  ArrayXd weights(n);
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < 3; ++k)
      pts(k, i) = internal::draw_urd(10.0);
    weights[i] = internal::draw_urd(0.5);
  }
  const Matrix4Xd lifted = lift(pts, weights);

  GEO::Delaunay3d del(4);
  del.set_keeps_infinite(true);
  del.set_vertices(n, lifted.data());

  EXPECT_GT(del.nb_finite_cells(), n);
  check_closed(del, n);

  for (index_t c = 0; c < del.nb_finite_cells(); ++c) {
    for (index_t lv = 0; lv < 4; ++lv) {
      const index_t v = del.cell_vertex(c, lv);
      ASSERT_NE(v, NO_INDEX);
      ASSERT_LT(v, n);
    }
  }
}
}  // namespace
}  // namespace nuri
