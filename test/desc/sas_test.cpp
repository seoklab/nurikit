//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>

#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/desc/surface.h"

namespace nuri {
namespace internal {
namespace {
using constants::kPi;
using constants::kTwoPi;

constexpr double kRp = 1.0;

struct Sas {
  SaPrep sa;
  SasGeometry geo;
};

Sas solve(const Matrix3Xd &pts, const ArrayXd &sar, const ArrayXb &active) {
  std::optional<SaPrep> sa = prepare(pts, sar, active, kRp);
  EXPECT_TRUE(sa);
  SasDelaunay del = triangulate(*sa);
  SasGeometry geo = build_sas(*sa, del);
  return { std::move(*sa), std::move(geo) };
}

Sas solve(const Matrix3Xd &pts, const ArrayXd &sar) {
  return solve(pts, sar, ArrayXb::Constant(sar.size(), true));
}

double sr_total(const Matrix3Xd &pts, const ArrayXd &sar) {
  return sr_sasa_impl(pts, sar, 5000, SrSasaMethod::kDirect).sum();
}

TEST(BuildSasTest, SingleSphere) {
  Matrix3Xd pts = Matrix3Xd::Zero(3, 1);
  ArrayXd sar = ArrayXd::Constant(1, 2.5);

  auto [sa, geo] = solve(pts, sar);
  ASSERT_EQ(geo.area.size(), 1);
  EXPECT_DOUBLE_EQ(geo.area[0], 4 * kPi * 2.5 * 2.5);
  EXPECT_EQ(geo.probes.pos.cols(), 0);
  EXPECT_TRUE(geo.arcs.empty());
}

TEST(BuildSasTest, TwoSpheresFullCircle) {
  Matrix3Xd pts(3, 2);
  pts << 0, 2.0, 0, 0, 0, 0;
  ArrayXd sar(2);
  sar << 1.8, 1.5;

  auto [sa, geo] = solve(pts, sar);
  ASSERT_EQ(geo.area.size(), 2);

  const double d = 2.0,
               a0 = (d * d + sar[0] * sar[0] - sar[1] * sar[1]) / (2 * d);
  EXPECT_NEAR(geo.area[sa.order[0] == 0 ? 0 : 1],
              4 * kPi * sar[0] * sar[0] - kTwoPi * sar[0] * (sar[0] - a0),
              1e-12);
  EXPECT_NEAR(geo.area[sa.order[0] == 1 ? 0 : 1],
              4 * kPi * sar[1] * sar[1] - kTwoPi * sar[1] * (sar[1] - (d - a0)),
              1e-12);

  EXPECT_EQ(geo.probes.pos.cols(), 0);
  ASSERT_EQ(geo.arcs.size(), 1);
  EXPECT_EQ(geo.arcs[0].beg, -1);
  EXPECT_EQ(geo.arcs[0].end, -1);
  EXPECT_DOUBLE_EQ(geo.arcs[0].dphi, kTwoPi);
}

TEST(BuildSasTest, CoplanarTriangle) {
  Matrix3Xd pts(3, 3);
  pts << 0, 2.4, 1.2, 0, 0, 2.1, 0, 0, 0;
  ArrayXd sar = ArrayXd::Constant(3, 1.9);

  auto [sa, geo] = solve(pts, sar);
  ASSERT_EQ(geo.area.size(), 3);
  EXPECT_NEAR(geo.area.sum(), sr_total(pts, sar), 1e-2 * geo.area.sum());

  ASSERT_EQ(geo.probes.pos.cols(), 2);
  EXPECT_NEAR(std::abs(geo.probes.pos(2, 0)), std::abs(geo.probes.pos(2, 1)),
              1e-12);
  EXPECT_NEAR(geo.probes.pos(2, 0) + geo.probes.pos(2, 1), 0, 1e-12);
  for (int p = 0; p < 2; ++p) {
    EXPECT_EQ(geo.probes.atoms.degree(p), 3);
    EXPECT_EQ(geo.probes.tan_off.degree(p), 3);
  }

  EXPECT_EQ(geo.arcs.size(), 3);
  for (const SasArc &arc: geo.arcs) {
    EXPECT_GE(arc.beg, 0);
    EXPECT_GE(arc.end, 0);
    EXPECT_NE(arc.beg, arc.end);
    EXPECT_GT(arc.dphi, 0);
    EXPECT_LT(arc.dphi, kTwoPi);
  }
}

TEST(BuildSasTest, CoveredSphere) {
  Matrix3Xd pts(3, 7);
  pts << 0, 1, -1, 0, 0, 0, 0,  //
      0, 0, 0, 1, -1, 0, 0,     //
      0, 0, 0, 0, 0, 1, -1;
  pts *= 2.2;
  ArrayXd sar = ArrayXd::Constant(7, 2.1);
  sar[0] = 1.5;

  auto [sa, geo] = solve(pts, sar);
  ASSERT_EQ(geo.area.size(), 7);
  const int center = static_cast<int>(
      std::find(sa.order.begin(), sa.order.end(), 0) - sa.order.begin());
  ASSERT_LT(center, sa.n_solve);
  EXPECT_NEAR(geo.area[center], 0, 1e-12);
  EXPECT_NEAR(geo.area.sum(), sr_total(pts, sar), 1e-2 * geo.area.sum());
}

TEST(BuildSasTest, FourSphereVertex) {
  const double reach = 2.0;
  Matrix3Xd pts(3, 4);
  pts << 1, 1, -1, -1,  //
      1, -1, 1, -1,     //
      1, -1, -1, 1;
  pts *= reach / std::sqrt(3.0);
  ArrayXd sar = ArrayXd::Constant(4, reach);

  auto [sa, geo] = solve(pts, sar);

  int at_origin = 0;
  for (int p = 0; p < geo.probes.pos.cols(); ++p) {
    if (geo.probes.pos.col(p).norm() > 1e-9)
      continue;

    ++at_origin;
    EXPECT_EQ(geo.probes.atoms.degree(p), 4);
  }
  EXPECT_EQ(at_origin, 1);
  EXPECT_NEAR(geo.area.sum(), sr_total(pts, sar), 1e-2 * geo.area.sum());
}

Matrix3Xd star(const int n_ring, const double polar_deg, const double reach) {
  const double polar = polar_deg * kPi / 180;
  Matrix3Xd pts(3, n_ring + 1);
  pts.col(0) << 0, 0, reach;
  for (int k = 0; k < n_ring; ++k) {
    const double az = kTwoPi * k / n_ring;
    pts.col(k + 1) << reach * std::sin(polar) * std::cos(az),
        reach * std::sin(polar) * std::sin(az), reach * std::cos(polar);
  }
  return pts;
}

int probes_at(const SasGeometry &geo, const Vector3d &x, const int n_atoms) {
  int n = 0;
  for (int p = 0; p < geo.probes.pos.cols(); ++p) {
    if ((geo.probes.pos.col(p) - x).norm() > 1e-9)
      continue;

    ++n;
    EXPECT_EQ(geo.probes.atoms.degree(p), n_atoms);
  }
  return n;
}

TEST(BuildSasTest, FiveSphereVertex) {
  const Matrix3Xd pts = star(4, 110, 2.0);
  const ArrayXd sar = ArrayXd::Constant(5, 2.0);

  auto [sa, geo] = solve(pts, sar);
  EXPECT_EQ(probes_at(geo, Vector3d::Zero(), 5), 1);
  EXPECT_NEAR(geo.area.sum(), sr_total(pts, sar), 1e-2 * geo.area.sum());
}

TEST(BuildSasTest, CoplanarSquareVertex) {
  Matrix3Xd pts(3, 4);
  pts << 1, 1, -1, -1,  //
      1, -1, 1, -1,     //
      0, 0, 0, 0;
  const ArrayXd sar = ArrayXd::Constant(4, 1.5);

  auto [sa, geo] = solve(pts, sar);
  EXPECT_EQ(geo.probes.pos.cols(), 2);
  EXPECT_EQ(probes_at(geo, Vector3d(0, 0, 0.5), 4), 1);
  EXPECT_EQ(probes_at(geo, Vector3d(0, 0, -0.5), 4), 1);
  EXPECT_NEAR(geo.area.sum(), sr_total(pts, sar), 1e-2 * geo.area.sum());
}

TEST(BuildSasTest, ApexTangentAtVertex) {
  const double reach = 2.0, circum = 1.5, rl = 1.8;
  Matrix3Xd pts(3, 4);
  for (int k = 0; k < 3; ++k) {
    const double az = kTwoPi * k / 3;
    pts.col(k) << circum * std::cos(az), circum * std::sin(az), 0;
  }
  const Vector3d x(0, 0, std::sqrt(reach * reach - circum * circum));
  pts.col(3) = x + rl * (x - pts.col(0)).normalized();
  ArrayXd sar(4);
  sar << reach, reach, reach, rl;

  auto [sa, geo] = solve(pts, sar);
  EXPECT_EQ(geo.probes.pos.cols(), 3);
  EXPECT_EQ(probes_at(geo, x, 4), 1);
  EXPECT_NEAR(geo.area.sum(), sr_total(pts, sar), 1e-2 * geo.area.sum());
}

TEST(BuildSasTest, TriangulationSharedAcrossMasks) {
  const int n = 24;
  Matrix3Xd pts(3, n);
  for (int i = 0; i < n; ++i) {
    const int ix = i % 4, iy = (i / 4) % 3, iz = i / 12;
    pts.col(i) << 1.7 * ix, 1.6 * iy + 0.3 * (i % 2), 1.5 * iz + 0.2 * (i % 3);
  }
  ArrayXd sar = ArrayXd::Constant(n, 1.5) + 0.05 * ArrayXd::LinSpaced(n, 0, 1);

  ArrayXb all = ArrayXb::Constant(n, true), half = all;
  half.tail(n / 2).setConstant(false);

  std::optional<SaPrep> sa_all = prepare(pts, sar, all, kRp),
                        sa_half = prepare(pts, sar, half, kRp);
  ASSERT_TRUE(sa_all && sa_half);

  const SasDelaunay del = triangulate(*sa_all);
  const SasGeometry geo_all = build_sas(*sa_all, del),
                    geo_half = build_sas(*sa_half, del);

  ArrayXd area_all(n);
  for (int p = 0; p < sa_all->n_solve; ++p)
    area_all[sa_all->order[p]] = geo_all.area[p];

  EXPECT_EQ(sa_half->n_active, n / 2);
  for (int p = 0; p < sa_half->n_active; ++p)
    EXPECT_DOUBLE_EQ(geo_half.area[p], area_all[sa_half->order[p]]) << p;
}
}  // namespace
}  // namespace internal
}  // namespace nuri
